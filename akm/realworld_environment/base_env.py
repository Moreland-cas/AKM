import cv2
from PIL import Image
import yaml
import os
import logging
import sapien.core as sapien
import numpy as np
from akm.representation.basic_structure import Frame
from akm.utility.constants import ASSET_PATH
from akm.utility.utils import visualize_pc

class BaseEnv():
    def __init__(self, cfg):     
        self.cfg = cfg
        self.exp_cfg = cfg["exp_cfg"]
        self.task_cfg = cfg["task_cfg"]
        self.base_env_cfg = cfg["base_env_cfg"]
        
        if self.exp_cfg["save_cfg"]:
            save_cfg_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                f'{self.task_cfg["task_id"]}.yaml'
            )
            os.makedirs(os.path.dirname(save_cfg_path), exist_ok=True)
            with open(save_cfg_path, "w") as f:
                yaml.dump(self.cfg, f, default_flow_style=False, sort_keys=False)
        
        self.asset_prefix = ASSET_PATH
        self.cur_steps = 0
        self.realworld = self.exp_cfg["realworld"]
        self.phy_timestep = self.base_env_cfg["phy_timestep"]
        self.planner_timestep = self.base_env_cfg["planner_timestep"]
        
        # 尽管是 realworld, 还是要 load 一个最基本的 scene, 从而方便后续 load franka 的 urdf 供 mplib 使用
        self.engine = sapien.Engine()
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.phy_timestep)
        self.scene.add_ground(0)
        
        # 设置 logger
        self.setup_logger(
            cfg=cfg, 
            txt_path=os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "log.txt"
            )
        )
        
        self.step = self.base_step
        self.load_camera()
    
    def setup_logger(self, cfg, txt_path):
        # 将 txt_path 生成
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        logger = logging.getLogger(f'logger_{cfg["task_cfg"]["task_id"]}')
        
        level = cfg["logging"]["level"]
        if level == "DEBUG":
            logger.setLevel(logging.DEBUG)
        elif level == "INFO":            
            logger.setLevel(logging.INFO)
        elif level == "WARNING":            
            logger.setLevel(logging.WARNING)
        elif level == "ERROR":            
            logger.setLevel(logging.ERROR)
        elif level == "CRITICAL":            
            logger.setLevel(logging.CRITICAL)
        
        # 设置格式化
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        
        # 设置流
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(txt_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger
    
    def load_camera(self):
        """
        定义 self.pipeline, self.camera_intrinsic, self.camera_extrinsic
        开启 realsense 视频流, 进行内参的读取和外参的标定
        """
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        
        self.frame_width = 640
        self.frame_height = 480
        
        # 启动管道
        profile = pipeline.start(config)
        
        # 设置 rgb 相机
        rgb_sensor = profile.get_device().first_color_sensor()
        rgb_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

        # 设置深度预设 (Default)
        """
        00: Custom
        01: Default
        02: Hand
        03: High Accuracy
        04: High Density
        """
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 1)     # Default
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

        # 设置 depth filters
        dec_filter = rs.decimation_filter()  # 解码滤波
        dec_filter.set_option(rs.option.filter_magnitude, 2)  

        spatial_filter = rs.spatial_filter()  # 空间滤波
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        spatial_filter.set_option(rs.option.holes_fill, 0)

        """
        00: Disabled
        01: Valid in 8/8
        02: Valid in 2/last 3
        03: Valid in 2/last 4
        04: Valid in 2/8
        05: Valid in 1/last 2
        06: Valid in 1/last 5
        07: Valid in 1/8
        08: Always on
        """
        temporal_filter = rs.temporal_filter()  # 时间滤波
        temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        temporal_filter.set_option(rs.option.holes_fill, 7)  # Valid in 1/8
        
        threshold_filter = rs.threshold_filter()  # 阈值滤波
        threshold_filter.set_option(rs.option.min_distance, 0.2)  # 最小深度 0.1 米
        threshold_filter.set_option(rs.option.max_distance, 4.0)
        
        """
        00: Fill from Left
        01: Farest from around
        02: Nearest from around
        """
        hole_filling = rs.hole_filling_filter()  # 孔洞填充
        hole_filling.set_option(rs.option.holes_fill, 1)

        # 设置 alignment 模块
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # 保存 handle
        self.pipeline = pipeline
        self.depth_filters = [
            # dec_filter,
            spatial_filter,
            temporal_filter,
            threshold_filter,
            hole_filling
        ]
        
        # calibrate 文件夹下读取内外参
        try:
            calib_folder = self.cfg["base_env_cfg"]["calib_folder"]
            K_path = os.path.join(calib_folder, "tmp/K.npy")
            Tc2w_path = os.path.join(calib_folder, "tmp/Tc2w.npy")
            
            self.camera_intrinsic = np.load(K_path)
            self.camera_extrinsic = np.linalg.inv(np.load(Tc2w_path)) # Tw2c
        except Exception as e:
            print("Catched Exception: ", str(e))
            self.camera_intrinsic = None
            self.camera_extrinsic = None
        
        # 暖机, 自适应估计曝光参数
        for _ in range(30):
            self.capture_frame(visualize=False, robot_mask=False)
        
    def capture_rgb(self):
        """
        读取一个 numpy 的 rgb 图片, uint8
        """
        rgb_numpy = self.captured_rgbd()[0]
        return rgb_numpy
    
    def capture_rgbd(self):
        """
        从 self.pipeline 和 self.align 中得到一个 RGB+D
        RGB: np uint8
        Depth: m
        """
        while True:
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            frames = self.align.process(frames)
            
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            
            for filter in self.depth_filters:
                depth_frame = filter.process(depth_frame)
                
            # depth_frame = frames.get_depth_frame() 
            # color_frame = frames.get_color_frame()
            
            depth_image_np = np.asanyarray(depth_frame.get_data()) / 1000.
            color_image_np = np.asanyarray(color_frame.get_data())
            break
            
        return color_image_np, depth_image_np
    
    def capture_frame(self, visualize=False):
        rgb_np, depth_np = self.capture_rgbd()
        
        frame = Frame(
            rgb=rgb_np,
            depth=depth_np,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
        )
        if visualize:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(frame.depth * 1000, alpha=0.1), cv2.COLORMAP_JET
            )
            depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            # Image.fromarray(depth_colormap).show()
            pcs, colors = frame.get_env_pc(
                use_robot_mask=False,
                use_height_filter=True,
                world_frame=True
            )
            visualize_pc(
                points=pcs,
                point_size=5,
                colors=colors/255.
            )
        return frame
        
    def base_step(self):
        """
        真实世界是自然进行 step 的, 因此需要在虚拟世界中通过 sleep 同步一下
        """
        # import time
        # time.sleep(self.phy_timestep)
        # self.cur_steps += 1
        pass
    
    def delete(self):
        pass
        

if __name__ == "__main__":
    import yaml
    cfg_path = "/home/user/Programs/akm/akm/realworld_environment/calibration/test.yaml"
    
    # open
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    env = BaseEnv(cfg)
    
    # env.step()
    for i in range(10):
        f = env.capture_frame(False)
        
    for i in range(10):
        f = env.capture_frame(True)
    
    
    # for i in range(100):
    # while True:
    #     env.step()
        # env.capture_rgbd_sapien2(visualize=True)