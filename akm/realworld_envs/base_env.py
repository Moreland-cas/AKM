
import os
import cv2
import yaml
import logging
import numpy as np
import sapien.core as sapien

from akm.utility.utils import visualize_pc
from akm.utility.constants import ASSET_PATH
from akm.representation.basic_structure import Frame


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
        
        # Even though it is realworld, we still need to load a basic scene to facilitate the subsequent loading of Franka's urdf for mplib to use.
        self.engine = sapien.Engine()
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(self.phy_timestep)
        self.scene.add_ground(0)
        
        # setup logger
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
        
        # setup formatter
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        
        # setup stream
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(txt_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger
    
    def load_camera(self):
        """
        Define self.pipeline, self.camera_intrinsic, and self.camera_extrinsic
        Enable the RealSense video stream, read the intrinsic parameters, and calibrate the extrinsic parameters.
        """
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        
        self.frame_width = 640
        self.frame_height = 480
        
        # activate pipeline 
        profile = pipeline.start(config)
        
        # setup RGB camera
        rgb_sensor = profile.get_device().first_color_sensor()
        rgb_sensor.set_option(rs.option.enable_auto_exposure, 1.0)

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

        # depth filters
        dec_filter = rs.decimation_filter()  
        dec_filter.set_option(rs.option.filter_magnitude, 2)  

        spatial_filter = rs.spatial_filter()  
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
        temporal_filter = rs.temporal_filter()  
        temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        temporal_filter.set_option(rs.option.holes_fill, 7)  # Valid in 1/8
        
        threshold_filter = rs.threshold_filter()  
        threshold_filter.set_option(rs.option.min_distance, 0.2)  
        threshold_filter.set_option(rs.option.max_distance, 4.0)
        
        """
        00: Fill from Left
        01: Farest from around
        02: Nearest from around
        """
        hole_filling = rs.hole_filling_filter()  
        hole_filling.set_option(rs.option.holes_fill, 1)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.pipeline = pipeline
        self.depth_filters = [
            # dec_filter,
            spatial_filter,
            temporal_filter,
            threshold_filter,
            hole_filling
        ]
        
        # read intrinsic and extrinsic
        try:
            calib_folder = self.cfg["base_env_cfg"]["calib_folder"]
            K_path = os.path.join(calib_folder, "tmp/K.npy")
            Tc2w_path = os.path.join(calib_folder, "tmp/Tc2w.npy")
            
            self.camera_intrinsic = np.load(K_path)
            Tc2w = np.load(Tc2w_path)
            # NOTE: TODO refine Tc2w, camera 在 world 的 x 坐标上应该 + 5cm
            # Tc2w[0, -1] += 0.04
            self.camera_extrinsic = np.linalg.inv(Tc2w) # Tw2c
        except Exception as e:
            print("Catched Exception: ", str(e))
            self.camera_intrinsic = None
            self.camera_extrinsic = None
        
        # Warm up, adaptively estimate exposure parameters
        # for _ in range(30):
            # self.capture_frame(visualize=False, robot_mask=False, Tph2w=False)
        
    def capture_rgb(self):
        """
        Read a numpy RGB image, uint8
        """
        rgb_numpy = self.captured_rgbd()[0]
        return rgb_numpy
    
    def capture_rgbd(self):
        """
        Get an RGB+D value from self.pipeline and self.align
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
        pass
    
    def delete(self):
        pass
        