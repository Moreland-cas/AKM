import os
import yaml
import logging
import numpy as np
import sapien.core as sapien
from PIL import Image, ImageColor
from sapien.utils.viewer import Viewer
from transforms3d.quaternions import qmult
from scipy.spatial.transform import Rotation as R

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
                
        self.offscreen = self.base_env_cfg["offscreen"]
        phy_timestep = self.base_env_cfg["phy_timestep"]
        self.phy_timestep = phy_timestep
        planner_timestep = self.base_env_cfg["planner_timestep"]
        use_sapien2 = self.base_env_cfg["use_sapien2"]
        self.use_sapien2 = use_sapien2
        
        self.asset_prefix = ASSET_PATH
        self.cur_steps = 0
        
        self.engine = sapien.Engine()  # Create a physical simulation engine
        self.renderer = sapien.SapienRenderer(offscreen_only=self.offscreen)  # Create a Vulkan renderer
        self.engine.set_renderer(self.renderer)  # Bind the renderer and the engine
        
        scene_config = sapien.SceneConfig()
        # follow video axis aligned (or RGBManip?)
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        # scene_config.contact_offset = 0.0001 # 0.1 mm
        scene_config.contact_offset = 0.02 
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 1
        
        self.scene = self.engine.create_scene(scene_config)
        self.phy_timestep = phy_timestep
        self.scene.set_timestep(phy_timestep)
        if planner_timestep is None:
            self.planner_timestep = self.phy_timestep
        else:
            self.planner_timestep = planner_timestep
        self.scene.add_ground(0)
        
        # add some lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        
        if not self.offscreen:
            self.viewer = Viewer(
                renderer=self.renderer,
                resolutions=(800, 600)
            )  # Create a viewer (window)
            self.viewer.set_scene(self.scene)  # Bind the viewer and the scene
            self.viewer.set_camera_xyz(x=-1, y=1, z=2)
            self.viewer.set_camera_rpy(r=0, p=-np.arctan2(1, 1), y=np.arctan2(1, 1))
            self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
            self.viewer.toggle_axes(False)
        
        if use_sapien2:
            self.capture_rgb = self.capture_rgb_sapien2
            self.capture_rgbd = self.capture_rgbd_sapien2
        else:
            self.capture_rgb = self.capture_rgb_sapien3
            self.capture_rgbd = self.capture_rgbd_sapien3
            
        self.step = self.base_step
        self.load_camera()
        
        # setup logger
        self.setup_logger(
            cfg=cfg, 
            txt_path=os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "log.txt"
            )
        )
    
    def setup_logger(self, cfg, txt_path):
        # os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        logger = logging.getLogger(f'logger_{cfg["task_cfg"]["task_id"]}')
        # print(f'**** {txt_path}  {cfg["task_cfg"]["task_id"]} ****')
        if logger.hasHandlers():
            logger.handlers.clear()
        
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
        
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(txt_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger
    
    def get_viewer_param(self):
        p = self.viewer.window.get_camera_position()
        q = qmult(self.viewer.window.get_camera_rotation(), [0.5, -0.5, 0.5, 0.5])
        return p, q
    
    def load_camera(self, pose=None):
        # camera config
        near, far = 0.1, 100
        width, height = 800, 600
        camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(90), # 1.57
            near=near,
            far=far,
        )
        if pose is None:
            pose = sapien.Pose(
                [-0.04706102,  0.47101435,  1.0205718],
                [0.9624247 ,  0.03577587,  0.21794608, -0.15798205]
            )
        camera.set_local_pose(pose)
        self.camera = camera
        
        self.camera_intrinsic = self.camera.get_intrinsic_matrix() # [3, 3], K
        Tw2c = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
        self.camera_extrinsic = Tw2c
    
    def capture_rgb_sapien2(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # render rgb image
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        return rgb_numpy
    
    def capture_rgb_sapien3(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        rgba = camera.get_picture("Color")  
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        return rgb_numpy
    
    def capture_rgbd_sapien2(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        rgba = camera.get_float_texture("Color") 
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
    
        # get pointcloud
        position = camera.get_float_texture("Position")  
        points_opengl = position[..., :3][position[..., 3] < 1] # num_valid_points, 3
        points_color = rgba[position[..., 3] < 1] # num_valid_points, 4
        # opengl camera to world, must be called after scene.update_render()
        model_matrix = camera.get_model_matrix() 
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3] # N. 3
        points_world = points_world.astype(np.float64)
        points_color = (np.clip(points_color, 0, 1)).astype(np.float64)
        points_color = points_color[..., :3]
        
        if visualize:
            visualize_pc(points_world, points_color, None)
            
        # get depth image
        depth = -position[..., 2] # H, W, in meters
        depth_numpy = np.array(depth)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
        
    def capture_rgbd_sapien3(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # get rgb image
        rgba = camera.get_picture("Color")  
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
    
        # get pointcloud
        position = camera.get_picture("Position")  # [H, W, 4]
        points_opengl = position[..., :3][position[..., 3] < 1] # num_valid_points, 3
        points_color = rgba[position[..., 3] < 1] # num_valid_points, 4
        model_matrix = camera.get_model_matrix() # opengl camera to world, must be called after scene.update_render()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3] # N. 3
        points_world = points_world.astype(np.float64)
        points_color = (np.clip(points_color, 0, 1)).astype(np.float64)
        points_color = points_color[..., :3]
        
        if visualize:
            visualize_pc(points_world, points_color, None)
            
        # get depth image
        depth = -position[..., 2] # H, W, in meters
        depth_numpy = np.array(depth)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
    
    def capture_segmentation(self):
        camera = self.camera
        camera.take_picture()
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        actor_np = seg_labels[..., 1].astype(np.uint8)  # actor-level [H, W]
        return actor_np # [H, W]
    
    def capture_frame(self, visualize=False) -> Frame:
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        frame = Frame(
            rgb=rgb_np,
            depth=depth_np,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
        )
        if visualize:
            frame.visualize()
            
        return frame
    
    def base_step(self):
        self.scene.step()
        self.scene.update_render() 
        if not self.offscreen:
            self.viewer.render()
        self.cur_steps += 1
    
    def delete(self):
        if not self.offscreen:
            self.viewer.close()
