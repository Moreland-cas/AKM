import numpy as np
from transforms3d.quaternions import qmult
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
from embodied_analogy.representation.basic_structure import Frame

from embodied_analogy.utility.utils import visualize_pc

class BaseEnv():
    def __init__(
            self,
            cfg,
            offscreen=False
        ):        
        self.cfg = cfg
        self.offscreen = offscreen
        phy_timestep = cfg["phy_timestep"]
        planner_timestep = cfg["planner_timestep"]
        use_sapien2 = cfg["use_sapien2"]
        
        self.asset_prefix = "/home/zby/Programs/Embodied_Analogy/assets"
        self.cur_steps = 0
        
        self.engine = sapien.Engine()  # Create a physical simulation engine
        self.renderer = sapien.SapienRenderer(offscreen_only=offscreen)  # Create a Vulkan renderer
        self.engine.set_renderer(self.renderer)  # Bind the renderer and the engine
        if False:
            from sapien.core import renderer as R
            self.renderer_context: R.Context = self.renderer._internal_context
        
        scene_config = sapien.SceneConfig()
        # follow video axis aligned (or RGBManip?)
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        # scene_config.contact_offset = 0.0001 # 0.1 mm
        scene_config.contact_offset = 0.02 # TODO: RGBManip 设置的为 0.02
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
        
        if not offscreen:
            self.viewer = Viewer(self.renderer)  # Create a viewer (window)
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
    
    def get_viewer_param(self):
        """
        返回
        """
        p = self.viewer.window.get_camera_position()
        q = qmult(self.viewer.window.get_camera_rotation(), [0.5, -0.5, 0.5, 0.5])
        return p, q
    
    def load_camera(self, pose=None):
        # camera config
        near, far = 0.1, 100
        width, height = 800, 600
        
        """ 
            intrinsic matrix should be:
            W, 0, W//2
            0, W, H//2
            0, 0, 1
            fx = fy = W
            fovy = 2 * np.arctan(H / (2*W))
        """
        
        camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(90), # 1.57
            near=near,
            far=far,
        )
        """
            set pose method 1
            sapien中的相机坐标系为: forward(x), left(y) and up(z)
            Camera to World 也就是相机在世界坐标系下的位姿
            C2W 中的 t 是相机在世界坐标系下的坐标
            C2W 中的 R 的三列从左到右依次是相机坐标系的 x,y,z 轴在世界坐标系下的坐标向量
        """
        # cam_pos = np.array([-1, 1, 2])
        # forward = -cam_pos / np.linalg.norm(cam_pos)
        # left = np.cross([0, 0, 1], forward)
        # left = left / np.linalg.norm(left)
        # up = np.cross(forward, left)
        # mat44 = np.eye(4)
        # mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        # mat44[:3, 3] = cam_pos
        # camera.entity.set_pose(sapien.Pose(mat44)) # C2W
        
        """
            set pose method 2
            这里的 set_local_pose 其实是指定了 Tc2w
        """
        if pose is None:
            pose = sapien.Pose(
                [-0.04706102,  0.47101435,  1.0205718],
                [0.9624247 ,  0.03577587,  0.21794608, -0.15798205]
            )
        camera.set_local_pose(pose)
        self.camera = camera
        
        # 记录相机的内参和外参
        self.camera_intrinsic = self.camera.get_intrinsic_matrix() # [3, 3], K
        Tw2c = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
        self.camera_extrinsic = Tw2c
    
    def capture_rgb_sapien2(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        # An alias
        # rgba = camera.get_color_rgba()  
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    
    def capture_rgb_sapien3(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    
    def capture_rgbd_sapien2(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染过程
        # get rgb image
        rgba = camera.get_float_texture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
    
        # get pointcloud
        position = camera.get_float_texture("Position")  # [H, W, 4], 格式为(x, y, z, render_depth), 其中 render_depth < 1 的点是有效的
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
        # depth_numpy = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_numpy)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
        
    def capture_rgbd_sapien3(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染过程
        # get rgb image
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
    
        # get pointcloud
        position = camera.get_picture("Position")  # [H, W, 4], 格式为(x, y, z, render_depth), 其中 render_depth < 1 的点是有效的
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
        # depth_numpy = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_numpy)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
    
    def capture_segmentation(self):
        camera = self.camera
        camera.take_picture()
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        mesh_np = seg_labels[..., 0].astype(np.uint8)  # mesh-level [H, W]
        actor_np = seg_labels[..., 1].astype(np.uint8)  # actor-level [H, W]
        
        # Or you can use aliases below
        # label0_image = camera.get_visual_segmentation()
        # label1_image = camera.get_actor_segmentation()
        
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array(
            [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        )
        mesh_pil = Image.fromarray(color_palette[mesh_np])
        actor_pil = Image.fromarray(color_palette[actor_np])
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
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        if not self.offscreen:
            self.viewer.render()
        self.cur_steps += 1


if __name__ == "__main__":
    env = BaseEnv()
    
    env.step()
    env.capture_frame()
    
    # for i in range(100):
    while True:
        env.step()
        # env.capture_rgbd_sapien2(visualize=True)