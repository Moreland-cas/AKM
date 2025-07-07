import mplib
import pygame
import numpy as np
from PIL import Image
import sapien.core as sapien
import transforms3d as t3d
from embodied_analogy.environment.manipulate_env import ManipulateEnv

    
class HumanControlEnv(ManipulateEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_pygame() 
    
    # get_viewer_param
    def setup_custom_camera(
        self,
        width=640,
        height=480,
        p=[-0.04706102,  0.47101435,  1.0205718],
        q=[0.9624247 ,  0.03577587,  0.21794608, -0.15798205]
    ):
        camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(90), # 1.57
            near=0.1,
            far=100,
        )
        pose = sapien.Pose(p, q)
        camera.set_local_pose(pose)
        self.custom_camera = camera
        
    def capture_custom_frame(self):
        camera = self.custom_camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        # An alias
        # rgba = camera.get_color_rgba()  
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_pil
    
    def setup_pygame(self):
        # initialize pygame for keyboard control
        pygame.init()
        # for sapien 3
        if self.use_sapien2:
            resolution = (self.camera.width, self.camera.height)
        else:
            resolution = (self.camera.get_width(), self.camera.get_height())
        
        self.pygame_screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Keyboard Control")
    
            
    def human_control(self):
        pos_scale_factor = 0.05
        
        while not self.viewer.closed:
            target_pos, target_quat = self.get_ee_pose()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        delta_pos = np.array([1, 0, 0]) 
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_DOWN:
                        delta_pos = np.array([-1, 0, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_LEFT:
                        delta_pos = np.array([0, 1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_RIGHT:
                        delta_pos = np.array([0, -1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP_PLUS:
                        delta_pos = np.array([0, 0, 1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP_MINUS:
                        delta_pos = np.array([0, 0, -1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                        
                    # 旋转变换的部分
                    elif event.key == pygame.K_KP1:  # 绕x轴+30度
                        delta_quat = t3d.euler.euler2quat(np.deg2rad(30), 0, 0, axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP4:  # 绕x轴-30度
                        delta_quat = t3d.euler.euler2quat(np.deg2rad(-30), 0, 0, axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP2:  # 绕y轴+30度
                        delta_quat = t3d.euler.euler2quat(0, np.deg2rad(30), 0, axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP5:  # 绕y轴-30度
                        delta_quat = t3d.euler.euler2quat(0, np.deg2rad(-30), 0, axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP3:  # 绕z轴+30度
                        delta_quat = t3d.euler.euler2quat(0, 0, np.deg2rad(30), axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP6:  # 绕z轴-30度
                        delta_quat = t3d.euler.euler2quat(0, 0, np.deg2rad(-30), axes="syxz")
                        target_quat = t3d.quaternions.qmult(target_quat, delta_quat)
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_x:  # "x"键归位
                        target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(90), np.deg2rad(90), axes="syxz")
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                        
                    # 夹爪控制部分
                    elif event.key == pygame.K_c: 
                        self.close_gripper()
                    elif event.key == pygame.K_o:  
                        self.open_gripper()
                    
                    # 退出 human control 部分
                    elif event.key == pygame.K_c: 
                        return
                    
                    continue
            
            self.base_step()
    