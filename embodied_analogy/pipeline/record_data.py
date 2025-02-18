import sapien.core as sapien
from sapien.utils.viewer import Viewer
import mplib
import numpy as np
import transforms3d as t3d
import trimesh
# from trimesh_utils import *
from datetime import datetime
from embodied_analogy.utility.utils import *
from embodied_analogy.environment.base_env import BaseEnv
    
class RecordEnv(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
            record_fps=30,
            use_sapien2=True
        ):
        super().__init__(
            phy_timestep,
            use_sapien2
        )
        self.phy_timestep = phy_timestep
        self.record_fps = record_fps
        
        # 根据record_fps计算record_interval
        self.record_interval = max(int(1. / self.phy_timestep / self.record_fps), 1)
        self.recorded_data = {}
        self.start_recording = False
        
        # setup camera before franka arm
        self.setup_camera()
        
        # setup pygame after camera
        self.setup_pygame() 
        
        # setup articulated object and franka arm
        # self.load_articulated_object()                
        # self.load_franka_arm()
        if use_sapien2:
            self.step = self.step_sapien2
        else:
            self.step = self.step_sapien3
        
    def setup_pygame(self):
        # initialize pygame for keyboard control
        pygame.init()
        # for sapien 3
        # resolution = (self.camera.get_width(), self.camera.get_height())
        
        # for sapien 2
        resolution = (self.camera.width, self.camera.height)
        self.pygame_screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Keyboard Control")
    
    def save_recoreded_data(self):
        assert isinstance(self.recorded_data, dict)
        prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"        
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        np.savez(prefix + f'/{timestamp}.npz', **self.recorded_data)
        
    def step_sapien3(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        
        self.cur_steps += 1
        self.cur_steps = self.cur_steps % self.record_interval
        
        # 如果机械手臂没有尝试关闭，则不录制
        # 关闭后按照 fps 的帧率录制
        if self.cur_steps == 0 and self.start_recording:
            # pose of panda_hand link
            ee_pos, ee_quat = self.get_ee_pose() # np array of size 3 and 4
            
            # contact point in 3d
            cp_3d = []
            valid_contact_link = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
            contacts = self.scene.get_contacts()
            for contact in contacts:
                name0, name1 = contact.bodies[0].entity.name, contact.bodies[1].entity.name
                if (name0 not in valid_contact_link) and (name1 not in valid_contact_link): 
                    continue
                for point in contact.points:
                    cp_3d.append(point.position) # numpy array in world frame
            
            # contact point projected in 2d camera (in uv pixel coordinates, normalized to [0, 1])
            cp_2d = []
            for cp in cp_3d:
                K = self.recorded_data["intrinsic"]
                Tw2c = self.recorded_data["extrinsic"]
                # w = self.camera.get_width()
                # h = self.camera.get_height()
                w = self.camera.width
                h = self.camera.height
                uv = world_to_image(cp[None], K, Tw2c, w, h, normalized_uv=True) # B, 2
                u, v = uv[0]
                cp_2d.append(np.array([u, v]))
                
            # record rgb image and display to pygame screen
            rgb_np, depth_np, _, _ = self.capture_rgbd(return_pc=False, visualize=False)
            rgb_np = self.capture_rgb()
            rgb_pil = Image.fromarray(rgb_np)
            update_image(self.pygame_screen, rgb_pil)
            
            # 在这里添加当前帧的 franka_arm 上的点的 franka_tracks3d 和 franka_tracks2d
            franka_tracks2d, franka_tracks3d = self.get_points_on_arm()
            
            cur_dict = {
                "fps": self.record_fps,
                "panda_hand_pos": ee_pos,
                "panda_hand_quat": ee_quat,
                "rgb_np": rgb_np, 
                "depth_np": depth_np,
                "contact_points_3d": np.array(cp_3d), # N x 3
                "contact_points_2d": np.array(cp_2d), # N x 2
                "franka_tracks2d": franka_tracks2d, # N x 2
                "franka_tracks3d": franka_tracks3d
            }
            if "traj" not in self.recorded_data.keys():
                self.recorded_data["traj"] = []
            self.recorded_data["traj"].append(cur_dict)

    def step_sapien2(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        
        self.cur_steps += 1
        self.cur_steps = self.cur_steps % self.record_interval
        
        # 打印 joint relative states, Tinit2cur
        print(self.get_joint_state())
        
        # 如果机械手臂没有尝试关闭，则不录制
        # 关闭后按照 fps 的帧率录制
        if self.cur_steps == 0 and self.start_recording:
            # pose of panda_hand link
            ee_pos, ee_quat = self.get_ee_pose() # np array of size 3 and 4
                
            # record rgb image and display to pygame screen
            rgb_np, depth_np, _, _ = self.capture_rgbd(return_pc=False, visualize=False)
            rgb_np = self.capture_rgb()
            rgb_pil = Image.fromarray(rgb_np)
            update_image(self.pygame_screen, rgb_pil)
            
            # 在这里添加当前帧的 franka_arm 上的点的 franka_tracks3d 和 franka_tracks2d
            franka_tracks2d, franka_tracks3d = self.get_points_on_arm()
            
            cur_dict = {
                "fps": self.record_fps,
                "panda_hand_pos": ee_pos,
                "panda_hand_quat": ee_quat,
                "rgb_np": rgb_np, 
                "depth_np": depth_np,
                # "contact_points_3d": np.array(cp_3d), # N x 3
                # "contact_points_2d": np.array(cp_2d), # N x 2
                "franka_tracks2d": franka_tracks2d, # N x 2
                "franka_tracks3d": franka_tracks3d
            }
            if "traj" not in self.recorded_data.keys():
                self.recorded_data["traj"] = []
            self.recorded_data["traj"].append(cur_dict)
            
    def manipulate_and_record(self):
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
                    elif event.key == pygame.K_x:  # "r"键归位
                        target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(90), np.deg2rad(90), axes="syxz")
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                        
                    # 夹爪控制部分
                    elif event.key == pygame.K_c: 
                        self.close_gripper()
                    elif event.key == pygame.K_o:  
                        self.open_gripper()
                        
                    # 数据保存部分
                    elif event.key == pygame.K_e:  
                        self.save_recoreded_data()
                        
                    elif event.key == pygame.K_r:  
                        self.start_recording = True
                        
                    continue
            
            self.step()
    
if __name__ == '__main__':
    record_env = RecordEnv()
    
    # drawer
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_1",
        "activate_joint": "joint_1"
    }
    
    # door 
    # obj_config = {
    #     "index": 9280,
    #     "scale": 0.7,
    #     "pose": [0.6, 0., 0.5],
    #     "active_link": "link_1",
    #     "activate_joint": "joint_0"
    # }
    
    # pot
    # obj_config = {
    #     "index": 100015,
    #     "scale": 0.3,
    #     "pose": [0.5, 0., 0.2],
    #     "active_link": "link_1",
    #     "activate_joint": "joint_0"
    # }
    
    record_env.load_articulated_object(obj_config)
    # record_env.load_articulated_object(index=9280, scale=0.7, pose=[0.6, 0., 0.4])
    record_env.load_franka_arm()
    # while True:
    #     record_env.step()
    record_env.manipulate_and_record()
