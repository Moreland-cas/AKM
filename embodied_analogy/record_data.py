import sapien.core as sapien
from sapien.utils.viewer import Viewer
import mplib
import numpy as np
import transforms3d as t3d
import trimesh
# from trimesh_utils import *
from datetime import datetime
from embodied_analogy.utils import *
from embodied_analogy.base_env import BaseEnv
    
class RecordEnv(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
            record_fps=30,
        ):
        super().__init__(phy_timestep)
        self.phy_timestep = phy_timestep
        self.record_fps = record_fps
        
        # setup camera before franka arm
        self.setup_camera()
        self.setup_record()
        
        # setup pygame after camera
        self.setup_pygame() 
        
        # load articulated object
        self.load_articulated_object(index=100051, scale=0.2)
        self.load_franka_arm()
        self.after_try_to_close = 0
        self.setup_planner()
        
    def setup_pygame(self):
        # initialize pygame for keyboard control
        pygame.init()
        resolution = (self.camera.get_width(), self.camera.get_height())
        self.pygame_screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Keyboard Control")
    
    def setup_record(self):
        # 根据record_fps计算record_interval
        self.record_interval = max(int(1. / self.phy_timestep / self.record_fps), 1)
        self.after_try_to_close = 0
        self.recorded_data = {}
        
        # 将相机的内参和外参保存到 self.recorded_data 中
        self.recorded_data["intrinsic"] = self.camera.get_intrinsic_matrix() # [3, 3]
        extrinsic = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
        extrinsic = np.vstack([extrinsic, np.array([0, 0, 0, 1])])
        self.recorded_data["extrinsic"] =  extrinsic
    
    def save_recoreded_data(self):
        assert isinstance(self.recorded_data, dict)
        prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"        
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        np.savez(prefix + f'/{timestamp}.npz', **self.recorded_data)
        
    def step(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        
        self.cur_steps += 1
        self.cur_steps = self.cur_steps % self.record_interval
        
        # 如果机械手臂没有尝试关闭，则不录制
        # 关闭后按照 fps 的帧率录制
        if self.cur_steps == 0 and self.after_try_to_close:
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
                w = self.camera.get_width()
                h = self.camera.get_height()
                u, v = world_to_normalized_uv(cp, K, Tw2c, w, h)
                cp_2d.append(np.array([u, v]))
                
            # record rgb image and display to pygame screen
            rgb_np, depth_np, _, _ = self.capture_rgbd(return_point_cloud=False, visualize=False)
            rgb_np = self.capture_rgb()
            rgb_pil = Image.fromarray(rgb_np)
            update_image(self.pygame_screen, rgb_pil)
            
            cur_dict = {
                "fps": self.record_fps,
                "after_close": self.after_try_to_close,
                "panda_hand_pos": ee_pos,
                "panda_hand_quat": ee_quat,
                "rgb_np": rgb_np, 
                "depth_np": depth_np,
                "contact_points_3d": np.array(cp_3d), # N x 3
                "contact_points_2d": np.array(cp_2d), # N x 2
            }
            if "traj" not in self.recorded_data.keys():
                self.recorded_data["traj"] = []
            self.recorded_data["traj"].append(cur_dict)
            self.recorded_data["object_image"] = None # TODO: add object image (without franka arm)

    def manipulate_and_record(self):
        pos_scale_factor = 0.1
        
        while not self.viewer.closed:
            target_pos, _ = self.get_ee_pose()
            target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
            delta_pos = np.array([0, 0, 0])
            # delta_rpy = np.array([0, 0, 0])
            
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
                    elif event.key == pygame.K_KP0: 
                        self.close_gripper()
                    elif event.key == pygame.K_KP1:  
                        self.open_gripper()
                    elif event.key == pygame.K_KP_ENTER:  
                        self.save_recoreded_data()
                    continue
            
            self.step()
    
if __name__ == '__main__':
    demo = RecordEnv()
    demo.manipulate_and_record()
