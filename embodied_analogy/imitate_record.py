import sapien
import mplib
from embodied_analogy.base_env import BaseEnv
from embodied_analogy.process_record import RecordDataReader
from embodied_analogy.utils import draw_red_dot, uv_to_camera, camera_to_world, visualize_pc
from embodied_analogy.image_matching import match_points
from PIL import Image
import numpy as np
import transforms3d as t3d

class ImitateEnv(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
            record_path_prefix="/home/zby/Programs/Embodied_Analogy/assets/recorded_data",
            record_file_name="/2024-12-12_11-23-38.npz"
        ):
        super().__init__(phy_timestep)
        
        # self.load_articulated_object(index=100015, pose=[0.4, 0.4, 0.2], scale=0.4)
        self.load_articulated_object(index=100051, pose=[0.4, 0.4, 0.2], scale=0.2)
        self.setup_camera()
        
        self.DataReader = RecordDataReader(record_path_prefix, record_file_name)
        self.DataReader.process_data()
        
    def spawn_cube(self, pose, color=[1, 0, 0]):
        # cube
        builder = self.scene.create_actor_builder()
        
        # builder.use_density = False
        builder.add_box_collision(half_size=[0, 0, 0])
        builder.add_box_visual(half_size=[0.01, 0.01, 0.01], material=color)
        cube = builder.build(name='cube')
        cube.set_pose(sapien.Pose(pose))
        
        # cube.get_links()[0].set_collision_enabled(False)

        # 取消重力
        # cube.set_gravity(False)

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        print("n_step:", n_step)
        for i in range(n_step):  
            for _ in range(int(250 / 30.) + 1):
                qf = self.robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)
                for j in range(7):
                    self.active_joints[j].set_drive_target(result['position'][i][j])
                    self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                self.step()
            
    def update_pointcloud_for_avoidance(self, point_cloud):
        # point_cloud: N, 3 in world space
        self.planner.update_point_cloud(point_cloud)
    
    def is_task_success(self):
        return False
    def imitate_from_record(self):
        # 让物体落下
        for i in range(100):
            self.step()
            
        # 获取target场景初始位置的rgb图
        target_img_np, target_depth_np, target_pc, target_pc_color = self.capture_rgb_depth(
            return_point_cloud=True,
            show_pc=False,
        )
        # 测试抓取
        # visualize_pc(target_pc, target_pc_color)
        
        # mask 掉地面
        ground_mask = target_pc[:, 2] > 0.01
        target_pc = target_pc[ground_mask] 
        target_pc_color = target_pc_color[ground_mask]
        
        self.detect_grasp_anygrasp(target_pc, target_pc_color, vis=True)
        
        # load franka after capture first image so that franka pc are not in the captured data
        self.load_franka_arm()
        
        # 让机械手臂复原
        for i in range(100):
            self.step()
        
        self.setup_planner()
        
        # update pointcloud to avoid collision
        # pc = trimesh.points.PointCloud(target_pc)
        # pc.show()
        self.update_pointcloud_for_avoidance(target_pc)
        
        target_img_pil = Image.fromarray(target_img_np)
        source_img_pil = self.DataReader.source_img
        source_u, source_v = self.DataReader.first_cp_2d
        
        target_uvs, target_probs, similarity_map = match_points(
            source_img_pil, 
            target_img_pil, 
            (source_u, source_v), 
            top_k=500, 
            max_return=500,
            resize=798, 
            device="cuda",
            is_pil=True,
            show_matching=True,
            nms_threshold=0.002
        )
        
        # for i in range(len(target_uvs)):
        #     target_uv = target_uvs[i]
        #     img = draw_red_dot(target_img_pil, target_uv[0], target_uv[1], radius=1)
        #     img.save(self.asset_prefix + f"/tmp/{i}.png")
        
        # 使用 segmentation map 进行过滤
        # similarity_map = None # H, W, 0-1之间
        actor_level_seg = self.capture_segmentation()
        
        ph_pos = self.DataReader.panda_hand_pos
        ph_quat = self.DataReader.panda_hand_quat
         
            
        for i in range(len(target_uvs)):
            # print(i)
            # 根据深度图找到三维空间点
            target_u, target_v = target_uvs[i]
            # target_u, target_v = target_uvs[34]
            # target_u, target_v = 0.4868421052631579, 0.5153508771929824
            depth_h, depth_w = target_depth_np.shape
            
            row = int(target_v * depth_h)
            col = int(target_u * depth_w)
            
            depth = target_depth_np[row, col]
            K = self.camera.get_intrinsic_matrix()
            
            point_camera = uv_to_camera(target_u, target_v, depth, K, depth_w, depth_h)
            extrinsic_matric = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
            contact_pos = camera_to_world(point_camera, extrinsic_matric)
            
            if actor_level_seg[row, col] < 2:
                self.spawn_cube(contact_pos, color=[1, 0, 0])
                print(f"{i} not in mask")
                continue
            else:
                self.spawn_cube(contact_pos, color=[0, 1, 0])
                
            contact_pos += np.array([0, 0, 0.08])
            self.open_gripper()
            
            target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
            target_pose = mplib.Pose(p=contact_pos, q=target_quat)
            status = self.move_to_pose(target_pose, wrt_world=True)
            
            if status < 0:
                print(f"{i} not reachable")
                continue
                
            self.close_gripper()
            
            cur_pos, cur_quat = self.get_ee_pose()
            # 只执行一半的lift motion
            for i in range(int(len(ph_pos) * 0.5)):
                self.move_to_pose(mplib.Pose(p=cur_pos + ph_pos[i], q=ph_quat[i]), wrt_world=True)
                
            if self.is_task_success():
                break
            else:
                self.reset_franka_arm()
        
        while not self.viewer.closed:
            self.step()        

if __name__ == '__main__':
    demo = ImitateEnv()
    demo.imitate_from_record()
    