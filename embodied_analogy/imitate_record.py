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
    
    def find_nearest_grasp(self, grasp_group, contact_point):
        '''
            grasp_group: graspnetAPI 
            contact_point: (3, )
        '''
        # 找到 grasp_group 中距离 contact_point 最近的 grasp 并返回
        # 首先根据 grasp 的 score 排序, 筛选出前20
        grasp_group = grasp_group.nms().sort_by_score()
        grasp_group = grasp_group[0:50]
        
        # 找到距离 contact_point 最近的 grasp
        translations = grasp_group.translations # N, 3
        distances = np.linalg.norm(translations - contact_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_index = int(nearest_index)
        return grasp_group[nearest_index]
    
    def imitate_from_record(self):
        # 让物体落下
        for i in range(100):
            self.step()
            
        # 获取target场景初始位置的rgb图
        target_img_np, target_depth_np, target_pc, target_pc_color = self.capture_rgbd(
            return_point_cloud=True,
            visualize=False,
        )
        # 测试抓取
        # visualize_pc(target_pc, target_pc_color)
        
        # mask 掉地面
        ground_mask = target_pc[:, 2] > 0.01
        target_pc = target_pc[ground_mask] 
        target_pc_color = target_pc_color[ground_mask]
        
        self.grasp_group = self.detect_grasp_anygrasp(target_pc, target_pc_color, visualize=False)
        # visualize_pc(target_pc, target_pc_color, self.grasp_group)
        
        # load franka after capture first image so that franka pc are not in the captured data
        self.load_franka_arm()
        
        # 让机械手臂复原
        for i in range(100):
            self.step()
        
        self.setup_planner()
        
        # update pointcloud to avoid collision
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
            nms_threshold=0.002,
            visualize=False
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
            contact_point = camera_to_world(point_camera, extrinsic_matric)
            
            # 如果 contact_point 没有落在物体上
            if actor_level_seg[row, col] < 2:
                self.spawn_cube(contact_point, color=[1, 0, 0])
                print(f"{i} not in mask")
                continue
            else:
                self.spawn_cube(contact_point, color=[0, 1, 0])
            
            # 找到与 contact_point 最近的 grasp, 即得到了 Tgrasp2w, 但这里的 grasp 和 panda_hand 坐标系还不同
            grasp = self.find_nearest_grasp(self.grasp_group, contact_point)
            visualize_pc(target_pc, target_pc_color, grasp)
            Tgrasp2w_R = grasp.rotation_matrix # 3, 3
            Tgrasp2w_t = grasp.translation # 3
            Tgrasp2w = np.hstack((Tgrasp2w_R, Tgrasp2w_t[..., None])) # 3, 4
            Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
            
            # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
            Tph2grasp = np.array([
                [0, 0, 1, -0.07], 
                [0, 1, 0, 0], 
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            Tph2w = Tgrasp2w @ Tph2grasp # 4, 4
            
            self.open_gripper()
            
            # target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
            # target_pose = mplib.Pose(p=contact_point, q=target_quat)
            target_pose = mplib.Pose(Tph2w)
            
            if False:
                step = 0
                ph = self.load_panda_hand(pos=Tph2w[:3, 3] + [0, 0, 2], quat=t3d.quaternions.mat2quat(Tph2w[:3, :3]))
                while not self.viewer.closed:
                    step += 1
                    step = step % 200
                    if step % 200 == 0:
                        self.scene.remove_articulation(ph)
                        ph = self.load_panda_hand(pos=Tph2w[:3, 3] + [0, 0, 0.3], quat=t3d.quaternions.mat2quat(Tph2w[:3, :3]))
                    self.step()   
            
            status = self.move_to_pose(target_pose, wrt_world=True)
            
            if status < 0:
                print(f"{i} not reachable")
                continue
                
            self.close_gripper()
            
            cur_pos, cur_quat = self.get_ee_pose()
            # 只执行一半的lift motion
            for i in range(int(len(ph_pos) * 0.25)):
                self.move_to_pose(mplib.Pose(p=cur_pos + ph_pos[i], q=t3d.quaternions.mat2quat(Tph2w[:3, :3])), wrt_world=True)
                
            if self.is_task_success():
                break
            else:
                self.reset_franka_arm()
        
        while not self.viewer.closed:
            self.step()        

if __name__ == '__main__':
    demo = ImitateEnv()
    demo.imitate_from_record()
    