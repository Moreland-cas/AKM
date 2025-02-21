"""
    应该说给定一个物体表示（由重建算法得到）
    然后给定物体的初始状态, 和要达到的状态
    控制机器人将物体操作到指定状态, 并进行评估

"""
import mplib
from graspnetAPI import Grasp
from PIL import Image
import numpy as np
import transforms3d as t3d
from embodied_analogy.environment.base_env import BaseEnv
from embodied_analogy.utility import *
from embodied_analogy.perception import *
from embodied_analogy.estimation.relocalization import relocalization
from embodied_analogy.estimation.utils import find_correspondences
from scipy.spatial.transform import Rotation as R

def find_nearest_grasp(grasp_group, contact_point):
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
    
def score_grasp_group(grasp_group, contact_region, joint_axis, grasp_pre_filter=False):
    '''
        找到离 contact region 中点最近的 grasp, 且越是垂直于 joint_axis 越好
        grasp_group: from graspnetAPI 
        contact_point: (N, 3), 也即是 moving part
    '''
    if grasp_pre_filter: # 保留前 50 的 grasp
        grasp_group = grasp_group.nms().sort_by_score()
        grasp_group = grasp_group[0:50]
        
    t_grasp2w = grasp_group.translations # N, 3
    pred_scores = grasp_group.scores # N
    
    _, distances, _ = find_correspondences(t_grasp2w, contact_region) # N
    distance_scores = np.exp(-2 * distances) 
    
    R_grasp2w = grasp_group.rotation_matrices # N, 3, 3
    def R2unitAxis(rotation_matrix):
        rotation = R.from_matrix(rotation_matrix)
        axis_angle = rotation.as_rotvec()
        rotation_axis = axis_angle / np.linalg.norm(axis_angle)
        return rotation_axis
    pred_axis = np.array([R2unitAxis(R_grasp2w[i]) for i in range(len(R_grasp2w))]) # N, 3
    angle_scores = np.abs(np.sum(pred_axis * joint_axis, axis=-1)) # N
    
    grasp_scores = pred_scores * distance_scores * angle_scores # N
    
    # 找到距离 contact_point 最近的 grasp
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group, grasp_scores[index]


class ManipulateEnv(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
            use_sapien2=True
        ):
        super().__init__(
            phy_timestep=phy_timestep,
            use_sapien2=use_sapien2
        )
        self.load_franka_arm()
            
        obj_config = {
            "index": 44962,
            "scale": 0.8,
            "pose": [1.0, 0., 0.5],
            "active_link": "link_1",
            "active_joint": "joint_1"
        }
        self.load_articulated_object(obj_config)
        
        # 随机初始化物体对应 joint 的状态
        cur_joint_state = self.asset.get_qpos()
        active_joint_names = [joint.name for joint in self.asset.get_active_joints()]
        initial_state = []
        for i, joint_name in enumerate(active_joint_names):
            if joint_name == obj_config["active_joint"]:
                limit = self.asset.get_active_joints()[i].get_limits() # (2, )
                initial_state.append(0.1)
            else:
                initial_state.append(cur_joint_state[i])
        self.asset.set_qpos(initial_state)
        
        self.setup_camera()
        
        # load obj representation
        recon_data_path = "/home/zby/Programs/Embodied_Analogy/assets/tmp/reconstructed_data.npz"
        self.obj_repr = np.load(recon_data_path)

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        print("n_step:", n_step)
        for i in range(n_step):  
            for _ in range(int(250 / 30.) + 1):
                qf = self.robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                for j in range(7):
                # for j in range(9):    
                    self.active_joints[j].set_drive_target(result['position'][i][j])
                    self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                self.step()
    
    def grasp2ph(self, grasp_input):
        # ph is panda hand for short
        # 输入一个 grasp, 输出为该 grasp 包含的 Tgrasp2w 转换为 Tph2w 的结果
        grasp = Grasp()
        grasp.grasp_array = np.copy(grasp_input.grasp_array)
        
        R_grasp2w = grasp.rotation_matrix # 3, 3
        t_grasp2w = grasp.translation # 3
        Tgrasp2w = np.hstack((R_grasp2w, t_grasp2w[..., None])) # 3, 4
        Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
        
        # 先 dynamic 的调整 approaching vector, 使其尽可能平行于 joint axis
        # 这个需要在 grasp 坐标系下调整比较好
        # 我们要做的是, 调整 R_grasp2w 使得其平行于 joint axis
        # 考虑到 grasp 坐标系, 我们需要绕着 y 轴旋转, 使得 x 轴平行于 joint axis
        
        # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
        def T_with_offset(offset):
            Tph2grasp = np.array([
                [0, 0, 1, -(0.045 + 0.069) + offset], 
                [0, 1, 0, 0], 
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            return Tph2grasp

        # 再 dynamic 的调整 offset, 使得找到一个离物体表面最近, 且可以规划得到的 Tph2w
        # offset_list = [0.05, 0.04, 0.03, 0.02, 0.01, 0.0]  # 从近到远的顺序, 对应 offset 从大到小的试
        offset_list = [0.05, 0.04, 0.03, 0.02]
        best_offset = None
        for offset in offset_list:
            result = self.plan_path(target_pose=Tgrasp2w @ T_with_offset(offset), wrt_world=True)
            if result is not None:
                best_offset = offset
                break
        
        if best_offset is None:
            return None, None
        # 最后得到 Tph2w_pre, 为远离 target grasp pose 一定距离的版本, 防止碰撞发生
        Tph2w = Tgrasp2w @ T_with_offset(best_offset)
        Tph2w_pre = Tgrasp2w @ T_with_offset(best_offset - 0.05)
        
        return Tph2w_pre, Tph2w
    
    def manipulate(self, delta_state=0):
        # 1) 首先估计出当前的 joint state
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        _, initial_mask = run_grounded_sam(
            rgb_image=rgb_np,
            text_prompt="drawer",
            positive_points=None, 
            negative_points=self.get_points_on_arm()[0], # N, 2
            num_iterations=5,
            acceptable_thr=0.9,
            visualize=False
        )
        query_dynamic = initial_mask.astype(np.int32) * MOVING_LABEL
        
        start_state, query_dynamic_updated = relocalization(
            K=self.camera_intrinsic, 
            query_dynamic=query_dynamic,
            query_depth=depth_np, 
            ref_depths=self.obj_repr["depth_seq"], 
            joint_type=self.obj_repr["joint_type"], 
            joint_axis_unit=self.obj_repr["joint_axis"], 
            ref_joint_states=self.obj_repr["joint_states"], 
            ref_dynamics=self.obj_repr["dynamic_mask_seq"],
            lr=5e-3,
            tol=1e-7,
            icp_select_range=0.1,
            visualize=True
        )
        print("cur state est: ", start_state)      
        
        # 根据 target_joint_state 计算出 delta joint state
        target_state = start_state + delta_state
        
        # 根据当前的 depth 找到一些能 grasp 的地方, 要求最好是落在 moving part 中, 且方向垂直于 moving part
        start_pc_c = depth_image_to_pointcloud(depth_np, query_dynamic_updated == MOVING_LABEL, self.camera_intrinsic) # N, 3
        # start_pc_c = depth_image_to_pointcloud(depth_np, query_dynamic == MOVING_LABEL, self.camera_intrinsic) # N, 3
        start_pc_w = camera_to_world(start_pc_c, self.camera_extrinsic)
        joint_axis_c = self.obj_repr["joint_axis"]
        Rc2w = np.linalg.inv(self.camera_extrinsic)[:3, :3]
        joint_axis_w = Rc2w @ joint_axis_c
        joint_axis_outward_w = -joint_axis_w
        start_color = rgb_np[query_dynamic > 0] / 256.
        
        grasp_group = self.detect_grasp_anygrasp(
            start_pc_w, 
            start_color, 
            joint_axis=joint_axis_outward_w, 
            visualize=True
        )
        # self.reset_franka_arm()
        
        # 筛选出评分比较高的 grasp
        contact_region_c = depth_image_to_pointcloud(depth_np, query_dynamic_updated == MOVING_LABEL, self.camera_intrinsic)
        contact_region_w = camera_to_world(contact_region_c, self.camera_extrinsic)
        self.sorted_grasps, grasp_scores = score_grasp_group(
            grasp_group=grasp_group, 
            contact_region=contact_region_w, 
            joint_axis=joint_axis_w, 
            grasp_pre_filter=False
        )

        if False:
            visualize_pc(
                np.concatenate([start_pc_w, contact_region_w + 0.02], axis=0), 
                np.concatenate([start_color, np.array([[1, 0, 0]] * len(contact_region_w))], axis=0),
                self.sorted_grasps[:10]
            )
        
        # 更新 planner 的点云, 让 panda_hand 可以成功到达抓取位置
        self.planner.update_point_cloud(start_pc_w, resolution=0.01)
        # visualize_pc(start_pc_w)
        self.open_gripper()
        # 将抓取姿势从 Tgrasp2w 转换到 Tph2w, 从而可以移动 panda_hand
        for grasp in self.sorted_grasps:
            # visualize_pc(start_pc_w, start_color, grasp)
            
            Tph2w_pre, Tph2w = self.grasp2ph(grasp)
            
            if Tph2w_pre is None:
                continue
            # visualize_pc(start_pc_w, start_color, grasp)
            # 先移动到 result_pre
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            self.follow_path(result_pre)
            
            # 再移动到 result
            result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            # visualize_pc(start_pc_w, start_color, grasp)
            self.follow_path(result)
            break
        
        self.close_gripper()
        
        # 关闭点云, 保持抓取姿势，向着 axis 移动（此时需要关闭点云遮挡）
        
        # 进行 reset, 并进行状态估计
        while True:
            self.step()
            
        pass
    
    def evaluate(self):
        # 从环境中获取当前的 joint state
        pass
    
    def manipulate_deprecated(self):
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
        
        # while not self.viewer.closed:
        #     self.step()      
            
        self.setup_planner()
        
        # update pointcloud to avoid collision
        self.update_pointcloud_for_avoidance(target_pc)
        
        target_img_pil = Image.fromarray(target_img_np)
        source_img_pil = self.DataReader.get_img(idx=0)
        source_u, source_v = self.DataReader.first_cp_2d
        
        similarity_map = match_points_dino_featup(
        # similarity_map = match_points_dift_sd(
            source_img_pil, 
            target_img_pil, 
            (source_u, source_v), 
            resize=224 * 3, 
            device="cuda",
            visualize=True
        )
        
        self.similarity_map = SimilarityMap(similarity_map, alpha=20)
        target_uvs = self.similarity_map.sample(num_samples=50, visualize=True)
        
        # 使用 segmentation map 进行过滤
        # similarity_map = None # H, W, 0-1之间
        actor_level_seg = self.capture_segmentation()
        
        ph_pos = self.DataReader.panda_hand_pos
        ph_quat = self.DataReader.panda_hand_quat
         
            
        for i in range(len(target_uvs)):
            # 根据深度图找到三维空间点
            target_u, target_v = target_uvs[i]
            depth_h, depth_w = target_depth_np.shape
            
            row = int(target_v * depth_h)
            col = int(target_u * depth_w)
            
            depth = target_depth_np[row, col]
            K = self.camera.get_intrinsic_matrix()
            
            point_camera = image_to_camera(target_u, target_v, depth, K, depth_w, depth_h)
            extrinsic_matric = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
            contact_point = camera_to_world(point_camera, extrinsic_matric)
            
            # 如果 contact_point 没有落在物体上
            if actor_level_seg[row, col] < 2:
                # self.spawn_cube(contact_point, color=[1, 0, 0])
                print(f"{i} not in mask")
                continue
            else:
                pass
                # self.spawn_cube(contact_point, color=[0, 1, 0])
            
            # 找到与 contact_point 最近的 grasp, 即得到了 Tgrasp2w, 但这里的 grasp 和 panda_hand 坐标系还不同
            grasp = find_nearest_grasp(self.grasp_group, contact_point)
            visualize_pc(target_pc, target_pc_color, grasp)
            Tgrasp2w_R = grasp.rotation_matrix # 3, 3
            Tgrasp2w_t = grasp.translation # 3
            Tgrasp2w = np.hstack((Tgrasp2w_R, Tgrasp2w_t[..., None])) # 3, 4
            Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
            
            # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
            offset = 0.03 # [0.01, 0.02, 0.03, 0.04, 0.05]
            Tph2grasp = np.array([
                [0, 0, 1, -(0.045 + 0.069 - offset)], 
                [0, 1, 0, 0], 
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            Tph2w = Tgrasp2w @ Tph2grasp # 4, 4
            
            self.open_gripper()
            
            # target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
            # target_pose = mplib.Pose(p=contact_point, q=target_quat)
            target_pose = mplib.Pose(Tph2w)
            status = self.move_to_pose(target_pose, wrt_world=True)
            
            if status < 0:
                print(f"{i} not reachable")
                continue
            
            self.close_gripper()
            
            # traj imitation
            cur_pos, cur_quat = self.get_ee_pose()
            for i in range(int(len(ph_pos) * 0.25)):
                self.move_to_pose(mplib.Pose(p=cur_pos + ph_pos[i], q=t3d.quaternions.mat2quat(Tph2w[:3, :3])), wrt_world=True)
                
            self.reset_franka_arm()
        
        while not self.viewer.closed:
            self.step()        

if __name__ == '__main__':
    demo = ManipulateEnv()
    
    demo.manipulate()
    
    # while True:
    #     demo.step()
    