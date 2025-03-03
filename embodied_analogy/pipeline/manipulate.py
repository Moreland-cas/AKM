"""
    应该说给定一个物体表示（由重建算法得到）
    然后给定物体的初始状态, 和要达到的状态
    控制机器人将物体操作到指定状态, 并进行评估

"""
import numpy as np
from graspnetAPI import Grasp
from scipy.spatial.transform import Rotation as R

from embodied_analogy.environment.base_env import BaseEnv
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    compute_bbox_from_pc,
    sample_points_on_bbox_surface,
    visualize_pc
)
initialize_napari()
from embodied_analogy.grasping.anygrasp import (
    detect_grasp_anygrasp,
    sort_grasp_group
)


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
                    self.active_joints[j].set_drive_target(result['position'][i][j])
                    self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                self.step()
    
    def anyGrasp2executable_ph(self, grasp_input, pc_obs, reserved_distance=0.05):
        """
            根据 
        """
        self.planner.update_point_cloud(pc_obs)
        
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
        Tph2w_pre = Tgrasp2w @ T_with_offset(best_offset - reserved_distance)
        
        return Tph2w_pre, Tph2w
    
    def manipulate(self, delta_state=0, reserved_distance=0.05):
        from embodied_analogy.estimation.relocalization import relocalization
        
        # 1) 首先估计出当前的 joint state
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        start_state, query_dynamic, query_dynamic_updated = relocalization(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np,
            query_depth=depth_np, 
            ref_depths=self.obj_repr["depth_seq"], 
            joint_type=self.obj_repr["joint_type"], 
            joint_axis_unit=self.obj_repr["joint_axis"], 
            ref_joint_states=self.obj_repr["joint_states"], 
            ref_dynamics=self.obj_repr["dynamic_mask_seq"],
            lr=5e-3,
            tol=1e-7,
            icp_select_range=0.1,
            text_prompt="drawer",
            negative_points=self.get_points_on_arm()[0],
            visualize=False
        )
        print("current state estimation: ", start_state)      
        print("target state estimation: ", start_state + delta_state)
        
        # 根据当前的 depth 找到一些能 grasp 的地方, 要求最好是落在 moving part 中, 且方向垂直于 moving part
        start_pc_c = depth_image_to_pointcloud(depth_np, query_dynamic == MOVING_LABEL, self.camera_intrinsic) # N, 3
        start_pc_w = camera_to_world(start_pc_c, self.camera_extrinsic)
        joint_axis_c = self.obj_repr["joint_axis"]
        Rc2w = np.linalg.inv(self.camera_extrinsic)[:3, :3]
        joint_axis_w = Rc2w @ joint_axis_c
        joint_axis_outward_w = -joint_axis_w
        start_color = rgb_np[query_dynamic > 0] / 256.
        
        grasp_group = detect_grasp_anygrasp(
            start_pc_w, 
            start_color, 
            joint_axis=joint_axis_outward_w, 
            visualize=True
        )
        
        # 筛选出评分比较高的 grasp
        contact_region_c = depth_image_to_pointcloud(depth_np, query_dynamic_updated == MOVING_LABEL, self.camera_intrinsic)
        contact_region_w = camera_to_world(contact_region_c, self.camera_extrinsic)
        self.sorted_grasps, _ = sort_grasp_group(
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
        
        self.open_gripper()
        
        # 将抓取姿势从 Tgrasp2w 转换到 Tph2w, 从而可以移动 panda_hand
        for grasp in self.sorted_grasps:
            # visualize_pc(start_pc_w, start_color, grasp)
            
            Tph2w_pre, _ = self.anyGrasp2executable_ph(grasp, start_pc_w, reserved_distance=reserved_distance)
            
            if Tph2w_pre is None:
                continue
            visualize_pc(start_pc_w, start_color, grasp)
            
            # 先移动到 pre_grasp_pose
            start_bbox_min, start_bbox_max = compute_bbox_from_pc(start_pc_w, offset=0.02)
            start_collision_points = sample_points_on_bbox_surface(start_bbox_min, start_bbox_max, num_samples=1000)
            self.planner.update_point_cloud(start_collision_points)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            self.follow_path(result_pre)
            
            # 再从 pre_grasp_pose 向前移动一段距离
            self.planner.update_point_cloud(np.array([[0, 0, -1]]))
            self.move_forward(reserved_distance)
            break
        
        self.close_gripper()
        
        # 关闭点云, 保持抓取姿势，向着 axis 移动（此时需要关闭点云遮挡）
        self.planner.update_point_cloud(np.array([[0, 0, -1]]))
        self.move_along_axis(joint_axis_outward_w, delta_state)
        
        # TODO：在这里加入 arm 的 reset 过程
        
        # 在这里进行一个状态估计, 输出算法 predict 的当前状态
        # TODO：同时输出 gt 的 joint 状态
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        state_after_manip, _, _ = relocalization(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np,
            query_depth=depth_np, 
            ref_depths=self.obj_repr["depth_seq"], 
            joint_type=self.obj_repr["joint_type"], 
            joint_axis_unit=self.obj_repr["joint_axis"], 
            ref_joint_states=self.obj_repr["joint_states"], 
            ref_dynamics=self.obj_repr["dynamic_mask_seq"],
            lr=5e-3,
            tol=1e-7,
            icp_select_range=0.1,
            text_prompt="drawer",
            negative_points=self.get_points_on_arm()[0],
            visualize=False
        )
        print("state estimation after manipulate: ", state_after_manip)   
        
        while True:
            self.step()
    def reset_franka_arm_with_pc(self):        
        # 先打开 gripper, 再撤退一段距离
        self.move_forward(-0.05) # 向后撤退 5 cm
        
        # 读取一帧 rgbd， 经过 sam 得到 pc， 对 pc 进行处理
        self.planner.update_point_cloud(pc)
        self.reset_franka_arm()

        # 重置 point cloud
        self.planner.update_point_cloud(np.array([]))
        
    def evaluate(self):
        # 从环境中获取当前的 joint state
        pass
    
    

if __name__ == '__main__':
    demo = ManipulateEnv()
    demo.manipulate(delta_state=-0.1)
    
    # while True:
    #     demo.step()
    