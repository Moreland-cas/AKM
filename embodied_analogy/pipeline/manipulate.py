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
    visualize_pc,
    get_depth_mask
)
initialize_napari()
from embodied_analogy.grasping.anygrasp import (
    detect_grasp_anygrasp,
    sort_grasp_group
)

class ManipulateEnv(BaseEnv):
    def __init__(
            self,
            obj_config,
            instruction,
            obj_repr_path=None,
            phy_timestep=1/250.,
            use_sapien2=True
        ):
        super().__init__(
            phy_timestep=phy_timestep,
            use_sapien2=use_sapien2
        )
        self.load_robot()
        self.load_object(obj_config)
        
        # 随机初始化物体对应 joint 的状态
        cur_joint_state = self.asset.get_qpos()
        active_joint_names = [joint.name for joint in self.asset.get_active_joints()]
        initial_state = []
        for i, joint_name in enumerate(active_joint_names):
            if joint_name == obj_config["active_joint"]:
                limit = self.asset.get_active_joints()[i].get_limits() # (2, )
                # initial_state.append(0.1)
                initial_state.append(0.0)
            else:
                initial_state.append(cur_joint_state[i])
        self.asset.set_qpos(initial_state)
        self.setup_camera()
        
        self.instruction = instruction
        self.obj_description = self.instruction.split(" ")[-1] # TODO: 改为从 instruction 中提取
    
        # load obj representation
        if obj_repr_path is not None:
            self.load_obj_repr(obj_repr_path)
    
    def load_obj_repr(self, obj_repr_path, visualize=False):
        self.obj_repr = np.load(obj_repr_path)
        if visualize:
            pass
        
    def anyGrasp2ph(self, grasp):
        """
            从 anygrasp 输出的 grasp 中提取出 Tph2w, 也就是做一个 Tgrasp2w 到 Tph2w 的转换
            grasp: Grasp 对象
        """
        # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
        def T_with_offset(offset):
            Tph2grasp = np.array([
                [0, 0, 1, -(0.045 + 0.069) + offset],  # TODO: offset 设置为 0.014?
                [0, 1, 0, 0], 
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            return Tph2grasp
        
        R_grasp2w = grasp.rotation_matrix # 3, 3
        t_grasp2w = grasp.translation # 3
        Tgrasp2w = np.hstack((R_grasp2w, t_grasp2w[..., None])) # 3, 4
        Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
        # Tph2w = Tgrasp2w @ T_with_offset(0.03)
        Tph2w = Tgrasp2w @ T_with_offset(0.014)
        return Tph2w
    
    def manipulate(self, delta_state=0, reserved_distance=0.05, visualize=False):
        from embodied_analogy.estimation.relocalization import relocalization
        
        # 1) 首先估计出当前的 joint state
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        start_state, obj_mask, query_dynamic = relocalization(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np,
            query_depth=depth_np, 
            ref_depths=self.obj_repr["depth_seq"], 
            joint_type=self.obj_repr["joint_type"], 
            joint_axis_c=self.obj_repr["joint_axis_c"], 
            ref_joint_states=self.obj_repr["joint_states"], 
            ref_dynamics=self.obj_repr["dynamic_seq"],
            lr=1e-3,
            tol=1e-7,
            icp_select_range=0.1,
            obj_description=self.obj_description,
            negative_points=self.get_points_on_arm()[0],
            visualize=False
        )
        print("start state: ", start_state)      
        
        # 根据当前的 depth 找到一些能 grasp 的地方, 要求最好是落在 moving part 中, 且方向垂直于 moving part
        # TODO: 在这里进行一个 depth_np 的进一步过滤
        
        depth_mask = get_depth_mask(depth_np, self.camera_intrinsic, self.camera_extrinsic, height=0.02)
        start_pc_c = depth_image_to_pointcloud(depth_np, obj_mask & depth_mask, self.camera_intrinsic) # N, 3
        start_pc_w = camera_to_world(start_pc_c, self.camera_extrinsic)
        
        joint_axis_w = self.obj_repr["joint_axis_w"]
        start_color = rgb_np[obj_mask] / 256.
        
        grasp_group = detect_grasp_anygrasp(
            start_pc_w, 
            start_color, 
            dir_out=joint_axis_w, 
            visualize=False
        )
        
        # 筛选出评分比较高的 grasp
        contact_region_c = depth_image_to_pointcloud(depth_np, query_dynamic == MOVING_LABEL, self.camera_intrinsic)
        contact_region_w = camera_to_world(contact_region_c, self.camera_extrinsic)
        sorted_grasps, _ = sort_grasp_group(
            grasp_group=grasp_group, 
            contact_region=contact_region_w, 
            axis=joint_axis_w, 
            grasp_pre_filter=False
        )

        if False:
            visualize_pc(
                np.concatenate([start_pc_w, contact_region_w + 0.02], axis=0), 
                np.concatenate([start_color, np.array([[1, 0, 0]] * len(contact_region_w))], axis=0),
                sorted_grasps[:10]
            )
        
        # 将抓取姿势从 Tgrasp2w 转换到 Tph2w, 从而可以移动 panda_hand
        for grasp in sorted_grasps:
            
            # 旋转 grasp 使得其尽肯能平行于 joint axis
            grasp = self.get_rotated_grasp(grasp, axis_out_w=joint_axis_w)
            Tph2w = self.anyGrasp2ph(grasp)
            
            # visualize_pc(start_pc_w, start_color, grasp)
            
            result_test = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if not result_test:
                continue
            
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            # visualize_pc(start_pc_w, start_color, grasp)
            
            # 先移动到 pre_grasp_pose
            self.planner.update_point_cloud(start_pc_w)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if not result_pre:
                continue
            
            # visualize_pc(start_pc_w, start_color, grasp)
            self.follow_path(result_pre)
            
            self.open_gripper()
            self.clear_planner_pc()
            self.move_forward(reserved_distance)
            self.close_gripper()
            
            self.move_along_axis(joint_axis_w, delta_state)
            break
        
        # 在这里进行一个状态估计, 输出算法 predict 的当前状态
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        end_state, _, _ = relocalization(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np,
            query_depth=depth_np, 
            ref_depths=self.obj_repr["depth_seq"], 
            joint_type=self.obj_repr["joint_type"], 
            joint_axis_c=self.obj_repr["joint_axis_c"], 
            ref_joint_states=self.obj_repr["joint_states"], 
            ref_dynamics=self.obj_repr["dynamic_seq"],
            lr=5e-3,
            tol=1e-7,
            icp_select_range=0.1,
            obj_description="drawer",
            negative_points=self.get_points_on_arm()[0],
            visualize=True
        )
        print("predicted delta: ", end_state - start_state)   
        
        while True:
            self.step()
    def reset_franka_arm_with_pc(self, pc):        
        # 先打开 gripper, 再撤退一段距离
        self.move_forward(-0.05) # 向后撤退 5 cm
        
        # 读取一帧 rgbd， 经过 sam 得到 pc， 对 pc 进行处理
        self.planner.update_point_cloud(pc)
        self.reset_franka_arm()

        # 重置 point cloud
        self.clear_planner_pc()
        
    def evaluate(self):
        # 从环境中获取当前的 joint state
        pass
    
    

if __name__ == '__main__':
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        "active_joint": "joint_2"
    }
    demo = ManipulateEnv(
        obj_config=obj_config,
        obj_repr_path="/home/zby/Programs/Embodied_Analogy/assets/tmp/reconstruct/obj_repr.npz",
        instruction="open the drawer"
    )
    demo.manipulate(delta_state=0.1)
    
    # while True:
    #     demo.step()
    