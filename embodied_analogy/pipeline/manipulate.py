import numpy as np
from graspnetAPI import Grasp
from scipy.spatial.transform import Rotation as R

from embodied_analogy.environment.base_env import BaseEnv
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    visualize_pc,
    get_depth_mask,
    remove_dir_component
)
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr

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
                initial_state.append(np.deg2rad(0))
            else:
                initial_state.append(cur_joint_state[i])
        self.asset.set_qpos(initial_state)
        self.setup_camera()
        
        self.instruction = instruction
        self.obj_description = self.instruction.split(" ")[-1] # TODO: 改为从 instruction 中提取
    
        # load obj representation
        if obj_repr_path is not None:
            self.obj_repr = Obj_repr.load(obj_repr_path)
    
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
        # Tph2w = Tgrasp2w @ T_with_offset(0.02)
        Tph2w = Tgrasp2w @ T_with_offset(0.014)
        return Tph2w
    
    def manipulate(self, delta_state=0, reserved_distance=0.05, visualize=False):
        from embodied_analogy.estimation.relocalization import relocalization
        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(self.camera_extrinsic)
        
        # 首先进行机械手的 reset, 因为当前可能还处在 explore 阶段末尾的抓取阶段
        # 此时并不需要点云
        self.reset_robot()
        
        # 然后估计出 initial_frame 的 joint_state, 并根据 delta_state 计算出 target_state
        initial_frame_update = relocalization(
            obj_repr=self.obj_repr,
            query_frame=self.obj_repr.initial_frame,
            # 这里可以 update 一下 initial_frame 的 dynamic mask (当然也可以不 update)
            update_query_dynamic=True, 
            # 但是不要 update contact2d 和 3d
            update_query_contact=False,
            visualize=visualize
        )
        self.obj_repr.initial_frame = initial_frame_update
        self.target_state = initial_frame_update.joint_state + delta_state
        
        # 然后估计出 cur_state
        self.base_step()
        cur_frame = self.capture_frame()
        cur_frame = relocalization(
            obj_repr=self.obj_repr,
            query_frame=cur_frame,
            update_query_dynamic=True,
            update_query_contact=True,
            visualize=visualize
        )
        cur_state = cur_frame.joint_state
        
        # 根据 cur_frame 中的 contact3d 选择抓取位姿, 并沿着 joint_dir 进行移动 （仿照 explore_once 函数）
        cur_frame.detect_grasp(visualize=True)
        
        # 对于 dir_out_c 进行定制化修改
        joint_dir = self.obj_repr.joint_dict["joint_dir"]
        if self.obj_repr.joint_dict["joint_type"] == "prismatic":
            dir_out_c = joint_dir
        elif self.obj_repr.joint_dict["joint_type"] == "revolute":
            dir_out_c = remove_dir_component(cur_frame.dir_out, joint_dir, return_normalized=True)
        dir_out_w = Tc2w[:3, :3] @ dir_out_c # 3
        
        if cur_frame.grasp_group is None:
            assert "detected grasp is None"
            import pdb;pdb.set_trace()
        
        grasps_w = cur_frame.grasp_group.transform(Tc2w) # Tgrasp2w
        
        result_pre = None
        depth_mask = get_depth_mask(cur_frame.depth, self.camera_intrinsic, Tw2c, height=0.02)
        pc_collision_c = depth_image_to_pointcloud(cur_frame.depth, cur_frame.obj_mask & depth_mask, self.camera_intrinsic) # N, 3
        pc_colors = cur_frame.rgb[cur_frame.obj_mask & depth_mask]
        pc_collision_w = camera_to_world(pc_collision_c, Tw2c)
        self.planner.update_point_cloud(pc_collision_w)
            
        for grasp_w in grasps_w:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            
            # result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            
            # if (result is not None) and (result_pre is not None):
            if (result_pre is not None):
                break
        
        if result_pre is None:
            print("result_pre is None")
            import pdb;pdb.set_trace()
            
        # if visualize:
        if True:
            contact3d_w = camera_to_world(
                point_camera=cur_frame.contact3d[None],
                extrinsic_matrix=Tw2c
            )[0]
            visualize_pc(
                points=pc_collision_w, 
                colors=pc_colors / 255,
                grasp=grasp, 
                contact_point=contact3d_w, 
                post_contact_dirs=[dir_out_w]
            )
        
        # 实际执行
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(reserved_distance)
        self.close_gripper()
        
        # 在 close gripper 之后再开始录制数据
        joint_dir_c = self.obj_repr.joint_dict["joint_dir"]
        joint_start_c = self.obj_repr.joint_dict["joint_start"]
        joint_dir_w = Tc2w[:3, :3] @ joint_dir_c
        joint_start_w = Tc2w[:3, :3] @ joint_start_c + Tc2w[:3, 3]
        self.move_along_axis(
            joint_type=self.obj_repr.joint_dict["joint_type"],
            joint_axis=joint_dir_w,
            joint_start=joint_start_w,
            moving_distance=self.target_state-cur_state
        )
        
        # TODO: 在这里进行一个状态估计, 输出算法 predict 的当前状态 (改为一个 close-loop 的状态)
        pass
        
        while True:
            self.step()
    def reset_robot_with_pc(self, pc):        
        # 先打开 gripper, 再撤退一段距离
        self.move_forward(-0.05) # 向后撤退 5 cm
        
        # 读取一帧 rgbd， 经过 sam 得到 pc， 对 pc 进行处理
        self.planner.update_point_cloud(pc)
        self.reset_robot()

        # 重置 point cloud
        self.clear_planner_pc()
        
    def evaluate(self):
        # 从环境中获取当前的 joint state
        pass
    

if __name__ == '__main__':
    # drawer
    # obj_config = {
    #     "index": 44962,
    #     "scale": 0.8,
    #     "pose": [1.0, 0., 0.5],
    #     "active_link": "link_2",
    #     "active_joint": "joint_2"
    # }
    
    # microwave
    obj_config = {
        "index": 7221,
        "scale": 0.4,
        "pose": [0.8, 0.1, 0.6],
        "active_link": "link_0",
        "active_joint": "joint_0"
    }
    
    obj_index = obj_config["index"]
    # instruction="open the drawer"
    instruction="open the microwave"
    
    demo = ManipulateEnv(
        obj_config=obj_config,
        obj_repr_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/reconstruct/recon_data.pkl",
        instruction=instruction
    )
    demo.manipulate(
        # delta_state=-0.1,
        delta_state=np.deg2rad(+30),
        visualize=False
    )
    
    while True:
        demo.base_step()
    