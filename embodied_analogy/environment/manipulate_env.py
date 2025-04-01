import numpy as np

from embodied_analogy.environment.reconstruct_env import ReconEnv
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    camera_to_world,
    initialize_napari,
    joint_data_to_transform_np,
)
initialize_napari()
from embodied_analogy.representation.basic_structure import Frame


class ManipulateEnv(ReconEnv):
    def __init__(
            self,
            base_cfg={
                "phy_timestep": 1/250.,
                "planner_timestep": None,
                "use_sapien2": True 
            },
            robot_cfg={},
            explore_cfg={
                "record_fps": 30,
                "pertubation_distance": 0.1,
                "max_tries": 10,
            },
            recon_cfg={
                "num_initial_pts": 1000,
                "num_kframes": 5,
                "fine_lr": 1e-3
            },
            manip_cfg={
                "reloc_lr": 3e-3,
                "reserved_distance": 0.05
            },
            task_cfg={
                "instruction": "open the drawer",
                "obj_description": "drawer",
                "delta_state": 0.2,
                "obj_cfg": {
                    "index": 44962,
                    "scale": 0.8,
                    "pose": [1.0, 0., 0.5],
                    "active_link": "link_2",
                    "active_joint": "joint_2"
                },
            }
        ):        
        super().__init__(
            base_cfg=base_cfg,
            robot_cfg=robot_cfg,
            task_cfg=task_cfg,
            explore_cfg=explore_cfg,
            recon_cfg=recon_cfg
        )
        self.reloc_lr = manip_cfg["reloc_lr"]
        self.reserved_distance = manip_cfg["reserved_distance"]
        self.delta_state = task_cfg["delta_state"]
    
    def set_goal(self, visualize=False):
        """
        对于 obj_repr.initial_frame 进行重定位, 并根据要操作的值, 计算 target joint state
        """    
        self.obj_repr.initial_frame = self.obj_repr.reloc(
            query_frame=self.obj_repr.initial_frame,
            update_query_dynamic=True,
            reloc_lr=self.reloc_lr,
            visualize=visualize
        )
        self.initial_state = self.obj_repr.initial_frame.joint_state
        self.target_state = self.initial_state + self.delta_state
    
    def transfer_ph_pose(self, ref_frame: Frame, tgt_frame: Frame, visualize=False):
        """
        将 ref_frame 中的 panda_hand grasp_pose 转换到 target_frame 中去
        
        NOTE: 
            由于现在的抓取模块不是很强, 所以需要这个函数, 也就是说我们 transfer 的不是 contact_3d, 
            而是 explore 阶段已经证实比较好的一个 panda_hand grasp pose
        """
        Tph2w_ref = ref_frame.Tph2w
        Tph2c_ref = self.obj_repr.Tw2c @ Tph2w_ref
        
        # Tref2tgt 是 camera 坐标系下的一个变换
        Tref2tgt_c = joint_data_to_transform_np(
            joint_type=self.obj_repr.fine_joint_dict["joint_type"],
            joint_dir=self.obj_repr.fine_joint_dict["joint_dir"],
            joint_start=self.obj_repr.fine_joint_dict["joint_start"],
            joint_state_ref2tgt=tgt_frame.joint_state-ref_frame.joint_state
        )
        
        Tph2c_tgt = Tref2tgt_c @ Tph2c_ref
        Tph2w_tgt = np.linalg.inv(self.obj_repr.Tw2c) @ Tph2c_tgt
        tgt_frame.Tph2w = Tph2w_tgt
        
        if visualize:
            pass
    def manipulate(self, visualize=False):
        """
        manipulate 的执行逻辑:
        首先重定位出 initial frame 的状态, 并根据 instruction 得到 target state
        
        对机械臂进行归位, 对当前状态进行定位, 计算出当前帧的抓取位姿 (使用 manipulate first frame)
        
        移动到该位姿, 并根据 target_state 进行操作
        """
        Tc2w = np.linalg.inv(self.camera_extrinsic)
        
        self.set_goal()
        
        # 首先进行机械手的 reset, 因为当前可能还处在 explore 阶段末尾的抓取阶段
        self.reset_robot_with_pc(pc=None)
        
        # 然后估计出 cur_state
        self.base_step()
        cur_frame = self.capture_frame()
        cur_frame = self.obj_repr.reloc(
            query_frame=cur_frame,
            update_query_dynamic=True,
            reloc_lr=self.reloc_lr,
            visualize=visualize
        )
        self.cur_state = cur_frame.joint_state
        
        self.transfer_ph_pose(
            ref_frame=self.obj_repr.kframes[0],
            tgt_frame=cur_frame,
            visualize=visualize
        )
        
        pc_c, _ = cur_frame.get_pc(only_obj=True)
        pc_w = camera_to_world(pc_c, self.obj_repr.Tw2c)
        self.planner.update_point_cloud(pc_w)
        
        Tph2w_pre = self.get_translated_ph(cur_frame.Tph2w, -self.reserved_distance)
        result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
        
        # 实际执行
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(self.reserved_distance)
        self.close_gripper()
        
        # 转换 joint_dict 到世界坐标系
        self.move_along_axis(
            joint_type=self.obj_repr.fine_joint_dict["joint_type"],
            joint_axis=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_dir"],
            joint_start=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_start"] + Tc2w[:3, 3],
            moving_distance=self.target_state-cur_frame.joint_state
        )
        
    def manipulate_main(self):
        self.explore_stage()
        self.recon_stage()
        self.manipulate()
        

if __name__ == '__main__':
    task_cfg={
        "instruction": "open the drawer",
        "obj_description": "drawer",
        "delta_state": -0.2,
        "obj_cfg": {
            "index": 44962,
            "scale": 0.8,
            "pose": [1.0, 0., 0.5],
            "active_link": "link_2",
            "active_joint": "joint_2",
            "joint_limit": None,
            "init_state": 0.3
        },
    }
    # task_cfg={
    #     "instruction": "open the microwave",
    #     "obj_description": "microwave",
    #     "delta_state": np.deg2rad(30),
    #     "obj_cfg": {
    #         "index": 7221,
    #         "scale": 0.4,
    #         "pose": [0.8, 0.1, 0.6],
    #         "active_link": "link_0",
    #         "active_joint": "joint_0",
    #         "joint_limit": None,
    #         "init_state": 0
    #     }
    # }
    me = ManipulateEnv(task_cfg=task_cfg)
    # me.obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/reconstruct/recon_data.pkl")
    me.manipulate_main()
    
    while True:
        me.base_step()
    