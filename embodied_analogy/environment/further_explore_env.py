import os
import json
import pickle
import math
import numpy as np
from embodied_analogy.utility.utils import (
    camera_to_world,
    initialize_napari,
    visualize_pc,
)
initialize_napari()
from embodied_analogy.environment.obj_env import ObjEnv
from embodied_analogy.environment.reconstruct_env import ReconEnv
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.representation.traj import Traj, Trajs


class FurtherExploreEnv(ReconEnv):
    def __init__(
            self,
            cfg
        ):        
        # TODO: cfg 需要包含 obj_folder_path_further_explore
        
        ObjEnv().__init__(cfg=cfg)
        self.cfg = cfg
        print("loading further explore env, using cfg:", cfg)
        
        # self.record_fps = cfg["record_fps"]
        # self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = cfg["pertubation_distance"]
        self.valid_thresh = cfg["valid_thresh"]
        
        # self.instruction = cfg["instruction"]
        
        # self.update_sigma = cfg["update_sigma"]
        # self.max_tries = cfg["max_tries"]
        # self.obj_description = cfg["obj_description"]
        # self.has_valid_explore = False
        
        # load reconstructed obj_repr
        self.obj_repr = Obj_repr.load(os.path.join(cfg["obj_folder_path_reconstruct"], "obj_repr.npy"))
        
        # 读取和保存所用的变量
        # self.joint_type = cfg["joint_type"]
        self.save_prefix = cfg["obj_folder_path_further_explore"]
        os.makedirs(self.save_prefix, exist_ok=True)
        
        # FurtherExplore 阶段独特的参数
        assert self.obj_repr.kframes[0].joint_state == 0
        if self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            self.reloc_freq = np.deg2rad(10) # 10 degree
        elif self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            self.reloc_freq = 0.1 # 1 dm
        self.stable_thresh = 0.1 * self.reloc_freq

        # 一些中间变量
        self.num_further_explore = 0
        self.max_further_explore = 20

    def get_stable_grasp_proposal(self):
        """
        返回基于当前状态的最好的 grasp
        要求在 cur_state 下首先检测可行的 grasp, 然后利用 dynamic_mask 和 历史记录对于 grasp 进行筛选, 返回最好的 grasp proposal
        """
        pass

    def make_a_move(self, cur_state, delta_state, manip_type):
        # 返回的是 一个 list of traj, 最后一个 traj 的 valid 一般是 False
        trajs = []
        # TODO 在这里面增加 self.num_further_explore
        return trajs
    
    def further_explore_loop(self):
        # 初始化 explore_traj
        self.obj_repr: Obj_repr
        joint_type = self.obj_repr.fine_joint_dict("joint_type")
        self.trajs = Trajs(
            joint_type=joint_type
        )
        self.trajs: Trajs
        
        # 一开始这个并不严格, 随便设置一个好了
        if joint_type == "prismatic":
            initial_traj = Traj(
                start_state=0,
                manip_type="open",
                Tph2w=self.obj_repr.kframes[0].Tph2w,
                goal_state=0.1,
                end_state=0.1,
                valid_mask=True
            )
        else:
            initial_traj = Traj(
                start_state=0,
                manip_type="open",
                Tph2w=self.obj_repr.kframes[0].Tph2w,
                goal_state=np.deg2rad(10),
                end_state=np.deg2rad(10),
                valid_mask=True
            )
        self.trajs.update(initial_traj)
        
        # 然后开始探索
        while not self.trajs.is_range_covered() and self.num_further_explore < self.max_further_explore:
            # 首先获取当前状态
            self.update_cur_frame()
            is_explored, current_range = self.trajs.get_current_range(self.cur_state)
            
            if is_explored:
                # 一开始默认处于 reset_pose
                if current_range[0] > self.trajs.min_state:
                    # 向左探索
                    target_state = current_range[0]
                    manip_type = "close"
                elif current_range[1] < self.trajs.max_state:
                    # 向右探索
                    target_state = current_range[1]
                    manip_type = "open"
                else:
                    continue
                
                # 此时 panda_hand 应该没有抓住物体
                Tph2w = self.get_stable_grasp_proposal(self.cur_state)
                self.open_gripper()
                self.move_to_pose_safe(Tph2w)
                self.close_gripper()
                
                self.update_cur_frame()
                trajs = self.make_a_move(
                    cur_state=self.cur_state,
                    delta_state=target_state-self.cur_state,
                )
                self.trajs.update(trajs)
                
                # 松开并复位
                self.reset_robot_safe()
                # 然后开始真正的尝试
                self.update_cur_frame()
                Tph2w = self.get_stable_grasp_proposal(self.cur_state)
                
                self.open_gripper()
                self.move_to_pose_safe(Tph2w)
                self.close_gripper()
                
                self.update_cur_frame()
                trajs = self.make_a_move(
                    cur_state=self.cur_state,
                    delta_state=self.trajs.min_state-self.cur_state if manip_type == "close" else self.trajs.max_state-self.cur_state,
                )
                self.trajs.update(trajs)
            else:
                # 一开始默认处于 reset_pose
                # 直接向 current_range[0] 探索
                target_state = current_range[0]
                manip_type = "close"
                
                Tph2w = self.get_stable_grasp_proposal(self.cur_state)
                self.open_gripper()
                self.move_to_pose_safe(Tph2w)
                self.close_gripper()
                
                self.update_cur_frame()
                trajs = self.make_a_move(
                    cur_state=self.cur_state,
                    delta_state=self.trajs.min_state-self.cur_state,
                )
                self.trajs.update(trajs)
        
    
if __name__ == "__main__":
    
    exploreEnv = FurtherExploreEnv(
        cfg={
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "valid_thresh": 0.5,
    "instruction": "open the cabinet",
    "num_initial_pts": 1000,
    "obj_description": "cabinet",
    "joint_type": "revolute",
    "obj_index": "45162",
    "joint_index": "0",
    "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/45162_link_0",
    "active_link_name": "link_0",
    "active_joint_name": "joint_0",
    "load_pose": [
        0.8806502223014832,
        0.0,
        0.6088799834251404
    ],
    "load_quat": [
        1.0,
        0.0,
        0.0,
        0.0
    ],
    "load_scale": 1
}
    )
    exploreEnv.explore_stage(visualize=False)
    # exploreEnv.save(file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore/explore_data.pkl")
    