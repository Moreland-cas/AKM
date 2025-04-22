import os
import json
import pickle
import math
import numpy as np
from embodied_analogy.utility.utils import (
    camera_to_world,
    initialize_napari,
    visualize_pc,
    custom_linspace
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
        
        ObjEnv.__init__(self, cfg=cfg)
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
        # self.save_prefix = cfg["obj_folder_path_further_explore"]
        # os.makedirs(self.save_prefix, exist_ok=True)
        
        # FurtherExplore 阶段独特的参数
        assert self.obj_repr.kframes[0].joint_state == 0
        if self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            self.reloc_interval = np.deg2rad(10) # 10 degree
        elif self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            self.reloc_interval = 0.1 # 1 dm
        self.stable_thresh = 0.1 * self.reloc_interval

        # 一些中间变量
        self.num_further_explore = 0
        self.max_further_explore = 20

    def get_stable_grasp_proposal(self):
        """
        基于 self.cur_frame 和 self.trajs, 返回一个 stable_grasp proposal TODO 需要修改
        """
        # 假设 cur_frame 处于 explored_range 中, 那么直接用现成的 grasp
        # 否则 detect_grasp, 并且根据历史信息筛选。方式为随机 sampe, 但是通过 history traj 影响 sample 的概率分布
        # self.cur_frame.detect_grasp()
        
        # if self.cur_frame.grasp_group is None:
        #     assert "detect no grasp on current frame"
        #     return None
        # else:
        #     grasps_w = self.cur_frame.grasp_group.transform(np.linalg.inv(self.cur_frame.Tw2c))
        #     for grasp_w in grasps_w:
        #         # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
        #         # grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
        #         Tph2w = self.anyGrasp2ph(grasp=grasp_w)        
        #         Tph2w_pre = self.get_translated_ph(Tph2w, -self.cfg["reserved_distance"])
        #         result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
        #         if result_pre is not None:
        #             return grasp_w
        #     return None
        self.ref_ph_to_tgt(
            ref_frame=self.obj_repr.kframes[0],
            tgt_frame=self.cur_frame
        )
        return self.cur_frame.Tph2w
            
    def move_along_axis_with_reloc(self, target_state, update_traj=False):
        """
        move_along_axis 的封装版本, 额外增加了每隔一段距离 reloc 判断执行状态的功能
        NOTE: 该函数假设机器人此时已经抓住了物体, 只需要进行一个位移操作, cur_state 是 close gripper 后检测出的 joint 状态
        """
        # 这里虽然进行了 reloc, 但是不要更新 move_along_axis 的 cur_state
        interpolate_targets = custom_linspace(self.cur_state, target_state, self.reloc_interval)
        # 初始夹爪关闭状态下的 panda_hand pose (TODO 是不是存储关闭前的会更好)
        Tph2w = self.get_ee_pose(as_matrix=True)
        
        if target_state > self.cur_state:
            sign = 1
        else:
            sign = -1
        
        traj = Traj(
            start_state=self.cur_state,
            manip_type="open" if target_state > self.cur_state else "close",
            Tph2w=Tph2w,
            goal_state=None,
            end_state=None,
            valid_mask=None
        )
        
        joint_dict_w = self.obj_repr.get_joint_param(resolution="fine", frame="world")
        for i, interpolate_target in enumerate(interpolate_targets):
            if i != len(interpolate_targets) - 1:
                self.move_along_axis(
                    joint_type=joint_dict_w["joint_type"],
                    joint_axis=joint_dict_w["joint_dir"],
                    joint_start=joint_dict_w["joint_start"],
                    moving_distance=sign*self.reloc_interval,
                )
            else:
                if len(interpolate_targets) == 1:
                    moving_distance = interpolate_target - self.cur_state
                else:
                    moving_distance = interpolate_target - interpolate_targets[-2]
                self.move_along_axis(
                    joint_type=joint_dict_w["joint_type"],
                    joint_axis=joint_dict_w["joint_dir"],
                    joint_start=joint_dict_w["joint_start"],
                    moving_distance=moving_distance,
                )
            
            # 跑 reloc 并生成 traj
            self.update_cur_frame()
            if abs(self.cur_state - interpolate_target) > self.stable_thresh:
                break
            else:
                traj.end_state = self.cur_state
                traj.goal_state = interpolate_target
                
        if traj.end_state is None:
            traj.valid_mask = False
        else:
            traj.valid_mask = True
        
        if update_traj:
            self.trajs.update(traj)
    
    def grasp_and_move(self, target_state):
        # NOTE 从 reset 状态开始, 抓取到当前状态的物体上, 并进行 make_a_move 操作
        Tph2w = self.get_stable_grasp_proposal()
        self.open_gripper()
        self.move_to_pose_safe(Tph2w)
        self.close_gripper()
        
        self.update_cur_frame()
        trajs = self.move_along_axis_with_reloc(target_state=target_state)
        self.trajs.update(trajs)
        
        # 为防止循环不停止
        self.num_further_explore += 1
    
    def further_explore_loop(self):
        # NOTE 保证调用这个函数的时候 franke 处于 reset 状态
        
        # 初始化 explore_traj
        self.obj_repr: Obj_repr
        joint_type = self.obj_repr.fine_joint_dict["joint_type"]
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
                self.grasp_and_move(target_state=target_state)
                
                # 松开并复位
                self.reset_robot_safe()
                # 然后开始真正的尝试
                self.update_cur_frame()
                self.grasp_and_move(
                    target_state=self.trajs.min_state if manip_type == "close" else self.trajs.max_state
                )
                self.reset_robot_safe()
            else:
                # 一开始默认处于 reset_pose
                # 直接向 current_range[0] 探索
                target_state = current_range[0]
                manip_type = "close"
                
                self.grasp_and_move(target_state=target_state)
                self.reset_robot_safe()
        
    
if __name__ == "__main__":
    Env = FurtherExploreEnv(
        cfg={
    "obj_folder_path_explore": "/media/zby/MyBook/embody_analogy_data/assets/logs/explore_4_16/45135_1_prismatic",
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "valid_thresh": 0.5,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "instruction": "open the cabinet",
    "num_initial_pts": 1000,
    "obj_description": "cabinet",
    "joint_type": "prismatic",
    "obj_index": "45135",
    "joint_index": "1",
    "init_joint_state": "0",
    "asset_path": "/media/zby/MyBook/embody_analogy_data/assets/dataset/one_drawer_cabinet/45135_link_1",
    "active_link_name": "link_1",
    "active_joint_name": "joint_1",
    "load_pose": [
        0.8806247711181641,
        0.0,
        0.6068519949913025
    ],
    "load_quat": [
        1.0,
        0.0,
        0.0,
        0.0
    ],
    "load_scale": 1,
    "obj_folder_path_reconstruct": "/media/zby/MyBook/embody_analogy_data/assets/logs/recon_4_16_1/45135_1_prismatic/",
    "num_kframes": 3,
    "fine_lr": 0.001,
    "save_memory": True,
            "reloc_lr": 3e-3,
        }
    )
    Env.further_explore_loop()
    