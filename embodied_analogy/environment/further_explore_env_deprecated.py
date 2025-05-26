import os
import random
import numpy as np
from embodied_analogy.utility.utils import (
    initialize_napari,
    custom_linspace,
    dis_point_to_range,
    distance_between_transformation
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

    def reject_similar_faild_grasp(self, available_grasp):
        # 并且剔除那种跟 cur_state 附近的失败尝试拥有相似 pre_gpos 的 (因为 pre_qpos 几乎直接决定了 Tph2w)
        grasp_move_thresh = 0.1
        nearby_failed_grasps = self.trajs.failed_grasp_around(self.cur_state)
        if nearby_failed_grasps != None:
            # 去除掉 available_grasp 中, 那些与 nearby_failed_grasps 相似的 grasp_move
            for nearby_failed_grasp in nearby_failed_grasps:
                # 计算两个 grasp_move 的距离
                dis = 0
                # Tph2w TODO
                if available_grasp[2] is not None and nearby_failed_grasp[2] is not None:
                    dis += distance_between_transformation(available_grasp[2], nearby_failed_grasp[2])
                # pre_qpos
                if available_grasp[1] is not None and nearby_failed_grasp[1] is not None:
                    dis += np.linalg.norm(available_grasp[1] - nearby_failed_grasp[1])
                grasp_move_thresh = 0.1
                if dis < grasp_move_thresh:
                    # reject
                    return True
        return False
    
    def get_stable_grasp_proposal(self, utilize_history_first=False, n_init_qpos=5):
        """
        基于 self.cur_frame 和 self.trajs, 返回一个 stable_grasp proposal (由 Grasp 直接转换得到的 Tph2w)
        
        NOTE 
            我们的策略是优先尝试已有 grasp, 以及已有 grasp 的不同 pre_qpos
            如果都不行 (traj 中有明确的尝试情况且都是失败的)
            那么就重新在当前 frame 下运行 Anygrasp, 并得到一个新的 proposal
        """
        print("Tring to get a stable grasp proposal...")
        available_grasp = None
        
        if utilize_history_first:
            valid_grasps, _ = self.trajs.get_all_grasp()
            for valid_grasp in valid_grasps:
                ref_state, _, Tph2w_ref = valid_grasp
                Tph2w_tgt = self.transfer_Tph2w(
                    Tph2w_ref=Tph2w_ref,
                    ref_state=ref_state,
                    tgt_state=self.cur_state
                )
                pre_Tph2w_tgt = self.get_translated_ph(
                    Tph2w=Tph2w_tgt,
                    distance=-self.cfg["reserved_distance"]
                )
                self.planner.update_point_cloud(
                    self.cur_frame.get_env_pc(
                        use_robot_mask=True,
                        world_frame=True
                    )[0]
                )
                # 在这里计算 tgt_state 下的 Tph2w 对应的 pre_qpos 
                pre_qpos_list = self.IK(
                    target_pose=pre_Tph2w_tgt,
                    wrt_world=True,
                    n_init_qpos=n_init_qpos
                )
                if pre_qpos_list != None:
                    for pre_qpos in pre_qpos_list:
                        available_grasp = (self.cur_state, pre_qpos, Tph2w_tgt)
                        if not self.reject_similar_faild_grasp(available_grasp):
                            break
                if available_grasp != None:
                    break
                
        if available_grasp is None and False:
            # 直接在当前 frame 检测新的
            self.cur_frame.segment_obj(
                obj_description=self.obj_repr.obj_description
            )
            grasp_group = self.cur_frame.detect_grasp_moving(
                crop_thresh=0.1,
                visualize=True
            )
            if grasp_group is not None:
                # 遍历使得后续 IK 成功的概率大大增加
                pre_qpos = None
                for i in range(len(grasp_group)):
                    selected_grasp = grasp_group[i] 
                    grasps_w = selected_grasp.transform(np.linalg.inv(self.cur_frame.Tw2c)) # Tgrasp2w
                    Tph2w = self.anyGrasp2ph(grasps_w)
                    pre_Tph2w = self.get_translated_ph(
                        Tph2w=Tph2w,
                        distance=-self.cfg["reserved_distance"]
                    )
                    self.planner.update_point_cloud(
                        self.cur_frame.get_env_pc(
                            use_robot_mask=True,
                            world_frame=True
                        )[0]
                    )
                    pre_qpos_list = self.IK(
                        target_pose=pre_Tph2w,
                        wrt_world=True,
                        n_init_qpos=n_init_qpos
                    )
                    # 最后使用 IK 跑一个 pre_gpos 出来 # TODO: 如果他就是None应该怎么办
                    if pre_qpos_list is not None:
                        for pre_qpos in pre_qpos_list:
                            available_grasp = (self.cur_state, pre_qpos, Tph2w)
                            if not self.reject_similar_faild_grasp(available_grasp):
                                break
                    if available_grasp is not None:
                        break
            else:
                print("Anygrasp did not detect any valid grasp after filtering by moving part in current frame")
                
        if available_grasp is None:
            return None, None
        
        # grasp_move 类的成员变量包含 start_state, pre_qpos 和 Tph2w
        _, pre_qpos, Tph2w = available_grasp
        return Tph2w, pre_qpos
    
            
    def move_along_axis_with_reloc(self, target_state, traj: Traj, drop_large_move):
        """
        move_along_axis 的封装版本, 额外增加了每隔一段距离 reloc 判断执行状态的功能
        NOTE: 该函数假设机器人此时已经抓住了物体, 只需要进行一个位移操作, cur_state 是 close gripper 后检测出的 joint 状态
        """
        # 这里虽然进行了 reloc, 但是不要更新 move_along_axis 的 cur_state
        self.update_cur_frame()
        interpolate_targets = custom_linspace(self.cur_state, target_state, self.reloc_interval)
        print(f"The interpolate_targets is {interpolate_targets}")
        
        if target_state > self.cur_state:
            sign = 1
        else:
            sign = -1
        
        joint_dict_w = self.obj_repr.get_joint_param(resolution="fine", frame="world")
        for i, interpolate_target in enumerate(interpolate_targets):
            if i != len(interpolate_targets) - 1:
                self.move_along_axis(
                    joint_type=joint_dict_w["joint_type"],
                    joint_axis=joint_dict_w["joint_dir"],
                    joint_start=joint_dict_w["joint_start"],
                    moving_distance=sign*self.reloc_interval,
                    drop_large_move=drop_large_move
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
                    drop_large_move=drop_large_move
                )
            # 跑 reloc 并生成 traj
            self.update_cur_frame(init_guess=interpolate_target)
            if abs(self.cur_state - interpolate_target) > self.stable_thresh:
                print("Not working well with reloc checkpoint, break from move_along_axis_with_reloc")
                break
            else:
                print("It currently works well with reloc checkpoint, keep manipulating to see the bound")
                traj.end_state = self.cur_state
                traj.goal_state = interpolate_target
            print(f"*** The initialized traj is curretly {traj}")
                
        if traj.end_state is None:
            traj.valid_mask = False
        else:
            traj.valid_mask = True
        
    def grasp_and_move(self, target_state, update_traj=False):
        # NOTE 从 reset 状态开始, 抓取到当前状态的物体上, 并进行 make_a_move 操作
        Tph2w, pre_qpos = self.get_stable_grasp_proposal(utilize_history_first=True, n_init_qpos=5)
        if Tph2w is None or pre_qpos is None:
            print("Not a single stable grasp proposal is found, return from grasp_and_move")
            return 
        traj = Traj(
            start_state=self.cur_state,
            manip_type="open" if target_state > self.cur_state else "close",
            Tph2w=Tph2w,
            pre_qpos=pre_qpos,
            goal_state=target_state,
            end_state=None,
            valid_mask=None
        )
        print(f"*** Init a new traj: ")
        print(traj)
        
        self.open_gripper()
        # 需要这个函数返回一下 pre_Tph2w 对应的 qpos
        self.move_to_qpos_safe(pre_qpos)
        # 然后关闭环境点云, 并且控制 panda_hand 向前移动 5cm
        self.clear_planner_pc()
        self.move_forward(
            moving_distance=self.cfg["reserved_distance"], 
            drop_large_move=False
        )
        self.close_gripper()
        print("Start move_along_axis_with_reloc...")
        self.move_along_axis_with_reloc(
            target_state=target_state,
            traj=traj,
            drop_large_move=True
        )
        
        if update_traj:
            self.trajs.update(traj)
        
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
        print("Curret trajs: ", self.trajs)
        
        self.update_cur_frame()
        # 然后开始探索
        while self.trajs.exist_large_unexplored_range() and self.num_further_explore < self.max_further_explore:
            print(f"[{self.num_further_explore + 1}/{self.max_further_explore}] Strat further explore...")
            print(f"current trajs:")
            print(self.trajs)
            # 首先获取当前状态, 这里的 initial_guess 来自 grasp_and_move 最后估计出的 cur_state
            self.update_cur_frame(self.cur_state)
            nearest_unexplored_range = self.trajs.find_nearest_unexplored_range(self.cur_state)
            print(f"Start exploring range {nearest_unexplored_range}")
            """
            这里需要加一个简单的分类逻辑
            """
            if dis_point_to_range(self.cur_state, nearest_unexplored_range) != 0:
                # cur_state 在 nearest_unexplored_range 外, 那就尝试到 nearest_unexplored_range 的附近, 然后开始探索
                print(f"Currently outside nearest_unexplored_range {nearest_unexplored_range}")
                if nearest_unexplored_range[0] > self.cur_state:
                    manip_type = "open"
                    target_state = nearest_unexplored_range[0]
                else:
                    manip_type = "close"
                    target_state = nearest_unexplored_range[1]
                    
                # 如果距离已经很近了, 那就直接开始探索, 否则先到 nearest_unexplored_range 附近 
                joint_type = self.obj_repr.fine_joint_dict["joint_type"]
                thresh = 0.025 if joint_type == "prismatic" else np.deg2rad(3)
                
                if abs(dis_point_to_range(self.cur_state, nearest_unexplored_range)) > thresh:
                    print(f"Since it's a little bit further, first move to state {target_state}")
                    self.grasp_and_move(target_state=target_state, update_traj=False) # TODO 这里是不是不应该用 explore 版本的 grasp_and move 而是应该用 exploit 版本的
                    # 松开并复位
                    self.reset_robot_safe()
                    self.update_cur_frame(init_guess=target_state)
                
                # 然后开始真正的尝试
                print("Start exploring...")
                self.grasp_and_move(
                    target_state=self.trajs.min_state if manip_type == "close" else self.trajs.max_state,
                    update_traj=True
                )
                print("Trajs after updating: ", self.trajs)
                self.reset_robot_safe()
                # self.update_cur_frame()
            else:
                # cur_state 在 nearest_unexplored_range 内
                # 无非是向左或者向右探索, 选择那个区间更长的就完事了
                print(f"Currently within nearest_unexplored_range {nearest_unexplored_range}")
                if nearest_unexplored_range[1] - self.cur_state > self.cur_state - nearest_unexplored_range[0]:
                    # 向右探索
                    print("Explore to open...")
                    self.grasp_and_move(
                        target_state=self.trajs.max_state,
                        update_traj=True
                    )
                else:
                    # 向左探索
                    print("Explore to close...")
                    self.grasp_and_move(
                        target_state=self.trajs.min_state,
                        update_traj=True
                    )
                # 松开并复位
                print("Trajs after updating: ", self.trajs)
                self.reset_robot_safe()
                # self.update_cur_frame()
            
            # 在这里限制 while 循环无限执行
            self.num_further_explore += 1
        
    
if __name__ == "__main__":
    cfg={
    "obj_folder_path_explore": "/media/zby/MyBook/embody_analogy_data/assets/logs/explore_4_16/45636_3_prismatic",
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
    "obj_index": "45636",
    "joint_index": "3",
    "init_joint_state": "0",
    "asset_path": "/media/zby/MyBook/embody_analogy_data/assets/dataset/one_drawer_cabinet/45636_link_3",
    "active_link_name": "link_3",
    "active_joint_name": "joint_3",
    "load_pose": [
        0.8754647374153137,
        0.0,
        0.47468623518943787
    ],
    "load_quat": [
        1.0,
        0.0,
        0.0,
        0.0
    ],
    "load_scale": 1,
    "obj_folder_path_reconstruct": "/media/zby/MyBook/embody_analogy_data/assets/logs/recon_4_16/45636_3_prismatic/",
    "num_kframes": 5,
    "fine_lr": 0.001,
    "save_memory": True
}
    cfg["reloc_lr"] = 3e-3
    Env = FurtherExploreEnv(
        cfg=cfg
    )
    Env.further_explore_loop()
    
    while True:
        Env.step()
    