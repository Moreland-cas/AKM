import os
import numpy as np

from embodied_analogy.utility.utils import (
    initialize_napari,
    joint_data_to_transform_np,
    custom_linspace
)
initialize_napari()

from embodied_analogy.environment.obj_env import ObjEnv
from embodied_analogy.environment.reconstruct_env import ReconEnv
from embodied_analogy.utility.constants import *

from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.representation.obj_repr import Obj_repr


class ManipulateEnv(ReconEnv):
    def __init__(
        self, 
        cfg
    ):       
        """
        TODO: 还可能需要一些 ICP 的参数, 尤其是 ICP range 之类的
        """
        # 首先 load 物体
        ObjEnv.__init__(self, cfg)
        self.reloc_lr = cfg["reloc_lr"]
        self.reserved_distance = cfg["reserved_distance"]
        
        
        if cfg["manipulate_type"] == "close":
            self.goal_delta = -cfg["manipulate_distance"]
        elif cfg["manipulate_type"] == "open":
            self.goal_delta = cfg["manipulate_distance"]
            
        if cfg["joint_type"] == "revolute":
            self.goal_delta = np.deg2rad(self.goal_delta)
        
        # NOTE: 在 ObjEnv.__init__() 中会对 cfg 中的这个值进行修改
        self.init_joint_state = cfg["init_joint_state"] 
        
        # NOTE: 这里要将 target_state 初始化为 None, 然后在第一次调用 not_good_enough 的时候进行设置
        self.target_state = None
        
        self.obj_description = cfg["obj_description"]
        self.obj_repr = Obj_repr.load(os.path.join(cfg["obj_folder_path_reconstruct"], "obj_repr.npy"))
        
    def manip_once(self):
        print("Start one manipulation run ...")
        """
        首先重定位出 initial frame 的状态, 并根据 instruction 得到 target state
        
        对机械臂进行归位, 对当前状态进行定位, 计算出当前帧的抓取位姿 (使用 manipulate first frame)
        
        移动到该位姿, 并根据 target_state 进行操作
        """
        self.obj_repr : Obj_repr 
        
        # NOTE: 感觉每次失败的时候没必要完全 reset, 只要撤回一段距离, 然后再次尝试就好了
        print("\topen gripper and move back a little bit...")
        self.open_gripper()
        self.move_forward(
            moving_distance=-self.reserved_distance,
            drop_large_move=False
        )
        
        # 在这里是不是需要再跑一遍 reloc, 从而防止 open_gripper + move_backward 对于物体状态的影响
        self.update_cur_frame()
        
        # print("reset robot safe...")
        # self.reset_robot_safe()
        
        self.ref_ph_to_tgt(
            ref_frame=self.obj_repr.kframes[0],
            tgt_frame=self.cur_frame
        )
        
        # self.cur_frame.segment_obj(obj_description=self.obj_description, visualize=visualize)
        self.cur_frame: Frame
        pc_w, _ = self.cur_frame.get_env_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
        
        self.planner.update_point_cloud(pc_w)
        
        Tph2w_pre = self.get_translated_ph(self.cur_frame.Tph2w, -self.reserved_distance)
        result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
        
        if result_pre is None:
            print("Get None planning result in manip_once()...")
        else:
            # 实际执行
            self.follow_path(result_pre)
            self.open_gripper()
            self.clear_planner_pc()
            self.move_forward(
                moving_distance=self.reserved_distance,
                drop_large_move=False
            )
            self.close_gripper()
            
            # 转换 joint_dict 到世界坐标系
            Tc2w = np.linalg.inv(self.camera_extrinsic)
            self.move_along_axis(
                joint_type=self.obj_repr.fine_joint_dict["joint_type"],
                joint_axis=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_dir"],
                joint_start=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_start"] + Tc2w[:3, 3],
                moving_distance=self.target_state-self.cur_frame.joint_state,
                drop_large_move=False
            )
        
        # 这里进行重定位, 如果离自己的目标差太多, 就重新执行
        result_dict = self.evaluate()
        # print(result_dict)
        return result_dict
        
    def not_good_enough(self, visualize=False):
        print("Check if current state is good enough...")
        # NOTE: cur_frame 代表刚开始一轮操作时捕获的 frame
        # 然后估计出 cur_state
        self.update_cur_frame(visualize=visualize)
        
        # NOTE: 仅在第一次调用 not_good_enough 的时候设置 target_state
        if self.target_state is None:
            self.target_state = self.cur_state + self.goal_delta
        
        if self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            return abs(self.cur_state - self.target_state) > 1e-2 # 1cm
        elif self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            return abs(self.cur_state - self.target_state) > np.deg2rad(5) # 5 degree 
        
    def manipulate_close_loop(self, visualize=False):
        self.max_manip = cfg["max_manip"]
        print("Start manipulation Loop ...")
        num_manip = 0
        results = []
        while(self.not_good_enough(visualize=visualize) and (num_manip < self.max_manip)):
            print(f"Start manipulating, round {num_manip + 1}...")
            result = self.manip_once()
            num_manip = num_manip + 1
            print(result)
            results.append(result)
        
        if num_manip == self.max_manip:
            print(f"After {num_manip} round, Stopped since num_manip reach max_manip...")
        else:
            print(f"After {num_manip} round, Stopped since the robot thinks it is good enough...")
        return results
        
    ###################################### 底下是新版本 ######################################
    def manipulate_close_loop_intermediate(self):
        """
        给定 target_state, 每隔一段距离进行一个 close_loop, 而不是整个执行完一个 close_loop
        """
        # while True: # 当没有执行完
        #     pass
            # 1）得到自身位置，并根据目标位置计算一系列 intermediate_targets (当前状态下 gripper 没碰到物体)
            # 2) 依次执行到每个 intermediate_target
            #    并且在执行到每个的位置时候，进行判断是否距离足够小
            #    如果足够小，那就继续执行 intermediate_targets 的下一个，但是 cur_state 更新为当前的
            #    否则重置 gripper, 并返回 1)
            
        # 配置参数（保持不变）
        self.joint_type = self.obj_repr.fine_joint_dict["joint_type"]
        if self.joint_type == "prismatic":
            step = self.cfg["prismatic_reloc_interval"]
            tolerance = self.cfg["prismatic_reloc_tolerance"]
        else:
            step = np.deg2rad(self.cfg["revolute_reloc_interval"])
            tolerance = np.deg2rad(self.cfg["revolute_reloc_tolerance"])
        
        max_attempts = self.cfg["max_attempts"]
        global_attempt = 0
        
        self.init_guess = None
        self.update_cur_frame(init_guess=None)
        self.init_guess = self.cur_state
        self.target_state = self.cur_state + self.goal_delta
        
        self._move_to_grasp_and_close()
        
        while global_attempt < max_attempts:
            # NOTE 在这里的状态应该是已经 close_gripper, 且 cur_state 已经更新为当前的 state
            # === 核心修改点1：每次循环都打印状态 ===
            # self.update_cur_frame(init_guess=self.init_guess) # 这句似乎在第一次不需要
            print(f"\n[状态更新] 当前: {self.cur_state:.4f} | 目标: {self.target_state:.4f}")
            
            # 终止检查
            if abs(self.cur_state - self.target_state) <= tolerance:
                print(">> 成功抵达最终目标！")
                return True

            # === 核心修改点2：显示中间点生成过程 ===
            intermediates = custom_linspace(self.cur_state, self.target_state, step)
            print(f"[路径规划] 新路径长度: {len(intermediates)} 步, 首点: {intermediates[0]:.4f} 末点: {intermediates[-1]:.4f}")

            # 执行路径
            for target in intermediates:
                print(f"\n[执行阶段] 当前目标: {target:.4f}")
                if not self._execute_validate_step(target, tolerance):
                    # === 核心修改点3：恢复后强制重新生成路径 ===
                    print("!! 步骤执行失败，执行恢复协议...")
                    self._enhanced_recovery_protocol()
                    global_attempt += 1
                    print(f"!! 全局尝试次数: {global_attempt}/{max_attempts}")
                    break  # 退出当前路径循环，触发外层循环重新生成路径
            else:
                print(">> 完整路径执行成功！")
                return True

        print("## 超过最大尝试次数，操作终止")
        return False
    
    def _move_to_grasp_and_close(self):
        """
        移动到当前抓取位置, 并关闭夹爪 + update cur_frame
        """
        Tph2w_ref = self.obj_repr.kframes[0].Tph2w
        Tph2w_tgt = self.transfer_Tph2w(
            Tph2w_ref=Tph2w_ref,
            ref_state=self.obj_repr.kframes[0].joint_state,
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
        self.open_gripper()
        self.move_to_pose(
            pose=pre_Tph2w_tgt,
            wrt_world=True
        )
        self.clear_planner_pc()
        self.move_forward(
            moving_distance=self.reserved_distance,
            drop_large_move=False
        )
        self.close_gripper()
        self.update_cur_frame(init_guess=self.init_guess)
        self.init_guess = self.cur_state
        
    def _execute_validate_step(self, target, tolerance):
        """执行并验证单个步骤"""
        print(f"执行步骤 → {target:.4f}")
        
        # 获取运动参数
        joint_config = self.obj_repr.get_joint_param("fine", "world")
        
        # 执行运动
        self.move_along_axis(
            joint_type=joint_config["joint_type"],
            joint_axis=joint_config["joint_dir"],
            joint_start=joint_config["joint_start"],
            moving_distance=target - self.cur_state,
            drop_large_move=True,
        )
        
        # 立即验证实际位置
        self.init_guess = target
        self.update_cur_frame(init_guess=self.init_guess)
        self.init_guess = self.cur_state
        error = abs(self.cur_state - target)
        print(f"预期：{target:.4f} | 实际：{self.cur_state:.4f} | 误差：{error:.4f}")
    
        return error <= tolerance
    
    def _enhanced_recovery_protocol(self):
        """增强版恢复协议（明确状态刷新）"""
        print("\n--- 开始恢复协议 ---")
        # 回撤操作
        self.open_gripper()
        self.move_forward(
            moving_distance=-self.reserved_distance,
            drop_large_move=False
        ) # 保持原有回撤逻辑
        
        # 核心修改点：强制状态刷新并验证
        print("强制重新定位...")
        self.update_cur_frame(init_guess=self.init_guess)
        # 然后进行抓取 
        self._move_to_grasp_and_close()
        print(f"恢复后最新状态: {self.cur_state:.4f}")
        print("--- 恢复协议完成 ---\n")
            
    def evaluate(self):
        # 评测 manipulate 的好坏
        actual_delta = self.get_active_joint_state() - self.init_joint_state
        diff = actual_delta - self.goal_delta
        result_dict = {
            "diff": diff,
            "actual_delta": actual_delta,
            "goal_delta": self.goal_delta
        }
        return result_dict
            
        
if __name__ == '__main__':
    import json
    # close
    # cfg_path = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/manip_4_16/46134_1_revolute/close/scale_30/cfg.json"
    cfg_path = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/manip_4_16_hard/45135_1_prismatic/open/scale_0.45/cfg.json"
    
    # open
    # cfg_path = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/manip_4_16/49133_1_revolute/open/scale_30/cfg.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    for k, v in cfg.items():
        if isinstance(v, str):
            cfg[k] = v.replace("MyBook*/", "MyBook1/")
    # cfg["asset_path"] = cfg["asset_path"].replace("MyBook", "MyBook1")
    me = ManipulateEnv(cfg)
    me.manipulate_close_loop_intermediate()
    
    while True:
        me.base_step()
    