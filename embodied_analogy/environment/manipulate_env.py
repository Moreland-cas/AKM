import os
import numpy as np

from embodied_analogy.utility.utils import (
    initialize_napari,
    joint_data_to_transform_np,
)
initialize_napari()

from embodied_analogy.environment.obj_env import ObjEnv
from embodied_analogy.utility.constants import *

from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.representation.obj_repr import Obj_repr


class ManipulateEnv(ObjEnv):
    def __init__(
        self, 
        cfg
    ):       
        """
        TODO: 还可能需要一些 ICP 的参数, 尤其是 ICP range 之类的
        """
        # 首先 load 物体
        super().__init__(cfg)
        self.reloc_lr = cfg["reloc_lr"]
        self.reserved_distance = cfg["reserved_distance"]
        self.max_manip = cfg["max_manip"]
        
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
    
    def transfer_ph_pose(self, ref_frame: Frame, tgt_frame: Frame):
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
        self.move_forward(-self.reserved_distance)
        
        # print("reset robot safe...")
        # self.reset_robot_safe()
        
        self.transfer_ph_pose(
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
            self.move_forward(self.reserved_distance)
            self.close_gripper()
            
            # 转换 joint_dict 到世界坐标系
            Tc2w = np.linalg.inv(self.camera_extrinsic)
            self.move_along_axis(
                joint_type=self.obj_repr.fine_joint_dict["joint_type"],
                joint_axis=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_dir"],
                joint_start=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_start"] + Tc2w[:3, 3],
                moving_distance=self.target_state-self.cur_frame.joint_state
            )
        
        # 这里进行重定位, 如果离自己的目标差太多, 就重新执行
        result_dict = self.evaluate()
        # print(result_dict)
        return result_dict
    
    def not_good_enough(self, visualize=False):
        print("Check if current state is good enough...")
        # NOTE: cur_frame 代表刚开始一轮操作时捕获的 frame
        # 然后估计出 cur_state
        self.base_step()
        cur_frame = self.capture_frame()
        cur_frame = self.obj_repr.reloc(
            query_frame=cur_frame,
            reloc_lr=self.reloc_lr,
            visualize=visualize
        )
        self.cur_state = cur_frame.joint_state
        self.cur_frame = cur_frame
        
        # NOTE: 仅在第一次调用 not_good_enough 的时候设置 target_state
        if self.target_state is None:
            self.target_state = self.cur_state + self.goal_delta
        
        if self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            return abs(self.cur_state - self.target_state) > 1e-2 # 1cm
        elif self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            return abs(self.cur_state - self.target_state) > np.deg2rad(5) # 5 degree 
        
    def manipulate_close_loop(self, visualize=False):
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
    cfg_prismatic = {
        "phy_timestep": 0.004,
        "planner_timestep": 0.01,
        "use_sapien2": True,
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "max_tries": 10,
        "update_sigma": 0.05,
        "reserved_distance": 0.05,
        "logs_path": "/home/zby/Programs/Embodied_Analogy/assets/logs",
        "run_name": "4_14",
        "valid_thresh": 0.5,
        "instruction": "open the cabinet",
        "num_initial_pts": 1000,
        "obj_description": "cabinet",
        "joint_type": "prismatic",
        "obj_index": "45135",
        "joint_index": "0",
        "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_drawer_cabinet/45135_link_0",
        "active_link_name": "link_0",
        "active_joint_name": "joint_0",
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
        "obj_folder": "/home/zby/Programs/Embodied_Analogy/assets/logs/4_14/45135_0_prismatic",
        "num_kframes": 5,
        "fine_lr": 0.001,
        "save_memory": True
    }
    cfg_revolute = {
        "phy_timestep": 0.004,
        "planner_timestep": 0.01,
        "use_sapien2": True,
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "max_tries": 10,
        "update_sigma": 0.05,
        "reserved_distance": 0.05,
        "logs_path": "/home/zby/Programs/Embodied_Analogy/assets/logs",
        "run_name": "4_14",
        "valid_thresh": 0.5,
        "instruction": "open the cabinet",
        "num_initial_pts": 1000,
        "obj_description": "cabinet",
        "joint_type": "revolute",
        "obj_index": "45984",
        "joint_index": "0",
        "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/45984_link_0",
        "active_link_name": "link_0",
        "active_joint_name": "joint_0",
        "load_pose": [
            0.8842315077781677,
            0.0,
            0.63484126329422
        ],
        "load_quat": [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        "load_scale": 1,
        "obj_folder": "/home/zby/Programs/Embodied_Analogy/assets/logs/4_14/45984_0_revolute",
        "num_kframes": 5,
        "fine_lr": 0.001,
        "save_memory": True
    }
    
    cfg_rev_bug = {
        "phy_timestep": 0.004,
        "planner_timestep": 0.01,
        "use_sapien2": True,
        "record_fps": 30,
        "pertubation_distance": 0.1,
        "max_tries": 10,
        "update_sigma": 0.05,
        "reserved_distance": 0.05,
        "logs_path": "/home/zby/Programs/Embodied_Analogy/assets/logs",
        "run_name": "4_14",
        "valid_thresh": 0.5,
        "instruction": "open the cabinet",
        "num_initial_pts": 1000,
        "obj_description": "cabinet",
        "joint_type": "revolute",
        "obj_index": "45168",
        "joint_index": "1",
        "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/45168_link_1",
        "active_link_name": "link_1",
        "active_joint_name": "joint_1",
        "load_pose": [
            0.951462984085083,
            0.0,
            0.5911185145378113
        ],
        "load_quat": [
            1.0,
            0.0,
            0.0,
            0.0
        ],
        "load_scale": 1,
        "obj_folder": "/home/zby/Programs/Embodied_Analogy/assets/logs/4_14/45168_1_revolute",
        "num_kframes": 5,
        "fine_lr": 0.001,
        "save_memory": True
    }
    cfg = cfg_prismatic
    # cfg = cfg_rev_bug
    
    cfg.update({
        "reloc_lr": 3e-3,
        "init_joint_state": 0.,
        "goal_delta": 0.5,
        # "init_joint_state": np.deg2rad(0),
        # "goal_delta": np.deg2rad(60),
        "obj_repr_path": os.path.join(cfg["obj_folder"], "reconstruct", "obj_repr.npy")
    })
    me = ManipulateEnv(cfg)
    me.manipulate_close_loop()
    
    while True:
        me.base_step()
    