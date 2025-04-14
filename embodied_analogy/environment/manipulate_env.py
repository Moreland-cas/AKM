import numpy as np

from embodied_analogy.environment.obj_env import ObjEnv
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    initialize_napari,
    joint_data_to_transform_np,
)
initialize_napari()
from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.representation.obj_repr import Obj_repr


class ManipulateEnv(ObjEnv):
    def __init__(
        self, 
        cfg
    ):       
        """
        cfg 需要包含:
            init_joint_state
            goal_delta
            obj_repr_path
        TODO: 还可能需要一些 ICP 的参数, 尤其是 ICP range 之类的
        """
        # 首先 load 物体
        super().__init__(cfg)
        self.reloc_lr = cfg["reloc_lr"]
        self.reserved_distance = cfg["reserved_distance"]
        
        self.goal_delta = cfg["goal_delta"]
        self.init_joint_state = cfg["init_joint_state"]
        
        self.obj_description = cfg["obj_description"]
        self.obj_repr = Obj_repr.load(cfg["obj_repr_path"])
    
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
        
    def manip_stage(self, load_path=None, evaluate=True, visualize=False):
        """
        manipulate 的执行逻辑:
        首先重定位出 initial frame 的状态, 并根据 instruction 得到 target state
        
        对机械臂进行归位, 对当前状态进行定位, 计算出当前帧的抓取位姿 (使用 manipulate first frame)
        
        移动到该位姿, 并根据 target_state 进行操作
        """
        if load_path is not None:
            self.obj_repr = Obj_repr.load(load_path)
        
        self.obj_repr : Obj_repr 
        Tc2w = np.linalg.inv(self.camera_extrinsic)
        
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
        self.target_state = self.cur_state + self.goal_delta
        
        self.transfer_ph_pose(
            ref_frame=self.obj_repr.kframes[0],
            tgt_frame=cur_frame,
            visualize=visualize
        )
        
        cur_frame.segment_obj(obj_description=self.obj_description, visualize=visualize)
        pc_w, _ = cur_frame.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
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
        if evaluate:
            result_dict = self.evaluate()
            print(result_dict)
    
    def evaluate(self):
        # 评测 manipulate 的好坏
        actual_delta = self.get_active_joint_state() - self.init_joint_state
        diff = actual_delta - self.goal_delta
        loss = np.abs(diff)
        result_dict = {
            "diff": diff,
            "l1_loss": loss,
            "actual_delta": actual_delta,
            "goal_delta": self.goal_delta
        }
        return result_dict
    
    def main(self, visualize=False):
        result = {
            "exception": "",
            "loss": None,
        }
        
        try:
            self.explore_stage(visualize=visualize)
            self.recon_stage(visualize=visualize)
            self.manip_stage(visualize=visualize)
            loss = self.evaluate()
            result["loss"] = loss
        except Exception as e:
            print(e)
            result["exception"] = str(e)
        
        return result
            
        
if __name__ == '__main__':
    cfg = {
        'base_cfg': {
            'phy_timestep': 0.004, 
            'planner_timestep': 0.01, 
            'use_sapien2': True
            }, 
        'robot_cfg': {}, 
        'explore_cfg': {
            'record_fps': 30, 
            'pertubation_distance': 0.1, 
            "valid_thresh": 0.5,
            'max_tries': 10, 
            'update_sigma': 0.05
            }, 
        'recon_cfg': {
            'num_initial_pts': 1000, 
            'num_kframes': 5, 
            'fine_lr': 0.001
            }, 
        'manip_cfg': {
            'reloc_lr': 0.003, 
            'reserved_distance': 0.05
            }, 
        'task_cfg': {
            'instruction': 'open the cabinet', 
            'obj_description': 'cabinet', 
            'delta': 0.2617993877991494, 
            'obj_cfg': {
                'asset_path': '/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/46277_link_1', 
                'scale': 1.0, 
                'active_link_name': 'link_1', 
                'active_joint_name': 'joint_1', 
                # 'pose': Pose([0.917415, 0.148381, 0.512052], [0.988918, 0, 0, 0.148461]), 
                # 'init_joint_state': 0.0
                }
            }
        }
    me = ManipulateEnv(
        explore_cfg=cfg["explore_cfg"],
        recon_cfg=cfg["recon_cfg"],
        task_cfg=cfg["task_cfg"]
    )
    # obj_index = task_cfg["obj_cfg"]["asset_path"].split("/")[-1].split("_")[0]
    # me.explore_stage()
    # me.recon_stage(save_path="/home/zby/Prograwms/Embodied_Analogy/assets/tmp/47578/recon_data.pkl")
    result = me.main(visualize=False)
    
    while True:
        me.base_step()
    