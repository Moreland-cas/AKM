import json
import os
import logging
import numpy as np
from embodied_analogy.utility.utils import (
    initialize_napari,
    joint_data_to_transform_np,
    numpy_to_json
)
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.environment.explore_env import ExploreEnv
from embodied_analogy.representation.basic_structure import Frame


class ReconEnv(ExploreEnv):
    def __init__(self, cfg):     
        super().__init__(cfg=cfg)
        self.recon_env_cfg = cfg["recon_env_cfg"]
        
        self.num_kframes = self.recon_env_cfg["num_kframes"]
        self.fine_lr = self.recon_env_cfg["fine_lr"]
        self.reloc_lr = self.recon_env_cfg["reloc_lr"]
    
    def update_cur_frame(self, init_guess=None, visualize=False):
        self.obj_repr: Obj_repr
        self.base_step()
        cur_frame = self.capture_frame()
        cur_frame = self.obj_repr.reloc(
            query_frame=cur_frame,
            reloc_lr=self.reloc_lr,
            init_guess=init_guess,
            visualize=visualize
        )
        self.cur_state = cur_frame.joint_state
        self.cur_frame = cur_frame
        # 这里打印的 Extimated Current State 是 recon 算法预测的 offset + init_joint_state 得到的
        self.logger.log(logging.DEBUG, f'Extimated Current State: {self.cur_state + self.obj_env_cfg["init_joint_state"]}')
        self.logger.log(logging.DEBUG, f"GT Current State: {self.get_active_joint_state()}")
        
    def transform_grasp(self, Tph2w_ref, ref_state, tgt_state):
        """
        根据 joint_dict_w 将 ref_state 下的 grasp_pose (Tph2w_ref) 转换到 target_state 下得到 Tph2w_tgt
        """
        joint_dict_w = self.obj_repr.get_joint_param(
            resolution="fine",
            frame="world"
        )
        Tref2tgt_w = joint_data_to_transform_np(
            joint_type=joint_dict_w["joint_type"],
            joint_dir=joint_dict_w["joint_dir"],
            joint_start=joint_dict_w["joint_start"],
            joint_state_ref2tgt=tgt_state-ref_state
        )
        Tph2w_tgt = Tref2tgt_w @ Tph2w_ref
        return Tph2w_tgt

    def ref_ph_to_tgt(self, ref_frame: Frame, tgt_frame: Frame):
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
        
    def transfer_Tph2w(self, Tph2w_ref, ref_state, tgt_state):
        """
        将 Tph2w 从 ref_state 转换到 tgt_state
        """
        Tph2c_ref = self.obj_repr.Tw2c @ Tph2w_ref
        
        # Tref2tgt 是 camera 坐标系下的一个变换
        Tref2tgt_c = joint_data_to_transform_np(
            joint_type=self.obj_repr.fine_joint_dict["joint_type"],
            joint_dir=self.obj_repr.fine_joint_dict["joint_dir"],
            joint_start=self.obj_repr.fine_joint_dict["joint_start"],
            joint_state_ref2tgt=tgt_state-ref_state
        )
        
        Tph2c_tgt = Tref2tgt_c @ Tph2c_ref
        Tph2w_tgt = np.linalg.inv(self.obj_repr.Tw2c) @ Tph2c_tgt
        return Tph2w_tgt

    def recon_stage(self, load_path=None, visualize=False):
        if load_path is not None:
            self.obj_repr = Obj_repr.load(load_path)
        
        self.recon_result = self.obj_repr.reconstruct(
            num_kframes=self.num_kframes,
            obj_description=self.obj_description,
            fine_lr=self.fine_lr,
            visualize=visualize,
            evaluate=True
        )
        self.logger.log(logging.INFO, self.recon_result)
        
        if self.exp_cfg["save_obj_repr"]:
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr.npy"
            )
            self.obj_repr.save(save_path)
        return self.recon_result
    
    ###########################################################
    def main(self):
        super().main()
        
        try:
        # if True:
            self.recon_result = {}
            
            if self.explore_result["has_valid_explore"]:
                self.logger.log(logging.INFO, f"Valid explore detected, thus start reconstruction...") 
                self.recon_result = self.recon_stage()
                self.recon_result["has_valid_recon"] = True
            else:
                self.logger.log(logging.INFO, f"No Valid explore, thus skip reconstruction...") 
                self.recon_result["has_valid_recon"] = False
                self.recon_result["exception"] = "No valid explore."
            
        except Exception as e:
            self.logger.log(logging.ERROR, f"Exception occured during Reconstruct_stage: {e}", exc_info=True)
            self.recon_result["has_valid_recon"] = False
            self.recon_result["exception"] = str(e)
        
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "recon_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.recon_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
        
if __name__ == "__main__":
    cfg = {
        "num_kframes": 5,
        "fine_lr": 1e-3,
    }
    env = ReconEnv(cfg)
    
    env.recon_stage(
        load_path=f"/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore_4_11/40147_1_prismatic/explore/obj_repr.npy",    
        save_path=None,
        evaluate=True,
        visualize=False
    )