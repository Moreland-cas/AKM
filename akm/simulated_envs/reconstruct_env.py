import os
import json
import logging
import numpy as np

from akm.utility.utils import (
    joint_data_to_transform_np,
    numpy_to_json
)
from akm.representation.obj_repr import Obj_repr
from akm.representation.basic_structure import Frame
from akm.simulated_envs.explore_env import ExploreEnv


class ReconEnv(ExploreEnv):
    def __init__(self, cfg):     
        super().__init__(cfg=cfg)
        self.recon_env_cfg = cfg["recon_env_cfg"]
        
        self.num_kframes = self.recon_env_cfg["num_kframes"]
        self.fine_lr = self.recon_env_cfg["fine_lr"]
        self.reloc_lr = self.recon_env_cfg["reloc_lr"]
    
    def update_cur_frame(self, init_guess=None, visualize=False):
        """
        Will update self.cur_frame and self.cur_state
        """
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
        # The Extimated Current State printed here is the offset predicted by the recon algorithm + load_joint_state
        self.logger.log(logging.DEBUG, f'Estimated Current State: {self.cur_state + self.obj_env_cfg["load_joint_state"]}')
        self.logger.log(logging.DEBUG, f"GT Current State: {self.get_active_joint_state()}")
        
    def transform_grasp(self, Tph2w_ref, ref_state, tgt_state):
        """
        According to joint_dict_w, grasp_pose (Tph2w_ref) under ref_state is converted to target_state to obtain Tph2w_tgt
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

    def ref_ph_to_tgt(self, ref_frame: Frame, tgt_frame: Frame, use_gt_joint_dict: bool = False):
        """
        Transfer the panda_hand grasp_pose in the ref_frame to the target_frame.

        NOTE:
        This function is necessary because the current grasping module is not very powerful. 
        This means we are not transferring the contact_3d pose,
        but rather the panda_hand grasp pose that has been proven to be good during the exploration phase.
        """
        Tph2w_ref = ref_frame.Tph2w
        Tph2c_ref = self.obj_repr.Tw2c @ Tph2w_ref
        
        if use_gt_joint_dict:
            fine_joint_dict = self.obj_repr.get_joint_param(
                resolution="gt",
                frame="camera"
            )
        else:
            fine_joint_dict = self.obj_repr.fine_joint_dict
            
        Tref2tgt_c = joint_data_to_transform_np(
            joint_type=fine_joint_dict["joint_type"],
            joint_dir=fine_joint_dict["joint_dir"],
            joint_start=fine_joint_dict["joint_start"],
            joint_state_ref2tgt=tgt_frame.joint_state-ref_frame.joint_state
        )
        
        Tph2c_tgt = Tref2tgt_c @ Tph2c_ref
        Tph2w_tgt = np.linalg.inv(self.obj_repr.Tw2c) @ Tph2c_tgt
        tgt_frame.Tph2w = Tph2w_tgt
        
    def transfer_Tph2w(self, Tph2w_ref, ref_state, tgt_state):
        """
        Transition Tph2w from ref_state to tgt_state
        """
        Tph2c_ref = self.obj_repr.Tw2c @ Tph2w_ref
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
            num_R_augmented=self.recon_env_cfg["num_R_augmented"],
            evaluate=True
        )
        self.logger.log(logging.INFO, self.recon_result)
        return self.recon_result
    
    def main(self):
        super().main()
        
        try:
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
