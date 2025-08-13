import os
import yaml
import json
import logging
import numpy as np

from akm.utility.utils import (
    numpy_to_json,
    visualize_pc,
    joint_data_to_transform_np,
    depth_image_to_pointcloud
)
from akm.project_config import MOVING_LABEL
from akm.representation.obj_repr import Obj_repr
from akm.representation.basic_structure import Frame
from akm.realworld_envs.explore_env import ExploreEnv


class ReconEnv(ExploreEnv):
    def __init__(self, cfg):     
        super().__init__(cfg=cfg)
        self.recon_env_cfg = cfg["recon_env_cfg"]
        
        self.num_kframes = self.recon_env_cfg["num_kframes"]
        self.fine_lr = self.recon_env_cfg["fine_lr"]
        self.reloc_lr = self.recon_env_cfg["reloc_lr"]
        
        obj_repr_path = os.path.join(self.cfg["exp_cfg"]["exp_folder"], str(self.cfg["task_cfg"]["task_id"]), "obj_repr_explore.npy")
        self.obj_repr = Obj_repr.load(obj_repr_path)
    
    def update_cur_frame(self, init_guess=None, visualize=False):
        """
        Will update self.cur_frame and self.cur_state
        """
        self.obj_repr: Obj_repr
        for _ in range(30):
            self.capture_frame()
            
        cur_frame = self.capture_frame()
        
        cur_frame.segment_obj(
            obj_description=self.obj_description,
            post_process_mask=True,
            filter=True,
        )
        
        cur_frame = self.obj_repr.reloc(
            query_frame=cur_frame,
            reloc_lr=self.reloc_lr,
            init_guess=init_guess,
            visualize=False
        )
        self.cur_state = cur_frame.joint_state
        self.cur_frame = cur_frame
        # The Extimated Current State printed here is the offset predicted by the recon algorithm + load_joint_state
        self.logger.log(logging.DEBUG, f'Estimated Current State: {self.cur_state + self.obj_env_cfg["load_joint_state"]}')
        
        if visualize:
            points, colors = cur_frame.get_obj_pc(
                use_robot_mask=True, 
                use_height_filter=True,
                world_frame=False,
                visualize=False
            )
            colors = colors / 255
            fine_jonint_dict = self.obj_repr.get_joint_param(
                resolution="fine",
                frame="camera"
            )
            first_kframe = self.obj_repr.kframes[0]
            first_kframe_pc = depth_image_to_pointcloud(
                first_kframe.depth,
                first_kframe.dynamic_mask == MOVING_LABEL,
                first_kframe.K,
            )
            from akm.utility.utils import joint_data_to_transform_np
            Tref2tgt = joint_data_to_transform_np(
                joint_type=fine_jonint_dict["joint_type"], 
                joint_dir=fine_jonint_dict["joint_dir"], 
                joint_start=fine_jonint_dict["joint_start"], 
                joint_state_ref2tgt=self.cur_state - first_kframe.joint_state
            )
            translated_mobile_pc = first_kframe_pc @Tref2tgt[:3, :3].T + Tref2tgt[:3, -1]
            mobile_pc_color = np.array([0, 255, 0]) / 255
            visualize_pc(
                points=np.concatenate([points, translated_mobile_pc], axis=0),
                colors=np.concatenate([colors, np.broadcast_to(mobile_pc_color, (translated_mobile_pc.shape[0], 3))], axis=0),
                point_size=5,
                online_viewer=True
            )
        
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
        This function is necessary because the current grasping module is not very powerful. This means we are not transferring the contact_3d pose,
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
            visualize=False,
            num_R_augmented=self.recon_env_cfg["num_R_augmented"],
            evaluate=False
        )
        self.logger.log(logging.INFO, self.recon_result)
        
        if visualize:
            coarse_jonint_dict = self.obj_repr.get_joint_param(
                resolution="coarse",
                frame="camera"
            )
            fine_jonint_dict = self.obj_repr.get_joint_param(
                resolution="fine",
                frame="camera"
            )
            points, colors = self.obj_repr.kframes[0].get_obj_pc(
                use_robot_mask=True, 
                use_height_filter=True,
                world_frame=False,
                visualize=False
            )
            # coarse
            visualize_pc(
                points=points,
                colors=colors/255,
                point_size=5,
                pivot_point=coarse_jonint_dict["joint_start"],
                joint_axis=coarse_jonint_dict["joint_dir"],
                online_viewer=True
            )
            # fine
            visualize_pc(
                points=points,
                colors=colors/255,
                point_size=5,
                pivot_point=fine_jonint_dict["joint_start"],
                joint_axis=fine_jonint_dict["joint_dir"],
                online_viewer=True
            )
                
        if self.recon_result is None:
            self.recon_result = {}
        return self.recon_result
    
    def main(self):
        self.obj_repr = Obj_repr.load(
            os.path.join(self.exp_cfg["exp_folder"], str(self.task_cfg["task_id"]), "obj_repr_explore.npy")
        )
        self.recon_result = self.recon_stage(visualize=True)
        
        if self.exp_cfg["save_obj_repr"]:
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr_recon.npy"
            )
            self.obj_repr.save(save_path)
            
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "recon_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.recon_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
        
if __name__ == "__main__":
    # with open("/home/user/Programs/AKM/akm/realworld_envs/cabinet.yaml", "r") as f:
    with open("/home/user/Programs/AKM/akm/realworld_envs/drawer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    reconEnv = ReconEnv(cfg=cfg)
    reconEnv.main()
    