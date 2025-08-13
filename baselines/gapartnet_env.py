import os
import sys
import json
import logging
import numpy as np

from akm.utility.utils import numpy_to_json
from akm.representation.basic_structure import Frame
from akm.simulated_envs.explore_env import ExploreEnv
from akm.simulated_envs.manipulate_env import ManipulateEnv

sys.path.append(os.path.join(os.path.dirname(__file__)))
from gapartnet_utils import gapartnet_reconstruct


class GAPartNet_ManipEnv(ManipulateEnv):
    """
    The GeneralFlow baseline strategy performs a contact transfer with joint_state = 0 and estimates the joint parameters. 
    Subsequent operations are performed using a contact transfer to obtain the grasping pose, 
    and then the operation is performed according to the estimated model when joint state = 0.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        ManipulateEnv.__init__(self, cfg)
        
        # Clear the joint_dict in obj_repr from the embodied analogy
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.coarse_joint_dict = None
        self.obj_repr.fine_joint_dict = None
        
        self.logger.log(logging.INFO, "Running gpnet baseline...")
    
    def recon_stage_gapartnet(self, visualize=False):
        """
        Estimate the part bbox based on the first and last frames of self.obj_repr frames and get the estimate of the joint model
        """
        joint_dict = gapartnet_reconstruct(
            obj_repr=self.obj_repr,
            gapartnet_model=None,
            visualize=visualize,
            use_gt_joint_type=self.recon_env_cfg["use_gt_joint_type"]
        )

        self.obj_repr.coarse_joint_dict = joint_dict
        self.obj_repr.fine_joint_dict = joint_dict
        
        result = self.obj_repr.compute_joint_error(skip_states=True)
        self.logger.log(logging.INFO, "Reconstruction Result:")
        for k, v in result.items():
            self.logger.log(logging.INFO, f"{k}, {v}")
        return result
    
    def recon_main(self):
        try:
            self.recon_result = {}
            if self.explore_result["has_valid_explore"]:
                self.logger.log(logging.INFO, f"Valid explore detected, thus start reconstruction...") 
                self.recon_result = self.recon_stage_gapartnet(visualize=False)
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
        
        if self.exp_cfg["save_obj_repr"]:
            # NOTE You also need to add the matched bbox
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr.npy"
            )
            self.obj_repr.save(save_path)
                
    def manip_gapartnet(self):
        """
        According to the initial_frame grasp migration to cur_frame, and follow the articulation mdoel operation
        """
        self.base_step()
        self.cur_frame: Frame = self.capture_frame()
        
        # use gt reloc joint state
        self.obj_repr.frames[0].joint_state = self.obj_repr.frames[0].gt_joint_state
        self.cur_frame.joint_state = self.cur_frame.gt_joint_state
        self.ref_ph_to_tgt(
            ref_frame=self.obj_repr.frames[0],
            tgt_frame=self.cur_frame,
            use_gt_joint_dict=self.manip_env_cfg["use_gt_ref_ph_to_tgt"]
        )
        # 
        pc_collision_w, pc_colors = self.cur_frame.get_env_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        Tph2w_pre = self.get_translated_ph(self.cur_frame.Tph2w, -self.reserved_distance)
        result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
        
        if result_pre is None:
            self.logger.log(logging.INFO, "Get None planning result in manip_once(), thus do nothing")
        else:
            self.follow_path(result_pre)
            self.open_gripper()
            self.clear_planner_pc()
            self.move_forward(
                moving_distance=self.reserved_distance,
                drop_large_move=False
            )
            self.close_gripper()
            
            Tc2w = np.linalg.inv(self.camera_extrinsic)
            self.move_along_axis(
                joint_type=self.obj_repr.fine_joint_dict["joint_type"],
                joint_axis=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_dir"],
                joint_start=Tc2w[:3, :3] @ self.obj_repr.fine_joint_dict["joint_start"] + Tc2w[:3, 3],
                moving_distance=self.goal_delta,
                drop_large_move=False
            )
            
        result_dict = {
            "1": self.evaluate()
        }
        self.logger.log(logging.INFO, result_dict)
        return result_dict
    
    def main(self):
        ExploreEnv.main(self)
        self.recon_main()
        self.manip_result = {}
        
        for k, v in self.manip_env_cfg["tasks"].items():
            manip_start_state = v["manip_start_state"]
            manip_end_state = v["manip_end_state"]
            tmp_manip_result = {
                "manip_start_state": manip_start_state,
                "manip_end_state": manip_end_state
            }
            self.prepare_task_env(
                manip_start_state=manip_start_state,
                manip_end_state=manip_end_state
            )
            tmp_manip_result.update({0: self.evaluate()})
            try:
                if self.recon_result["has_valid_recon"]:
                    self.logger.log(logging.INFO, f'Valid reconstruction detected, thus start manipulation...')
                    tmp_manip_result.update(self.manip_gapartnet())
                
            except Exception as e:
                self.logger.log(logging.ERROR, f'Encouter {e} when manipulating, thus only save current state', exc_info=True)
                tmp_manip_result["exception"] = str(e)

            self.manip_result.update({
                k: tmp_manip_result
            })
            
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "manip_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.manip_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
                