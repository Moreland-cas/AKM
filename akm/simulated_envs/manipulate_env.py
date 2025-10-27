import os
import time
import copy
import json
import logging
import numpy as np

from akm.utility.constants import *
from akm.utility.utils import numpy_to_json
from akm.representation.obj_repr import Obj_repr
from akm.representation.basic_structure import Frame
from akm.simulated_envs.reconstruct_env import ReconEnv
from akm.utility.timer import Timer


class ManipulateEnv(ReconEnv):
    def __init__(self, cfg):       
        super().__init__(cfg)
        self.manip_env_cfg = cfg["manip_env_cfg"]
        
        # Timer
        self.manip_timer = Timer()
          
    def manip_once(self):
        """
        First, relocate the initial frame state and obtain the target state according to the instruction.
        Home the robot arm, locate the current state, and calculate the grasping pose for the current frame (using manipulate first frame).
        Move to this pose and perform operations based on target_state.
        """
        self.logger.log(logging.INFO, "Run manip_once()...")
        self.obj_repr : Obj_repr 
        self.reset_robot_safe()
        
        # You need to run reloc again here to prevent open_gripper + move_backward from affecting the object state
        self.update_cur_frame(init_guess=self.cur_state)
        self.ref_ph_to_tgt(
            ref_frame=self.obj_repr.kframes[0],
            tgt_frame=self.cur_frame
        )
        
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
                moving_distance=self.target_state-self.cur_frame.joint_state,
                drop_large_move=False
            )
        
        # Reposition here. If it is too far from your target, re-execute
        result_dict = self.evaluate()
        self.logger.log(logging.INFO, result_dict)
        return result_dict
        
    def not_good_enough(self, range_transition=None, visualize=False):
        self.logger.log(logging.INFO, "Check if current state is good enough...")
        # NOTE: cur_frame represents the frame captured at the beginning of the first round of operations.
        # Then estimate cur_state
        if self.target_state is None:
            self.update_cur_frame(
                init_guess=None,
                visualize=visualize
            )
        else:
            self.update_cur_frame(
                init_guess=self.target_state,
                visualize=visualize
            )
        
        # NOTE: Set target_state only when target_state is None
        if self.target_state is None:
            self.obj_repr.reloc(
                query_frame=self.obj_repr.initial_frame,
                reloc_lr=self.recon_env_cfg["reloc_lr"],
                init_guess=None,
                visualize=visualize
            )
            # NOTE Here you need to modify repositioning to use the current frame instead of the initial frame
            self.target_state = self.cur_state + self.goal_delta
        
        if self.exp_cfg["save_obj_repr"]:
            if range_transition not in self.obj_repr.save_for_vis:
                self.obj_repr.save_for_vis[range_transition] = []
            self.cur_frame.target_state = self.target_state
            self.obj_repr.save_for_vis[range_transition].append(copy.deepcopy(self.cur_frame))
            
        if self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            not_good = abs(self.cur_state - self.target_state) > self.manip_env_cfg["prismatic_whole_traj_success_thresh"] # 1cm
        elif self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            not_good = abs(self.cur_state - self.target_state) > np.deg2rad(self.manip_env_cfg["revolute_whole_traj_success_thresh"]) # 5 degree 
        if not_good:
            self.logger.log(logging.INFO, "Not good enough")
        else:
            self.logger.log(logging.INFO, "Good enough")
        self.logger.log(logging.INFO, f"cur_state: {self.cur_state}, target_state: {self.target_state}")
        return not_good
        
    def manipulate_close_loop(self, range_transition=None, visualize=False):
        self.max_manip = self.manip_env_cfg["max_manip"]
        self.logger.log(logging.INFO, "Start manipulation Loop ...")
        num_manip = 0
        # For cases where manipulation is successful, add a result
        results = {
            num_manip: self.evaluate()
        }
        
        not_good = True
        while(not_good and (num_manip < self.max_manip)):
            reloc_time_start = time.time()
            not_good = self.not_good_enough(range_transition=range_transition, visualize=visualize)
            reloc_time_end = time.time()
            self.manip_timer.update(f"reloc_{range_transition}", reloc_time_end - reloc_time_start)
        
            self.logger.log(logging.INFO, f"Start manipulating, round {num_manip + 1}...")
            manip_move_start = self.cur_steps * self.phy_timestep
            result = self.manip_once()
            manip_move_end = self.cur_steps * self.phy_timestep
            self.manip_timer.update(f"manip_move_{range_transition}", manip_move_end - manip_move_start)
            num_manip = num_manip + 1
            results[num_manip] = result
        
        if num_manip == self.max_manip:
            self.logger.log(logging.INFO, f"After {num_manip} round, Stopped since num_manip reach max_manip...")
        else:
            self.logger.log(logging.INFO, f"After {num_manip} round, Stopped since the robot thinks it is good enough...")
        self.logger.log(logging.INFO, results)
        return results
        
    def evaluate(self):
        actual_delta = self.get_active_joint_state() - self.manip_start_state
        diff = actual_delta - self.goal_delta
        result_dict = {
            "diff": diff,
            "actual_delta": actual_delta,
            "goal_delta": self.goal_delta
        }
        return result_dict
    
    def prepare_task_env(self, manip_start_state, manip_end_state):
        """
        After exploration and reconstructing, a manipulation task is executed.
        manip_start_state: The object's joints are initialized to this value (joint_state = 0 for a fully closed state).
        manip_end_state: The algorithm manipulates the object so that its joints reach this value.
        """
        self.logger.log(logging.INFO, f"Start manip task: {manip_start_state:.2f} -> {manip_end_state:.2f}")
        self.manip_start_state = manip_start_state
        self.manip_end_state = manip_end_state
        
        self.reset_robot()
        
        self.set_active_joint_state(joint_state=manip_start_state)
        self.goal_delta = manip_end_state - manip_start_state
        # NOTE You need to set target_state to None before manipulate_close_loop
        self.target_state = None
        
    def main(self):
        super().main()
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
                    tmp_manip_result.update(self.manipulate_close_loop(range_transition=k))
                    
                else:
                    self.logger.log(logging.INFO, f'No valid reconstruction, thus skip manipulation...')
                    tmp_manip_result["exception"] = "No valid reconstruct."
                    
                
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
        
        if self.exp_cfg["save_obj_repr"]:
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr.npy"
            )
            self.obj_repr.save(save_path)
        
        # save Timer info
        self.manip_timer.save(os.path.join(self.exp_cfg["exp_folder"], str(self.task_cfg["task_id"]), "manip_timer.json"))
    