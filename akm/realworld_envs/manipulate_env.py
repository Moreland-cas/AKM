import os
import yaml
import logging
import numpy as np
from franky import Reaction, JointStopMotion

from akm.utility.constants import *
from akm.utility.utils import clean_pc_np
from akm.representation.obj_repr import Obj_repr
from akm.realworld_envs.reconstruct_env import ReconEnv


class ManipulateEnv(ReconEnv):
    def __init__(self, cfg):       
        super().__init__(cfg)
        self.manip_env_cfg = cfg["manip_env_cfg"]
        obj_repr_path = os.path.join(self.exp_cfg["exp_folder"], str(self.task_cfg["task_id"]), "obj_repr_recon.npy")
        self.obj_repr = Obj_repr.load(obj_repr_path)
          
    def not_good_enough(self, visualize=False):
        self.logger.log(logging.INFO, "Check if current state is good enough...")
        
        self.update_cur_frame(
            init_guess=None,
            visualize=visualize
        )
        
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
        
    def manipulate_close_loop(self, visualize=True):
        """
        reloc
        update_collision
        move to pre_grasp_pose
        open_gripper
        approach
        close_gripper
        # reloc
        move_along_axis
        if reloc not good:
            open_gripper
            draw_back
            goto first line
        """
        self.max_manip = self.manip_env_cfg["max_manip"]
        self.logger.log(logging.INFO, "Start manipulation Loop ...")
        num_manip = 0
        
        while num_manip < self.max_manip:
            num_manip = num_manip + 1
            
            self.update_cur_frame(
                init_guess=None,
                visualize=visualize
            )
            pc_w = self.cur_frame.get_obj_pc(
                use_robot_mask=True,
                use_height_filter=True,
                world_frame=True
            )[0]
            self.update_point_cloud_with_wall(clean_pc_np(pc_w))

            self.ref_ph_to_tgt(
                ref_frame=self.obj_repr.kframes[0],
                tgt_frame=self.cur_frame
            )
            
            Tph2w_pre = self.get_translated_ph(self.cur_frame.Tph2w, -self.reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            
            if result_pre is None:
                self.logger.log(logging.INFO, "Get None planning result in manip_once(), thus do nothing")
                self.reset_safe(distance=-self.reserved_distance)
                break
            else:
                self.switch_mode("cartesian_impedance")
                self.move_to(Tph2w=Tph2w_pre)
                
                # self.switch_mode("joint_impedance")
                # self.follow_path(result_pre)
                
                self.clear_planner_pc()
                self.open_gripper(target=0.08)
                
                self.switch_mode("cartesian_impedance")
                self.approach(distance=self.reserved_distance + 0.01, speed=0.02)
                
                # 改为 safe_close
                self.close_gripper_safe()
                # self.close_gripper(target=0.0, gripper_force=4)
                
                self.update_cur_frame(
                    init_guess=None,
                    visualize=visualize
                )
                
                fine_dict_w = self.obj_repr.get_joint_param(
                    resolution="fine",
                    frame="world"
                )
                
                self.switch_mode("cartesian_impedance")
                self.move_along_axis(
                    joint_type=fine_dict_w["joint_type"],
                    joint_axis=fine_dict_w["joint_dir"],
                    joint_start=fine_dict_w["joint_start"],
                    moving_distance=self.target_state-self.cur_frame.joint_state,
                )
                    
                if self.not_good_enough(visualize=visualize):
                    self.reset_safe(distance=-self.reserved_distance)
                else:
                    break
    
    def prepare_task_env(self, goal_delta):
        """
        After exploration and reconstructing, a manipulation task is executed.
        manip_start_state: The object's joints are initialized to this value (joint_state = 0 for a fully closed state).
        manip_end_state: The algorithm manipulates the object so that its joints reach this value.
        """
        self.goal_delta = goal_delta
        self.reset()
        self.update_cur_frame(
            init_guess=None,
            visualize=False
        )
        self.target_state = self.cur_state + self.goal_delta
        
    def main(self):
        self.obj_repr = Obj_repr.load(
            os.path.join(self.exp_cfg["exp_folder"], str(self.task_cfg["task_id"]), "obj_repr_recon.npy")
        )
        goal_delta = self.manip_env_cfg["goal_delta"]
        if self.joint_type == "revolute":
            goal_delta = np.deg2rad(goal_delta)
        self.prepare_task_env(goal_delta=goal_delta)        
        self.manipulate_close_loop(visualize=False)


if __name__ == '__main__':
    # cfg_path = "/home/user/Programs/AKM/cfgs/realworld_cfgs/drawer.yaml"
    cfg_path = "/home/user/Programs/AKM/cfgs/realworld_cfgs/cabinet.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    manipEnv = ManipulateEnv(cfg=cfg)
    manipEnv.main()
    input("Type anything:")
    manipEnv.reset_safe()
    manipEnv.delete()
    