import os
import logging
import numpy as np
from franky import Reaction, JointStopMotion

from embodied_analogy.realworld_environment.robot_env import clean_pc_np
from embodied_analogy.realworld_environment.reconstruct_env import ReconEnv
from embodied_analogy.utility.constants import *

from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.representation.obj_repr import Obj_repr

# 要修改的地方: 安全性保证
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
                break
            else:
                self.follow_path(result_pre)
                self.clear_planner_pc()
                self.open_gripper(target=0.06)
                self.approach(distance=self.reserved_distance)
                # self.approach_safe(distance=self.reserved_distance)
                # self.close_gripper_safe(target=0.02, gripper_force=4)
                self.close_gripper(target=0.0, gripper_force=4)
                
                # self.update_cur_frame(
                #     init_guess=None,
                #     visualize=visualize
                # )
                
                # 转换 joint_dict 到世界坐标系
                fine_dict_w = self.obj_repr.get_joint_param(
                    resolution="fine",
                    frame="world"
                )
                
                # try:
                reaction_motion = Reaction(self.get_force() > 15, JointStopMotion())
                self.move_along_axis(
                    joint_type=fine_dict_w["joint_type"],
                    joint_axis=fine_dict_w["joint_dir"],
                    joint_start=fine_dict_w["joint_start"],
                    moving_distance=self.target_state-self.cur_frame.joint_state,
                    drop_large_move=False,
                    reaction_motion=reaction_motion
                )
                # except Exception as e:
                #     self.open_gripper(target=0.06)
                #     self.franky_robot.recover_from_errors()
                #     self.move_dz(-self.reserved_distance)
                #     continue
                    
                if self.not_good_enough(visualize=visualize):
                    if self.goal_delta < 0 and num_manip == 1:
                        self.move_dz(-self.reserved_distance)
                        continue
                    
                    self.open_gripper(target=0.06)
                    self.move_dz(-self.reserved_distance)
                else:
                    break
    
    def prepare_task_env(self, goal_delta):
        """
        在 explore 和 reconstruct 结束后执行一次 manipulate 任务
        manip_start_state: 物体关节被初始化到这个值 (完全关闭的 joint_state = 0)
        manip_end_state: 算法要操作物体使得其关节变到这个值
        """
        self.goal_delta = goal_delta
        # 进行 robot 的 reset
        # input("Please reset robot and init joint state before you continue: ")
        self.reset_robot()
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
    import yaml
    with open("/home/user/Programs/Embodied_Analogy/embodied_analogy/realworld_environment/drawer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    manipEnv = ManipulateEnv(cfg=cfg)
    # manipEnv.update_cur_frame(visualize=True)
    manipEnv.main()
    manipEnv.reset_robot_safe()
    