import os
import logging
import sys
import numpy as np
import json
from akm.project_config import PROJECT_ROOT, ASSET_PATH
from akm.environment.explore_env import ExploreEnv
from akm.environment.manipulate_env import ManipulateEnv
from akm.representation.basic_structure import Frame
from akm.utility.utils import numpy_to_json

sys.path.append(os.path.join(os.path.dirname(__file__)))
from gapartnet_utils import gapartnet_reconstruct


class GAPartNet_ManipEnv(ManipulateEnv):
    """
    GeneralFlow baseline 的策略是在 joint_state = 0 的情况下进行一次的 contact transfer, 并进行 joint param 的估计
    之后操作时通过一次 contact transfer 得到抓取位姿, 然后按照 joint state = 0 时估计的 model 进行操作
    """
    def __init__(self, cfg):
        """
        这里的 cfg 来自于 akm 测试的那些 cfg
        """
        self.cfg = cfg
        ManipulateEnv.__init__(self, cfg)
        
        # 清空来自 embodied analogy 的 obj_repr 中的 joint_dict
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.coarse_joint_dict = None
        self.obj_repr.fine_joint_dict = None
        
        self.logger.log(logging.INFO, "Running gpnet baseline...")
    
    def recon_stage_gapartnet(self, visualize=False):
        """
        基于 self.obj_repr 的 frames 的 first 和 last frame 估计 part bbox, 并得到 joint model 的估计
        """
        joint_dict = gapartnet_reconstruct(
            obj_repr=self.obj_repr,
            gapartnet_model=None,
            visualize=visualize,
            use_gt_joint_type=self.recon_env_cfg["use_gt_joint_type"]
        )

        # 将 joint_dict 转换到世界坐标系下
        self.obj_repr.coarse_joint_dict = joint_dict
        self.obj_repr.fine_joint_dict = joint_dict
        
        result = self.obj_repr.compute_joint_error(skip_states=True)
        self.logger.log(logging.INFO, "Reconstruction Result:")
        for k, v in result.items():
            self.logger.log(logging.INFO, f"{k}, {v}")
        return result
    
    def recon_main(self):
  
        try:
        # if True:
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
            # NOTE 同时需要把 matched bbox 也加上去
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr.npy"
            )
            self.obj_repr.save(save_path)
                
    def manip_gapartnet(self):
        """
        根据将 initial_frame 的 grasp 迁移到 cur_frame, 并按照 articulation mdoel 进行操作
        """
        self.base_step()
        self.cur_frame: Frame = self.capture_frame()
        
        # 使用 gt 的 reloc joint state
        self.obj_repr.frames[0].joint_state = self.obj_repr.frames[0].gt_joint_state
        self.cur_frame.joint_state = self.cur_frame.gt_joint_state
        self.ref_ph_to_tgt(
            # ref_frame=self.obj_repr.kframes[0],
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
                moving_distance=self.goal_delta,
                drop_large_move=False
            )
            
        result_dict = {
            "1": self.evaluate()
        }
        self.logger.log(logging.INFO, result_dict)
        return result_dict
    
    def main(self):
        # 保存 explore_result, 和 initial_frame 的 contact point, grasp pose
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
            # if True:
                if self.recon_result["has_valid_recon"]:
                    self.logger.log(logging.INFO, f'Valid reconstruction detected, thus start manipulation...')
                    # manip_type = "open" if manip_end_state > manip_start_state else "close"
                    tmp_manip_result.update(self.manip_gapartnet())
                
            except Exception as e:
                self.logger.log(logging.ERROR, f'Encouter {e} when manipulating, thus only save current state', exc_info=True)
                # self.manip_result["has_valid_manip"] = False
                tmp_manip_result["exception"] = str(e)
                # tmp_manip_result.update({0: self.evaluate()})

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
                