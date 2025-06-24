import os
import logging
import sys
import numpy as np
import json
from embodied_analogy.project_config import PROJECT_ROOT, ASSET_PATH
sys.path.append(os.path.join(PROJECT_ROOT, "third_party", "GAPartNet"))
# gapartnet related import 

from embodied_analogy.utility.estimation.coarse_joint_est import (
    coarse_t_from_tracks_3d,
    coarse_estimation,
    coarse_R_from_tracks_3d_augmented
)
from embodied_analogy.environment.manipulate_env import ManipulateEnv
sys.path.append(os.path.join(os.path.dirname(__file__)))
from gapartnet_utils import gapartnet_reconstruct_fixed


class GAPartNet_ManipEnv(ManipulateEnv):
    """
    GeneralFlow baseline 的策略是在 joint_state = 0 的情况下进行一次的 contact transfer, 并进行 joint param 的估计
    之后操作时通过一次 contact transfer 得到抓取位姿, 然后按照 joint state = 0 时估计的 model 进行操作
    """
    def __init__(self, cfg):
        """
        这里的 cfg 来自于 embodied_analogy 测试的那些 cfg
        """
        self.cfg = cfg
        ManipulateEnv.__init__(self, cfg)
        
        # 清空来自 embodied analogy 的 obj_repr 中的 joint_dict
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.coarse_joint_dict = None
        self.obj_repr.fine_joint_dict = None
        
        self.logger.log(logging.INFO, "Running gflow baseline...")
    
    def get_contact_2d(self, frame, manip_type, visualize=False):
        """
        获取当前帧上的 contact 2d
        manip_type: "open" or "close"
        """
        from embodied_analogy.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        # frame: Frame
        # self.base_step()
        # self.initial_frame = self.capture_frame()
        
        # 只在第一次进行 contact transfer, 之后直接进行复用
        self.logger.log(logging.INFO, "Start transfering 2d contact pount to current frame...")
        
        # self.task_cfg["instruction"]
        modified_instruction = manip_type + " the " + self.obj_env_cfg["obj_description"]
        affordance_map_2d = get_ram_affordance_2d(
            query_rgb=frame.rgb,
            instruction=modified_instruction,  
            obj_description=self.obj_description,
            fully_zeroshot=self.explore_env_cfg["fully_zeroshot"],
            visualize=visualize,
            logger=self.logger
        )
        obj_mask = affordance_map_2d.get_obj_mask(visualize=False)
        contact_uv = affordance_map_2d.sample_highest(visualize=visualize)
        
        frame.obj_mask = obj_mask
        frame.contact2d = contact_uv
    
    def recon_stage_general_flow(self, visualize=False):
        """
        基于 self.obj_repr 的 initial_frame 估计 generalFlow, 并得到 joint model 的估计
        """
        # self.base_step()
        # frame = self.capture_frame()
        
        # if frame.contact2d is None or frame.obj_mask is None:
        #     self.get_contact_2d(frame=frame, manip_type="open", visualize=visualize)
        
        frame = self.obj_repr.frames[0]
        
        # (M, T, 3) in camera frame
        general_flow = get_generalFlow(
            frame=frame, 
            # 由于 coarse estimation 本身就是假设轨迹是 open 的, 所以这里设置为固定的 open instruction
            instruction="open_Storage_Furniture", 
            visualize=visualize
        )
        # (M, T, 3) - > (T, M, 3)
        general_flow = np.transpose(general_flow, (1, 0, 2))
        self.general_flow_c = general_flow
        
        if self.recon_env_cfg["use_gt_joint_type"]:
            joint_type = self.obj_repr.gt_joint_dict["joint_type"]
            if joint_type == "revolute":
                joint_dict, _ = coarse_R_from_tracks_3d_augmented(
                    tracks_3d=general_flow,
                    visualize=False,
                    logger=self.logger,
                    num_R_augmented=self.recon_env_cfg["num_R_augmented"]
                )
                joint_dict["joint_type"] = "revolute"
            elif joint_type == "prismatic":
                joint_dict, _ = coarse_t_from_tracks_3d(tracks_3d=general_flow, visualize=False, logger=self.logger)
                joint_dict["joint_type"] = "prismatic"
        else:
            joint_dict = coarse_estimation(
                tracks_3d=general_flow,
                visualize=False,
                logger=self.logger,
                num_R_augmented=self.recon_env_cfg["num_R_augmented"]
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
        # self.explore_result = {
        #     "num_tries": 1,
        #     "has_valid_explore": True,
        #     "joint_type": self.obj_env_cfg["joint_type"],
        #     "joint_state_start": 0,
        #     # 设置为 1, 方便 manipulate 认定 explore 一定成功
        #     "joint_state_end": 1 
        # }
        
        # if self.exp_cfg["save_result"]:
        #     save_json_path = os.path.join(
        #         self.exp_cfg["exp_folder"],
        #         str(self.task_cfg["task_id"]),
        #         "explore_result.json"
        #     )
        #     with open(save_json_path, 'w', encoding='utf-8') as json_file:
        #         json.dump(self.explore_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
                
        # try:
        if True:
            self.recon_result = {}
            
            if self.explore_result["has_valid_explore"]:
                self.logger.log(logging.INFO, f"Valid explore detected, thus start reconstruction...") 
                self.recon_result = self.recon_stage_general_flow()
                self.recon_result["has_valid_recon"] = True
            else:
                self.logger.log(logging.INFO, f"No Valid explore, thus skip reconstruction...") 
                self.recon_result["has_valid_recon"] = False
                self.recon_result["exception"] = "No valid explore."
            
        # except Exception as e:
        #     self.logger.log(logging.ERROR, f"Exception occured during Reconstruct_stage: {e}", exc_info=True)
        #     self.recon_result["has_valid_recon"] = False
        #     self.recon_result["exception"] = str(e)
        
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "recon_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.recon_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
                
    def manip_general_flow_deprecated(self, manip_type, visualize=False):
        """
        根据将 cur_frame 找到一个 grasp, 并按照 articulation mdoel 进行操作
        """
        self.base_step()
        frame: Frame = self.capture_frame()
        
        if frame.contact2d is None or frame.obj_mask is None:
            self.get_contact_2d(frame=frame, manip_type=manip_type, visualize=visualize)
            
        # 确保 frame 的 contact2d 和 obj_mask 已经准备好
        frame.detect_grasp(
            use_anygrasp=self.cfg["algo_cfg"]["use_anygrasp"],
            world_frame=True,
            visualize=visualize,
            asset_path=ASSET_PATH,
            logger=self.logger
        )
        # 
        pc_collision_w, pc_colors = frame.get_env_pc(
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        if frame.grasp_group is None or len(frame.grasp_group) == 0: 
            raise Exception("No grasp_group found for current frame in manipulation stage.")
        
        for grasp_w in frame.grasp_group:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            # 从 general_flow_c 中找到 dir_out_c
            first_frame_flow_c = self.general_flow_c[1, :, :] - self.general_flow_c[0, :, :] # M, 3
            # 由于已经运行了 detect_grasp, 所以 contact_3d 已经被设置过了
            nearest_flow = np.argmin(
                np.linalg.norm(frame.contact3d - self.general_flow_c[0], axis=-1) # M
            ) # M
            # 由于我们产生 flow 的 prompt 固定是 open, 所以现在的 flow 天然的就是 out 
            dir_out_c = first_frame_flow_c[nearest_flow]
            dir_out_c = dir_out_c / max(np.linalg.norm(dir_out_c), 1e-8)
            dir_out_w = np.linalg.inv(frame.Tw2c)[:3, :3] @ dir_out_c
            
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.explore_env_cfg["reserved_distance"])
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            
            if result_pre is not None:
                if visualize:
                    obj_pc, pc_colors = frame.get_obj_pc(
                        use_robot_mask=True,
                        use_height_filter=True,
                        world_frame=True
                    )
                    Tc2w = np.linalg.inv(frame.Tw2c)
                    contact3d_w = Tc2w[:3, :3] @ frame.contact3d + Tc2w[:3, -1]
                    visualize_pc(
                        points=obj_pc, 
                        colors=pc_colors / 255,
                        grasp=grasp, 
                        contact_point=contact3d_w, 
                        post_contact_dirs=[dir_out_w]
                    )
                break
        
        # 没有成功的 planning 结果, 直接返回 False
        if result_pre is None:
            raise Exception("No valid planning result for detected grasp_group during manipulation stage...")
        
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(
            moving_distance=self.explore_env_cfg["reserved_distance"],
            drop_large_move=False
        )
        self.close_gripper()
        
        joint_dict_w = self.obj_repr.get_joint_param(resolution="coarse", frame="world")
        self.move_along_axis(
            joint_type=joint_dict_w["joint_type"],
            joint_axis=joint_dict_w["joint_dir"],
            joint_start=joint_dict_w["joint_dir"],
            moving_distance=self.goal_delta,
            drop_large_move=False
        )
        
        result_dict = {
            "1": self.evaluate()
        }
        self.logger.log(logging.INFO, result_dict)
        return result_dict

    def manip_general_flow(self):
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
            tgt_frame=self.cur_frame
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
                    tmp_manip_result.update(self.manip_general_flow())
                
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