import os
import time
import copy
import json
import math
import logging
import numpy as np

from akm.utility.utils import (
    camera_to_world,
    visualize_pc,
    numpy_to_json
)
from akm.simulated_envs.obj_env import ObjEnv
from akm.utility.constants import ASSET_PATH
from akm.utility.timer import Timer


class ExploreEnv(ObjEnv):
    def __init__(self, cfg):    
        super().__init__(cfg=cfg)
            
        self.explore_env_cfg = cfg["explore_env_cfg"]
        self.algo_cfg = cfg["algo_cfg"]
        
        self.record_fps = self.explore_env_cfg["record_fps"]
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = self.explore_env_cfg["pertubation_distance"]
        self.reserved_distance = self.explore_env_cfg["reserved_distance"]
        self.valid_thresh = self.explore_env_cfg["valid_thresh"]
        self.instruction = self.task_cfg["instruction"]
        
        self.update_sigma = self.explore_env_cfg["update_sigma"]
        self.max_tries = self.explore_env_cfg["max_tries"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.has_valid_explore = False
        
        self.joint_type = self.obj_env_cfg["joint_type"]
        
        # Timer
        self.explore_timer = Timer()
        
    def explore_stage(self, visualize=False):
        """
        Explore multiple times until a matching sequence of operations is found, or exit after enough attempts.
        NOTE: Currently, this is direct reuse; it may need to be switched to fusion later.
        """
        # First get affordance_map_2d, then start exploring and modifying affordance_map_2d
        from akm.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        initial_frame = super().capture_frame()
        
        self.logger.log(logging.INFO, "Start transfering 2d contact affordance map...")
        get_ram_affordance_2d_start = time.time()
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=initial_frame.rgb,
            instruction=self.task_cfg["instruction"],
            obj_description=self.obj_description,
            fully_zeroshot=self.explore_env_cfg["fully_zeroshot"],
            visualize=visualize,
            logger=self.logger
        )
        get_ram_affordance_2d_end = time.time()
        self.explore_timer.update("get_ram_affordance_2d", get_ram_affordance_2d_end - get_ram_affordance_2d_start)
        
        if not self.explore_env_cfg["contact_analogy"]:
            self.affordance_map_2d.uninit_cosmap()
            self.logger.log(logging.INFO, "Detected contact_analogy flag = False, disable Contact Analogy")
        else:
            self.logger.log(logging.INFO, "Detected contact_analogy flag = True, use Contact Analogy")
        
        if self.exp_cfg["save_obj_repr"]:
            self.obj_repr.save_for_vis.update({
                "explore_cos_map": [np.copy(self.affordance_map_2d.cos_map)]
            })
            
        self.obj_repr.obj_description = self.obj_description
        self.obj_repr.K = self.camera_intrinsic
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.frames.K = self.camera_intrinsic
        self.obj_repr.frames.Tw2c = self.camera_extrinsic
        self.obj_repr.initial_frame = initial_frame
        
        self.max_tries = self.explore_env_cfg["max_tries"]
        self.logger.log(logging.INFO, f"Start exploring..., you have {self.max_tries} chances to explore...")
        
        if self.max_tries == 1:
            self.logger.log(logging.INFO, "Only try once, Disable Interactive perception")
        else:
            self.logger.log(logging.INFO, "Enable Interactive perception")
            
        self.num_tries = 0
        while self.num_tries < self.max_tries:
            if self.num_tries >= 1:
                if self.exp_cfg["save_obj_repr"]:
                    self.obj_repr.save_for_vis[str(self.num_tries)] = [
                        copy.deepcopy(self.obj_repr.frames[0]),
                        copy.deepcopy(self.obj_repr.frames[-1])
                    ]
                    
            self.obj_repr.clear_frames()
            
            if self.num_tries == 0:
                self.reset_robot()
            else:
                self.reset_robot_safe()
            
            self.logger.log(logging.INFO, f"[{self.num_tries + 1}|{self.max_tries}] Start exploring once...")
            # explore_once_start = time.time()
            actually_tried, explore_uv = self.explore_once(visualize=visualize)
            # explore_once_end = time.time()
            # self.explore_timer.update("explore_once", explore_once_end - explore_once_start)
            self.num_tries += 1
            if not actually_tried:
                self.logger.log(logging.INFO, "The planning path is not valid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                    
                    if self.exp_cfg["save_obj_repr"]:
                        self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))
                continue
            
            check_valid_start = time.time()
            is_valid = self.check_valid(visualize=visualize)
            check_valid_end = time.time()
            self.explore_timer.update("check_valid", check_valid_end - check_valid_start)
            if is_valid:
                self.logger.log(logging.INFO, "Check valid, break explore loop")
                break
            else:
                self.logger.log(logging.INFO, "Check invalid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                
                if self.exp_cfg["save_obj_repr"]:
                    self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))

        if self.exp_cfg["save_obj_repr"]:
            self.obj_repr.save_for_vis["aff_map"] = copy.deepcopy(self.affordance_map_2d)
                    
        # save explore data
        if visualize:
            self.obj_repr.visualize()
        
        if self.obj_repr.frames.num_frames() == 0:
            joint_state_end = 0
        else:
            joint_state_end = self.get_active_joint_state() - self.obj_repr.frames[0].gt_joint_state
        result_dict = {
            "num_tries": self.num_tries,
            "has_valid_explore": self.has_valid_explore,
            "joint_type": self.joint_type,
            "joint_state_start": 0,
            "joint_state_end": joint_state_end
        }
        self.logger.log(logging.INFO, f"exploration stage result: {result_dict}")
        
        if not self.has_valid_explore:
            self.logger.log(logging.INFO, "In summary, no valid exploration during explore phase!")
        else:
            self.logger.log(logging.INFO, "In summary, get valid exploration during explore phase!")
        
        return result_dict
    
    def explore_once(self, visualize=False):
        """
        Performs an exploration in the current state. By default, the robot arm is in the reset state.
        Returns explore_ok, explore_uv:
            explore_ok: bool, indicating whether the plan phase was successful.
            explore_uv: np.array([2,]), representing the UVs of the contact points in this attempt.
        """        
        explore_once_prepare_start = time.time()
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        self.base_step()
        cur_frame = self.capture_frame()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        
        if self.explore_env_cfg["use_IOR"]:
            self.logger.log(logging.INFO, "Detected use_IOR flag = True, use Inhibition of Return")
            contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        else:
            # sample_prob returns a list of N, 2. The larger the alpha, the denser the sampling.
            self.logger.log(logging.INFO, "Detected use_IOR flag = False, do not update affordance map")
            contact_uv = self.affordance_map_2d.sample_prob(alpha=1, num_samples=1, return_rgb_frame=True, visualize=False)[0]
        
        cur_frame.obj_mask = obj_mask
        cur_frame.contact2d = contact_uv
        
        # Here rgb_np, depth_np may be different from those stored in affordance_map_2d, but it should not be too different
        detect_grasp_start = time.time()
        cur_frame.detect_grasp(
            use_anygrasp=self.algo_cfg["use_anygrasp"],
            world_frame=True,
            visualize=visualize,
            asset_path=ASSET_PATH,
            logger=self.logger
        )
        detect_grasp_end = time.time()
        self.explore_timer.update("detect_grasp", detect_grasp_end - detect_grasp_start)
        
        if cur_frame.grasp_group is None:
            return False, contact_uv
        
        contact3d_w = camera_to_world(
            point_camera=cur_frame.contact3d[None],
            extrinsic_matrix=Tw2c
        )[0]
        dir_out_w = Tc2w[:3, :3] @ cur_frame.dir_out # 3
        
        result_pre = None
        # NOTE: get_obj_pc is not used here, because each explore will have a new cur_frame, so it does not always have the latest obj_mask information
        pc_collision_w, pc_colors = cur_frame.get_env_pc(
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        for grasp_w in cur_frame.grasp_group:
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if result_pre is not None:
                if visualize:
                    visualize_pc(
                        points=pc_collision_w, 
                        colors=pc_colors / 255,
                        grasp=grasp, 
                        contact_point=contact3d_w, 
                        post_contact_dirs=[dir_out_w]
                    )
                break
        
        if result_pre is None:
            return False, contact_uv
        
        explore_once_prepare_end = time.time()
        self.explore_timer.update("explore_once_prepare", explore_once_prepare_end - explore_once_prepare_start)
        
        explore_once_move_start = self.cur_steps * self.phy_timestep
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(
            moving_distance=self.reserved_distance,
            drop_large_move=False
        )
        self.close_gripper()
        
        self.step = self.explore_step
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=dir_out_w,
            joint_start=None,
            moving_distance=self.pertubation_distance,
            drop_large_move=False
        )
        explore_once_move_end = self.cur_steps * self.phy_timestep
        self.explore_timer.update("explore_once_move", explore_once_move_end - explore_once_move_start)
        self.step = self.base_step 
        return True, contact_uv
    
    def explore_step(self):
        self.base_step()
        
        cur_steps = self.cur_steps % self.record_interval
        if cur_steps == 0:
            cur_frame = self.capture_frame()
            self.obj_repr.frames.append(cur_frame)
    
    def check_valid(self, visualize=False): 
        # Track the frames and determine whether the moving_part is moving a lot based on the clustering results. If so, consider it valid.
        # Compared to the deprecated version, this function can better handle situations like "a crack in a cabinet, no significant change in joint state, but a sudden change in crack depth."
        # This function writes the tracks2d, tracks3d, and moving mask of obj_repr
        if self.obj_repr.frames.num_frames() == 0:
            return False
        
        self.obj_repr.frames[0].segment_obj(
            obj_description=self.obj_env_cfg["obj_description"],
            post_process_mask=True,
            filter=True,
            visualize=visualize
        )
        self.obj_repr.frames[0].sample_points(num_points=self.explore_env_cfg["num_initial_pts"], visualize=visualize)
        self.obj_repr.frames.track_points(visualize=visualize)
        self.obj_repr.frames.track2d_to_3d(filter=True, visualize=visualize)
        self.obj_repr.frames.cluster_track3d(visualize=visualize)
        
        moving_tracks = self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :]
        mean_delta = np.linalg.norm(moving_tracks[-1] - moving_tracks[0], axis=-1).mean()
        if mean_delta > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
        
    def main(self):
        try:
        # if True:
            self.explore_result = {}
            self.explore_result = self.explore_stage()
        # pass
        except Exception as e:
            self.logger.log(logging.ERROR, f"Explore exception occured: {e}", exc_info=True)
            
            self.explore_result["has_valid_explore"] = False
            self.explore_result["joint_type"] = self.obj_env_cfg["joint_type"]
            self.explore_result["exception"] = str(e)

        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "explore_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.explore_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
                
            # save Timer info
            self.explore_timer.save(os.path.join(self.exp_cfg["exp_folder"], str(self.task_cfg["task_id"]), "explore_timer.json"))