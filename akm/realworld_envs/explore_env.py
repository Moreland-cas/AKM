import os
import yaml
import copy
import json
import logging
import threading
import numpy as np
from franky import (
    CartesianMotion,
    Affine,
    ReferenceType,
    Reaction,
    CartesianStopMotion,
)

from akm.utility.utils import (
    camera_to_world,
    visualize_pc,
    numpy_to_json
)
from akm.realworld_envs.obj_env import ObjEnv
from akm.utility.utils import clean_pc_np
from akm.utility.constants import ASSET_PATH
from akm.representation.basic_structure import Frame


class ExploreEnv(ObjEnv):
    def __init__(self, cfg):    
        super().__init__(cfg=cfg)
            
        self.explore_env_cfg = cfg["explore_env_cfg"]
        self.algo_cfg = cfg["algo_cfg"]
        
        self.pertubation_distance = self.explore_env_cfg["pertubation_distance"]
        self.reserved_distance = self.explore_env_cfg["reserved_distance"]
        self.valid_thresh = self.explore_env_cfg["valid_thresh"]
        self.instruction = self.task_cfg["instruction"]
        
        self.update_sigma = self.explore_env_cfg["update_sigma"]
        self.max_tries = self.explore_env_cfg["max_tries"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.has_valid_explore = False
        
        self.joint_type = self.obj_env_cfg["joint_type"]
        
    def explore_stage(self, visualize=False):
        """
        Explore multiple times until a sequence of operations that meets the requirements is found, or exit after trying enough times
        """
        # First get affordance_map_2d, then start to explore and modify affordance_map_2d
        from akm.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        self.reset_robot()
        initial_frame = self.capture_frame()
        
        # Only perform contact transfer for the first time, and reuse directly thereafter
        self.logger.log(logging.INFO, "Start transfering 2d contact affordance map...")
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=initial_frame.rgb,
            instruction=self.task_cfg["instruction"],
            obj_description=self.obj_description,
            fully_zeroshot=self.explore_env_cfg["fully_zeroshot"],
            visualize=visualize,
            logger=self.logger
        )
        
        if not self.explore_env_cfg["contact_analogy"]:
            self.affordance_map_2d.uninit_cosmap()
            self.logger.log(logging.INFO, "Detected contact_analogy flag = False, disable Contact Analogy")
        else:
            self.logger.log(logging.INFO, "Detected contact_analogy flag = True, use Contact Analogy")
        
        # Save the cos_map obtained when initializing the affordance map
        if self.exp_cfg["save_vis"]:
            self.obj_repr.save_for_vis.update({
                "explore_cos_map": [np.copy(self.affordance_map_2d.cos_map)]
            })
            
        # Save the first frame here
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
            # Initialize related states, you need to clear the frames obtained previously
            self.obj_repr.clear_frames()
            if self.num_tries == 0:
                self.reset_robot()
            else:
                self.reset_robot_safe()
            
            self.logger.log(logging.INFO, f"[{self.num_tries + 1}|{self.max_tries}] Start exploring once...")
            actually_tried, explore_uv = self.explore_once(visualize=visualize, idx=self.num_tries)
            self.num_tries += 1
            if not actually_tried:
                self.logger.log(logging.INFO, "The planning path is not valid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=True)
                    
                    # Save the cos_map of the affordance map after update
                    if self.exp_cfg["save_vis"]:
                        self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))
                continue
            
            if self.check_valid(visualize=visualize):
                self.logger.log(logging.INFO, "Check valid, break explore loop")
                break
            else:
                self.logger.log(logging.INFO, "Check invalid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=True)
                
                # Save the cos_map of the affordance map after update
                if self.exp_cfg["save_vis"]:
                    self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))

        if self.exp_cfg["save_vis"]:
            self.obj_repr.save_for_vis["aff_map"] = copy.deepcopy(self.affordance_map_2d)
                    
        # save explore data
        if visualize:
            self.obj_repr.visualize()
        
        if self.obj_repr.frames.num_frames() == 0:
            joint_state_end = 0
            
        result_dict = {
            "num_tries": self.num_tries,
            "has_valid_explore": self.has_valid_explore,
            "joint_type": self.joint_type,
        }
        self.logger.log(logging.INFO, f"exploration stage result: {result_dict}")
        
        if not self.has_valid_explore:
            self.logger.log(logging.INFO, "In summary, no valid exploration during explore phase!")
        else:
            self.logger.log(logging.INFO, "In summary, get valid exploration during explore phase!")
        
        if not self.has_valid_explore:
            raise Exception("No valid explore found!")
        
        return result_dict
    
    def explore_once(self, visualize=False, idx=0):
        """
        Performs an exploration in the current state. By default, the robot arm is in the reset state.
        Returns explore_ok, explore_uv:
            explore_ok: bool, indicating whether the plan phase was successful.
            explore_uv: np.array([2,]), representing the UVs of the contact points in this attempt.
        """        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        for _ in range(30):
            self.capture_frame()
        cur_frame = self.capture_frame()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        
        if self.explore_env_cfg["use_IOR"]:
            self.logger.log(logging.INFO, "Detected use_IOR flag = True, use Inhibition of Return")
            contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        else:
            # sample_prob returns a list of N, 2. The larger the alpha, the denser the sampling.
            self.logger.log(logging.INFO, "Detected use_IOR flag = False, do not update affoedance map")
            contact_uv = self.affordance_map_2d.sample_prob(alpha=1, num_samples=1, return_rgb_frame=True, visualize=False)[0]
        
        cur_frame.obj_mask = obj_mask
        cur_frame.contact2d = contact_uv
        
        # Here rgb_np, depth_np may be different from those stored in affordance_map_2d, but it should not be too different
        cur_frame.detect_grasp(
            use_anygrasp=self.algo_cfg["use_anygrasp"],
            world_frame=True,
            visualize=visualize,
            asset_path=ASSET_PATH,
            logger=self.logger
        )
        
        if cur_frame.grasp_group is None:
            return False, contact_uv
        
        contact3d_w = camera_to_world(
            point_camera=cur_frame.contact3d[None],
            extrinsic_matrix=Tw2c
        )[0]
        dir_out_w = Tc2w[:3, :3] @ cur_frame.dir_out # 3
        
        result_pre = None
        # NOTE: get_obj_pc is not used here, because each explore will have a new cur_frame, so it does not always have the latest obj_mask information
        cur_frame: Frame
        pc_collision_w, pc_colors = cur_frame.get_obj_pc(
            use_robot_mask=False, 
            use_height_filter=True,
            world_frame=True
        )
        # The pc here still contains franka's, so it needs to be modified. In addition, the point cloud of the wall and camera needs to be added
        self.update_point_cloud_with_wall(clean_pc_np(pc_collision_w))
        
        for grasp_w in cur_frame.grasp_group:
            # Get the poses of pre_ph_grasp and ph_grasp based on best grasp
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w =self.anyGrasp2ph(grasp=grasp)
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if result_pre is not None:
                if visualize or True:
                    visualize_pc(
                        points=pc_collision_w, 
                        point_size=5,
                        colors=pc_colors / 255,
                        grasp=grasp, 
                        contact_point=contact3d_w, 
                        post_contact_dirs=[dir_out_w]
                    )
                break
        
        # Actually execute the proposal and record data during the process
        if result_pre is None:
            return False, contact_uv
        
        self.follow_path(result_pre)
        self.open_gripper(target=0.06)
        self.clear_planner_pc()
        
        self.approach(distance=self.reserved_distance)
        self.close_gripper(target=0.0, gripper_force=4)
        
        # Execute move asynchronously, and set self.recording to False after execution.
        self.dir_out_w = dir_out_w
        move_thread = threading.Thread(target=self.async_move_out, daemon=True)
        move_thread.start()
        self.recording = True
        
        while self.recording:
            cur_frame = self.capture_frame()
            self.obj_repr.frames.append(copy.deepcopy(cur_frame))
    
        return True, contact_uv
    
    def async_move_out(self):
        """
        Keep moving when the distance is not enough
        """
        init_Tph2w = self.robot.get_ee_pose(as_matrix=True)
        def get_moved_distance():
            cur_Tph2w = self.robot.get_ee_pose(as_matrix=True)
            distance = np.linalg.norm(init_Tph2w[:3, -1] - cur_Tph2w[:3, -1])
            return distance
        
        num_tries = 2
        cur_tries = 0
        while (cur_tries < num_tries) and (get_moved_distance() < self.pertubation_distance * 0.8):
            pull_out = CartesianMotion(
                Affine([0, 0, -self.pertubation_distance / num_tries]), 
                ReferenceType.Relative, 
                relative_dynamics_factor=0.05
            )
            reaction = Reaction(self.get_force() > 20, CartesianStopMotion())
            pull_out.add_reaction(reaction)
            self.franky_robot.move(pull_out, asynchronous=True)
            self.franky_robot.join_motion()
            cur_tries += 1
            self.open_gripper(0.06)
            self.franky_robot.recover_from_errors()
            self.close_gripper(target=0.0, gripper_force=4)
                
        self.recording = False
    
    def check_valid(self, visualize=False): 
        """
        Track the frames and determine whether the moving_part is moving a lot based on the clustering results. If so, consider it valid.
        Compared to the deprecated version, this function can better handle situations like "a crack in a cabinet, no significant change in joint state, but a sudden change in crack depth.
        This function writes the tracks2d, tracks3d, and moving mask of obj_repr
        """
        
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
        
        if visualize:
            visualize_pc(
                points=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[0],
                point_size=5,
                colors=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[1] / 255.,
                tracks_3d=self.obj_repr.frames.track3d_seq,
                tracks_n_step=None,
                tracks_t_step=3,
                tracks_norm_threshold=0.2e-2,
            )
            
            visualize_pc(
                points=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[0],
                point_size=5,
                colors=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[1] / 255.,
                tracks_3d=self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :],
                tracks_n_step=None,
                tracks_t_step=3,
                tracks_norm_threshold=0.2e-2,
            )
            
        moving_tracks = self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :]
        mean_delta = np.linalg.norm(moving_tracks[-1] - moving_tracks[0], axis=-1).mean()
        if mean_delta > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
    
    def main(self):
        try:
            self.explore_result = {}
            self.explore_result = self.explore_stage(visualize=False)
                    
        except Exception as e:
            self.logger.log(logging.ERROR, f"Explore exception occured: {e}", exc_info=True)
            
            self.explore_result["has_valid_explore"] = False
            self.explore_result["joint_type"] = self.obj_env_cfg["joint_type"]
            self.explore_result["exception"] = str(e)

        self.reset_robot_safe()
        
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "explore_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.explore_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
        
        if self.exp_cfg["save_vis"]:
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr_explore.npy"
            )
            self.obj_repr.save(save_path)
                    
if __name__ == "__main__":
    with open("/home/user/Programs/AKM/akm/realworld_envs/drawer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    exploreEnv = ExploreEnv(cfg=cfg)
    exploreEnv.main()