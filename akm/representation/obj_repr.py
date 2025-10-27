import time
import copy
import logging
import numpy as np

from akm.utility.utils import (
    farthest_scale_sampling,
    joint_data_to_transform_np,
    get_depth_mask,
    camera_to_image,
    depth_image_to_pointcloud,
    line_to_line_distance
)
from akm.utility.constants import *
from akm.representation.basic_structure import Data, Frame, Frames
from akm.utility.estimation.fine_joint_est import fine_estimation
from akm.utility.estimation.coarse_joint_est import coarse_estimation


class Obj_repr(Data):
    def __init__(self):
        
        self.obj_description = None
        self.initial_frame = Frame()
        self.frames = Frames()
        self.K = None
        self.Tw2c = None
        
        self.kframes = Frames()
        self.track_type = None # either "open" or "close"
        
        # The coarse_joint_dict and fine_joint_dict and gt_joint_dict here are estimated in the camera coordinate system
        self.coarse_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
        self.fine_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
        self.gt_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
        
        # for visualization
        self.save_for_vis = {}
    
    def setup_logger(self, logger):
        self.logger = logger
    
    def get_joint_param(self, resolution="coarse", frame="world"):
        assert resolution in ["coarse", "fine", "gt"]
        assert frame in ["world", "camera"]
        if resolution == "coarse":
            if self.coarse_joint_dict["joint_type"] is None:
                assert "coarse joint dict is not found"
            joint_dict = copy.deepcopy(self.coarse_joint_dict)
            if frame == "world":
                Tc2w = np.linalg.inv(self.Tw2c)
                joint_dict["joint_dir"] = Tc2w[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tc2w[:3, :3] @ joint_dict["joint_start"] + Tc2w[:3, 3]
        elif resolution == "fine":
            if self.fine_joint_dict["joint_type"] is None:
                assert "fine joint dict is not found"
            joint_dict = copy.deepcopy(self.fine_joint_dict)
            if frame == "world":
                Tc2w = np.linalg.inv(self.Tw2c)
                joint_dict["joint_dir"] = Tc2w[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tc2w[:3, :3] @ joint_dict["joint_start"] + Tc2w[:3, 3]
        elif resolution == "gt":
            if self.gt_joint_dict["joint_type"] is None:
                assert "Ground truth joint dict is not found"
            joint_dict = copy.deepcopy(self.gt_joint_dict)
            if frame == "camera":
                Tw2c = self.Tw2c
                joint_dict["joint_dir"] = Tw2c[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tw2c[:3, :3] @ joint_dict["joint_start"] + Tw2c[:3, 3]
        return joint_dict
    
    def compute_joint_error(self, skip_states=False):
        """
        Calculate the difference between coarse_joint_dict, fine_joint_dict, and gt_joint_dict, print, and save.
        Also calculate joint_state_error for frames and kframes.
        """
        coarse_w = self.get_joint_param(resolution="coarse", frame="world")
        fine_w = self.get_joint_param(resolution="fine", frame="world")
        gt_w = self.get_joint_param(resolution="gt", frame="world")
        result = {
            "coarse_w": coarse_w,
            "fine_w": fine_w,
            "gt_w": gt_w,
            "coarse_loss": {
                "type_err": 0,
                "angle_err": 0,
                "pos_err": 0,
            },
            "fine_loss": {
                "type_err": 0,
                "angle_err": 0,
                "pos_err": 0,
            }
        }
        # NOTE: Here we need to take a minimum for angle_err
        result["coarse_loss"]["type_err"] = 1 if coarse_w["joint_type"] != gt_w["joint_type"] else 0
        coarse_dir_dot = np.clip(np.dot(coarse_w["joint_dir"], gt_w["joint_dir"]), -1, 1)
        if np.arccos(coarse_dir_dot) > np.pi / 2:
            result["coarse_loss"]["angle_err"] = np.pi - np.arccos(coarse_dir_dot)
        else:
            result["coarse_loss"]["angle_err"] = np.arccos(coarse_dir_dot)
        
        result["fine_loss"]["type_err"] = 1 if fine_w["joint_type"] != gt_w["joint_type"] else 0
        fine_dir_dot = np.clip(np.dot(fine_w["joint_dir"], gt_w["joint_dir"]), -1, 1)
        if np.arccos(fine_dir_dot) > np.pi / 2:
            result["fine_loss"]["angle_err"] = np.pi - np.arccos(fine_dir_dot)
        else:
            result["fine_loss"]["angle_err"] = np.arccos(fine_dir_dot)
        
        if gt_w["joint_type"] == "revolute":
            result["coarse_loss"]["pos_err"] = line_to_line_distance(
                P1=coarse_w["joint_start"],
                d1=coarse_w["joint_dir"],
                P2=gt_w["joint_start"],
                d2=gt_w["joint_dir"]
            )
            result["fine_loss"]["pos_err"] = line_to_line_distance(
                P1=fine_w["joint_start"],
                d1=fine_w["joint_dir"],
                P2=gt_w["joint_start"],
                d2=gt_w["joint_dir"]
            )
        if skip_states:
            return result
        
        coarse_joint_state_recon = self.coarse_joint_dict["joint_states"]
        fine_joint_state_recon = self.fine_joint_dict["joint_states"]
        
        coarse_joint_state_gt = self.frames.get_gt_joint_states()
        coarse_joint_state_gt -= coarse_joint_state_gt[0]
        
        fine_joint_state_gt = self.kframes.get_gt_joint_states()
        fine_joint_state_gt -= fine_joint_state_gt[0]
        
        result["gt_w"]["fine_joint_states"] = fine_joint_state_gt
        result["gt_w"]["coarse_joint_states"] = coarse_joint_state_gt
        
        # Calculate state_err, which is the average value of l1 loss
        result["coarse_loss"]["state_err"] = np.abs(coarse_joint_state_recon - coarse_joint_state_gt).mean()
        result["fine_loss"]["state_err"] = np.abs(fine_joint_state_recon - fine_joint_state_gt).mean()
        
        return result
    
    def clear_frames(self):
        self.frames.clear()
        
    def clear_kframes(self):
        self.kframes.clear()
    
    def initialize_kframes(self, num_kframes, save_memory=True):
        self.clear_kframes()
        
        self.kframes.fps = self.frames.fps
        self.kframes.K = self.frames.K
        self.kframes.Tw2c = self.frames.Tw2c
        
        self.kframes.moving_mask = self.frames.moving_mask
        self.kframes.static_mask = self.frames.static_mask
        
        # NOTE: Here we need to ensure that frames[0] is selected into kframes, because ph_pose is stored in frames[0]
        kf_idxs = farthest_scale_sampling(
            arr=self.coarse_joint_dict["joint_states"],
            M=num_kframes,
            include_first=True
        )
        self.kf_idxs = kf_idxs
        
        self.kframes.track2d_seq = self.frames.track2d_seq[kf_idxs, ...]
        
        for i, kf_idx in enumerate(kf_idxs):
            tmp_frame = copy.deepcopy(self.frames[kf_idx])
            self.kframes.append(tmp_frame)
        
        if save_memory:
            self.clear_frames()
    
    def coarse_joint_estimation(self, visualize=False, num_R_augmented=100):
        coarse_joint_dict = coarse_estimation(
            tracks_3d=self.frames.track3d_seq[:, self.frames.moving_mask, :], 
            visualize=visualize,
            logger=self.logger,
            num_R_augmented=num_R_augmented,
        )
        self.coarse_joint_dict = coarse_joint_dict
        self.frames.write_joint_states(coarse_joint_dict["joint_states"])
    
    def fine_joint_estimation(self, lr=1e-3, visualize=False):
        joint_type = self.coarse_joint_dict["joint_type"]
        fine_joint_dict = fine_estimation(
            K=self.K,
            joint_type=self.coarse_joint_dict["joint_type"],
            joint_dir=self.coarse_joint_dict["joint_dir"],
            joint_start=self.coarse_joint_dict["joint_start"],
            joint_states=self.kframes.get_joint_states(),
            depth_seq=self.kframes.get_depth_seq(),
            dynamic_seq=self.kframes.get_dynamic_seq(),
            opti_joint_dir=True,
            opti_joint_start=(joint_type=="revolute"),
            opti_joint_states_mask=np.arange(self.kframes.num_frames())!=0,
            lr=lr, # 1mm
            gt_joint_dict=self.get_joint_param(resolution="gt", frame="camera"),
            visualize=visualize,
            logger=self.logger
        )
        # Write the updated joint_dict and joint_states back to obj_repr here
        self.kframes.write_joint_states(fine_joint_dict["joint_states"])
        self.fine_joint_dict = fine_joint_dict
        
        # By default, the track is open in the explore phase
        track_type = "open"
        self.track_type = track_type
        
    def reconstruct(
        self,
        num_kframes=5,
        obj_description="drawer",
        fine_lr=1e-3,
        evaluate=False,
        save_memory=False,
        visualize=True,
        num_R_augmented=100
    ):
        """
        Recover the joint state dict from frames
        """
        coarse_start_time = time.time()
        self.coarse_joint_estimation(visualize=visualize, num_R_augmented=num_R_augmented)
        coarse_end_time = time.time()
        fine_start_time = time.time()
        self.initialize_kframes(num_kframes=num_kframes, save_memory=save_memory)
        self.kframes.segment_obj(obj_description=obj_description, visualize=visualize)
        self.kframes.classify_dynamics(
            filter=True,
            joint_dict=self.coarse_joint_dict,
            visualize=visualize
        )
        self.fine_joint_estimation(lr=fine_lr, visualize=visualize)
        fine_end_time = time.time()
        result = None
        if evaluate:
            if self.gt_joint_dict["joint_type"] is None:
                raise Exception("evaluate=True in obj_repr.reconstruct() but no gt_joint_dict found!")
            result = self.compute_joint_error()
            self.logger.log(logging.INFO, "Reconstruction Result:")
            for k, v in result.items():
                self.logger.log(logging.INFO, f"{k}, {v}")
            result["coarse_time"] = coarse_end_time - coarse_start_time
            result["fine_time"] = fine_end_time - fine_start_time
        return result
    
    def update_state(self, query_frame: Frame):
        """
        Make a rough estimate of the joint_state of the query_frame.
        The strategy is to sample multiple joint states, then obtain multiple transformed_moving_pcs in turn.
        Then find the joint state closest to the query_frame.

        NOTE: Since we force kframes[0] to be the manipulated first frame, and the joint state of the first frame is 0,
        then all subsequent joint_deltas are directly added to the joint_state of kframes[0].
        """
        # Select the sample range based on the joint type
        if self.coarse_joint_dict["joint_type"] == "revolute":
            joint_delta = np.pi / 2.
        elif self.coarse_joint_dict["joint_type"] == "prismatic":
            joint_delta = 0.5
        
        sampled_states = np.linspace(0, joint_delta, 15)
        # Get the moving_part in kframes[0]
        moving_pc = depth_image_to_pointcloud(
            depth_image=self.kframes[0].depth, 
            mask=self.kframes[0].dynamic_mask == MOVING_LABEL, 
            K=self.K
        ) # N, 3
        
        best_err = 1e10
        best_matched_idx = -1
        for i, sampled_state in enumerate(sampled_states):
            Tref2tgt = joint_data_to_transform_np(
                joint_type=self.fine_joint_dict["joint_type"], 
                joint_dir=self.fine_joint_dict["joint_dir"],
                joint_start=self.fine_joint_dict["joint_start"],
                joint_state_ref2tgt=sampled_state
            )
            tf_moving_pc = (Tref2tgt[:3, :3] @ moving_pc.T).T + Tref2tgt[:3, 3] # N, 3
            tf_moving_uv, moving_depth = camera_to_image(tf_moving_pc, self.K) # (N, 2), (N, )
            
            # Filter tf_moving_uv that exceeds the depth range
            in_img_mask = (tf_moving_uv[:, 0] >= 0) & (tf_moving_uv[:, 0] < query_frame.depth.shape[1]) & \
                          (tf_moving_uv[:, 1] >= 0) & (tf_moving_uv[:, 1] < query_frame.depth.shape[0])
            tf_moving_uv, moving_depth = tf_moving_uv[in_img_mask], moving_depth[in_img_mask]
            
            # Read the depth of query_frame at tf_moving_uv and compare it with moving_depth
            query_depth = query_frame.depth[tf_moving_uv[:, 1].astype(np.int32), tf_moving_uv[:, 0].astype(np.int32)]
            cur_mean_err = np.abs(query_depth - moving_depth).mean()
            if cur_mean_err < best_err:
                best_err = cur_mean_err
                best_matched_idx = i
        query_state = sampled_states[best_matched_idx] 
        
        self.logger.log(logging.INFO, f"Guess query state through sampling: {query_state}")
        query_frame.joint_state = query_state
    
    def update_dynamic(self, query_frame: Frame, visualize=False):
        """
        Use the kframes dynamics in obj_repr to update the query_frame dynamics.
        NOTE:
        Requires the query_frame joint_state to be non-null.
        """
        K = self.K
        ref_joint_states = self.kframes.get_joint_states()
        assert len(ref_joint_states) == len(self.kframes)
        
        num_ref = len(self.kframes)
        ref_depths = self.kframes.get_depth_seq()
        ref_dynamics = self.kframes.get_dynamic_seq()
        
        query_moving = np.zeros_like(query_frame.depth).astype(np.bool_) # H, W
        for i in range(num_ref):
            ref_moving_pc = depth_image_to_pointcloud(ref_depths[i], ref_dynamics[i]==MOVING_LABEL, K) # N, 3
            Tref2query = joint_data_to_transform_np(
                joint_type=self.fine_joint_dict["joint_type"],
                joint_dir=self.fine_joint_dict["joint_dir"],
                joint_start=self.fine_joint_dict["joint_start"],
                joint_state_ref2tgt=query_frame.joint_state-ref_joint_states[i]
            )
            ref_moving_pc_aug = np.concatenate([ref_moving_pc, np.ones((len(ref_moving_pc), 1))], axis=1) # N, 4
            moving_pc = (ref_moving_pc_aug @ Tref2query.T)[:, :3] # N, 3
            moving_uv, _ = camera_to_image(moving_pc, K) # N, 2
            moving_uv = moving_uv.astype(np.int32)
            
            in_img_mask = (moving_uv[:, 0] >= 0) & (moving_uv[:, 0] < query_frame.depth.shape[1]) & \
                          (moving_uv[:, 1] >= 0) & (moving_uv[:, 1] < query_frame.depth.shape[0])
            moving_uv = moving_uv[in_img_mask]
            
            tmp_moving = np.zeros_like(query_moving)
            tmp_moving[moving_uv[:, 1], moving_uv[:, 0]] = True # H, W
            query_moving = query_moving | tmp_moving
            break
        
        # Filter query_dynamic with depth_mask and robot_mask
        depth_mask = get_depth_mask(query_frame.depth, K, query_frame.Tw2c)
        query_moving = query_moving & depth_mask & (~query_frame.robot_mask)
        
        # First assign the entire image to UNKNOWN
        query_dynamic = np.ones_like(query_moving) * UNKNOWN_LABEL  
        
        # Then assign the moving part to MOVING_LABEL
        query_dynamic[query_moving] = MOVING_LABEL
        query_frame.dynamic_mask = query_dynamic
        
        if visualize:
            viewer = napari.Viewer()
            viewer.title = "update dynamic of query frame"
            self.kframes[0]._visualize(viewer, prefix="initial_kframe")
            query_frame._visualize(viewer, prefix="query")
            napari.run()
    
    def reloc(
        self,
        query_frame: Frame,
        init_guess=None,
        reloc_lr=3e-3,
        visualize=False
    ) -> Frame:
        """
        Restore the joint_state and dynamics of the query frame.
        Dynamics are restored from sam2 and kframes.

        query_frame:
            Requires query_depth and query_dynamic
        """
        self.logger.log(logging.INFO, "Start Relocalizing...")
        # First get the mask of the object in the current frame. 
        num_ref = len(self.kframes)
        
        self.update_state(query_frame)
        self.update_dynamic(query_frame, visualize=visualize)
        
        # Write query_frame into obj_repr.kframes, and then reuse fine_estimation to optimize the initial frame
        self.kframes.frame_list.insert(0, query_frame)
        fine_joint_dict = fine_estimation(
            K=self.K,
            joint_type=self.fine_joint_dict["joint_type"],
            joint_dir=self.fine_joint_dict["joint_dir"],
            joint_start=self.fine_joint_dict["joint_start"],
            joint_states=self.kframes.get_joint_states(),
            depth_seq=self.kframes.get_depth_seq(),
            dynamic_seq=self.kframes.get_dynamic_seq(),
            opti_joint_dir=False,
            opti_joint_start=False,
            opti_joint_states_mask=np.arange(num_ref+1)==0,
            lr=reloc_lr,
            visualize=visualize,
            logger=self.logger
        )
        # Then spit out query_frame from keyframes here
        query_frame = self.kframes.frame_list.pop(0)
        if fine_joint_dict == {}:
            self.logger.log(logging.ERROR, "Fine estimation failed, return state as 0 or initial_guess if given")
            
            if init_guess is not None:
                query_frame.joint_state = init_guess
            else:
                query_frame.joint_state = 0
            return query_frame
        
        query_frame.joint_state = fine_joint_dict["joint_states"][0]
        self.logger.log(logging.INFO, f"Fine estimated joint state: {query_frame.joint_state}")
        
        if self.frames is not None:
            manip_first_frame_gt_joint_state = self.frames[0].gt_joint_state
        else:
            manip_first_frame_gt_joint_state = self.kframes[0].gt_joint_state
        self.logger.log(logging.INFO, f"GT joint state: {query_frame.gt_joint_state - manip_first_frame_gt_joint_state}")
       
        self.update_dynamic(query_frame, visualize=visualize)
        return query_frame
    
    def visualize_joint(self):
        tmp_frame: Frame = None
        if len(self.frames) > 0:
            tmp_frame = self.frames[0]
        elif len(self.kframes) > 0:
            tmp_frame = self.kframes[0]
        env_pc, colors = tmp_frame.get_env_pc(
            use_robot_mask=True, 
            use_height_filter=True,
            world_frame=True,
        )
        joint_dict_w = self.get_joint_param(resolution="fine", frame="world")
        joint_dir = joint_dict_w["joint_dir"]
        joint_start = joint_dict_w["joint_start"]
        
        joint_dict_gt_w = self.get_joint_param(resolution="gt", frame="world")
        joint_dir_gt = joint_dict_gt_w["joint_dir"]
        joint_start_gt = joint_dict_gt_w["joint_start"]
        
        viewer = napari.Viewer(ndisplay=3)
        viewer.title = "fine joint est visualization"
        
        joint_start[-1] *= -1
        joint_dir[-1] *= -1
        
        viewer.add_points(env_pc, size=0.01, name='predicted tracks 3d', opacity=0.8, face_color="green")

        viewer.add_shapes(
            data=np.array([joint_start, joint_start + joint_dir * 0.2]),
            name="est joint dir",
            shape_type="line",
            edge_width=0.005,
            face_color="blue",
            edge_color="blue"
        )
        viewer.add_points(
            data=joint_start,
            name="est joint start",
            size=0.02,
            face_color="blue",
            border_color="red",
        )
        
        viewer.add_shapes(
            data=np.array([joint_start_gt, joint_start_gt + joint_dir_gt * 0.2]),
            name="gt joint dir",
            shape_type="line",
            edge_width=0.005,
            face_color="green",
            edge_color="green"
        )
        viewer.add_points(
            data=joint_start_gt,
            name="gt joint start",
            size=0.02,
            face_color="green",
            border_color="red",
        )
        napari.run()
            
    def _visualize(self, viewer, prefix=""):
        pass
    
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "object representation"
        self.initial_frame._visualize(viewer, prefix="initial")
        if len(self.frames) > 0:
            self.frames._visualize_f(viewer, prefix="frames")
        if len(self.kframes) > 0:
            self.kframes._visualize_kf(viewer, prefix="kframes")
        self._visualize(viewer)
        napari.run()
