import os
import pickle
import numpy as np
from graspnetAPI import GraspGroup

from akm.utility.grasp.anygrasp import (
    detect_grasp_anygrasp,
    filter_grasp_group,
    crop_grasp,
    crop_grasp_by_moving,
    sort_grasp_group
)
from akm.utility.utils import (
    napari_time_series_transform,
    image_to_camera,
    get_depth_mask,
    classify_dynamic,
    get_depth_mask_seq,
    depth_image_to_pointcloud,
    crop_nearby_points,
    fit_plane_ransac,
    visualize_pc,
    make_bbox,
    sample_points_within_bbox_and_mask,
    extract_tracked_depths,
    filter_tracks_by_consistency,
    filter_dynamic,
    camera_to_world,
)
from akm.utility.constants import *
from akm.utility.perception.online_cotracker import track_any_points
from akm.utility.estimation.clustering import cluster_tracks_3d_spectral as cluster_tracks_3d


class Data():
    def save(self, file_path):
        # Save this class to file_path. If the folder path where file_path is located does not exist, create it.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
        
        
class Frame(Data):
    def __init__(
        self,
        rgb=None,
        depth=None,
        K=None,
        Tw2c=None,
        joint_state=None,
        gt_joint_state=None,
        joint_state_offset=None,
        obj_bbox=None,
        obj_mask=None,
        dynamic_mask=None,
        contact2d=None,
        contact3d=None,
        dir_out=None,
        grasp_group: GraspGroup = None,
        track2d=None,
        robot2d=None,
        robot3d=None,
        robot_mask=None,
        Tph2w=None
    ):
        """
        joint_state: The joint_state predicted by the reconstruction algorithm
        joint_state_offset: The gt_joint_state corresponding to the first frame of the explore_video
        gt_joint_state: The actual joint_state. If the prediction is good enough, then joint_state + joint_state_offset = gt_joint_state
        """
        self.rgb = rgb
        self.depth = depth
        self.K = K
        self.Tw2c = Tw2c
        
        self.obj_bbox = obj_bbox
        self.obj_mask = obj_mask
        self.dynamic_mask = dynamic_mask
        self.joint_state = joint_state
        self.gt_joint_state = gt_joint_state
        self.joint_state_offset = joint_state_offset
        
        # NOTE: These are all in the camera coordinate system
        self.track2d = track2d
        self.contact2d = contact2d
        self.contact3d = contact3d
        self.dir_out = dir_out
        self.grasp_group = grasp_group
        
        self.robot2d = robot2d
        self.robot3d = robot3d
        self.robot_mask = robot_mask
        
        # NOTE: Tph2w is in world coordinates (4, 4)
        self.Tph2w = Tph2w
        
        # save for vis
        self.target_state = None
        
    def _visualize(self, viewer, prefix=""):
        viewer.add_image(self.rgb, rgb=True, name=f"{prefix}_rgb")
        
        if self.dynamic_mask is not None:
            viewer.add_labels(self.dynamic_mask.astype(np.int32), name=f"{prefix}_dynamic_mask")
            
        if self.robot_mask is not None:
            viewer.add_labels(self.robot_mask, name=f"{prefix}_robot_mask")
            
        if self.obj_mask is not None:
            viewer.add_labels(self.obj_mask, name=f"{prefix}_obj_mask")
        
        if self.contact2d is not None:
            u, v = self.contact2d
            viewer.add_points((v, u), face_color="red", name=f"{prefix}_contact2d")
            
        if self.obj_bbox is not None:
            # obj_bbox in order [u_left, v_left, u_right, v_right]
            #  [min_row, min_column, max_row, max_column]
            tmp = np.array([self.obj_bbox[1], self.obj_bbox[0], self.obj_bbox[3], self.obj_bbox[2]])[:, None]
            bbox_rect = make_bbox(tmp)
            viewer.add_shapes(
                bbox_rect,
                face_color='transparent',
                edge_color='green',
                edge_width=5,
                text="bbox",
                name=f"{prefix}_bbox",
            )
    
    def get_obj_pc(
        self,
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=False,
        visualize=False
    ):
        """
        Returns the point cloud in the camera coordinate system. 
        You can choose whether to return only the points where obj_mask is True
        """
        if use_height_filter:
            height_filter = 0.02
        else:
            height_filter = 0.0
        depth_mask = get_depth_mask(self.depth, self.K, self.Tw2c, height=height_filter)
        
        assert self.obj_mask is not None
        depth_mask = self.obj_mask & depth_mask
        
        if use_robot_mask:
            depth_mask = depth_mask & (~self.robot_mask)
            
        obj_pc = depth_image_to_pointcloud(
            depth_image=self.depth,
            mask = depth_mask,
            K=self.K
        )
        pc_colors = self.rgb[depth_mask]
        
        if world_frame:
            obj_pc = camera_to_world(obj_pc, self.Tw2c)
        
        if visualize:
            visualize_pc(obj_pc, pc_colors / 255)
        return obj_pc, pc_colors
    
    def get_env_pc(
        self,
        use_robot_mask=True, 
        use_height_filter=False,
        world_frame=False,
        visualize=False
    ):
        """
        Returns the point cloud in the camera coordinate system. 
        You can choose whether to return only the points where obj_mask is True
        """
        if use_height_filter:
            height_filter = 0.02
        else:
            height_filter = 0.0
        depth_mask = get_depth_mask(self.depth, self.K, self.Tw2c, height=height_filter)
        
        if use_robot_mask:
            depth_mask = depth_mask & (~self.robot_mask)
            
        obj_pc = depth_image_to_pointcloud(
            depth_image=self.depth,
            mask = depth_mask,
            K=self.K
        )
        pc_colors = self.rgb[depth_mask]
        
        if world_frame:
            obj_pc = camera_to_world(obj_pc, self.Tw2c)
        
        if visualize:
            visualize_pc(obj_pc, pc_colors / 255)
        return obj_pc, pc_colors
        
    def detect_grasp(self, use_anygrasp=True, world_frame=False, visualize=False, asset_path=None, logger=None) -> GraspGroup:
        """
        Returns a graspGroup near contact2d on the point cloud of the current frame. The default frame is Tgrasp2c.
        If world_frame = True, return Tgrasp2w
        """
        assert (self.contact2d is not None) and (self.obj_mask is not None)
        c2d_int = self.contact2d.astype(np.int32)
        if self.contact3d is None:
            contact3d = image_to_camera(
                uv=self.contact2d[None], # 1, 3
                depth=np.array(self.depth[c2d_int[1], c2d_int[0]])[None], # 1, 1
                K=self.K,
            )[0]
            self.contact3d = contact3d
        contact3d = self.contact3d
        
        # Find points near contact3d
        obj_pc, pc_colors = self.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=True,
            world_frame=False
        )
        
        crop_mask = crop_nearby_points(
            point_clouds=obj_pc,
            contact3d=self.contact3d,
            radius=0.1
        )
        cropped_pc = obj_pc[crop_mask]
        cropped_colors = pc_colors[crop_mask]
        
        # Fit a local normal as the grasp direction
        plane_normal = fit_plane_ransac(
            points=cropped_pc,
            threshold=0.005, 
            max_iterations=100,
            visualize=False
        )
        if (plane_normal * np.array([0, 0, -1])).sum() > 0:
            dir_out = plane_normal
        else:
            dir_out = -plane_normal
        self.dir_out = dir_out
        
        #  detect grasp on full pc
        if use_anygrasp:
            gg = detect_grasp_anygrasp(
                points=obj_pc, 
                colors=pc_colors / 255.,
                dir_out=dir_out, 
                augment=True,
                visualize=False,
                logger=logger
            )  
        else: 
            # use gsnet
            from akm.utility.grasp.gsnet import detect_grasp_gsnet
            gg = detect_grasp_gsnet(
                gsnet=None,
                points=obj_pc,
                colors=None,
                nms=True,
                keep=1e6,
                visualize=visualize,
                asset_path=asset_path,
                logger=logger
            )
            
        # Initial constraints
        max_radius = 0.5 # Maximum acceptable radius
        max_degree_thre = 90 # Maximum acceptable degree_thre
        radius_step = 0.05 # Step size for each radius increment
        degree_step = 10 # Step size for each degree_thre increment
        radius = 0.1 # Initial radius
        degree_thre = 30 # Initial degree_thre

        while True:
            gg_cropped = crop_grasp(
                grasp_group=gg,
                contact_point=self.contact3d,
                radius=radius,
            )

            gg_filtered = filter_grasp_group(
                grasp_group=gg_cropped,
                degree_thre=degree_thre,
                dir_out=dir_out,
            )

            if gg_filtered is not None and len(gg_filtered) > 0:
                break

            # If no grasp is found, gradually relax restrictions
            if radius < max_radius:
                radius += radius_step
            if degree_thre < max_degree_thre:
                degree_thre += degree_step

            if radius >= max_radius and degree_thre >= max_degree_thre:
                break

        gg = gg_filtered

        # Then use the distance from contact3d and the score predicted by the detector itself to make a sort
        gg_sorted = sort_grasp_group(
            grasp_group=gg_filtered,
            contact_region=self.contact3d[None],
        )
        self.grasp_group = gg_sorted
        
        if visualize:
            visualize_pc(
                points=obj_pc, 
                colors=pc_colors / 255,
                grasp=None if gg_sorted is None else gg_sorted[0], 
                contact_point=self.contact3d, 
                post_contact_dirs=[self.dir_out]
            )
        if world_frame:
            # Tgrasp2w
            Tc2w = np.linalg.inv(self.Tw2c)
            if self.grasp_group is not None and len(self.grasp_group) > 0:
                self.grasp_group = self.grasp_group.transform(Tc2w) 
            
    def detect_grasp_moving(self, crop_thresh=0.1, visualize=False, logger=None) -> GraspGroup:
        """
        Returns a graspGroup near the moving_part on the point cloud of the current frame
        """
        assert (self.dynamic_mask is not None) and (self.obj_mask is not None)
        
        obj_pc, pc_colors = self.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=True,
            world_frame=False
        )
        
        # Here we use the -z axis of the camera coordinate system as dir_out
        dir_out = np.linalg.inv(self.Tw2c)[:3, :3] @ np.array([0, 0, -1])
        gg = detect_grasp_anygrasp(
            points=obj_pc, 
            colors=pc_colors / 255.,
            dir_out=dir_out, 
            augment=True,
            visualize=False,
            logger=logger
        )  
        # This is the grasp detected on all object point clouds. We need to crop out the ones that are close to the moving_mask in the dynamic_mask
        gg = crop_grasp_by_moving(
            grasp_group=gg,
            contact_region=depth_image_to_pointcloud(self.depth, self.dynamic_mask == MOVING_LABEL, self.K),
            crop_thresh=crop_thresh
        )
        
        if visualize:
            visualize_pc(
                points=obj_pc, 
                colors=pc_colors / 255,
                grasp=None if gg is None else gg, 
                contact_point=None, 
                post_contact_dirs=[self.dir_out]
            )
        return gg

    def segment_obj(
        self, 
        obj_description=None,
        post_process_mask=True,
        filter=True,
        visualize=False,
    ):
        from akm.utility.perception.grounded_sam import run_grounded_sam
        obj_bbox, obj_mask = run_grounded_sam(
            rgb_image=self.rgb,
            obj_description=obj_description,
            positive_points=None,  
            negative_points=None,
            num_iterations=3,
            acceptable_thr=0.9,
            dino_model=None,
            sam2_image_model=None,
            post_process_mask=post_process_mask,
            visualize=False,
        )
        self.obj_bbox = obj_bbox
        
        if filter:
            assert self.robot_mask is not None
            obj_mask = obj_mask & (~self.robot_mask)
            depth_mask = get_depth_mask(
                depth=self.depth,
                K=self.K,
                Tw2c=self.Tw2c,
                height=0.02
            )
            obj_mask = obj_mask & depth_mask
        
        self.obj_mask = obj_mask
            
        if visualize:
            self.visualize()
    
    def sample_points(self, num_points=1000, visualize=False):
        initial_uvs = sample_points_within_bbox_and_mask(self.obj_bbox, self.obj_mask, num_points)
        self.track2d = initial_uvs
        if visualize:
            viewer = napari.view_image(self.rgb)
            viewer.title = "initial uvs on intial rgb"
            initial_uvs_vis = initial_uvs[:, [1, 0]]
            viewer.add_points(initial_uvs_vis, size=3, name="initial_uvs", face_color="green")
            napari.run()
            
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "frame visualization"
        self._visualize(viewer)
        napari.run()
    
    
class Frames(Data):
    def __init__(
        self, 
        frame_list=[], 
        fps=30, 
        K=None, 
        Tw2c=None,
        track2d_seq=None,
        track3d_seq=None,
        moving_mask=None, # Used to record semantic classification of tracks
        static_mask=None,
        obj_mask_seq=None,
        dynamic_seq=None,
    ):
        """
        frame_list: list of class Frame
        """
        self.frame_list = frame_list
        self.fps = fps
        self.K = K
        self.Tw2c = Tw2c
        
        self.track2d_seq = track2d_seq
        self.track3d_seq = track3d_seq
        
        self.moving_mask = moving_mask
        self.static_mask = static_mask
        
        self.obj_mask_seq = obj_mask_seq
        self.dynamic_seq = dynamic_seq
    
    def num_frames(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        return self.frame_list[idx]
    
    def __len__(self):
        return len(self.frame_list)
    
    def clear(self):
        self.frame_list = []
        self.track2d_seq = None
        self.track3d_seq = None
        self.moving_mask = None
        self.static_mask = None
        
    def append(self, frame: Frame):
        self.frame_list.append(frame)
        
    def get_rgb_seq(self):
        # T, H, W, C
        if len(self.frame_list) == 0:
            return None
        rgb_seq = np.stack([self.frame_list[i].rgb for i in range(self.num_frames())]) 
        return rgb_seq
        
    def get_depth_seq(self):
        # T, H, W
        depth_seq = np.stack([self.frame_list[i].depth for i in range(self.num_frames())]) 
        return depth_seq
        
    def get_robot2d_seq(self):
        # T, N, 2
        robot2d_seq = np.stack([self.frame_list[i].robot2d for i in range(self.num_frames())]) 
        return robot2d_seq
    
    def get_joint_states(self):
        # T
        joint_states = np.array([self.frame_list[i].joint_state for i in range(self.num_frames())]) 
        return joint_states
    
    def get_gt_joint_states(self):
        # T
        joint_states = np.array([self.frame_list[i].gt_joint_state for i in range(self.num_frames())]) 
        return joint_states
    
    def get_dynamic_seq(self):
        # T
        dynamic_seq = np.array([self.frame_list[i].dynamic_mask for i in range(self.num_frames())]) 
        return dynamic_seq
    
    def write_joint_states(self, joint_states):
        assert len(joint_states) == self.num_frames()
        for i, joint_state in enumerate(joint_states):
            self.frame_list[i].joint_state = joint_state
    
    def write_dynamic_seq(self, dynamic_seq):
        for i, dynamic in enumerate(dynamic_seq):
            self.frame_list[i].dynamic_mask = dynamic
    
    def get_robot_mask_seq(self):
        # T, H, W
        robot_mask_seq = np.stack([self.frame_list[i].robot_mask for i in range(self.num_frames())]) 
        return robot_mask_seq
    
    def track_points(self, visualize=False):
        """
        Use cotracker to track the track2d on the first frame to get the track2d on other frames
        """
        assert self.frame_list[0].track2d is not None
        # (T, M, 2), (T, M)
        track2d_seq, pred_visibility = track_any_points(
            rgb_frames=self.get_rgb_seq(),
            queries=self.frame_list[0].track2d,
            visualize=visualize
        ) 
        self.track2d_seq = track2d_seq
    
    def track2d_to_3d(self, filter=True, visualize=False):
        # T, M, 2
        track2d_seq = self.track2d_seq
        T, M, _ = track2d_seq.shape
        # T, M
        track_depths = extract_tracked_depths(
            depth_seq=self.get_depth_seq(),
            pred_tracks=track2d_seq
        )
        # T, M, 3
        track3d_seq = image_to_camera(
            uv=track2d_seq.reshape(T * M, -1), 
            depth=track_depths.reshape(-1), 
            K=self.K
        ).reshape(T, M, 3)
        self.track3d_seq = track3d_seq
        
        # Filter the obtained track3d_seq
        if filter:
            def filter_pixel_tracks(boolean_mask, pixel_tracks):
                T, H, W = boolean_mask.shape
                M = pixel_tracks.shape[1]
                output_mask = np.ones(M, dtype=bool)
                for m in range(M):
                    for t in range(T):
                        u, v = pixel_tracks[t, m]
                        u, v = int(u), int(v)
                        # Checks if coordinate (u, v) is True in the Boolean mask
                        if u < 0 or u >= W or v < 0 or v >= H or not boolean_mask[t, v, u]:
                            output_mask[m] = False
                            break
                return output_mask
            
            robot_filter = filter_pixel_tracks(~self.get_robot_mask_seq(), track2d_seq)
            consis_filter = filter_tracks_by_consistency(track3d_seq, threshold=0.02) # M
            
            filter_mask = robot_filter & consis_filter
            
            # But if the number of points filtered by filter_mask is too small, no filtering will be performed.
            if filter_mask.sum() < 100:
                filter_mask = np.ones(M, dtype=bool)
                
            track2d_seq = track2d_seq[:, filter_mask]
            track3d_seq = track3d_seq[:, filter_mask]
            self.track2d_seq = track2d_seq
            self.track3d_seq = track3d_seq
    
    def cluster_track3d(self, feat_type="diff", visualize=False):
        moving_mask, static_mask = cluster_tracks_3d(
            self.track3d_seq, 
            feat_type=feat_type,
            visualize=visualize, 
        )
        self.moving_mask = moving_mask
        self.static_mask = static_mask
    
    def segment_obj(self, obj_description=None, filter=True, visualize=False):
        from akm.utility.perception.mask_obj_from_video import mask_obj_from_video_with_image_sam2
        obj_mask_seq = mask_obj_from_video_with_image_sam2(
            rgb_seq=self.get_rgb_seq(), 
            obj_description=obj_description,
            positive_tracks2d=None, 
            negative_tracks2d=None, 
            visualize=visualize
        ) 
        
        if filter:
            depth_mask_seq = get_depth_mask_seq(
                depth_seq=self.get_depth_seq(),
                K=self.K,
                Tw2c=self.Tw2c,
                height=0.02 # 2 cm
            )
            obj_mask_seq = obj_mask_seq & depth_mask_seq
            robot_mask_seq = self.get_robot_mask_seq()
            obj_mask_seq = obj_mask_seq & (~robot_mask_seq)
        
        self.obj_mask_seq = obj_mask_seq
    
    def filter_dynamics(self, depth_tolerance=0.05, joint_dict=None, visualize=False):
        dynamic_seq = self.get_dynamic_seq()
        depth_seq = self.get_depth_seq()
        joint_states = self.get_joint_states()
        T = self.num_frames()
        
        dynamic_seq_updated = dynamic_seq.copy()
        
        for i in range(T):
            query_depth = depth_seq[i]
            query_dynamic = dynamic_seq[i]
            query_state = joint_states[i]
            
            other_mask = np.arange(T) != i
            ref_depths = depth_seq[other_mask]
            ref_states = joint_states[other_mask]
            
            query_dynamic_updated = filter_dynamic(
                K=self.K, 
                query_depth=query_depth, # H, W
                query_dynamic=query_dynamic, # H, W
                ref_depths=ref_depths,  # T, H, W
                joint_type=joint_dict["joint_type"],
                joint_dir=joint_dict["joint_dir"],
                joint_start=joint_dict["joint_start"],
                query_state=query_state,
                ref_states=ref_states,
                depth_tolerance=depth_tolerance, 
                visualize=False
            )
            dynamic_seq_updated[i] = query_dynamic_updated
        
        if visualize:
            import napari 
            viewer = napari.Viewer()
            viewer.title = "filter dynamic seq (moving part)"
            viewer.add_labels(dynamic_seq.astype(np.int32), name='before filtering')
            viewer.add_labels(dynamic_seq_updated.astype(np.int32), name='after filtering')
            napari.run()
        
        self.write_dynamic_seq(dynamic_seq_updated)
        
    def classify_dynamics(self, filter=False, joint_dict=None, visualize=False):
        dynamic_seq = []
        
        for i in range(self.num_frames()):
            dynamic = classify_dynamic(
                mask=self.obj_mask_seq[i], 
                moving_points=self.track2d_seq[i, self.moving_mask], 
                static_points=self.track2d_seq[i, self.static_mask],
            )
            dynamic_seq.append(dynamic)
        
        dynamic_seq = np.array(dynamic_seq)
        self.write_dynamic_seq(dynamic_seq)
        
        if filter:
            self.filter_dynamics(
                depth_tolerance=0.03, 
                joint_dict=joint_dict,
                visualize=visualize
            ) 
           
    def _visualize_f(self, viewer, prefix="", visualize_robot2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        self.frame_list[0]._visualize(viewer, prefix="start_frame")
        self.frame_list[-1]._visualize(viewer, prefix="end_frame")
    
    def _visualize_kf(self, viewer, prefix="", visualize_robot2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        if self.obj_mask_seq is not None:
            viewer.add_labels(self.obj_mask_seq, name=f"{prefix}_obj_mask_seq")
        
        if self.dynamic_seq is not None:
            viewer.add_labels(self.dynamic_seq.astype(np.int32), name=f"{prefix}_dynamic_mask_seq")
        
        if visualize_robot2d:
            robot2d_seq = self.get_robot2d_seq()
            link_names = [
                'panda_link0', 'panda_link1', 'panda_link2', 
                'panda_link3', 'panda_link4', 'panda_link5', 
                'panda_link6', 'panda_link7', 'panda_link8', 
                'panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 
                'panda_rightfinger', 'camera_base_link', 'camera_link'
            ]
            robot2d_data = napari_time_series_transform(robot2d_seq) # T*M, (1+2)
            robot2d_data = robot2d_data[:, [0, 2, 1]]
            for i in range(len(link_names)):
                T = len(rgb_seq)
                M = len(link_names)
                viewer.add_points(robot2d_data[i::M, :], face_color="red", name=f"{prefix}_{link_names[i]}")
                
    def visualize(self, is_k=False):
        viewer = napari.Viewer()
        if is_k:
            viewer.title = "kframes visualization"
            self._visualize_kf(viewer)
        else:
            viewer.title = "frames visualization"
            self._visualize_f(viewer)
        napari.run()