import os
import napari
import pickle
import numpy as np
from graspnetAPI import GraspGroup
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
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
    camera_to_world
)
from embodied_analogy.utility.perception.online_cotracker import track_any_points

from embodied_analogy.utility.estimation.clustering import cluster_tracks_3d_spectral as cluster_tracks_3d
from embodied_analogy.utility.grasp.anygrasp import (
    detect_grasp_anygrasp,
    filter_grasp_group,
    crop_grasp,
    crop_grasp_by_moving,
    sort_grasp_group
)

class Data():
    def save(self, file_path):
        # 保存这个类到 file_path, 如果 file_path 所在的文件夹路径不存在, 则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
        
    # def load(self, file_path):
    #     with open(file_path, "rb") as f:
    #         loaded_data = pickle.load(f)
    #         # 递归更新所有属性
    #         for key, value in loaded_data.__dict__.items():
    #             if hasattr(self, key) and isinstance(getattr(self, key), object):
    #                 # 这里可以添加更复杂的递归更新逻辑
    #                 pass
    #             setattr(self, key, value)
        
class Frame(Data):
    def __init__(
        self,
        rgb=None,
        depth=None,
        K=None,
        Tw2c=None,
        joint_state=None,
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
        
        self.rgb = rgb
        self.depth = depth
        self.K = K
        self.Tw2c = Tw2c
        
        self.obj_bbox = obj_bbox
        self.obj_mask = obj_mask
        self.dynamic_mask = dynamic_mask
        self.joint_state = joint_state
        
        # NOTE: 这几个都在相机坐标系下
        self.track2d = track2d
        self.contact2d = contact2d
        self.contact3d = contact3d
        self.dir_out = dir_out
        self.grasp_group = grasp_group
        
        self.robot2d = robot2d
        self.robot3d = robot3d
        self.robot_mask = robot_mask
        
        # NOTE: Tph2w 是在世界坐标系下的 (4, 4)
        self.Tph2w = Tph2w
        
    def _visualize(self, viewer: napari.Viewer, prefix=""):
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
                # features=features,
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
        返回 camera 坐标系下的点云, 可以选择是否仅返回 obj_mask 为 True 的点
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
        返回 camera 坐标系下的点云, 可以选择是否仅返回 obj_mask 为 True 的点
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
        
    def detect_grasp(self, use_anygrasp=True, world_frame=False, visualize=False, asset_path=None) -> GraspGroup:
        """
        返回当前 frame 的点云上 contact2d 附近的一个 graspGroup, default frame is Tgrasp2c
        if world_frame = True, return Tgrasp2w
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
        
        # 找到 contact3d 附近的点
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
        
        # fit 一个局部的 normal 做为 grasp 的方向
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
        
        # 1) 在 crop 出的点云上检测 grasp
        # gg = detect_grasp_anygrasp(
        #     points=cropped_pc, 
        #     colors=cropped_colors / 255.,
        #     dir_out=dir_out, 
        #     augment=True,
        #     visualize=False
        # )  
        # 2) 在完整的点云上检测 grasp
        if use_anygrasp:
            gg = detect_grasp_anygrasp(
                points=obj_pc, 
                colors=pc_colors / 255.,
                dir_out=dir_out, 
                augment=True,
                visualize=False, # still have bug visualize this
                asset_path=asset_path
            )  
        else: # use gsnet
            from embodied_analogy.utility.grasp.gsnet import detect_grasp_gsnet
            gg = detect_grasp_gsnet(
                gsnet=None,
                points=obj_pc,
                colors=None,
                nms=True,
                keep=1e6,
                visualize=visualize,
                asset_path=asset_path
            )
            
        gg = crop_grasp(
            grasp_group=gg,
            contact_point=self.contact3d,
            radius=0.1,
        )

        # 先用 dir_out 进行一个 hard filter, 保留角度在 30度 内的 grasp
        gg_filtered = filter_grasp_group(
            grasp_group=gg,
            degree_thre=30,
            dir_out=dir_out,
        )

        # 再用距离 contact3d 的距离和 detector 本身预测的分数做一个排序
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
            self.grasp_group = self.grasp_group.transform(Tc2w) 
            
    def detect_grasp_moving(self, crop_thresh=0.1, visualize=False) -> GraspGroup:
        """
        返回当前 frame 的点云上 moving_part 附近的一个 graspGroup
        """
        assert (self.dynamic_mask is not None) and (self.obj_mask is not None)
        
        obj_pc, pc_colors = self.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=True,
            world_frame=False
        )
        
        # 这里用相机坐标系的 -z 轴作为 dir_out
        dir_out = np.linalg.inv(self.Tw2c)[:3, :3] @ np.array([0, 0, -1])
        gg = detect_grasp_anygrasp(
            points=obj_pc, 
            colors=pc_colors / 255.,
            dir_out=dir_out, 
            augment=True,
            visualize=False
        )  
        # 这是在所有物体点云上检测得到的 grasp, 我们需要把距离 dynamic_mask 中 moving_mask 近的裁剪出来
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
        visualize=False
    ):
        from embodied_analogy.utility.perception.grounded_sam import run_grounded_sam
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
            # 使用 depth_mask 进行过滤
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
        # 用于记录 tracks 的语义分类
        moving_mask=None,
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
        使用 cotracker 对于第一个 frame 上的 track2d 进行跟踪得到其他 frame 上的 track2d
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
        
        # 对得到的 track3d_seq 进行过滤
        if filter:
            def filter_pixel_tracks(boolean_mask, pixel_tracks):
                T, H, W = boolean_mask.shape
                M = pixel_tracks.shape[1]
                # 创建一个大小为 M 的布尔掩码，初始化为 True
                output_mask = np.ones(M, dtype=bool)
                for m in range(M):
                    for t in range(T):
                        u, v = pixel_tracks[t, m]
                        u, v = int(u), int(v)
                        # 检查坐标 (u, v) 是否在布尔掩码中为 True
                        if u < 0 or u >= H or v < 0 or v >= W or not boolean_mask[t, v, u]:
                            output_mask[m] = False
                            break  # 一旦发现不满足条件，就可以停止检查
                return output_mask
            
            robot_filter = filter_pixel_tracks(~self.get_robot_mask_seq(), track2d_seq)
            consis_filter = filter_tracks_by_consistency(track3d_seq, threshold=0.02) # M
            
            filter_mask = robot_filter & consis_filter
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
        from embodied_analogy.utility.perception.mask_obj_from_video import mask_obj_from_video_with_image_sam2
        # 对于所有 frames 进行物体分割
        obj_mask_seq = mask_obj_from_video_with_image_sam2(
            rgb_seq=self.get_rgb_seq(), 
            obj_description=obj_description,
            # positive_tracks2d=self.track2d_seq, 
            # negative_tracks2d=self.get_robot2d_seq(), 
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
                K=self.K, # 相机内参
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
            # viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
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
                depth_tolerance=0.03, # 假设 coarse 阶段的误差估计在 5 cm 内
                joint_dict=joint_dict,
                visualize=visualize
            ) 
        
        # for i in range(self.num_frames()):
        #     self.frame_list[i].obj_mask = self.obj_mask_seq[i]
        #     self.frame_list[i].dynamic_mask = dynamic_seq_updated[i]
    
    # 底下估计 open/close 的这一部分并不是很 robust, 因此删除, 并且默认 explore 阶段得到的都是 open 的轨迹
    # 根据追踪的 3d 轨迹判断是 "open" 还是 "close"
    # track_type = classify_open_close(
    #     tracks3d=tracks3d_filtered,
    #     moving_mask=moving_mask,
    #     visualize=visualize
    # )
           
    def _visualize_f(self, viewer: napari.Viewer, prefix="", visualize_robot2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        self.frame_list[0]._visualize(viewer, prefix="start_frame")
        self.frame_list[-1]._visualize(viewer, prefix="end_frame")
    
    def _visualize_kf(self, viewer: napari.Viewer, prefix="", visualize_robot2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        # 添加绘制 mask
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


if __name__ == "__main__":
    # frame = Frame.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/grasp/init_frame_drawer.npy")
    # frame = Frame.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/grasp/init_frame_micro.npy")
    # frame.detect_grasp(True)
    # frame.segment_obj(obj_description="drawer", visualize=True, remove_robot=False)
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets_zby/logs/explore_51/44781_1_revolute/obj_repr.npy")
    # print(obj_repr.frames.track2d_seq.shape)
    # obj_repr.visualize()
    
    from embodied_analogy.utility.grasp.gsnet import detect_grasp_gsnet
    from embodied_analogy.utility.utils import visualize_pc
    pc = obj_repr.frames[0].get_env_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=True,
    )[0]
    visualize_pc(pc)
    detect_grasp_gsnet(None, pc, visualize=True)