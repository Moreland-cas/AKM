import os
import time
import numpy as np

from embodied_analogy.utility.utils import (
    initialize_napari,   
    set_random_seed,
    depth_image_to_pointcloud,
    camera_to_world,
    sample_points_within_bbox_and_mask,
    image_to_camera,
    filter_tracks_by_visibility,
    filter_tracks2d_by_depth_mask_seq,
    filter_tracks_by_consistency,
    extract_tracked_depths,
    farthest_scale_sampling,
    get_dynamic_seq,
    get_depth_mask_seq,
    visualize_pc
)

initialize_napari()

import napari
from embodied_analogy.representation.basic_structure import Frames
from embodied_analogy.representation.obj_repr import Obj_repr

from embodied_analogy.perception.online_cotracker import track_any_points
from embodied_analogy.perception.grounded_sam import run_grounded_sam
from embodied_analogy.perception.mask_obj_from_video import (
    mask_obj_from_video_with_image_sam2,
    # mask_obj_from_video_with_video_sam2
)
from embodied_analogy.estimation.clustering import cluster_tracks_2d, cluster_tracks_3d
from embodied_analogy.estimation.coarse_joint_est import coarse_joint_estimation
from embodied_analogy.estimation.fine_joint_est import (
    fine_joint_estimation_seq,
    filter_dynamic_seq
)
from embodied_analogy.visualization.vis_tracks_2d import vis_tracks2d_napari
from embodied_analogy.utility.constants import *
set_random_seed(SEED)

def reconstruct(
    obj_repr: Obj_repr,
    num_initial_uvs=1000,
    num_key_frames=5,
    obj_description="drawer",
    file_path=None,
    gt_joint_axis=None,
    visualize=True,
):
    """
        读取 exploration 阶段获取的视频数据, 进而对物体结构进行恢复
        返回一个 obj_repr, 包含物体表示和 joint 估计
    """
    rgb_seq = obj_repr.frames.get_rgb_seq()
    depth_seq = obj_repr.frames.get_depth_seq()
    franka2d_seq = obj_repr.frames.get_franka2d_seq()
    K = obj_repr.initial_frame.K
    Tw2c = obj_repr.initial_frame.Tw2c
        
    depth_mask_seq = get_depth_mask_seq(
        depth_seq=depth_seq,
        K=K,
        Tw2c=Tw2c,
        height=0.02 # 2 cm
    )
    """
        根据物体初始状态的图像, 得到一些初始跟踪点, initial_uvs
    """
    # 根据 rgb_seq[0], 得到 initial_mask
    initial_bbox, initial_mask=run_grounded_sam(
        rgb_image=rgb_seq[0],
        obj_description=obj_description,
        positive_points=None, 
        negative_points=franka2d_seq[0],
        num_iterations=3,
        acceptable_thr=0.9,
        visualize=visualize,
    )
    # 在 initial_bbox 内均匀采样
    initial_uvs = sample_points_within_bbox_and_mask(initial_bbox, initial_mask, num_initial_uvs)

    if visualize:
        viewer = napari.view_image(rgb_seq[0])
        viewer.title = "initial uvs on intial rgb"
        initial_uvs_vis = initial_uvs[:, [1, 0]]
        viewer.add_points(initial_uvs_vis, size=2, name="initial_uvs", face_color="green")
        napari.run()
    """
        对于 initial_uvs 进行追踪得到 tracks2d 
    """
    # [T, M, 2], [T, M]
    tracks2d, pred_visibility = track_any_points(
        rgb_frames=rgb_seq,
        queries=initial_uvs,
        visiualize=visualize
    ) 
    """
        将 tracks2d 升维到 tracks3d, 并在 3d 空间对 tracks 进行聚类
    """
    # 首先将 tracks2d 升维到 tracks3d
    T, M, _ = tracks2d.shape
    tracks_depth = extract_tracked_depths(depth_seq, tracks2d) # T, M
    tracks3d = image_to_camera(tracks2d.reshape(T * M, -1), tracks_depth.reshape(-1), K) # T*M, 3
    tracks3d = tracks3d.reshape(T, M, 3)
    # 对于 tracks3d 进行过滤
    # TODO: 使用 franka 的分割进行过滤 + 使用 pred_visibility 进行过滤
    # 
    # 使用 3d consistency 进行过滤
    consis_mask = filter_tracks_by_consistency(tracks3d, threshold=0.02) # M
    tracks2d_filtered = tracks2d[:, consis_mask]
    tracks3d_filtered = tracks3d[:, consis_mask]
    if visualize:
        vis_tracks2d_napari(rgb_seq, tracks2d_filtered, viewer_title="filter tracks by 3d consistency")
    # 在 3d 空间对 tracks 进行聚类
    moving_mask, static_mask = cluster_tracks_3d(
        tracks3d_filtered, 
        use_diff=True, 
        visualize=visualize, 
        viewer_title="clustering 3d tracks(filtered) into moving and static part"
    )
    """
        coarse joint estimation with tracks3d_filtered
    """
    joint_type, joint_axis_c, joint_states = coarse_joint_estimation(
        tracks_3d=tracks3d_filtered[:, moving_mask, :], 
        visualize=visualize
    )
    """
        根据 rgb_seq 和 tracks2d 得到 obj_mask_seq (可以用 sam 或者 sam2)
    """
    # 根据 coarse joint estimation 挑选出有信息量的一些帧, 进行 fine joint estimation
    kf_idx = farthest_scale_sampling(joint_states, M=num_key_frames)
    # obj_mask_seq by sam2 image mode
    obj_mask_seq = mask_obj_from_video_with_image_sam2(
        rgb_seq=rgb_seq[kf_idx], 
        obj_description=obj_description,
        positive_tracks2d=tracks2d_filtered[kf_idx], 
        negative_tracks2d=franka2d_seq[kf_idx], 
        visualize=visualize
        # visualize=True
    ) 
    # 对 obj_mask_seq 进行 depth 过滤, 可选可不选
    obj_mask_seq = obj_mask_seq & depth_mask_seq[kf_idx]
    """
        根据 tracks2d 和 obj_mask_seq 得到 dynamic_seq
    """
    dynamic_seq = get_dynamic_seq(
        mask_seq=obj_mask_seq, 
        moving_points_seq=tracks2d_filtered[kf_idx][:, moving_mask, :],
        static_points_seq=tracks2d_filtered[kf_idx][:, static_mask, :], 
        visualize=visualize
    )
    """
        根据 dynamic_seq 中的 moving_part, 利用 ICP 估计出精确的 joint params
    """
    # filter dynamic mask seq
    dynamic_seq_updated = filter_dynamic_seq(
        K=K,
        depth_seq=depth_seq[kf_idx],
        dynamic_seq=dynamic_seq,
        joint_type=joint_type,
        joint_axis_c=joint_axis_c,
        joint_states=joint_states[kf_idx],
        depth_tolerance=0.05, # 假设 coarse 阶段的误差估计在 5 cm 内
        visualize=visualize
        # visualize=True
    ) 
    # fine estimation
    joint_axis_c_updated, joint_states_updated = fine_joint_estimation_seq(
        K=K,
        depth_seq=depth_seq[kf_idx], 
        dynamic_seq=dynamic_seq_updated,
        joint_type=joint_type, 
        joint_axis_c=joint_axis_c, 
        joint_states=joint_states[kf_idx],
        max_icp_iters=200, # ICP 最多迭代多少轮
        optimize_joint_axis=True,
        optimize_state_mask=np.arange(num_key_frames)!=0,
        # 这里设置不迭代的优化 dynamic_mask
        update_dynamic_mask=np.zeros(num_key_frames).astype(np.bool_),
        lr=5e-3, # 5 mm
        icp_select_range=0.1,
        visualize=visualize
    )

    Rc2w = Tw2c[:3, :3].T # 3, 3
    joint_axis_w = Rc2w @ joint_axis_c
    joint_axis_w_updated = Rc2w @ joint_axis_c_updated # 3
    
    # 在这里调整 joint_axis_w_updated 的方向, 使得其指向使得物体点云方差变大的方向
    # TODO：似乎跟 joint_type 还有关系
    if joint_type == "prismatic":
        tracks_3d_moving_c = tracks3d_filtered[:, moving_mask, :] 
        moving_mean_start = tracks_3d_moving_c[0].mean(0)
        moving_mean_end = tracks_3d_moving_c[-1].mean(0)
        tracks_3d_static_c = tracks3d_filtered[:, static_mask, :]
        static_mean_start = tracks_3d_static_c[0].mean(0)
        static_mean_end = tracks_3d_static_c[-1].mean(0)
        
        # 首先根据 tracks3d_moving 和 tracks3d_static 类中心的方差判断当前 track 随着时间是 open 还是 close
        # if np.var(tracks3d_filtered[0]) > np.var(tracks3d_filtered[-1]):
        if np.linalg.norm(moving_mean_start - static_mean_start) > np.linalg.norm(moving_mean_end - static_mean_end):
            track_type = "close"
        else:
            track_type = "open"
            
        # 然后判断估计出的 joint_axis 与当前  tracks3d 变化的对应关系
        moving_dir_c = (tracks_3d_moving_c - tracks_3d_moving_c[0]).reshape(-1, 3) # T*N, 3
        # moving_dir_w 为 tracks3d 在世界坐标系下的运动方向
        moving_dir_w = camera_to_world(moving_dir_c, Tw2c) # T*N, 3
        dot_product_with_joint_axis = np.mean(moving_dir_w * joint_axis_w_updated)
            
        if dot_product_with_joint_axis < 0:
            # 代表估计出的 joint 方向与重建数据的变化方向相反, 此时如果重建数据的方向为 open, 则估计出的 joint 实际指向为 close 方向, 此时需要反转
            if track_type == "open":
                joint_axis_w_updated = -joint_axis_w_updated
                joint_axis_c = - joint_axis_c
                joint_axis_c_updated = -joint_axis_c_updated
                joint_states = -joint_states
                joint_states_updated = -joint_states_updated
            else:
                joint_axis_w_updated = joint_axis_w_updated
        else:
            if track_type == "close":
                joint_axis_w_updated = -joint_axis_w_updated
                joint_axis_c = - joint_axis_c
                joint_axis_c_updated = -joint_axis_c_updated
                joint_states = -joint_states
                joint_states_updated = -joint_states_updated
            else:
                joint_axis_w_updated = joint_axis_w_updated
    else:
        assert "revolute not implemented yet"
        
    if file_path is not None:
        # 更新 obj_repr, 并且进行保存
        _key_frames = [obj_repr.frames[kf_idx_i] for kf_idx_i in kf_idx]
        # 将 obj_mask 和 dynamic_mask 更新到 key_frames 中
        for i, frame in enumerate(_key_frames):
            frame.obj_mask = obj_mask_seq[i]
            frame.dynamic_mask = dynamic_seq_updated[i]
            
        obj_repr.key_frames = Frames(frame_list=_key_frames)
        obj_repr.clear_frames()
        obj_repr.track_type = track_type
        obj_repr.joint_axis_c = joint_axis_c_updated
        obj_repr.joint_axis_w = joint_axis_w_updated
        obj_repr.joint_states = joint_states_updated
        obj_repr.joint_type = joint_type
        obj_repr.save(file_path)
    
    if gt_joint_axis is not None:
        print(f"\tgt axis: {gt_joint_axis}")

        dot_before = np.dot(gt_joint_axis, joint_axis_w)
        print(f"\tbefore: {np.degrees(np.arccos(dot_before))}")
        print("\tjoint axis: ", joint_axis_w)
        print("\tjoint states: ", joint_states[kf_idx])

        dot_after = np.dot(gt_joint_axis, joint_axis_w_updated)
        print(f"\tafter : {np.degrees(np.arccos(dot_after))}")
        print("\tjoint axis: ", joint_axis_w_updated)
        print("\tjoint states: ", joint_states_updated)
    

if __name__ == "__main__":
    obj_idx = 44962
    obj_repr_path = f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/explore/explore_data.pkl"
    
    reconstruct(
        obj_repr=Obj_repr.load(obj_repr_path),
        num_initial_uvs=1000,
        num_key_frames=3,
        visualize=False,
        gt_joint_axis=np.array([-1, 0, 0]),
        # file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_idx}/reconstruct/recon_data.pkl"
        file_path = None
    )
