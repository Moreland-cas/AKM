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
    filter_tracks2d_by_visibility,
    filter_tracks2d_by_depth_mask_seq,
    filter_tracks2d_by_depth_diff_seq,
    extract_tracked_depths,
    farthest_scale_sampling,
    get_dynamic_seq,
    
)

initialize_napari()

from embodied_analogy.perception.online_cotracker import track_any_points
from embodied_analogy.perception.grounded_sam import run_grounded_sam
from embodied_analogy.perception.whole_obj_masking import (
    whole_obj_masking_sam,
    whole_obj_masking_sam2
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
    explore_data,
    num_initial_uvs=1000,
    num_key_frames=5,
    obj_description="drawer",
    save_dir="/home/zby/Programs/Embodied_Analogy/assets/tmp/reconstruct/",
    gt_joint_axis=None,
    visualize=True,
):
    """
        读取 exploration 阶段获取的视频数据, 进而对物体结构进行恢复
        返回一个 obj_repr, 包含物体表示和 joint 估计
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rgb_seq = explore_data["rgb_seq"]
    depth_seq = explore_data["depth_seq"]
    franka_seq = explore_data["franka_seq"]
    K = explore_data["K"]
    Tw2c = explore_data["Tw2c"]

    # 对于 depth_seq 进行处理, 得到 depth_mask_seq, 用于标记其中depth为 0, 或者重投影在地面上的位置
    depth_mask_seq = depth_seq > 0
    for i in range(depth_seq.shape[0]):
        _, H, W = depth_seq.shape
        pc_camera = depth_image_to_pointcloud(depth_seq[i], None, K) # H*W, 3
        pc_world = camera_to_world(pc_camera, Tw2c)
        pc_height_mask = (pc_world[:, 2] > 0.03).reshape(H, W) # H*W
        depth_mask_seq[i] = depth_mask_seq[i] & pc_height_mask

    """
        根据物体初始状态的图像, 得到一些初始跟踪点, initial_uvs
    """
    # 根据 rgb_seq[0], 得到 initial_mask
    initial_bbox, initial_mask=run_grounded_sam(
        rgb_image=rgb_seq[0],
        obj_description=obj_description,
        positive_points=None, 
        negative_points=franka_seq[0],
        num_iterations=3,
        acceptable_thr=0.9,
        visualize=visualize,
    )
    # 在 initial_bbox 内均匀采样
    initial_uvs = sample_points_within_bbox_and_mask(initial_bbox, initial_mask, num_initial_uvs)

    if visualize:
        import napari
        viewer = napari.view_image(rgb_seq[0])
        viewer.title = "initial uvs on intial rgb"
        initial_uvs_vis = initial_uvs[:, [1, 0]]
        viewer.add_points(initial_uvs_vis, size=2, name="initial_uvs", face_color="green")
        napari.run()
    """
        对于 initial_uvs 进行追踪得到 tracks_2d 
        根据 depth filter 得到 tracks2d_filtered
    """
    # [T, M, 2], [T, M]
    tracks_2d, pred_visibility = track_any_points(
        rgb_frames=rgb_seq,
        queries=initial_uvs,
        visiualize=visualize
    ) 
    # 将 tracks_2d 进行聚类
    # TODO: 这个要不要转移到 filter 之后
    # TODO: 要不要对于所有的 3d track 进行聚类, 或是对于 2d filter 后的进行聚类
    moving_mask_2d, static_mask_2d = cluster_tracks_2d(
        rgb_seq=rgb_seq,
        tracks_2d=tracks_2d,
        use_diff=True,
        visualize=visualize,
        viewer_title="dynamic clustering tracks2d"
    )
    # filter tracks2d by visibility
    tracks2d_filtered = filter_tracks2d_by_visibility(tracks_2d, pred_visibility)
    if visualize:
        vis_tracks2d_napari(rgb_seq, tracks_2d, viewer_title="filter tracks2d by visibility score")
        
    # filter tracks2d by depthSeq_mask
    # TODO: 这里不应该是 depth_mask_seq, 而应该是 depth_tracks
    tracks2d_filtered = filter_tracks2d_by_depth_mask_seq(tracks2d_filtered, depth_mask_seq)
    # tracks2d_filtered = filter_tracks2d_by_depth_diff_seq(tracks2d_filtered, depth_mask_seq)
    if visualize:
        vis_tracks2d_napari(rgb_seq, tracks_2d, viewer_title="filter tracks2d by depth diff")
    """
        dynamic segment tracks3d_filtered
    """  
    # 根据 tracks2d 得到 tracks3d
    T, M, _ = tracks2d_filtered.shape
    tracks2d_depth = extract_tracked_depths(depth_seq, tracks2d_filtered) # T, M
    tracks3d_filtered = image_to_camera(tracks2d_filtered.reshape(T*M, -1), tracks2d_depth.reshape(-1), K) # T*M, 3
    tracks3d_filtered = tracks3d_filtered.reshape(T, M, 3) 
    moving_mask_3d, static_mask_3d = cluster_tracks_3d(
        tracks3d_filtered, 
        use_diff=True, 
        visualize=visualize, 
        viewer_title="dynamic clustering tracks3d_filtered"
    )
    """
        coarse joint estimation with tracks3d_filtered
    """
    joint_type, joint_axis_c, joint_states = coarse_joint_estimation(tracks3d_filtered[:, moving_mask_3d, :], visualize)
    """
        根据 rgb_seq 和 tracks2d 得到 obj_mask_seq (可以用 sam 或者 sam2)
    """
    # 根据 coarse joint estimation 挑选出有信息量的一些帧, 进行 fine joint estimation
    kf_idx = farthest_scale_sampling(joint_states, M=num_key_frames)
    # obj_mask_seq by sam2 image mode
    obj_mask_seq = whole_obj_masking_sam(
        rgb_seq[kf_idx], 
        tracks2d_filtered[kf_idx], 
        franka_seq[kf_idx], 
        visualize
    ) 
    # obj_mask_seq by sam2 video mode
    # obj_mask_seq = whole_obj_masking_sam2(
    #     rgb_folder,
    #     kf_idx,
    #     tracks2d_filtered[kf_idx], 
    #     franka_seq[kf_idx], 
    #     visualize
    # )

    # 对 obj_mask_seq 进行 depth 过滤, 可选可不选
    obj_mask_seq = obj_mask_seq & depth_mask_seq[kf_idx]
    """
        根据 tracks2d 和 obj_mask_seq 得到 dynamic_seq
    """
    dynamic_seq = get_dynamic_seq(
        obj_mask_seq, 
        tracks_2d[kf_idx][:, moving_mask_2d, :],
        tracks_2d[kf_idx][:, static_mask_2d, :], 
        visualize
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
        joint_axis_unit=joint_axis_c,
        joint_states=joint_states[kf_idx],
        depth_tolerance=0.05, # 假设 coarse 阶段的误差估计在 5 cm 内
        visualize=visualize
    ) 
    # fine estimation
    joint_axis_c_updated, jonit_states_updated = fine_joint_estimation_seq(
        K=K,
        depth_seq=depth_seq[kf_idx], 
        dynamic_seq=dynamic_seq_updated,
        joint_type=joint_type, 
        joint_axis_unit=joint_axis_c, 
        joint_states=joint_states[kf_idx],
        max_icp_iters=200, # ICP 最多迭代多少轮
        optimize_joint_axis=True,
        optimize_state_mask=np.arange(num_key_frames)!=0,
        # 这里设置不迭代的优化 dynamic_mask
        update_dynamic_mask=np.zeros(num_key_frames).astype(np.bool_),
        lr=5e-3, # 5 mm
        tol=1e-8,
        icp_select_range=0.05,
        visualize=visualize
    )

    Rc2w = Tw2c[:3, :3].T # 3, 3
    joint_axis_w = Rc2w @ joint_axis_c
    joint_axis_w_updated = Rc2w @ joint_axis_c_updated
        
    if save_dir is not None:
        np.savez(
            save_dir + "reconstructed_data.npz",
            K=K,
            rgb_seq=rgb_seq[kf_idx],
            depth_seq=depth_seq[kf_idx],
            dynamic_seq=dynamic_seq_updated,
            joint_axis_w=joint_axis_w_updated,
            joint_states=jonit_states_updated,
            joint_type=joint_type,
            franka_seq=franka_seq[kf_idx],
        )
    
    if gt_joint_axis is not None:
        print(f"\tgt axis: {gt_joint_axis}")

        dot_before = np.dot(gt_joint_axis, joint_axis_w)
        dot_after = np.dot(gt_joint_axis, joint_axis_w_updated)
        print(f"\tbefore: {np.degrees(np.arccos(dot_before))}")
        print("\tjoint axis: ", joint_axis_w)
        print("\tjoint states: ", joint_states[kf_idx])

        print(f"\tafter : {np.degrees(np.arccos(dot_after))}")
        print("\tjoint axis: ", joint_axis_w_updated)
        print("\tjoint states: ", jonit_states_updated)
    

if __name__ == "__main__":
    reconstruct(
        explore_data=np.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/explore/explore_data.npz"),
        visualize=False,
        gt_joint_axis=np.array([-1, 0, 0])
    )
