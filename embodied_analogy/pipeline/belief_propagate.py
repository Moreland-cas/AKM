"""
Input:
    rgb_seq: [t, h, w]
    depth_seq: [t, h, w]
    object_mask: [t, h, w]
    franka_mask: [t, h, w]
Output:
    moving_mask: [t, h, w]
    joint_states: [t, 1]
    joint_params:
        (3,) for prismatic joint
        (6,) for revolute joint
"""
import os
import torch
import numpy as np
from PIL import Image
import sklearn.cluster as cluster
from embodied_analogy.pipeline.process_record import RecordDataReader
from embodied_analogy.perception.online_cotracker import track_any_points
from embodied_analogy.visualization.vis_tracks_3d import (
    vis_tracks3d_napari,
    vis_pointcloud_series_napari
)
from embodied_analogy.estimation.utils import (
    filter_2d_tracks_by_visibility, 
    filter_2d_tracks_by_depthSeq_mask, 
    extract_tracked_depths, 
    visualize_2d_tracks_on_video,
)
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud, 
    visualize_pc, 
    camera_to_image,
    camera_to_world,
    image_to_camera,
    tracks3d_variance
)
from embodied_analogy.estimation.coarse_joint_est import coarse_joint_estimation
from embodied_analogy.estimation.fine_joint_est import fine_joint_estimation 

"""
    读取 exploration 阶段获取的视频数据, 进而对物体进行感知和推理理解
"""
# 读取数据
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-01-07_18-06-10.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = np.squeeze(dr.depth) # T H W
K = dr.intrinsic # 3, 3
Tw2c = dr.data["extrinsic"] # 4, 4
object_mask_0 = dr.seg
visualize = False
save_intermidiate = True

# 对于 depth_seq 进行处理, 得到 depth_seq_mask, 用于标记其中depth为 0, 或者重投影在地面上的位置
depth_seq_mask = np.ones_like(depth_seq, dtype=np.bool_) # T, H, W
depth_seq_mask = depth_seq_mask & (depth_seq > 0)

for i in range(depth_seq.shape[0]):
    _, H, W = depth_seq.shape
    pc_camera = depth_image_to_pointcloud(depth_seq[i], None, K) # H*W, 3
    pc_world = camera_to_world(pc_camera, Tw2c)
    pc_height_mask = (pc_world[:, 2] > 0.03).reshape(H, W) # H*W
    depth_seq_mask[i] = depth_seq_mask[i] & pc_height_mask

# 初始化中间过程的存储位置 tmp_folder
tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
rgb_folder = os.path.join(tmp_folder, "rgbs")
depth_folder = os.path.join(tmp_folder, "depths")
depth_mask_folder = os.path.join(tmp_folder, "depth_masks")
sam2_mask_folder = os.path.join(tmp_folder, "sam2_masks")
vis_folder = os.path.join(tmp_folder, "vis")
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(depth_mask_folder, exist_ok=True)
os.makedirs(sam2_mask_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

if save_intermidiate:
    for i, rgb in enumerate(rgb_seq):
        Image.fromarray(rgb).save(os.path.join(rgb_folder, f"{i}.jpg"))
    for i, depth in enumerate(depth_seq):
        np.save(os.path.join(depth_folder, f"{i}.npy"), depth) # H, W, 1
    for i, depth_mask in enumerate(depth_seq_mask):
        depth_mask_255 = (depth_mask * 255).astype(np.uint8)
        Image.fromarray(depth_mask_255).save(os.path.join(depth_mask_folder, f"{i}.png"))
        
if visualize and False:
    # 展示初始的 pointcloud sequence
    T = depth_seq.shape[0]
    raw_pc_list = []
    for i in range(T):
        pc_i = depth_image_to_pointcloud(depth_seq[i], depth_seq_mask[i], K) # N, 3
        indices = np.random.choice(pc_i.shape[0], size=5000, replace=False)
        # 使用这些索引获取对应的 100 x 3 数组
        pc_i = pc_i[indices]
        pc_i_world = camera_to_world(pc_i, Tw2c)
        raw_pc_list.append(pc_i_world)
    # raw_pc_list = np.stack(raw_pc_list)
    vis_pointcloud_series_napari(raw_pc_list)

"""
    根据物体初始状态的图像, 得到一些初始跟踪点
"""
pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255. # N,3   

# 对初始帧中的物体点云聚类得到 num_clusters 个点簇, 并计算 num_clusters 个簇中心在图像上的投影
num_clusters = 600
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centroids, labels, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

if visualize and False:
    # 可视化初始点云
    visualize_pc(pc_0, rgb_0)
    color_map = np.random.rand(num_clusters, 3)
    colors = color_map[labels]
    # 可视化聚类结果
    visualize_pc(pc_0, colors)

"""
    对初始跟踪点进行 rgb tracking, 得到一些 2d tracks 
"""
centroids_camera = centroids[:, :3] # num_clusters, 3
centroids_image, _ = camera_to_image(centroids_camera, K) # num_clusters, 2
pred_tracks_2d, pred_visibility = track_any_points(rgb_seq, centroids_image) # [T, M, 2], [T, M]

if save_intermidiate:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "track2d_initial_results", vis_folder)

# 筛选出一直能跟踪到的点 
pred_tracks_2d = filter_2d_tracks_by_visibility(pred_tracks_2d, pred_visibility)

if save_intermidiate:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "tracks2d_filtered_by_visibility", vis_folder)

# 筛选出不曾重投影到地面上的点
# depth_tracks = extract_tracked_depths(depth_seq, pred_tracks_2d) # T, M
# pred_tracks_2d, depth_tracks = filter_by_depthSeq(pred_tracks_2d, depth_tracks) # [T, M, 2], [T, M]
pred_tracks_2d = filter_2d_tracks_by_depthSeq_mask(pred_tracks_2d, depth_seq_mask)
depth_tracks = extract_tracked_depths(depth_seq, pred_tracks_2d) # T, M
    
if save_intermidiate:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "tracks2d_filtered_by_depth_mask", vis_folder)
        
"""
    将 tracks2d 聚类为 static 和 moving 两类
"""
# 根据 tracks2d 得到 tracks3d
T, M, _ = pred_tracks_2d.shape
pred_tracks_3d = image_to_camera(pred_tracks_2d.reshape(T*M, -1), depth_tracks.reshape(-1), K) # T * M, 3
pred_tracks_3d = pred_tracks_3d.reshape(T, M, 3) # T, M, 3

# 根据 tracks3d 将所有点分类为两类 
motion_for_kmeans = pred_tracks_3d.permute(1, 0, 2).reshape(M, -1) # M T*3
_, moving_labels, _ = cluster.k_means(motion_for_kmeans.cpu().numpy(), init="k-means++", n_clusters=2)

# 确定哪一类是 moving, 那一类是 static, 依据是轨迹的 variance
part1 = pred_tracks_3d[:, moving_labels == 0, :] # T, M1, 3
part2 = pred_tracks_3d[:, moving_labels == 1, :] # T, M2, 3

if tracks3d_variance(part1) > tracks3d_variance(part2):
    moving_mask = (moving_labels == 0)
    static_mask = (moving_labels == 1)
else:
    moving_mask = (moving_labels == 1)
    static_mask = (moving_labels == 0)

if visualize:
    red_and_green = np.array([[1, 0, 0], [0, 1, 0]])
    rigid_part_colors = red_and_green[moving_labels] # M, 3
    vis_tracks3d_napari(pred_tracks_3d, rigid_part_colors)
        
"""
    根据 tracks2d 初步估计出 joint params
"""
joint_type, joint_axis_camera, joint_states = coarse_joint_estimation(pred_tracks_3d[:, moving_mask, :], visualize)

Rc2w = Tw2c[:3, :3].T # 3, 3
axis_world = Rc2w @ joint_axis_camera
    
# 保存 joint states 到 tmp_folder
if save_intermidiate:
    np.savez(
        os.path.join(tmp_folder, "joint_state.npz"), 
        joint_type=joint_type,
        translation_c=joint_axis_camera, 
        translation_w=axis_world, 
        scales=joint_states
    )

"""
    运行 sam2 得到 articulated objects 整体随着时间变化的 sam2_video_mask
"""
from embodied_analogy.perception.sam2_masking import run_sam2_whole
from embodied_analogy.visualization.vis_sam2_mask import visualize_sam2_mask
# TODO: 把这里的 point_prompt 改为 bbox
point_prompt = pred_tracks_2d[0, :2, :].cpu() #2, 2
sam2_video_masks = run_sam2_whole(rgb_folder, initial_point_prompt=point_prompt) # T, H, W

# TODO: 在这里添加一个迭代逻辑，就是如果发现 tracks_2d 上的点在某个时刻没有在 sam2_video_masks 中, 那就在该位置添加 positive point prompt, 
# 如果发现 机械臂上的点 或者 !depthSeq_mask 的点在 mask 里, 那就添加 negative point prompt, 然后运行一次 sam2 propagate

# sam2 mask filtering: 根据 depth_seq 对 sam2 得到的 mask 进行过滤, 使得过滤后的 mask 不包含深度为 0, 或者重投影后会落在地面上的点
sam2_video_masks = sam2_video_masks & depth_seq_mask

if visualize:
    visualize_sam2_mask(rgb_seq, sam2_video_masks)

if save_intermidiate:
    for i, sam2_mask in enumerate(sam2_video_masks):
        sam2_mask_255 = (sam2_mask * 255).astype(np.uint8)
        Image.fromarray(sam2_mask_255).save(os.path.join(sam2_mask_folder, f"{i}.png"))
        # np.save(os.path.join(mask_folder, f"{i}.npy"), mask)
        
"""
    根据 sam2_video_mask 初步估计出 joint params, 利用 ICP 估计出精确的 joint params 和 object model
"""
# 根据 coarse joint estimation 挑选出有信息量的两帧, 进行 fine joint estimation
coarse_joint_axis = joint_axis_camera
translation_w_gt = np.array([0, 0, 1])
translation_c_gt = Rc2w.T @ translation_w_gt

for i in range(T):
    depth_ref = depth_seq[0]
    depth_tgt = depth_seq[i]
    obj_mask_ref = sam2_video_masks[0]
    obj_mask_tgt = sam2_video_masks[i]
    coarse_joint_state_ref2tgt = joint_states[i] - joint_states[0]
    fine_joint_axis, fine_joint_state_ref2tgt = fine_joint_estimation(
        K,
        depth_ref, depth_tgt,
        obj_mask_ref, obj_mask_tgt,
        joint_type, coarse_joint_axis, coarse_joint_state_ref2tgt,
    )
    print(f"{i}th frame:")
    print(f"\t before: {np.dot(translation_c_gt, coarse_joint_axis)}")
    print(f"\t after : {np.dot(translation_c_gt, fine_joint_axis)}")

