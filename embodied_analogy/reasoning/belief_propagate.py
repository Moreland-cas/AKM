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
from embodied_analogy.reasoning.utils import (
    filter_2d_tracks_by_visibility, 
    filter_2d_tracks_by_depthSeq_mask, 
    # extract_tracked_depths, 
    visualize_2d_tracks_on_video,
)
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud, 
    visualize_pc, 
    camera_to_image,
    image_to_camera,
    tracks3d_variance
)

"""
    读取 exploration 阶段获取的视频数据, 进而对物体进行感知和推理理解
"""
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-01-07_18-06-10.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = dr.depth # T H W 1
K = dr.intrinsic # 3, 3
Tw2c = dr.data["extrinsic"] # 4, 4
object_mask_0 = dr.seg
visualize = True
save_intermidiate = False

# 对于 depth_seq 进行处理, 得到 depth_seq_mask, 用于标记其中depth为 0, 或者重投影在地面上的位置

# 初始化中间过程的存储位置 tmp_folder
tmp_folder = "/home/zby/Programs/Embodied_Analogy/assets/tmp/"
rgb_folder = os.path.join(tmp_folder, "rgbs")
depth_folder = os.path.join(tmp_folder, "depths")
mask_folder = os.path.join(tmp_folder, "masks")
vis_folder = os.path.join(tmp_folder, "vis")
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

if save_intermidiate:
    for i, rgb in enumerate(rgb_seq):
        Image.fromarray(rgb).save(os.path.join(rgb_folder, f"{i}.jpg"))
    for i, depth in enumerate(depth_seq):
        np.save(os.path.join(depth_folder, f"{i}.npy"), depth) # H, W, 1
        
if visualize:
    # 展示初始的 pointcloud sequence
    T = depth_seq.shape[0]
    raw_pc_list = []
    for i in range(T):
        pc_i = depth_image_to_pointcloud(depth_seq[i].squeeze(), None, K) # N, 3
        indices = np.random.choice(pc_i.shape[0], size=5000, replace=False)
        # 使用这些索引获取对应的 100 x 3 数组
        pc_i = pc_i[indices]
        raw_pc_list.append(pc_i)
    raw_pc_list = np.stack(raw_pc_list)
    # vis_tracks_3d_napari(raw_pc_list)

"""
    根据物体初始状态的图像, 得到一些初始跟踪点
"""
pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255. # N,3   

# 对初始帧中的物体点云聚类得到 num_clusters 个点簇, 并计算 num_clusters 个簇中心在图像上的投影
num_clusters = 600
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centroids, labels, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

if visualize:
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

if visualize:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "track2d_initial_results", vis_folder)

# 筛选出一直能跟踪到的点 
pred_tracks_2d = filter_2d_tracks_by_visibility(pred_tracks_2d, pred_visibility)

if visualize:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "tracks2d_filtered_by_visibility", vis_folder)

# 筛选出不曾重投影到地面上的点
# depth_tracks = extract_tracked_depths(depth_seq, pred_tracks_2d) # T, M
# pred_tracks_2d, depth_tracks = filter_by_depthSeq(pred_tracks_2d, depth_tracks) # [T, M, 2], [T, M]
pred_tracks_2d, depth_tracks = filter_2d_tracks_by_depthSeq_mask(pred_tracks_2d, depth_seq_mask)
    
if visualize:
    visualize_2d_tracks_on_video(rgb_seq, pred_tracks_2d, "tracks2d_filtered_by_depth_mask", vis_folder)

"""
    将 tracks2d 聚类为 static 和 moving 两类
"""
# 根据 tracks2d 得到 tracks3d
T, M, _ = pred_tracks_2d.shape
pred_tracks_3d = image_to_camera(pred_tracks_2d.reshape(T*M, -1), depth_tracks.reshape(-1), K) # T * M, 3
pred_tracks_3d = pred_tracks_3d.reshape(T, M, 3) # T, M, 3

if visualize:
    vis_tracks3d_napari(pred_tracks_3d)
    
motion_for_kmeans = pred_tracks_3d.permute(1, 0, 2).reshape(M, -1) # M T*3
_, moving_labels, _ = cluster.k_means(motion_for_kmeans.cpu().numpy(), init="k-means++", n_clusters=2)

# 在这里将分类后的两类点进行判断, 判断哪一类是静止的, 哪一类是运动的 (M = M1 + M2)
part1 = pred_tracks_3d[:, moving_labels == 0, :] # T, M1, 3
part2 = pred_tracks_3d[:, moving_labels == 1, :] # T, M2, 3

# 计算点的轨迹的 variance, 将 variance 小的点认为是静止的, variance 大的点认为是运动的
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
    
# if visualize:
#     visualize_pc(pred_tracks_3d[:M, :].cpu().numpy(), rigid_part_colors)
    
# 在这里跑 sam2, 得到 moving_part 和 static_part 的分割
from embodied_analogy.perception.sam2_masking import run_sam2_whole
from embodied_analogy.visualization.vis_sam2_mask import visualize_sam2_mask
moving_tracks_2d = pred_tracks_2d[:, moving_mask, :] # T, N, 2
static_tracks_2d = pred_tracks_2d[:, static_mask, :] # T, N, 2
point_prompt = torch.stack([moving_tracks_2d[0][0], static_tracks_2d[0][0]]).cpu() #2, 2
video_masks = run_sam2_whole(rgb_folder, initial_point_prompt=point_prompt)
if visualize:
    visualize_sam2_mask(rgb_seq, video_masks)

# sam2 mask filtering: 根据 depth_seq 对 sam2 得到的 mask 进行过滤, 使得过滤后的 mask 不包含深度为 0, 或者重投影后会落在地面上的点
# TODO：

if save_intermidiate:
    for i, mask in enumerate(video_masks):
        mask_255 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_255).save(os.path.join(mask_folder, f"{i}.png"))
        # np.save(os.path.join(mask_folder, f"{i}.npy"), mask)

# 5) 初步估计出 joint parameters
from embodied_analogy.estimation.joint_estimate_w_corr import estimate_translation_from_tracks
    
translation_c, scales = estimate_translation_from_tracks(pred_tracks_3d[:, moving_mask, :])
Rc2w = Tw2c[:3, :3].T # 3, 3
translation_w = Rc2w @ translation_c

# for debug, visualize 原始的跟踪点，和第一frame的moving points 按照我们的估计的joint param 计算出的轨迹
if visualize:
    tracks_3d = pred_tracks_3d.cpu().numpy() # T, M, 3
    moving_points = tracks_3d[0, moving_mask, :] # M, 3
    # static_points = tracks_3d[0, moving_labels == 0, :]
    points = []
    colors = []
    for i in range(len(tracks_3d)):
        points_tmp = []
        colors_tmp = []
        points_tmp.extend(tracks_3d[i])
        colors_tmp.extend([[0, 0, 1]] * len(tracks_3d[i]))
        points_tmp.extend(moving_points + scales[i] * translation_c)
        colors_tmp.extend([[1, 0, 0]] * len(moving_points))
        points.append(np.array(points_tmp))
        colors.append(np.array(colors_tmp))
        
    vis_pointcloud_series_napari(points, colors)
# 将 joint states 保存到 tmp_folder
if save_intermidiate:
    np.savez(
        os.path.join(tmp_folder, "joint_state.npz"), 
        translation_c=translation_c, 
        translation_w=translation_w, 
        scales=scales
    )

# 6) 对于第一帧剩下的 k-m 个点簇进行分类验证, 分为 static, moving 和 unknown 三类
# static 和 moving 里又分为一直能看到的, 和随着运动新发现的 


# 7) 用所有 known 的点簇重新估计 joint parameters