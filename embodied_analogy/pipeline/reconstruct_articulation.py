import os
import numpy as np
from PIL import Image
import sklearn.cluster as cluster

from embodied_analogy.pipeline.process_recorded_data import RecordDataReader
from embodied_analogy.perception import *
from embodied_analogy.estimation import *
from embodied_analogy.utility import *


################################# PARAMS #################################
visualize = False
whole_obj_masking_with_sam = True
##########################################################################


"""
    读取 exploration 阶段获取的视频数据, 进而对物体进行感知和推理理解
"""
# 读取数据
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
# file_name = "/2025-01-07_18-06-10.npz"
file_name = "/2025-02-08_14-57-26.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = np.squeeze(dr.depth) # T H W
franka_tracks_seq = dr.franka_tracks_2d # T, M, 2, 把 gripper 刚体对应的点去掉
K = dr.intrinsic # 3, 3
Tw2c = dr.data["extrinsic"] # 4, 4
object_mask_0 = dr.seg

# process depth
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
vis_folder = os.path.join(tmp_folder, "vis")
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(depth_mask_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

for i, rgb in enumerate(rgb_seq):
    Image.fromarray(rgb).save(os.path.join(rgb_folder, f"{i}.jpg"))
for i, depth in enumerate(depth_seq):
    np.save(os.path.join(depth_folder, f"{i}.npy"), depth) # H, W, 1
for i, depth_mask in enumerate(depth_seq_mask):
    depth_mask_255 = (depth_mask * 255).astype(np.uint8)
    Image.fromarray(depth_mask_255).save(os.path.join(depth_mask_folder, f"{i}.png"))
        
# if visualize:
#     # 展示初始的 pointcloud sequence
#     T = depth_seq.shape[0]
#     raw_pc_list = []
#     for i in range(T):
#         pc_i = depth_image_to_pointcloud(depth_seq[i], depth_seq_mask[i], K) # N, 3
#         indices = np.random.choice(pc_i.shape[0], size=5000, replace=False)
#         # 使用这些索引获取对应的 100 x 3 数组
#         pc_i = pc_i[indices]
#         pc_i_world = camera_to_world(pc_i, Tw2c)
#         raw_pc_list.append(pc_i_world)
#     # raw_pc_list = np.stack(raw_pc_list)
#     vis_pointcloud_series_napari(raw_pc_list)


"""
    根据物体初始状态的图像, 得到一些初始跟踪点
"""
pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255. # N,3   

# 对初始帧中的物体点云聚类得到 num_clusters 个点簇, 并计算 num_clusters 个簇中心在图像上的投影
num_clusters = 600
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centroids, labels, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

# if visualize:
#     # 可视化初始点云
#     visualize_pc(pc_0, rgb_0)
#     color_map = np.random.rand(num_clusters, 3)
#     colors = color_map[labels]
#     # 可视化聚类结果
#     visualize_pc(pc_0, colors)


"""
    对于 rgb_seq 操作得到 tracks_2d 和 tracks2d_filtered
"""
centroids_camera = centroids[:, :3] # num_clusters, 3
centroids_image, _ = camera_to_image(centroids_camera, K) # num_clusters, 2
tracks_2d, pred_visibility = track_any_points(rgb_seq, centroids_image, visiualize=visualize) # [T, M, 2], [T, M]

# 在这里将 tracks_2d 根据图像坐标的变换聚类为两类
moving_mask_2d, static_mask_2d = cluster_tracks_2d(rgb_seq, tracks_2d, use_diff=True, visualize=visualize, viewer_title="dynamic clustering tracks2d")

# filter tracks2d by visibility
tracks2d_filtered = filter_tracks2d_by_visibility(rgb_seq, tracks_2d, pred_visibility, visualize)

# filter tracks2d by depthSeq_mask
tracks2d_filtered = filter_tracks2d_by_depthSeq_mask(rgb_seq, tracks2d_filtered, depth_seq_mask, visualize)
    
        
"""
    dynamic segment tracks3d_filtered
"""  
# 根据 tracks2d 得到 tracks3d
T, M, _ = tracks2d_filtered.shape
tracks2d_filtered_depth = extract_tracked_depths(depth_seq, tracks2d_filtered) # T, M
tracks3d_filtered = image_to_camera(tracks2d_filtered.reshape(T*M, -1), tracks2d_filtered_depth.reshape(-1), K) # T*M, 3
tracks3d_filtered = tracks3d_filtered.reshape(T, M, 3) # T, M, 3
moving_mask_3d, static_mask_3d = cluster_tracks_3d(tracks3d_filtered, use_diff=True, visualize=visualize, viewer_title="dynamic clustering tracks3d_filtered")

        
"""
    coarse joint estimation with tracks3d_filtered
"""
joint_type, joint_axis_camera, joint_states = coarse_joint_estimation(tracks3d_filtered[:, moving_mask_3d, :], visualize)
Rc2w = Tw2c[:3, :3].T # 3, 3
joint_axis_world = Rc2w @ joint_axis_camera
    
    
"""
    根据 rgb_seq 和 tracks2d 得到 video_masks (可以用 sam 或者 sam2)
"""
# 根据 coarse joint estimation 挑选出有信息量的一些帧, 进行 fine joint estimation
informative_frame_idx = farthest_scale_sampling(joint_states, M=5)

if whole_obj_masking_with_sam:
    video_masks = whole_obj_masking_sam(
        rgb_seq[informative_frame_idx], 
        tracks2d_filtered[informative_frame_idx], 
        franka_tracks_seq[informative_frame_idx], 
        visualize
    ) # T, H, W
else:
    video_masks = whole_obj_masking_sam2(
        rgb_folder,
        informative_frame_idx,
        tracks2d_filtered[informative_frame_idx], 
        franka_tracks_seq[informative_frame_idx], 
        visualize
    )

# 对 video_masks 进行 depth 过滤, 可选可不选
video_masks = video_masks & depth_seq_mask[informative_frame_idx]
    
    
"""
    根据 tracks2d 和 whole obj video_masks 得到 dynamic_mask
"""
dynamic_mask_seq = get_dynamic_mask_seq(
    video_masks, 
    tracks_2d[informative_frame_idx][:, moving_mask_2d, :],
    tracks_2d[informative_frame_idx][:, static_mask_2d, :], 
    visualize
) # T, H, W


"""
    根据 dynamic_mask 中的 moving_part, 利用 ICP 估计出精确的 joint params
"""
# filter dynamic mask seq
from embodied_analogy.estimation.fine_joint_est import fine_joint_estimation_seq
joint_axis_updated, jonit_states_updated = fine_joint_estimation_seq(
    K,
    depth_seq[informative_frame_idx], 
    dynamic_mask_seq,
    joint_type, 
    joint_axis_unit=joint_axis_camera, 
    joint_states=joint_states[informative_frame_idx],
    max_icp_iters=200, # ICP 最多迭代多少轮
    lr=3e-4, # 0.1 mm
    tol=1e-8,
    visualize=visualize
)


translation_w_gt = np.array([0, 0, 1])
translation_c_gt = Rc2w.T @ translation_w_gt

print(f"\t before: {np.dot(translation_c_gt, joint_axis_camera)}")
print(f"\t after : {np.dot(translation_c_gt, joint_axis_updated)}")
pass
