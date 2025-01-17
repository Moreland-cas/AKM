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
import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.process_record import RecordDataReader
from embodied_analogy.utils import depth_image_to_pointcloud, visualize_pc

record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-01-07_18-06-10.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = dr.depth # T H W 1
K = dr.intrinsic # 3, 3
object_mask_0 = dr.seg

pc_0 = depth_image_to_pointcloud(depth_seq[0].squeeze(), object_mask_0, K) # N, 3
rgb_0 = rgb_seq[0][object_mask_0] / 255.
# visualize_pc(pc_0, rgb_0)

# 对初始帧中的物体点云聚类得到 k 个点簇, 并计算 k 个簇中心在图像上的投影
# TODO: 可以在一开始聚类的时候也考虑视觉特征
num_clusters = 600
feat_for_kmeans = np.concatenate([pc_0, rgb_0], axis=-1)
centers, per_point_class_idx, _ = cluster.k_means(feat_for_kmeans, init="k-means++", n_clusters=num_clusters)

color_map = np.random.rand(num_clusters, 3)
colors = color_map[per_point_class_idx]

visualize_pc(pc_0, colors)
# import pdb;pdb.set_trace()

# 然后对这 k 个点进行 rgb tracking, 筛选出稳定的 m 个点

# 根据这 m 个点的轨迹将其聚类为 static 和 moving 两类

# 对于剩下的 k-m 个点簇进行分类验证, 分为 static, moving 和 unknown 三类