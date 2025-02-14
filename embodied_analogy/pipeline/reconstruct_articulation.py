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
num_initial_uvs = 1000
num_informative_frame_idx = 5
text_prompt = "drawer"
whole_obj_masking_with_sam = True
##########################################################################


"""
    读取 exploration 阶段获取的视频数据, 进而对物体进行感知和推理理解
"""
# 读取数据
record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
file_name = "/2025-02-13_13-43-47.npz"
dr = RecordDataReader(record_path_prefix, file_name)
dr.process_data()

rgb_seq = dr.rgb # T H W C
depth_seq = np.squeeze(dr.depth) # T H W
franka_tracks_seq = dr.franka_tracks_2d # T, M, 2, 把 gripper 刚体对应的点去掉
K = dr.intrinsic # 3, 3
Tw2c = dr.data["extrinsic"] # 4, 4
# object_mask_0 = dr.seg
# TODO: initial rgb 里还可以看到 camera 的可视化框??
initial_rgb = dr.initial_rgb
initial_depth = dr.initial_depth

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

import time
algo_start = time.time()
"""
    根据物体初始状态的图像, 得到一些初始跟踪点, initial_uvs
"""
# 根据 rgb_seq[0], 先得到 initial_bbox
from embodied_analogy.perception.grounding_dino import run_groundingDINO
initial_bboxs, initial_bbox_scores = run_groundingDINO(
    image=rgb_seq[0],
    text_prompt=text_prompt,
    visualize=visualize
)
initial_bbox = initial_bboxs[0]

# 然后根据 initial_bbox 得到 initial_mask
from embodied_analogy.perception.sam_masking import run_sam_whole
initial_mask, _, _ = run_sam_whole(
    rgb_img=rgb_seq[0], # numpy
    positive_points=None,  # np.array([N, 2])
    positive_bbox=initial_bbox, # np.array([4]), [u_left, v_left, u_right, v_right]
    negative_points=franka_tracks_seq[0],
    visualize=visualize
)
# 在 initial_bbox 内均匀采样
initial_uvs = sample_points_within_bbox_and_mask(initial_bbox, initial_mask, num_initial_uvs)

if visualize:
    import napari
    viewer = napari.view_image(initial_rgb)
    viewer.title = "initial uvs on intial rgb"
    initial_uvs_vis = initial_uvs[:, [1, 0]]
    viewer.add_points(initial_uvs_vis, size=2, name="initial_uvs", face_color="green")
    napari.run()


"""
    对于 initial_uvs 进行追踪得到 tracks_2d 
    根据 depth filter 得到 tracks2d_filtered
"""
tracks_2d, pred_visibility = track_any_points(rgb_seq, initial_uvs, visiualize=visualize) # [T, M, 2], [T, M]

# 在这里将 tracks_2d 根据图像坐标的变换聚类为两类
# TODO: 这个要不要转移到 filter 之后
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
informative_frame_idx = farthest_scale_sampling(joint_states, M=num_informative_frame_idx)

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
dynamic_mask_seq_updated = filter_dynamic_mask_seq(
    K=K,
    depth_seq=depth_seq[informative_frame_idx],
    dynamic_mask_seq=dynamic_mask_seq,
    joint_type=joint_type,
    joint_axis_unit=joint_axis_camera,
    joint_states=joint_states[informative_frame_idx],
    visualize=visualize
) # T, H, W

# fine estimation
joint_axis_updated, jonit_states_updated = fine_joint_estimation_seq(
    K=K,
    depth_seq=depth_seq[informative_frame_idx], 
    dynamic_mask_seq=dynamic_mask_seq_updated,
    joint_type=joint_type, 
    joint_axis_unit=joint_axis_camera, 
    joint_states=joint_states[informative_frame_idx],
    max_icp_iters=200, # ICP 最多迭代多少轮
    optimize_joint_axis=False,
    optimize_state_mask=np.arange(num_informative_frame_idx)!=0,
    lr=3e-4, # 0.1 mm
    tol=1e-7,
    visualize=visualize
)

algo_end = time.time()

translation_w_gt = np.array([-1, 0, 0])
translation_c_gt = Rc2w.T @ translation_w_gt

dot_before = np.dot(translation_c_gt, joint_axis_camera)
dot_after = np.dot(translation_c_gt, joint_axis_updated)
print(f"\tbefore: {np.degrees(np.arccos(dot_before))}")
print("\tjoint axis: ", joint_axis_camera)
print("\tjoint states: ", joint_states[informative_frame_idx])

print(f"\tafter : {np.degrees(np.arccos(dot_after))}")
print("\tjoint axis: ", joint_axis_updated)
print("\tjoint states: ", jonit_states_updated)
print(f"time used: {algo_end - algo_start} s")
