import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import sys
from einops import rearrange, repeat

from embodied_analogy.project_config import PROJECT_ROOT, ASSET_PATH
sys.path.append(os.path.join(PROJECT_ROOT, "third_party", "GAPartNet"))
from structure.gapartnet import ObjIns
from structure.utils import draw_bbox_from_world, _load_perception_model
from gapartnet.structure.point_cloud import PointCloud
from gapartnet.dataset.gapartnet import apply_voxelization
from gapartnet.network.grouping_utils import filter_invalid_proposals, apply_nms
from gapartnet.misc.pose_fitting import estimate_pose_from_npcs


from embodied_analogy.representation.obj_repr import Obj_repr


def load_gapartnet_model(ckpt_path):
    """Load GAPartNet model from checkpoint"""
    print(f"Loading GAPartNet model from {ckpt_path}...")

    if not ckpt_path.exists():
        print(f"Checkpoint file {ckpt_path} does not exist")
        return None

    # Load the perception model
    gapartnet_model = _load_perception_model(str(ckpt_path))

    return gapartnet_model


def create_pointcloud_input_fixed(pc_xyz, frame_idx, device):
    """Create PointCloud input with PROPER GAPartNet coordinate handling"""
    print(f"  Creating PointCloud input for frame {frame_idx}...")

    # Store original statistics for inverse transformation
    center = np.mean(pc_xyz, axis=0)
    max_radius = np.max(np.linalg.norm(pc_xyz - center, axis=1))

    if max_radius < 1e-6:
        max_radius = 1.0

    # GAPartNet ball normalization
    centered_pc = pc_xyz - center
    normalized_pc = centered_pc / max_radius

    print(f"    Ball normalization: center={center}, max_radius={max_radius}")
    print(f"    Normalized range: {np.min(normalized_pc)} to {np.max(normalized_pc)}")

    # Create RGB colors
    fake_colors = np.ones((pc_xyz.shape[0], 3)) * 0.5
    pc_with_colors = np.concatenate([normalized_pc, fake_colors], axis=1)

    # Convert to tensor
    pc_tensor = torch.tensor(pc_with_colors, dtype=torch.float32, device=device)

    # Create labels
    num_points = pc_tensor.shape[0]
    sem_labels = torch.ones(num_points, dtype=torch.int64, device=device)
    instance_labels = torch.zeros(num_points, dtype=torch.int32, device=device)

    # Create instance regions
    xyz = pc_tensor[:, :3]
    mean_xyz = torch.mean(xyz, dim=0)
    min_xyz = torch.min(xyz, dim=0)[0]
    max_xyz = torch.max(xyz, dim=0)[0]

    instance_regions = torch.zeros(num_points, 9, dtype=torch.float32, device=device)
    instance_regions[:, 0:3] = mean_xyz
    instance_regions[:, 3:6] = min_xyz
    instance_regions[:, 6:9] = max_xyz

    num_points_per_instance = torch.tensor([num_points], dtype=torch.int32, device=device)
    instance_sem_labels = torch.tensor([1], dtype=torch.int32, device=device)

    # Create PointCloud
    pc = PointCloud(
        pc_id=f"frame_{frame_idx}",
        points=pc_tensor,
        obj_cat=1,
        sem_labels=sem_labels,
        instance_labels=instance_labels,
        gt_npcs=None,
        num_instances=1,
        instance_regions=instance_regions,
        num_points_per_instance=num_points_per_instance,
        instance_sem_labels=instance_sem_labels
    )

    # Apply voxelization
    pc = apply_voxelization(pc, voxel_size=(1. / 100, 1. / 100, 1. / 100))

    return pc, max_radius, center


def transform_gapartnet_bbox_to_camera(bbox_ball, max_radius, center):
    """CORRECT transformation from GAPartNet ball space to camera coordinates"""
    # GAPartNet bbox is in normalized ball space [-1, 1]
    # Transform back to original camera coordinates
    bbox_camera = bbox_ball * max_radius + center
    return bbox_camera


def fix_gapartnet_bbox_corner_ordering(bbox_from_gapartnet):
    """
    Fix the corner ordering from GAPartNet's pose fitting to match our visualization expectations.

    GAPartNet generates corners in this order:
    [[-x, -y, -z], [+x, -y, -z], [-x, +y, -z], [-x, -y, +z],
     [+x, +y, -z], [+x, -y, +z], [-x, +y, +z], [+x, +y, +z]]

    But our visualization expects:
    [0: -x,-y,-z], [1: +x,-y,-z], [2: +x,+y,-z], [3: -x,+y,-z],  # bottom face
    [4: -x,-y,+z], [5: +x,-y,+z], [6: +x,+y,+z], [7: -x,+y,+z]]  # top face
    """
    if bbox_from_gapartnet.shape[0] != 8:
        return bbox_from_gapartnet  # Return unchanged if not 8 corners

    # GAPartNet ordering to our expected ordering mapping
    # GAPartNet: [0,1,2,3,4,5,6,7] -> Our expected: [0,1,4,2,3,5,7,6]
    reorder_indices = [0, 1, 4, 2, 3, 5, 7, 6]

    return bbox_from_gapartnet[reorder_indices]


def run_gapartnet_inference_fixed(pc_xyz, gapartnet_model, pc_seg=None, use_gt_segmentation=False):
    """
    Run GAPartNet inference following the EXACT training procedure with optional GT segmentation
    """
    device = "cuda"

    print(f"  Processing point cloud...")
    print(f"    Input: {pc_xyz.shape[0]} points")
    print(f"    Range: {np.min(pc_xyz, axis=0)} to {np.max(pc_xyz, axis=0)}")

    # Create input following the EXACT training procedure
    pc, max_radius, center = create_pointcloud_input_fixed(pc_xyz, 0, device)

    # Run inference following _training_or_validation_step procedure
    with torch.no_grad():
        # Step 1: Forward backbone
        data_batch = PointCloud.collate([pc])
        points = data_batch.points
        batch_indices = data_batch.batch_indices
        pt_xyz = points[:, :3]

        pc_feature = gapartnet_model.forward_backbone(pc_batch=data_batch)

        # Step 2: Semantic segmentation
        if use_gt_segmentation and pc_seg is not None:
            print(f"    Using GT segmentation...")
            # Create GT semantic predictions
            sem_preds = torch.from_numpy(pc_seg).to(device)
            sem_preds = sem_preds + 1
        else:
            print(f"    Using model segmentation...")
            # Use standard GAPartNet model prediction (exactly as in model.py lines 497-499)
            sem_logits = gapartnet_model.forward_sem_seg(pc_feature)
            sem_preds = torch.argmax(sem_logits.detach(), dim=-1)
            print(f"    Model slider button points: {(sem_preds == 3).sum().item()}")

        # Step 3: Offset prediction
        offsets_preds = gapartnet_model.forward_offset(pc_feature)

        # Step 4: Proposal clustering and revoxelization
        print(f"    Running proposal clustering...")
        voxel_tensor, pc_voxel_id, proposals = gapartnet_model.proposal_clustering_and_revoxelize(
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            pt_features=pc_feature,
            sem_preds=sem_preds,
            offset_preds=offsets_preds,
            instance_labels=None,  # No ground truth during inference
        )

        if proposals is None:
            print(f"    No proposals generated")
            return [], [], []

        print(f"    Found {len(proposals.proposal_offsets) - 1} proposals")

        # Step 5: Proposal scoring
        print(f"    Running proposal scoring...")
        score_logits = gapartnet_model.forward_proposal_score(
            voxel_tensor, pc_voxel_id, proposals
        )
        proposal_offsets_begin = proposals.proposal_offsets[:-1].long()
        proposal_sem_labels = sem_preds[proposals.valid_mask][proposals.sorted_indices][proposal_offsets_begin].long()
        score_logits = score_logits.gather(1, (proposal_sem_labels[:, None] - 1).clamp(min=0)).squeeze(1)
        proposals.score_preds = score_logits.detach().sigmoid()
        proposals.sem_preds = sem_preds[proposals.valid_mask][proposals.sorted_indices]

        # Step 6: NPCS prediction
        print(f"    Running NPCS prediction...")
        npcs_logits = gapartnet_model.forward_proposal_npcs(voxel_tensor, pc_voxel_id)

        # Process NPCS like in training
        npcs_logits = npcs_logits.detach()
        npcs_logits = rearrange(npcs_logits, "n (k c) -> n k c", c=3)
        sem_preds_selected = sem_preds[proposals.valid_mask][proposals.sorted_indices]
        npcs_preds = npcs_logits.gather(1, index=repeat(sem_preds_selected.long() - 1, "n -> n one c", one=1, c=3)).squeeze(1)
        proposals.npcs_preds = npcs_preds

        # Step 7: Apply filtering and NMS like in training
        print(f"    Applying filtering and NMS (same as training)...")
        proposals = filter_invalid_proposals(
            proposals,
            score_threshold=0.09,  # Same as val_score_threshold in training
            min_num_points_per_proposal=3
        )
        proposals = apply_nms(proposals, 0.3)  # Same as val_nms_iou_threshold in training

        if proposals is None or len(proposals.proposal_offsets) <= 1:
            print(f"    No valid proposals after filtering and NMS")
            return [], [], []

        print(f"    {len(proposals.proposal_offsets) - 1} proposals remain after filtering and NMS")

        # Step 8: Generate bounding boxes using pose fitting
        print(f"    Generating bounding boxes from NPCS (with training-like filtering)...")

        proposal_offsets = proposals.proposal_offsets
        pt_xyz_proposals = proposals.pt_xyz
        npcs_preds = proposals.npcs_preds

        bboxes = []
        bbox_scores = []
        bbox_sem_labels = []

        for proposal_i in range(len(proposal_offsets) - 1):
            offset_begin = proposal_offsets[proposal_i].item()
            offset_end = proposal_offsets[proposal_i + 1].item()

            # Get proposal points and NPCS
            xyz_i = pt_xyz_proposals[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]
            npcs_i = npcs_i - 0.5  # Center NPCS coordinates

            # Transform points back to camera coordinates (following EXACT training code)
            xyz_camera = transform_gapartnet_bbox_to_camera(xyz_i.cpu().numpy(), max_radius, center)
            npcs_numpy = npcs_i.cpu().numpy()

            print(f"      Processing proposal {proposal_i}: {xyz_i.shape[0]} points")

            # Use pose fitting to get bbox
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(
                xyz_camera, npcs_numpy
            )

            if scale[0] is None:
                print(f"        Pose fitting failed")
                continue

            # bbox_xyz is already in camera coordinates
            bbox_size = np.max(bbox_xyz, axis=0) - np.min(bbox_xyz, axis=0)
            print(f"        Generated bbox with size: {bbox_size}")

            # Fix corner ordering for proper visualization
            bbox_xyz_fixed = fix_gapartnet_bbox_corner_ordering(bbox_xyz)
            bboxes.append(bbox_xyz_fixed)

            # Store score and semantic label
            bbox_scores.append(proposals.score_preds[proposal_i].item())
            bbox_sem_labels.append(proposal_sem_labels[proposal_i].item())

    print(f"    Final: {len(bboxes)} bounding boxes generated")
    return bboxes, bbox_scores, bbox_sem_labels


def draw_moving_bbox_with_effects(img, bbox, camera2world_translation, world2camera_rotation, K,
                                 main_color=(0, 255, 0), shadow_color=(0, 180, 0)):
    """Draw moving bounding box with special visual effects including shadow and thick lines"""
    if len(bbox) == 0:
        return img

    bbox = np.array(bbox)

    # Transform to camera coordinates if needed
    if camera2world_translation is not None:
        assert world2camera_rotation is not None
        bbox_camera = (bbox - camera2world_translation) @ world2camera_rotation
    else:
        bbox_camera = bbox

    # Project to image coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    point2image = np.concatenate([
        (np.around(bbox_camera[:, 0] * fx / bbox_camera[:, 2] + cx)).astype(dtype=int).reshape(-1, 1),
        (np.around(bbox_camera[:, 1] * fy / bbox_camera[:, 2] + cy)).astype(dtype=int).reshape(-1, 1)
    ], axis=1)

    # First draw shadow (offset by a few pixels)
    shadow_offset = 3
    shadow_points = point2image + shadow_offset

    # Draw shadow edges with thinner lines
    shadow_thickness = 3
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    for edge in edges:
        pt1 = tuple(shadow_points[edge[0]])
        pt2 = tuple(shadow_points[edge[1]])
        cv2.line(img, pt1, pt2, color=(int(shadow_color[0]), int(shadow_color[1]), int(shadow_color[2])),
                thickness=shadow_thickness)

    # Draw main bbox with thick lines
    main_thickness = 4

    # Standard edges
    for edge in edges:
        pt1 = tuple(point2image[edge[0]])
        pt2 = tuple(point2image[edge[1]])
        cv2.line(img, pt1, pt2, color=(int(main_color[0]), int(main_color[1]), int(main_color[2])),
                thickness=main_thickness)

    # Special colored edges to show orientation (even thicker)
    special_thickness = 5
    cv2.line(img, tuple(point2image[6]), tuple(point2image[7]), color=(255, 255, 255), thickness=special_thickness)  # White
    cv2.line(img, tuple(point2image[3]), tuple(point2image[7]), color=(255, 255, 0), thickness=special_thickness)   # Yellow
    cv2.line(img, tuple(point2image[4]), tuple(point2image[7]), color=(255, 0, 255), thickness=special_thickness)   # Magenta

    # Add "MOVING PART" text label
    center_2d = np.mean(point2image, axis=0).astype(int)
    cv2.putText(img, "MOVING", (center_2d[0] - 30, center_2d[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "PART", (center_2d[0] - 20, center_2d[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def draw_3d_moving_bbox_with_effects(ax, bbox, camera2world_translation, world2camera_rotation,
                                    main_color='lime', shadow_color='darkgreen', label="MOVING PART"):
    """Draw 3D moving bounding box with special visual effects"""
    if len(bbox) == 0:
        print(f"Warning: bbox is empty")
        return

    bbox = np.array(bbox)
    if bbox.shape[0] != 8:
        print(f"Warning: bbox has {bbox.shape[0]} corners, expected 8")
        return

    # Transform to camera coordinates if needed
    if camera2world_translation is not None and world2camera_rotation is not None:
        bbox_camera = (bbox - camera2world_translation) @ world2camera_rotation
    else:
        bbox_camera = bbox

    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    # First draw shadow (slightly offset)
    shadow_offset = np.array([0.01, 0.01, 0.01])  # 1cm offset
    bbox_shadow = bbox_camera + shadow_offset

    for edge in edges:
        points = bbox_shadow[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                 color=shadow_color, linewidth=3, alpha=0.6)

    # Draw main wireframe with thick lines
    for edge in edges:
        points = bbox_camera[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                 color=main_color, linewidth=6, alpha=0.9)

    # Draw special orientation edges
    special_edges = [[6, 7], [3, 7], [4, 7]]  # Same as 2D version
    special_colors = ['white', 'yellow', 'magenta']

    for edge, color in zip(special_edges, special_colors):
        points = bbox_camera[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                 color=color, linewidth=4, alpha=1.0)

    # Draw semi-transparent faces with highlight
    faces = [
        [bbox_camera[0], bbox_camera[1], bbox_camera[2], bbox_camera[3]],  # bottom
        [bbox_camera[4], bbox_camera[5], bbox_camera[6], bbox_camera[7]],  # top
        [bbox_camera[0], bbox_camera[1], bbox_camera[5], bbox_camera[4]],  # front
        [bbox_camera[2], bbox_camera[3], bbox_camera[7], bbox_camera[6]],  # back
        [bbox_camera[0], bbox_camera[3], bbox_camera[7], bbox_camera[4]],  # left
        [bbox_camera[1], bbox_camera[2], bbox_camera[6], bbox_camera[5]]   # right
    ]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    face_collection = Poly3DCollection(faces, alpha=0.3, facecolor=main_color, edgecolor=main_color)
    ax.add_collection3d(face_collection)

    # Add text label
    if label:
        center = np.mean(bbox_camera, axis=0)
        ax.text(center[0], center[1], center[2] + 0.05, label, fontsize=12,
               color=main_color, weight='bold')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_args():
    parser = argparse.ArgumentParser(description='GAPartNet-based object reconstruction testing')
    parser.add_argument('--obj_folder_path_explore', type=str, help='Folder where exploration data is loaded from')
    parser.add_argument('--obj_folder_path_reconstruct', type=str, help='Folder where reconstruction results will be stored')
    parser.add_argument('--num_kframes', type=int, help='Number of keyframes')
    parser.add_argument('--fine_lr', type=float, help='Learning rate for fine ICP optimization')
    parser.add_argument('--save_memory', type=str2bool, help='Whether to keep frames in memory')
    parser.add_argument('--visualize', type=str2bool, default=True, help='Whether to create visualization plots')
    parser.add_argument('--ckpt_path', type=str, default='/home/wangwei/Documents/Embodied_Analogy/GAPartNet/gapartnet/ckpt/release.ckpt', help='Path to GAPartNet checkpoint')
    parser.add_argument('--use_gt_segmentation', type=str2bool, default=False, help='GT segmentation option (ignored - always disabled for reconstruction)')

    return parser.parse_args()


def extract_point_clouds_with_segmentation(obj_repr):
    """
    从 explore 阶段得到的 frames 的首尾帧分别提取物体点云
    """
    print(f"Extracting point clouds from frames...")
    from embodied_analogy.representation.basic_structure import Frame
    from embodied_analogy.representation.obj_repr import Obj_repr
    
    obj_repr: Obj_repr

    # Get frames
    first_frame: Frame = obj_repr.frames[0]
    last_frame: Frame = obj_repr.frames[-1]

    # 先 check 一下两个 Frame 的 obj_mask 都是有的
    if first_frame.obj_mask is None:
        first_frame.segment_obj( 
            obj_description=obj_repr.obj_description,
            post_process_mask=True,
            filter=True,
            visualize=False
        )
    
    if last_frame.obj_mask is None:
        last_frame.segment_obj( 
            obj_description=obj_repr.obj_description,
            post_process_mask=True,
            filter=True,
            visualize=False
        )

    # Extract point clouds using the correct method - get object point clouds
    first_pc, _ = first_frame.get_obj_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=False
    ) 
    last_pc, _ = last_frame.get_obj_pc(
        use_robot_mask=True,
        use_height_filter=True,
        world_frame=False
    )

    return first_pc, last_pc


def calculate_bbox_size_metrics(bbox):
    """
    Calculate size metrics for a bounding box

    Args:
        bbox: numpy array of shape (8, 3) representing 8 corners of the bounding box

    Returns:
        dict with keys: 'dimensions', 'volume', 'surface_area', 'diagonal'
        or None if bbox is invalid
    """
    if bbox.shape[0] != 8:
        return None

    # Calculate dimensions
    min_pt = np.min(bbox, axis=0)
    max_pt = np.max(bbox, axis=0)
    dimensions = max_pt - min_pt

    # Volume
    volume = np.prod(dimensions)

    # Surface area
    surface_area = 2 * (dimensions[0] * dimensions[1] +
                       dimensions[1] * dimensions[2] +
                       dimensions[0] * dimensions[2])

    # Diagonal length
    diagonal = np.linalg.norm(dimensions)

    return {
        'dimensions': dimensions,
        'volume': volume,
        'surface_area': surface_area,
        'diagonal': diagonal
    }


def calculate_size_similarity_with_depth_compensation(bbox1, bbox2):
    """
    Calculate size similarity between two bboxes, compensating for depth differences

    This function accounts for the perspective effect where objects appear smaller
    when they are farther from the camera. It compensates bbox2's size as if it
    were at the same depth as bbox1.

    Args:
        bbox1, bbox2: numpy arrays of shape (8, 3) representing bounding box corners

    Returns:
        float: similarity score between 0 and 1 (1 = identical size, 0 = very different)
    """
    metrics1 = calculate_bbox_size_metrics(bbox1)
    metrics2 = calculate_bbox_size_metrics(bbox2)

    if metrics1 is None or metrics2 is None:
        return 0.0  # Poor match if invalid bbox

    # Get depths (Z coordinates) of bbox centers
    center1 = np.mean(bbox1, axis=0)
    center2 = np.mean(bbox2, axis=0)
    depth1 = center1[2]
    depth2 = center2[2]

    # Compensate for perspective scaling: farther objects appear smaller
    # Assume linear perspective model: apparent_size = real_size / depth
    if depth1 > 0 and depth2 > 0:
        depth_ratio = depth1 / depth2
        # Compensate bbox2 metrics as if it were at the same depth as bbox1
        compensated_volume = metrics2['volume'] * (depth_ratio ** 3)
        compensated_surface_area = metrics2['surface_area'] * (depth_ratio ** 2)
        compensated_diagonal = metrics2['diagonal'] * depth_ratio
        compensated_dimensions = metrics2['dimensions'] * depth_ratio
    else:
        # If depth is invalid, don't compensate
        compensated_volume = metrics2['volume']
        compensated_surface_area = metrics2['surface_area']
        compensated_diagonal = metrics2['diagonal']
        compensated_dimensions = metrics2['dimensions']

    # Calculate similarity scores (1.0 = identical, 0.0 = very different)
    def similarity_score(val1, val2):
        if val1 == 0 and val2 == 0:
            return 1.0
        if val1 == 0 or val2 == 0:
            return 0.0
        ratio = min(val1, val2) / max(val1, val2)
        return ratio

    # Multiple metrics for robustness
    volume_sim = similarity_score(metrics1['volume'], compensated_volume)
    surface_sim = similarity_score(metrics1['surface_area'], compensated_surface_area)
    diagonal_sim = similarity_score(metrics1['diagonal'], compensated_diagonal)

    # Dimension-wise similarity
    dim_sim = np.mean([similarity_score(metrics1['dimensions'][i], compensated_dimensions[i])
                      for i in range(3)])

    # Weighted combination
    overall_similarity = (0.3 * volume_sim + 0.2 * surface_sim +
                         0.2 * diagonal_sim + 0.3 * dim_sim)

    return overall_similarity


def identify_moving_bounding_box_with_tracking(bboxes_first, bboxes_last, obj_repr):
    """
    Use tracking information to identify the CORRECT moving bounding box with size constraints
    bboxes_first: N, 8, 3
    bboxes_last: N, 8, 3
    """
    if len(bboxes_first) == 0 or len(bboxes_last) == 0:
        return None, None

    print(f"Identifying moving bounding box using tracking data with size constraints...")
    print(f"  {len(bboxes_first)} bboxes in first frame, {len(bboxes_last)} in last frame")

    # Calculate bbox size metrics
    first_metrics = []
    last_metrics = []

    for i, bbox in enumerate(bboxes_first):
        metrics = calculate_bbox_size_metrics(bbox)
        first_metrics.append(metrics)
        center = np.mean(bbox, axis=0)
        print(f"  First bbox {i}: center={center}, volume={metrics['volume']:.6f}, dims={metrics['dimensions']}")

    for i, bbox in enumerate(bboxes_last):
        metrics = calculate_bbox_size_metrics(bbox)
        last_metrics.append(metrics)
        center = np.mean(bbox, axis=0)
        print(f"  Last bbox {i}: center={center}, volume={metrics['volume']:.6f}, dims={metrics['dimensions']}")

    # Get tracking data - this is the GROUND TRUTH for moving parts
    if hasattr(obj_repr.frames, 'track3d_seq') and obj_repr.frames.track3d_seq is not None:
        tracks_3d = obj_repr.frames.track3d_seq
        if len(tracks_3d) > 1:
            first_tracks = tracks_3d[0]  # N x 3
            last_tracks = tracks_3d[-1]  # N x 3

            # Calculate displacement for each tracked point
            track_displacement = np.linalg.norm(last_tracks - first_tracks, axis=1)

            # Identify moving points (threshold at 75th percentile)
            movement_threshold = np.percentile(track_displacement, 75)
            moving_mask = track_displacement > movement_threshold

            print(f"  Tracking analysis:")
            print(f"    Total tracked points: {len(track_displacement)}")
            print(f"    Movement threshold (75th percentile): {movement_threshold:.4f}")
            print(f"    Moving points: {np.sum(moving_mask)}")
            print(f"    Movement range: {np.min(track_displacement):.4f} to {np.max(track_displacement):.4f}")

            if np.sum(moving_mask) > 0:
                # Get center of moving points in first and last frames
                moving_center_first = np.mean(first_tracks[moving_mask], axis=0)
                moving_center_last = np.mean(last_tracks[moving_mask], axis=0)

                print(f"  Found {np.sum(moving_mask)} moving points")
                print(f"  Moving center first: {moving_center_first}")
                print(f"  Moving center last: {moving_center_last}")
                print(f"  Movement magnitude: {np.linalg.norm(moving_center_last - moving_center_first):.4f}")

                # Find bounding box closest to moving center with size similarity constraint
                def find_best_bbox_with_size_constraint(bboxes, metrics_list, target_center, other_bboxes):
                    if len(bboxes) == 0:
                        return None, -1

                    best_score = -1
                    best_idx = -1

                    print(f"    Finding best bbox to target: {target_center}")
                    for i, bbox in enumerate(bboxes):
                        bbox_center = np.mean(bbox, axis=0)
                        position_dist = np.linalg.norm(bbox_center - target_center)

                        # Normalize position score (closer is better)
                        position_score = max(0, 1.0 - position_dist / 0.5)  # 50cm normalization

                        # Calculate best size similarity with any bbox in the other frame
                        max_size_sim = 0
                        for other_bbox in other_bboxes:
                            size_sim = calculate_size_similarity_with_depth_compensation(bbox, other_bbox)
                            max_size_sim = max(max_size_sim, size_sim)

                        # Combined score: position + size similarity
                        # Weight can be adjusted: higher weight on size = more conservative matching
                        combined_score = 0.6 * position_score + 0.4 * max_size_sim

                        print(f"      Bbox {i}: pos_dist={position_dist:.4f}, pos_score={position_score:.3f}, "
                              f"max_size_sim={max_size_sim:.3f}, combined={combined_score:.3f}")

                        if combined_score > best_score:
                            best_score = combined_score
                            best_idx = i

                    return bboxes[best_idx] if best_idx >= 0 else None, best_idx

                # Find best bboxes with size constraints
                moving_bbox_first, first_idx = find_best_bbox_with_size_constraint(
                    bboxes_first, first_metrics, moving_center_first, bboxes_last)
                moving_bbox_last, last_idx = find_best_bbox_with_size_constraint(
                    bboxes_last, last_metrics, moving_center_last, bboxes_first)

                if moving_bbox_first is not None and moving_bbox_last is not None:
                    # Validate the size similarity between selected bboxes
                    size_similarity = calculate_size_similarity_with_depth_compensation(
                        moving_bbox_first, moving_bbox_last)

                    first_bbox_center = np.mean(moving_bbox_first, axis=0)
                    last_bbox_center = np.mean(moving_bbox_last, axis=0)

                    dist_first = np.linalg.norm(first_bbox_center - moving_center_first)
                    dist_last = np.linalg.norm(last_bbox_center - moving_center_last)

                    print(f"  Selected bbox {first_idx} in first frame (dist: {dist_first:.4f})")
                    print(f"  Selected bbox {last_idx} in last frame (dist: {dist_last:.4f})")
                    print(f"  Size similarity between selected bboxes: {size_similarity:.3f}")

                    # Stricter validation: both distance and size similarity should be reasonable
                    if (dist_first < 0.5 and dist_last < 0.5 and size_similarity > 0.3):
                        print(f"  Tracking-based selection successful (size similarity: {size_similarity:.3f})")
                        return moving_bbox_first, moving_bbox_last
                    else:
                        print(f"  Validation failed - dist1: {dist_first:.3f}, dist2: {dist_last:.3f}, size_sim: {size_similarity:.3f}")

    # Fallback to enhanced bipartite matching if tracking fails
    print(f"  Falling back to enhanced bipartite matching...")
    return identify_moving_bounding_box_bipartite(bboxes_first, bboxes_last, obj_repr)


def draw_bbox_wireframe_cv2(img, bbox, camera2world_translation, world2camera_rotation, K, color=(255, 0, 255)):
    """Draw bounding box using the EXACT method from GAPartNet's draw_bbox_from_world"""
    if len(bbox) == 0:
        return img

    bbox = np.array(bbox)

    # Transform to camera coordinates
    if camera2world_translation is not None:
        assert world2camera_rotation is not None
        bbox_camera = (bbox - camera2world_translation) @ world2camera_rotation
    else:
        bbox_camera = bbox

    # Project to image coordinates - FIXED GAPartNet method
    point2image = np.concatenate([
        (np.around(bbox_camera[:, 0] * K[0][0] / bbox_camera[:, 2] + K[0][2])).astype(dtype=int).reshape(-1, 1),
        (np.around(bbox_camera[:, 1] * K[1][1] / bbox_camera[:, 2] + K[1][2])).astype(dtype=int).reshape(-1, 1)
    ], axis=1)

    # Draw edges - EXACT pattern from GAPartNet
    cl = color
    thickness = 2

    # Standard edges (thickness=2)
    cv2.line(img, tuple(point2image[0]), tuple(point2image[1]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[0]), tuple(point2image[3]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[0]), tuple(point2image[4]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[1]), tuple(point2image[2]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[1]), tuple(point2image[5]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[2]), tuple(point2image[3]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[2]), tuple(point2image[6]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[3]), tuple(point2image[7]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[4]), tuple(point2image[5]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[4]), tuple(point2image[7]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)
    cv2.line(img, tuple(point2image[5]), tuple(point2image[6]), color=(int(cl[0]), int(cl[1]), int(cl[2])), thickness=thickness)

    # Special colored edges (thickness=3) to show orientation
    cv2.line(img, tuple(point2image[6]), tuple(point2image[7]), color=(0, 0, 255), thickness=3)  # Red
    cv2.line(img, tuple(point2image[3]), tuple(point2image[7]), color=(255, 0, 0), thickness=3)  # Blue
    cv2.line(img, tuple(point2image[4]), tuple(point2image[7]), color=(0, 255, 0), thickness=3)  # Green

    return img


def draw_3d_bbox_wireframe_fixed(ax, bbox, camera2world_translation, world2camera_rotation, color='red', label=None, alpha=0.7, linewidth=2):
    """Draw 3D bounding box wireframe with CORRECT coordinate transformation"""
    if len(bbox) == 0:
        print(f"Warning: bbox is empty")
        return

    bbox = np.array(bbox)
    if bbox.shape[0] != 8:
        print(f"Warning: bbox has {bbox.shape[0]} corners, expected 8")
        return

    # Transform to camera coordinates if needed
    if camera2world_translation is not None and world2camera_rotation is not None:
        bbox_camera = (bbox - camera2world_translation) @ world2camera_rotation
    else:
        bbox_camera = bbox

    # Define the 12 edges of a cube - CORRECTED to match our corner ordering
    # Our corner ordering after fix_gapartnet_bbox_corner_ordering:
    # [0: -x,-y,-z], [1: +x,-y,-z], [2: +x,+y,-z], [3: -x,+y,-z],  # bottom face
    # [4: -x,-y,+z], [5: +x,-y,+z], [6: +x,+y,+z], [7: -x,+y,+z]]  # top face
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face (corrected)
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face (corrected)
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]

    # Draw wireframe edges
    for edge in edges:
        points = bbox_camera[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                 color=color, linewidth=linewidth, alpha=alpha)

    # Draw transparent surfaces - CORRECTED to match our corner ordering
    faces = [
        [bbox_camera[0], bbox_camera[1], bbox_camera[2], bbox_camera[3]],  # bottom (corrected)
        [bbox_camera[4], bbox_camera[5], bbox_camera[6], bbox_camera[7]],  # top (corrected)
        [bbox_camera[0], bbox_camera[1], bbox_camera[5], bbox_camera[4]],  # front
        [bbox_camera[2], bbox_camera[3], bbox_camera[7], bbox_camera[6]],  # back
        [bbox_camera[0], bbox_camera[3], bbox_camera[7], bbox_camera[4]],  # left (corrected)
        [bbox_camera[1], bbox_camera[2], bbox_camera[6], bbox_camera[5]]   # right (corrected)
    ]

    face_collection = Poly3DCollection(faces, alpha=0.1, facecolor=color, edgecolor=color)
    ax.add_collection3d(face_collection)

    if label:
        center = np.mean(bbox_camera, axis=0)
        ax.text(center[0], center[1], center[2], label, fontsize=8, color=color)
        
        
def visualize_rgbd_with_bboxes(obj_repr, bboxes_first, bboxes_last,
                              moving_bbox_first, moving_bbox_last,
                              joint_estimation, save_path):
    """
    Create comprehensive RGBD visualization with CORRECTED coordinate transformations
    using the exact same methods as demo_style_fixed.py
    """
    print("Creating RGBD visualization with CORRECTED coordinate transformations...")

    # Get first and last frames
    first_frame = obj_repr.frames[0]
    last_frame = obj_repr.frames[-1]

    # Create figure with 2x3 subplots: RGB + Depth + 3D view for first/last frames
    fig = plt.figure(figsize=(24, 16))

    # Define colors for different bounding boxes
    bbox_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255), (128, 128, 128)]
    moving_bbox_color = (0, 255, 0)  # Bright lime green for moving bbox
    moving_bbox_shadow_color = (0, 180, 0)  # Darker green for shadow effect

    # Convert colors to matplotlib format for 3D plotting
    bbox_colors_mpl = ['red', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'gray']
    moving_bbox_color_mpl = 'lime'
    moving_bbox_shadow_color_mpl = 'darkgreen'

    # Process frames
    for frame_idx, (frame, bboxes, moving_bbox, title_prefix) in enumerate([
        (first_frame, bboxes_first, moving_bbox_first, "First Frame"),
        (last_frame, bboxes_last, moving_bbox_last, "Last Frame")
    ]):

        # Get RGB and depth images
        rgb_img = frame.rgb.copy()
        depth_img = frame.depth
        intrinsics = frame.K  # Camera intrinsics

        # IMPORTANT: GAPartNet bboxes are already in camera coordinates
        # No coordinate transformation needed for the predicted bboxes
        camera2world_translation = None  # Don't transform - already in camera coords
        world2camera_rotation = None     # Don't transform - already in camera coords

        # RGB with bounding boxes overlay using CORRECT GAPartNet method
        rgb_with_bbox = rgb_img.copy()

        # First draw regular bboxes (avoid drawing moving bbox twice)
        for i, bbox in enumerate(bboxes):
            if bbox is not None and len(bbox) > 0:
                bbox_array = np.array(bbox)
                # Skip if this is the moving bbox (we'll draw it specially)
                if moving_bbox is not None and np.array_equal(bbox_array, np.array(moving_bbox)):
                    continue
                if bbox_array.shape[0] == 8:  # Valid 8-corner bbox
                    color = bbox_colors[i % len(bbox_colors)]
                    # Use the EXACT GAPartNet method - no coordinate transformation needed
                    rgb_with_bbox = draw_bbox_wireframe_cv2(
                        rgb_with_bbox, bbox_array,
                        camera2world_translation, world2camera_rotation,
                        intrinsics, color=color
                    )

        # Draw moving bounding box with special effects
        if moving_bbox is not None and len(moving_bbox) > 0:
            moving_bbox_array = np.array(moving_bbox)
            if moving_bbox_array.shape[0] == 8:
                rgb_with_bbox = draw_moving_bbox_with_effects(
                    rgb_with_bbox, moving_bbox_array,
                    camera2world_translation, world2camera_rotation,
                    intrinsics, moving_bbox_color, moving_bbox_shadow_color
                )

        # RGB subplot
        ax_rgb = fig.add_subplot(2, 3, frame_idx * 3 + 1)
        ax_rgb.imshow(rgb_with_bbox)
        ax_rgb.set_title(f"{title_prefix} - RGB with GAPartNet Bboxes")
        ax_rgb.set_xlabel("X (pixels)")
        ax_rgb.set_ylabel("Y (pixels)")
        ax_rgb.axis('off')

        # Depth with bounding boxes overlay
        depth_with_bbox = np.stack([depth_img, depth_img, depth_img], axis=-1)
        depth_with_bbox = (np.clip(depth_with_bbox, 0, 2.0) / 2.0 * 255).astype(np.uint8)

        # First draw regular bboxes (avoid drawing moving bbox twice)
        for i, bbox in enumerate(bboxes):
            if bbox is not None and len(bbox) > 0:
                bbox_array = np.array(bbox)
                # Skip if this is the moving bbox (we'll draw it specially)
                if moving_bbox is not None and np.array_equal(bbox_array, np.array(moving_bbox)):
                    continue
                if bbox_array.shape[0] == 8:
                    color = bbox_colors[i % len(bbox_colors)]
                    depth_with_bbox = draw_bbox_wireframe_cv2(
                        depth_with_bbox, bbox_array,
                        camera2world_translation, world2camera_rotation,
                        intrinsics, color=color
                    )

        # Draw moving bounding box with special effects on depth
        if moving_bbox is not None and len(moving_bbox) > 0:
            moving_bbox_array = np.array(moving_bbox)
            if moving_bbox_array.shape[0] == 8:
                depth_with_bbox = draw_moving_bbox_with_effects(
                    depth_with_bbox, moving_bbox_array,
                    camera2world_translation, world2camera_rotation,
                    intrinsics, moving_bbox_color, moving_bbox_shadow_color
                )

        # Depth subplot
        ax_depth = fig.add_subplot(2, 3, frame_idx * 3 + 2)
        ax_depth.imshow(depth_with_bbox)
        ax_depth.set_title(f"{title_prefix} - Depth with GAPartNet Bboxes")
        ax_depth.set_xlabel("X (pixels)")
        ax_depth.set_ylabel("Y (pixels)")
        ax_depth.axis('off')

        # 3D visualization using CORRECT coordinate transformation
        ax_3d = fig.add_subplot(2, 3, frame_idx * 3 + 3, projection='3d')

        # Get point cloud for context
        pc, _ = frame.get_obj_pc(use_robot_mask=True, world_frame=False)
        if len(pc) > 5000:  # Downsample for visualization
            indices = np.random.choice(len(pc), 5000, replace=False)
            pc = pc[indices]

        # Plot point cloud
        ax_3d.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='gray', alpha=0.3, s=0.5)

        # Draw all 3D bounding boxes using CORRECT method
        for i, bbox in enumerate(bboxes):
            if bbox is not None and len(bbox) > 0:
                bbox_array = np.array(bbox)
                # Skip if this is the moving bbox (we'll draw it specially)
                if moving_bbox is not None and np.array_equal(bbox_array, np.array(moving_bbox)):
                    continue
                if bbox_array.shape[0] == 8:
                    color = bbox_colors_mpl[i % len(bbox_colors_mpl)]
                    # Use the CORRECT 3D drawing method - no coordinate transformation needed
                    draw_3d_bbox_wireframe_fixed(
                        ax_3d, bbox_array,
                        camera2world_translation, world2camera_rotation,
                        color=color, label=f"Part {i}", linewidth=2
                    )

        # Draw moving bounding box with special 3D effects
        if moving_bbox is not None and len(moving_bbox) > 0:
            moving_bbox_array = np.array(moving_bbox)
            if moving_bbox_array.shape[0] == 8:
                draw_3d_moving_bbox_with_effects(
                    ax_3d, moving_bbox_array,
                    camera2world_translation, world2camera_rotation,
                    moving_bbox_color_mpl, moving_bbox_shadow_color_mpl, "MOVING PART"
                )

        ax_3d.set_title(f"{title_prefix} - 3D Point Cloud + GAPartNet Bboxes")
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")

        # Set reasonable axis limits based on actual point cloud
        if len(pc) > 0:
            x_range = [np.min(pc[:, 0]) - 0.1, np.max(pc[:, 0]) + 0.1]
            y_range = [np.min(pc[:, 1]) - 0.1, np.max(pc[:, 1]) + 0.1]
            z_range = [np.min(pc[:, 2]) - 0.1, np.max(pc[:, 2]) + 0.1]
            ax_3d.set_xlim(x_range)
            ax_3d.set_ylim(y_range)
            ax_3d.set_zlim(z_range)

    # Add joint estimation info
    if joint_estimation:
        joint_info = f"""Joint Estimation Results (CORRECTED):
Type: {joint_estimation.get('joint_type', 'unknown')}
Direction: {joint_estimation.get('joint_dir', 'N/A')}
Start: {joint_estimation.get('joint_start', 'N/A')}
Movement: {joint_estimation.get('movement_magnitude', 'N/A'):.4f}"""

        fig.text(0.02, 0.02, joint_info, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved CORRECTED RGBD+bbox visualization to {save_path}")


def gapartnet_reconstruct_fixed(obj_repr, gapartnet_model, visualize=False):
    """COMPLETELY CORRECTED GAPartNet reconstruction (always uses model segmentation, never GT)"""
    print("Running COMPLETELY CORRECTED GAPartNet reconstruction...")

    # Get ground truth for guidance
    gt_joint_type = obj_repr.gt_joint_dict.get("joint_type", "prismatic")
    print(f"Ground truth joint type: {gt_joint_type}")
    print(f"Using model segmentation (GAPartNet model.py lines 497-499)")
    print(f"Note: GT segmentation is disabled for reconstruction pipeline")

    # Extract point clouds (no GT segmentation for reconstruction)
    first_pc, last_pc = extract_point_clouds_with_segmentation(obj_repr)

    # Run CORRECTED GAPartNet inference using model segmentation only
    bboxes_first, bbox_scores_first, bbox_sem_labels_first = run_gapartnet_inference_fixed(
        pc_xyz=first_pc, 
        gapartnet_model=gapartnet_model, 
        pc_seg=None, 
        use_gt_segmentation=False
    )
    bboxes_last, bbox_scores_last, bbox_sem_labels_last = run_gapartnet_inference_fixed(
        pc_xyz=last_pc, 
        gapartnet_model=gapartnet_model, 
        pc_seg=None, 
        use_gt_segmentation=False
    )

    # Create frame stats for compatibility
    frame_stats = [
        {'bboxes': len(bboxes_first), 'scores': bbox_scores_first, 'sem_labels': bbox_sem_labels_first},
        {'bboxes': len(bboxes_last), 'scores': bbox_scores_last, 'sem_labels': bbox_sem_labels_last}
    ]

    # Use tracking-based moving part identification
    moving_bbox_first, moving_bbox_last = identify_moving_bounding_box_with_tracking(bboxes_first, bboxes_last, obj_repr)

    # Estimate joint parameters
    joint_estimation = estimate_joint_from_bboxes(
        moving_bbox_first=moving_bbox_first,
        moving_bbox_last=moving_bbox_last,
        joint_type=gt_joint_type
    )

    print(f"  Joint dir: {joint_estimation['joint_dir']}")
    print(f"  Joint start: {joint_estimation['joint_start']}")

    # Create RGBD + bounding box visualization
    if visualize:
        save_folder = "/home/zby/Programs/Embodied_Analogy/assets/unit_test/gapartnet"
        rgbd_bbox_path = os.path.join(save_folder, 'gapartnet_rgbd_bboxes_fixed.png')
        visualize_rgbd_with_bboxes(
            obj_repr, bboxes_first, bboxes_last,
            moving_bbox_first, moving_bbox_last,
            joint_estimation, rgbd_bbox_path
        )

    return joint_estimation


    # Generate joint states
    num_frames = len(obj_repr.frames)
    max_joint_state = joint_estimation['movement_magnitude']
    joint_states = np.linspace(0, max_joint_state, num_frames)

    # Create joint dictionaries
    joint_dict_camera = {
        "joint_type": gt_joint_type,
        "joint_dir": joint_estimation['joint_dir'],
        "joint_start": joint_estimation['joint_start'],
        "joint_states": joint_states
    }

    # Set results - CRITICAL: Set coarse joint dict first before calling initialize_kframes
    obj_repr.coarse_joint_dict = joint_dict_camera.copy()
    obj_repr.fine_joint_dict = joint_dict_camera.copy()
    obj_repr.frames.write_joint_states(joint_states)

    # ✅ PROPER FIX: Call the standard reconstruction pipeline to set up all data structures
    # This ensures frames have proper dynamic_mask, moving_mask, static_mask, etc.
    print("Setting up proper frame data structures using coarse joint estimation...")

    # The frames should already have proper segmentation from the exploration phase
    # We just need to ensure the moving/static masks are properly set up
    if not hasattr(obj_repr.frames, 'moving_mask') or obj_repr.frames.moving_mask is None:
        print("Warning: No moving_mask found in frames, this may cause issues")

    if not hasattr(obj_repr.frames, 'static_mask') or obj_repr.frames.static_mask is None:
        print("Warning: No static_mask found in frames, this may cause issues")

    # Initialize keyframes using the standard method - this should work now that coarse_joint_dict is set
    num_kframes = recon_cfg.get('num_kframes', 5)
    obj_repr.initialize_kframes(num_kframes=num_kframes, save_memory=recon_cfg.get('save_memory', True))

    # ✅ CRITICAL MISSING STEP: Set up dynamic masks using the standard pipeline
    # This is what the original reconstruct() method does after initialize_kframes()
    print("Setting up dynamic masks using classify_dynamics...")
    obj_repr.kframes.segment_obj(obj_description=recon_cfg.get("obj_description", "drawer"), visualize=False)
    obj_repr.kframes.classify_dynamics(
        filter=True,
        joint_dict=obj_repr.coarse_joint_dict,
        visualize=False
    )

    # Compute error if ground truth available
    result = None
    if obj_repr.gt_joint_dict["joint_type"] is not None:
        result = obj_repr.compute_joint_error()
        result['gapartnet_joint_estimation'] = joint_estimation
        result['reconstruction_method'] = 'gapartnet_fully_corrected'
        result['has_valid_explore'] = True
        result['has_valid_reconstruct'] = True
        result['frame_statistics'] = frame_stats

        print("\nFULLY CORRECTED GAPartNet Reconstruction Result:")
        for k, v in result.items():
            if k not in ['gapartnet_joint_estimation', 'frame_statistics']:
                print(f"  {k}: {v}")
    else:
        result = {
            'gapartnet_joint_estimation': joint_estimation,
            'reconstruction_method': 'gapartnet_fully_corrected',
            'has_valid_explore': True,
            'has_valid_reconstruct': True,
            'frame_statistics': frame_stats,
            'coarse_w': obj_repr.get_joint_param(resolution="coarse", frame="world") if obj_repr.Tw2c is not None else None,
            'fine_w': obj_repr.get_joint_param(resolution="fine", frame="world") if obj_repr.Tw2c is not None else None,
            'gt_w': None,
            'coarse_loss': {'type_err': 0, 'angle_err': 0, 'pos_err': 0},
            'fine_loss': {'type_err': 0, 'angle_err': 0, 'pos_err': 0}
        }

    return result


def compare_with_original_reconstruct(original_explore_folder, recon_cfg):
    """Compare GAPartNet results with the original reconstruct method using original non-clean data"""
    print("\n=== Running Original Reconstruct Method for Comparison ===")

    # Load original (non-clean) data for comparison
    print(f"Loading original non-clean data from: {original_explore_folder}")
    original_obj_repr = Obj_repr.load(os.path.join(original_explore_folder, "obj_repr.npy"))

    # Run the original reconstruct method on non-clean data
    original_result = original_obj_repr.reconstruct(
        num_kframes=recon_cfg.get('num_kframes', 5),
        obj_description=recon_cfg.get('obj_description', 'drawer'),
        fine_lr=recon_cfg.get('fine_lr', 1e-3),
        evaluate=True,
        save_memory=recon_cfg.get('save_memory', True),
        visualize=False
    )

    print("Original reconstruct method completed successfully")
    if original_result:
        print("Original reconstruction results:")
        for k, v in original_result.items():
            if k not in ['coarse_w', 'fine_w', 'gt_w']:
                print(f"  {k}: {v}")

    return original_result


def convert_numpy_to_json(obj):
    """Convert numpy types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def identify_moving_bounding_box_bipartite(bboxes_first, bboxes_last, obj_repr):
    """Use enhanced bipartite graph matching with size similarity constraints"""
    if len(bboxes_first) == 0 or len(bboxes_last) == 0:
        return None, None

    print(f"Using enhanced bipartite matching for {len(bboxes_first)} bboxes in first frame, {len(bboxes_last)} in last frame")

    # Calculate centers and size metrics for all bounding boxes
    centers_first = np.array([np.mean(bbox, axis=0) for bbox in bboxes_first])
    centers_last = np.array([np.mean(bbox, axis=0) for bbox in bboxes_last])

    print(f"Calculating size metrics for all bounding boxes...")
    first_metrics = [calculate_bbox_size_metrics(bbox) for bbox in bboxes_first]
    last_metrics = [calculate_bbox_size_metrics(bbox) for bbox in bboxes_last]

    # Create enhanced cost matrix: weighted combination of distance and size dissimilarity
    num_first = len(centers_first)
    num_last = len(centers_last)
    distance_matrix = np.zeros((num_first, num_last))
    size_similarity_matrix = np.zeros((num_first, num_last))
    cost_matrix = np.zeros((num_first, num_last))

    for i in range(num_first):
        for j in range(num_last):
            # Position distance
            distance = np.linalg.norm(centers_first[i] - centers_last[j])
            distance_matrix[i, j] = distance

            # Size similarity with depth compensation
            size_sim = calculate_size_similarity_with_depth_compensation(bboxes_first[i], bboxes_last[j])
            size_similarity_matrix[i, j] = size_sim

            # Combined cost: normalize distance and use (1 - size_similarity) as cost
            # Lower cost = better match
            normalized_distance = min(distance / 0.5, 1.0)  # Normalize to 0-1 range, 50cm max
            size_dissimilarity = 1.0 - size_sim

            # Weighted combination: emphasize size similarity for better matching
            cost_matrix[i, j] = 0.4 * normalized_distance + 0.6 * size_dissimilarity

    print(f"Distance matrix (position distances):")
    for i in range(num_first):
        row_str = "  " + " ".join([f"{distance_matrix[i, j]:.3f}" for j in range(num_last)])
        print(f"  First bbox {i}: {row_str}")

    print(f"Size similarity matrix:")
    for i in range(num_first):
        row_str = "  " + " ".join([f"{size_similarity_matrix[i, j]:.3f}" for j in range(num_last)])
        print(f"  First bbox {i}: {row_str}")

    print(f"Combined cost matrix (lower = better match):")
    for i in range(num_first):
        row_str = "  " + " ".join([f"{cost_matrix[i, j]:.3f}" for j in range(num_last)])
        print(f"  First bbox {i}: {row_str}")

    # Use Hungarian algorithm to find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Analyze the assignments to find valid matches and identify the moving part
    movements = []
    assignments = []
    valid_assignments = []

    for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
        distance = distance_matrix[row_idx, col_idx]
        size_sim = size_similarity_matrix[row_idx, col_idx]
        cost = cost_matrix[row_idx, col_idx]

        assignments.append((row_idx, col_idx, distance, size_sim, cost))
        movements.append(distance)

        print(f"  Assignment {i}: First bbox {row_idx} -> Last bbox {col_idx}")
        print(f"    Distance: {distance:.4f}, Size similarity: {size_sim:.3f}, Cost: {cost:.3f}")

        # Only consider assignments with reasonable size similarity as valid
        if size_sim > 0.2:  # At least 20% size similarity
            valid_assignments.append((row_idx, col_idx, distance, size_sim, cost))
            print(f"    Valid assignment (good size similarity)")
        else:
            print(f"    Invalid assignment (poor size similarity: {size_sim:.3f})")

    if len(valid_assignments) == 0:
        print(f"  No valid assignments found with sufficient size similarity")
        return None, None

    movements = np.array(movements)

    # Among valid assignments, find the one with maximum movement (indicating moving part)
    valid_movements = [assign[2] for assign in valid_assignments]  # distance is index 2
    max_movement_idx = np.argmax(valid_movements)
    best_assignment = valid_assignments[max_movement_idx]

    moving_first_idx = best_assignment[0]
    moving_last_idx = best_assignment[1]
    max_movement = best_assignment[2]
    best_size_sim = best_assignment[3]

    print(f"\nEnhanced bipartite matching results:")
    print(f"  Total assignments: {len(assignments)}")
    print(f"  Valid assignments (size similarity > 0.2): {len(valid_assignments)}")
    print(f"  Movement range: {np.min(movements):.4f} to {np.max(movements):.4f}")
    print(f"  Mean movement: {np.mean(movements):.4f}")
    print(f"  Selected moving pair: First bbox {moving_first_idx} -> Last bbox {moving_last_idx}")
    print(f"  Moving pair movement: {max_movement:.4f}")
    print(f"  Moving pair size similarity: {best_size_sim:.3f}")

    # Additional validation with tracking if available
    if hasattr(obj_repr.frames, 'track3d_seq') and obj_repr.frames.track3d_seq is not None:
        tracks_3d = obj_repr.frames.track3d_seq
        if len(tracks_3d) > 1:
            first_tracks = tracks_3d[0]
            last_tracks = tracks_3d[-1]
            track_displacement = np.linalg.norm(last_tracks - first_tracks, axis=1)

            movement_threshold = np.percentile(track_displacement, 75)
            moving_mask = track_displacement > movement_threshold

            if np.sum(moving_mask) > 0:
                moving_center_first = np.mean(first_tracks[moving_mask], axis=0)
                moving_center_last = np.mean(last_tracks[moving_mask], axis=0)

                # Check if our enhanced bipartite matching result aligns with tracking
                selected_center_first = centers_first[moving_first_idx]
                selected_center_last = centers_last[moving_last_idx]

                dist_to_track_first = np.linalg.norm(selected_center_first - moving_center_first)
                dist_to_track_last = np.linalg.norm(selected_center_last - moving_center_last)

                print(f"  Validation with tracking:")
                print(f"    Distance to tracking center (first): {dist_to_track_first:.4f}")
                print(f"    Distance to tracking center (last): {dist_to_track_last:.4f}")

                if dist_to_track_first > 0.3 or dist_to_track_last > 0.3:
                    print(f"    Warning: Moderate distance to tracking centers")
                else:
                    print(f"    Good alignment with tracking data")

    # Final validation: ensure we have a reasonable match
    if best_size_sim < 0.2:
        print(f"  Final validation failed: size similarity too low ({best_size_sim:.3f})")
        return None, None

    print(f"  Enhanced bipartite matching successful")
    return bboxes_first[moving_first_idx], bboxes_last[moving_last_idx]


def estimate_joint_from_bboxes(moving_bbox_first, moving_bbox_last, joint_type):
    """
    Estimate joint parameters directly from GAPartNet bounding boxes
    
    返回一个 dict, 里面包含了 joint_dir 和 joint_start
    """
    if moving_bbox_first is None or moving_bbox_last is None:
        raise Exception("One of Input to estimate_joint_from_bboxes() is None.")

    print(f"Estimating {joint_type} joint from moving bounding boxes...")

    if joint_type == "prismatic":
        return estimate_prismatic_joint(moving_bbox_first, moving_bbox_last)
    elif joint_type == "revolute":
        return estimate_revolute_joint(moving_bbox_first, moving_bbox_last)
    

def estimate_prismatic_joint(bbox_first, bbox_last):
    """Estimate prismatic joint from bounding box movement"""
    # Calculate center movement
    center_first = np.mean(bbox_first, axis=0)
    center_last = np.mean(bbox_last, axis=0)
    translation = center_last - center_first
    translation_magnitude = np.linalg.norm(translation)

    if translation_magnitude < 1e-6:
        print("Warning: Very small translation detected")
        return None

    # Joint direction is the translation direction
    joint_dir = translation / translation_magnitude
    joint_start = center_first

    print(f"Estimated prismatic joint:")
    print(f"  Direction: {joint_dir}")
    print(f"  Movement magnitude: {translation_magnitude:.4f}")
    print(f"  Center first: {center_first}")
    print(f"  Center last: {center_last}")
    print(f"  Raw translation: {translation}")

    return {
        'joint_dir': joint_dir,
        'joint_start': joint_start,
        'movement_magnitude': translation_magnitude
    }


def estimate_revolute_joint(bbox_first, bbox_last):
    """Estimate revolute joint from bounding box transformation"""
    print(f"Estimating revolute joint from bounding box positions...")
    print(f"  First bbox shape: {bbox_first.shape}")
    print(f"  Last bbox shape: {bbox_last.shape}")

    # Handle case where bounding boxes have different number of points
    min_points = min(len(bbox_first), len(bbox_last))
    if min_points < 3:
        print("Warning: Not enough points for rotation estimation")
        return None

    # Sample points if too many (for computational efficiency)
    if min_points > 100:
        indices = np.random.choice(min_points, 100, replace=False)
        points_first = bbox_first[indices]
        points_last = bbox_last[indices]
    else:
        points_first = bbox_first[:min_points]
        points_last = bbox_last[:min_points]

    print(f"  Using {len(points_first)} points for rotation estimation")

    # Find the rotation axis and center using absolute positions
    # Try different potential rotation centers and find the best one
    center_first = np.mean(points_first, axis=0)
    center_last = np.mean(points_last, axis=0)

    # Try multiple potential rotation centers
    potential_centers = [
        center_first,
        center_last,
        (center_first + center_last) / 2,
        # Add some points along the line between centers
        center_first + 0.3 * (center_last - center_first),
        center_first + 0.7 * (center_last - center_first),
    ]

    best_rotation_center = None
    best_rotation_axis = None
    best_angle = None
    min_error = float('inf')

    for potential_center in potential_centers:
        # Center points around this potential rotation center
        first_centered = points_first - potential_center
        last_centered = points_last - potential_center

        # Calculate cross-covariance matrix for rotation estimation
        H = first_centered.T @ last_centered

        # SVD to find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Extract rotation axis and angle
        eigenvalues, eigenvectors = np.linalg.eig(R)
        real_eigenvalues = np.real(eigenvalues)
        closest_to_one = np.argmin(np.abs(real_eigenvalues - 1))
        rotation_axis = np.real(eigenvectors[:, closest_to_one])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Calculate rotation angle
        cos_angle = (np.trace(R) - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        rotation_angle = np.arccos(cos_angle)

        # Calculate error: how well does this rotation explain the transformation
        rotated_first = (R @ first_centered.T).T + potential_center
        error = np.mean(np.linalg.norm(rotated_first - points_last, axis=1))

        if error < min_error:
            min_error = error
            best_rotation_center = potential_center
            best_rotation_axis = rotation_axis
            best_angle = rotation_angle

    if best_rotation_center is None:
        print("Warning: Could not find valid rotation")
        return None

    print(f"Estimated revolute joint:")
    print(f"  Axis: {best_rotation_axis}")
    print(f"  Angle: {best_angle:.4f} rad ({np.degrees(best_angle):.1f}°)")
    print(f"  Center: {best_rotation_center}")
    print(f"  Rotation error: {min_error:.4f}")

    return {
        'joint_dir': best_rotation_axis,
        'joint_start': best_rotation_center,
        'movement_magnitude': best_angle
    }


def main():
    """Main process: GAPartNet reconstruction using pure bounding box predictions"""
    args = read_args()

    print("=== GAPartNet Pure Bounding Box Reconstruction ===")

    # Load exploration data
    explore_folder = args.obj_folder_path_explore
    with open(os.path.join(explore_folder, "cfg.json"), 'r', encoding='utf-8') as file:
        explore_cfg = json.load(file)

    with open(os.path.join(explore_folder, "result.pkl"), 'rb') as f:
        explore_result = pickle.load(f)

    if not explore_result["has_valid_explore"]:
        print("No valid explore data, exiting...")
        return

    # Load object representation (clean data for GAPartNet)
    obj_repr = Obj_repr.load(os.path.join(explore_folder, "obj_repr.npy"))
    print(f"Loaded obj_repr with {len(obj_repr.frames)} frames")

    # Determine original non-clean folder for comparison
    original_explore_folder = explore_result.get('original_folder', explore_folder)
    if original_explore_folder != explore_folder:
        print(f"Using clean data for GAPartNet: {explore_folder}")
        print(f"Will use original data for comparison: {original_explore_folder}")
    else:
        print(f"Using same data for both methods: {explore_folder}")

    # Debug information about the loaded data
    print(f"Object representation debug info:")
    print(f"  Has tracking data: {hasattr(obj_repr.frames, 'track3d_seq') and obj_repr.frames.track3d_seq is not None}")
    if hasattr(obj_repr.frames, 'track3d_seq') and obj_repr.frames.track3d_seq is not None:
        print(f"  Track3d shape: {obj_repr.frames.track3d_seq.shape}")
    print(f"  Has moving mask: {hasattr(obj_repr.frames, 'moving_mask') and obj_repr.frames.moving_mask is not None}")
    if hasattr(obj_repr.frames, 'moving_mask') and obj_repr.frames.moving_mask is not None:
        print(f"  Moving points: {np.sum(obj_repr.frames.moving_mask)}/{len(obj_repr.frames.moving_mask)}")
    print(f"  Camera intrinsics available: {obj_repr.K is not None}")
    print(f"  World-to-camera transform available: {obj_repr.Tw2c is not None}")
    print(f"  Ground truth joint type: {obj_repr.gt_joint_dict.get('joint_type', 'Unknown')}")

    # Update config
    recon_cfg = explore_cfg.copy()
    if args.obj_folder_path_reconstruct:
        recon_cfg['obj_folder_path_reconstruct'] = args.obj_folder_path_reconstruct
    if args.num_kframes:
        recon_cfg['num_kframes'] = args.num_kframes
    if args.fine_lr:
        recon_cfg['fine_lr'] = args.fine_lr
    if args.save_memory is not None:
        recon_cfg['save_memory'] = args.save_memory
    if args.visualize is not None:
        recon_cfg['visualize'] = args.visualize
    # Add configuration options (GT segmentation always disabled for reconstruction)
    recon_cfg['use_gt_segmentation'] = False  # Always False for reconstruction pipeline

    recon_cfg['explore_result'] = explore_result

    # Load GAPartNet model
    ckpt_path = Path(args.ckpt_path)
    gapartnet_model = load_gapartnet_model(ckpt_path)
    if gapartnet_model is None:
        print("Failed to load GAPartNet model, exiting...")
        return

    # Run GAPartNet reconstruction using pure bounding box predictions
    print("\n=== Running GAPartNet Pure Bounding Box Method ===")
    gapartnet_result = gapartnet_reconstruct_fixed(obj_repr, gapartnet_model, recon_cfg)

    if gapartnet_result is None:
        print("GAPartNet reconstruction failed")
        return

    # Run original reconstruct method for comparison (using original non-clean data)
    original_result = compare_with_original_reconstruct(original_explore_folder, recon_cfg)

    # Compare results
    if gapartnet_result and original_result:
        print("\n=== Comparison: GAPartNet vs Original Method ===")

        def extract_errors(result):
            if result and 'coarse_loss' in result:
                return {
                    'type_err': result['coarse_loss']['type_err'],
                    'angle_err': result['coarse_loss']['angle_err'],
                    'pos_err': result['coarse_loss']['pos_err']
                }
            return {'type_err': float('inf'), 'angle_err': float('inf'), 'pos_err': float('inf')}

        gapartnet_errors = extract_errors(gapartnet_result)
        original_errors = extract_errors(original_result)

        print(f"GAPartNet errors:    Type={gapartnet_errors['type_err']}, Angle={gapartnet_errors['angle_err']:.4f} rad ({np.degrees(gapartnet_errors['angle_err']):.1f}°), Pos={gapartnet_errors['pos_err']:.4f}")
        print(f"Original errors:     Type={original_errors['type_err']}, Angle={original_errors['angle_err']:.4f} rad ({np.degrees(original_errors['angle_err']):.1f}°), Pos={original_errors['pos_err']:.4f}")

        # Determine which method is better
        angle_improvement = original_errors['angle_err'] - gapartnet_errors['angle_err']
        pos_improvement = original_errors['pos_err'] - gapartnet_errors['pos_err']

        print(f"Improvements (Original - GAPartNet): Angle={angle_improvement:.4f} rad ({np.degrees(angle_improvement):.1f}°), Pos={pos_improvement:.4f}")

        if gapartnet_errors['angle_err'] < original_errors['angle_err']:
            print("✅ GAPartNet has better angle accuracy")
        else:
            print("❌ Original method has better angle accuracy")

        if gapartnet_errors['pos_err'] < original_errors['pos_err']:
            print("✅ GAPartNet has better position accuracy")
        else:
            print("❌ Original method has better position accuracy")

    # Save results
    recon_save_folder = args.obj_folder_path_reconstruct
    os.makedirs(recon_save_folder, exist_ok=True)

    # Save config
    recon_cfg_json = convert_numpy_to_json(recon_cfg)
    with open(os.path.join(recon_save_folder, "cfg.json"), 'w', encoding='utf-8') as f:
        json.dump(recon_cfg_json, f, ensure_ascii=False, indent=4)

    # Save obj_repr (with GAPartNet results)
    obj_repr.save(os.path.join(recon_save_folder, "obj_repr.npy"))

    # Save GAPartNet results
    with open(os.path.join(recon_save_folder, 'gapartnet_result.pkl'), 'wb') as f:
        pickle.dump(gapartnet_result, f)

    # Save comparison results if available
    if original_result:
        with open(os.path.join(recon_save_folder, 'original_result.pkl'), 'wb') as f:
            pickle.dump(original_result, f)

        # Save comparison summary
        comparison_summary = {
            'gapartnet_errors': extract_errors(gapartnet_result),
            'original_errors': extract_errors(original_result),
            'method_used': 'gapartnet_bbox_only_vs_original_reconstruct'
        }
        with open(os.path.join(recon_save_folder, 'comparison.pkl'), 'wb') as f:
            pickle.dump(comparison_summary, f)

    print(f"\nResults saved to {recon_save_folder}")
    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    """
    Enhanced GAPartNet-based object reconstruction with improved bounding box matching.

    Key improvements:
    1. Added bounding box size similarity calculation with depth compensation
    2. Enhanced tracking-based matching with size constraints
    3. Improved bipartite matching using both position and size similarity
    4. Prevents incorrect matching between different-sized parts (e.g., handle vs drawer)
    5. Accounts for perspective effects where distant objects appear smaller

    This helps solve the problem where drawer handles and drawers themselves
    get incorrectly matched due to only considering position distances.
    """
    # main()
    
    gapartnet_model = load_gapartnet_model(
        "/home/zby/Programs/Embodied_Analogy/assets/ckpts/gapartnet/release.ckpt"
    )
    obj_repr = Obj_repr.load(
        "/home/zby/Programs/Embodied_Analogy/assets/unit_test/gapartnet/obj_repr.npy"
    )
    gapartnet_result = gapartnet_reconstruct_fixed(obj_repr, gapartnet_model)
