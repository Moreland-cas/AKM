import os
import sys
import copy
import torch
import numpy as np
from einops import rearrange, repeat

from akm.project_config import PROJECT_ROOT, ASSET_PATH
sys.path.append(os.path.join(PROJECT_ROOT, "third_party", "GAPartNet"))
from structure.utils import _load_perception_model
from gapartnet.structure.point_cloud import PointCloud
from gapartnet.dataset.gapartnet import apply_voxelization
from gapartnet.network.grouping_utils import filter_invalid_proposals, apply_nms
from gapartnet.misc.pose_fitting import estimate_pose_from_npcs

from akm.utility.utils import visualize_pc
from akm.utility.estimation.coarse_joint_est import (
    coarse_R_from_tracks_3d_augmented,
    coarse_t_from_tracks_3d,
    coarse_estimation
)

def load_gapartnet_model(ckpt_path=None):
    """Load GAPartNet model from checkoint"""
    print(f"Loading GAPartNet model from {ckpt_path}...")

    # Load the perception model
    if ckpt_path is None:
        ckpt_path = os.path.join(ASSET_PATH, "ckpts", "gapartnet", "all_best.ckpt")
    gapartnet_model = _load_perception_model(ckpt_path)

    return gapartnet_model

def extract_point_clouds_with_segmentation(obj_repr):
    """
    Extract the object point cloud from the first and last frames of the frames obtained in the exploration phase
    """
    print(f"Extracting point clouds from frames...")
    from akm.representation.basic_structure import Frame
    from akm.representation.obj_repr import Obj_repr
    
    obj_repr: Obj_repr

    # Get frames
    first_frame: Frame = obj_repr.frames[0]
    last_frame: Frame = obj_repr.frames[-1]

    # First check that the obj_mask of both frames are present.
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
    first_pc, first_pc_colors = first_frame.get_obj_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=False
    ) 
    last_pc, last_pc_colors = last_frame.get_obj_pc(
        use_robot_mask=True,
        use_height_filter=True,
        world_frame=False
    )
    first_pc_colors = first_pc_colors / 255
    last_pc_colors = last_pc_colors / 255
    return first_pc, first_pc_colors, last_pc, last_pc_colors

def create_pointcloud_input(pc_xyz, pc_colors, device="cuda"):
    """
    Create PointCloud input with PROPER GAPartNet coordinate handling
    """
    # Store original statistics for inverse transformation
    center = np.mean(pc_xyz, axis=0)
    max_radius = np.max(np.linalg.norm(pc_xyz - center, axis=1))

    if max_radius < 1e-6:
        max_radius = 1.0

    # GAPartNet ball normalization
    centered_pc = pc_xyz - center
    normalized_pc = centered_pc / max_radius

    # Create RGB colors
    pc_with_colors = np.concatenate([normalized_pc, pc_colors], axis=1)

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
        pc_id=f"frame_0",
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
    def downsample(pc: PointCloud, *, max_points: int = 20000) -> PointCloud:
        pc = copy.copy(pc)
        num_points = pc.points.shape[0]

        if num_points > max_points:
            indices = np.random.choice(num_points, max_points, replace=False)
            pc.points = pc.points[indices]
        return pc

    pc = pc.to_tensor()
    pc = downsample(pc, max_points=20000) # zby
    pc = apply_voxelization(pc, voxel_size=(1. / 100, 1. / 100, 1. / 100))

    return pc, max_radius, center

def run_gapartnet_inference(pc_xyz, pc_colors, gapartnet_model, use_nms=True):
    """
    Run GAPartNet inference following the EXACT training procedure with optional GT segmentation
    """
    # Create input following the EXACT training procedure
    pc, max_radius, center = create_pointcloud_input(pc_xyz, pc_colors)

    # Run inference following _training_or_validation_step procedure
    with torch.no_grad():
        # Step 1: Forward backbone
        data_batch = PointCloud.collate([pc])
        points = data_batch.points
        batch_indices = data_batch.batch_indices
        pt_xyz = points[:, :3]
        pc_feature = gapartnet_model.forward_backbone(pc_batch=data_batch)

        # Step 2: Semantic segmentation
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
            raise Exception("No proposals generated")
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
        if use_nms:
            print(f"    Applying filtering and NMS (same as training)...")
            proposals_filtered = filter_invalid_proposals(
                proposals,
                score_threshold=0.09,  # Same as val_score_threshold in training
                min_num_points_per_proposal=3
            )
            proposals_filtered = apply_nms(proposals, 0.3)  # Same as val_nms_iou_threshold in training

            if proposals is None or len(proposals.proposal_offsets) <= 1:
                proposals = proposals
            else:
                proposals = proposals_filtered

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
            def transform_gapartnet_bbox_to_camera(bbox_ball, max_radius, center):
                """CORRECT transformation from GAPartNet ball space to camera coordinates"""
                # GAPartNet bbox is in normalized ball space [-1, 1]
                # Transform back to original camera coordinates
                bbox_camera = bbox_ball * max_radius + center
                return bbox_camera
            
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

            bboxes.append(bbox_xyz)

            # Store score and semantic label
            bbox_scores.append(proposals.score_preds[proposal_i].item())
            bbox_sem_labels.append(proposal_sem_labels[proposal_i].item())

    print(f"    Final: {len(bboxes)} bounding boxes generated")
    return bboxes, bbox_scores, bbox_sem_labels

def calculate_bbox_size_feature(bbox):
    """
    Calculate size metrics for a bounding box

    Args:
        bbox: numpy array of shape (8, 3) representing 8 corners of the bounding box

    Returns:
        dict with keys: 'dimensions', 'volume', 'surface_area', 'diagonal'
        or None if bbox is invalid
    """
    dimensions = [0, 0, 0]
    dimensions[0] = np.linalg.norm(bbox[1] - bbox[0])
    dimensions[1] = np.linalg.norm(bbox[2] - bbox[0])
    dimensions[2] = np.linalg.norm(bbox[3] - bbox[0])
    dimensions = sorted(dimensions)
    dimensions = np.array(dimensions)
    
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

def calculate_bbox_pair_score(bbox1, bbox2, moving_tracks3d):
    """
    Calculate bbox similarity
    Args:
        bbox1, bbox2: numpy arrays of shape (8, 3) representing bounding box corners

    Returns:
        float: similarity score between 0 and 1 (1 = identical size, 0 = very different)
    """
    metrics1 = calculate_bbox_size_feature(bbox1)
    metrics2 = calculate_bbox_size_feature(bbox2)
    # Calculate similarity scores (1.0 = identical, 0.0 = very different)
    def similarity_score(val1, val2):
        ratio = 1 / np.exp(abs(val1 - val2))
        return ratio

    # Multiple metrics for robustness
    volume_sim = similarity_score(metrics1['volume'], metrics2['volume'])
    surface_sim = similarity_score(metrics1['surface_area'], metrics2['surface_area'])
    diagonal_sim = similarity_score(metrics1['diagonal'], metrics2['diagonal'])

    # Dimension-wise similarity
    dim_sim = np.mean([similarity_score(metrics1['dimensions'][i], metrics2['dimensions'][i]) for i in range(3)])

    # Weighted combination
    size_score = 0.3 * volume_sim + 0.2 * surface_sim + 0.2 * diagonal_sim + 0.3 * dim_sim

    bbox1_center = np.mean(bbox1, axis=0)
    bbox2_center = np.mean(bbox2, axis=0)
    
    nearset_dis1 = np.min(np.linalg.norm(bbox1_center[None] - moving_tracks3d[0], axis=1))
    nearset_dis2 = np.min(np.linalg.norm(bbox2_center[None] - moving_tracks3d[-1], axis=1))
    moving_score = np.exp(-nearset_dis1) * np.exp(-nearset_dis2)
    
    # See if the center moves along the trajectory of the nearest tracks and stops near another center.
    nearest_idx_in_tracks = np.argmin(np.linalg.norm(bbox1_center[None] - moving_tracks3d[0], axis=1))
    transformed_center = bbox1_center + moving_tracks3d[-1, nearest_idx_in_tracks, :] - moving_tracks3d[0, nearest_idx_in_tracks, :]
    dynamic_score = np.exp(-np.linalg.norm(transformed_center - bbox2_center))
    
    score = size_score * moving_score * dynamic_score
    return score

def identify_matched_bboxes(bboxes_first, bboxes_last, obj_repr):
    """
    Use tracking information to identify the CORRECT moving bounding box with size constraints
    bboxes_first: N, 8, 3
    bboxes_last: N, 8, 3
    """
    if len(bboxes_first) == 0 or len(bboxes_last) == 0:
        raise Exception("Encounter empty bbox list input in identify_matched_bboxes().")

    assert obj_repr.frames.track3d_seq is not None
    print(f"Identifying moving bounding box using tracking data with size constraints...")
    print(f"  {len(bboxes_first)} bboxes in first frame, {len(bboxes_last)} in last frame")

    # Get tracking data - this is the GROUND TRUTH for moving parts
    tracks_3d = obj_repr.frames.track3d_seq
    moving_mask = obj_repr.frames.moving_mask
    moving_tracks = tracks_3d[:, moving_mask, :]
        
    sim_matrix = np.zeros((len(bboxes_first), len(bboxes_last)))
    for i, bbox_first in enumerate(bboxes_first):
        for j, bbox_last in enumerate(bboxes_last):
            sim_matrix[i, j] = calculate_bbox_pair_score(bbox_first, bbox_last, moving_tracks)
            
    # Find i, j that maximizes sim_matrix[i, j], and return bboxes_first[i] and bboxes_last[j]
    max_indices = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
    i, j = max_indices
    
    bbox_first = bboxes_first[i]
    bbox_last = bboxes_last[j]
    return bbox_first, bbox_last

def gapartnet_reconstruct(obj_repr, gapartnet_model=None, visualize=False, use_gt_joint_type=True):
    """COMPLETELY CORRECTED GAPartNet reconstruction (always uses model segmentation, never GT)"""
    print("Running COMPLETELY CORRECTED GAPartNet reconstruction...")

    if gapartnet_model is None:
        gapartnet_model = load_gapartnet_model("/home/zby/Programs/AKM/assets/ckpts/gapartnet/all_best.ckpt")
        
    # Get ground truth for guidance
    gt_joint_type = obj_repr.gt_joint_dict.get("joint_type", "prismatic")
    print(f"Ground truth joint type: {gt_joint_type}")
    print(f"Using model segmentation (GAPartNet model.py lines 497-499)")
    print(f"Note: GT segmentation is disabled for reconstruction pipeline")

    # Extract point clouds (no GT segmentation for reconstruction)
    first_pc, first_pc_colors, last_pc, last_pc_colors = extract_point_clouds_with_segmentation(obj_repr)

    # Run CORRECTED GAPartNet inference using model segmentation only
    bboxes_first, bbox_scores_first, bbox_sem_labels_first = run_gapartnet_inference(
        pc_xyz=first_pc, 
        pc_colors=first_pc_colors,
        gapartnet_model=gapartnet_model, 
        use_nms=True
    )
    bboxes_last, bbox_scores_last, bbox_sem_labels_last = run_gapartnet_inference(
        pc_xyz=last_pc, 
        pc_colors=last_pc_colors,
        gapartnet_model=gapartnet_model, 
        use_nms=True
    )

    # Use tracking-based moving part identification
    moving_bbox_first, moving_bbox_last = identify_matched_bboxes(bboxes_first, bboxes_last, obj_repr)

    if visualize:
        visualize_pc(first_pc, colors=first_pc_colors, bboxes=[moving_bbox_first, moving_bbox_last])
    
    # Estimate joint parameters
    if use_gt_joint_type:
        if gt_joint_type == "prismatic":
            joint_dict, loss = coarse_t_from_tracks_3d(np.stack([moving_bbox_first, moving_bbox_last], axis=0))
        else:
            joint_dict, loss = coarse_R_from_tracks_3d_augmented(np.stack([moving_bbox_first, moving_bbox_last], axis=0), num_R_augmented=1000)
        joint_dict["joint_type"] = gt_joint_type
    else:
        joint_dict = coarse_estimation(np.stack([moving_bbox_first, moving_bbox_last], axis=0))
    
    joint_dict["moving_bbox_first"] = moving_bbox_first
    joint_dict["moving_bbox_last"] = moving_bbox_last
    
    return joint_dict

def estimate_joint_from_bboxes(moving_bbox_first, moving_bbox_last, joint_type):
    """
    Estimate joint parameters directly from GAPartNet bounding boxes
    Returns a dict containing joint_dir and joint_start
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

    # Joint direction is the translation direction
    joint_dir = translation / translation_magnitude
    joint_start = center_first

    return {
        'joint_dir': joint_dir,
        'joint_start': joint_start,
    }

def estimate_revolute_joint(bbox_first, bbox_last):
    """Estimate revolute joint from bounding box transformation"""
    points_first = bbox_first
    points_last = bbox_last
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