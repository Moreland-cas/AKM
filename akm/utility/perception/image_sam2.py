import os
import sys
import torch
import random
import numpy as np
from scipy import ndimage
from skimage import morphology
from sklearn.cluster import KMeans

relative_path = os.path.join(PROJECT_ROOT, "third_party", "sam2")
sys.path.append(relative_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from akm.utility.constants import PROJECT_ROOT, ASSET_PATH


def select_farthest_point(unsatisfied_points, used_same_class_points, used_diff_class_points):
    if not used_same_class_points and not used_diff_class_points:
        return unsatisfied_points[np.random.choice(len(unsatisfied_points))]

    unsatisfied_points_array = np.array(unsatisfied_points)

    # Calculate the average distance from similar points
    if used_same_class_points:
        same_class_array = np.array(list(used_same_class_points))
        same_class_dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - same_class_array[None, :, :], axis=2)
        mean_same_class_dists = np.min(same_class_dists, axis=1)
    else:
        mean_same_class_dists = np.zeros(len(unsatisfied_points))

    # Calculate the average distance to outliers
    if used_diff_class_points:
        diff_class_array = np.array(list(used_diff_class_points))
        diff_class_dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - diff_class_array[None, :, :], axis=2)
        mean_diff_class_dists = np.min(diff_class_dists, axis=1)
    else:
        mean_diff_class_dists = np.zeros(len(unsatisfied_points))
    
    # Comprehensive consideration: the larger the average distance from the same type, the better; 
    # the farther the distance from the different types, the better
    combined_score = mean_same_class_dists - mean_diff_class_dists
    
    farthest_idx = np.argmax(combined_score)
    return unsatisfied_points[farthest_idx]


def select_dense_point(unsatisfied_points):
    if len(unsatisfied_points) == 1:
        return unsatisfied_points[0]

    unsatisfied_points_array = np.array(unsatisfied_points)
    
    # Calculate the distance matrix between all point pairs
    dists = np.linalg.norm(unsatisfied_points_array[:, None, :] - unsatisfied_points_array[None, :, :], axis=2)
    
    # Calculate the average distance from each point to all other points. The smaller the distance, the greater the density.
    mean_dists = np.mean(dists, axis=1)
    
    # Select the point with the smallest average distance, that is, the point with the highest density
    dense_idx = np.argmin(mean_dists)
    return unsatisfied_points[dense_idx]


def select_cluster_center_point(points, return_k_points=1):
    if len(points) <= return_k_points:
        return np.arange(len(points))

    if return_k_points == 1:
        k_clusters = 3
    else:
        k_clusters = return_k_points
        
    kmeans = KMeans(n_clusters=min(k_clusters, len(points)), init='k-means++', random_state=0).fit(points)
    centers = kmeans.cluster_centers_

    # Find the sample closest to each center point
    closest_indices = []
    for center in centers:
        closest_idx = np.argmin(np.linalg.norm(points - center, axis=1))
        closest_indices.append(closest_idx)
    
    if return_k_points == 1:
        return np.array([random.choice(closest_indices)])
    else:
        return np.array(closest_indices)


def load_sam2_image_model():
    sam2_model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=os.path.join(ASSET_PATH, "ckpts/sam2/sam2.1_hiera_large.pt"),
        device=torch.device("cuda")
    )
    return sam2_model
    

# Run the SAM2 model and gradually optimize the input points to improve the segmentation effect
def sam2_on_image(
    rgb_img, # numpy
    positive_points=None,  # np.array([N, 2])
    positive_bbox=None, # np.array([4]), [u_left, v_left, u_right, v_right]
    negative_points=None,
    num_iterations=5,
    acceptable_thr=0.9,
    sam2_image_model=None,
    post_process=False,
    visualize=False,
):
    sam2_model = sam2_image_model
    if sam2_model is None:
        sam2_model = load_sam2_image_model()
        
    assert num_iterations >= 1
    # At least one of the positive points or positive box is not None
    assert (positive_bbox is not None) or (positive_points is not None)
    
    # Handle None by default
    if positive_points is None:
        positive_points = np.empty((0, 2))  
    if negative_points is None:
        negative_points = np.empty((0, 2))  
        
    if isinstance(positive_points, torch.Tensor):
        positive_points = positive_points.cpu().numpy()
    if isinstance(negative_points, torch.Tensor):
        negative_points = negative_points.cpu().numpy()

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(rgb_img)

    input_point = np.empty((0, 2))
    input_label = np.empty(0)

    used_positive_points = set()
    used_negative_points = set()
    
    acceptable_score = acceptable_thr * len(positive_points) if positive_points is not None else 0
        
    cur_best_score = -1e6
    cur_best_mask = None
    tmp_best_mask = None
    last_logits = None

    # visualize
    if visualize:
        import napari
        viewer = napari.view_image(rgb_img, rgb=True)
        viewer.title = "sam results"
        viewer.add_points(positive_points[:, [1, 0]], face_color="green", name="input positive points")
        viewer.add_points(negative_points[:, [1, 0]], face_color="red", name="input negative points")
            
    for i in range(num_iterations):
        if cur_best_score >= acceptable_score:
            break
        # make prediction
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=positive_bbox,
            mask_input=last_logits[None, :, :] if last_logits is not None else None,
            multimask_output=True,
        )

        # sort by score
        tmp_best_mask = masks[np.argmax(scores)]
        last_logits = logits[np.argmax(scores)]

        # Score function, and returns the points that do not meet the conditions
        def score_mask(mask):
            if positive_points.shape[0] == 0 and negative_points.shape[0] == 0:
                return 0, [], []
            positive_int = positive_points.astype(np.int32)
            negative_int = negative_points.astype(np.int32)
            pos_score = np.sum(mask[positive_int[:, 1], positive_int[:, 0]])
            neg_score = np.sum(mask[negative_int[:, 1], negative_int[:, 0]])

            # Find the point of dissatisfaction
            unsatisfied_positive = [tuple(pt) for pt in positive_points if mask[int(pt[1]), int(pt[0])] == 0 and tuple(pt) not in used_positive_points]
            unsatisfied_negative = [tuple(pt) for pt in negative_points if mask[int(pt[1]), int(pt[0])] == 1 and tuple(pt) not in used_negative_points]
            # Increase the penalty for negative_point because negative points are rare.
            return pos_score - 10 * neg_score, unsatisfied_positive, unsatisfied_negative

        # Update the best segmentation result
        current_score, unsatisfied_positive, unsatisfied_negative = score_mask(tmp_best_mask)
        if current_score > cur_best_score:
            cur_best_mask = tmp_best_mask
            cur_best_score = current_score

        # Prioritize selecting new points from unsatisfied points for expansion
        if unsatisfied_negative:
            new_negative_idx = select_cluster_center_point(unsatisfied_negative, return_k_points=1)[0]
            new_negative_point = unsatisfied_negative[new_negative_idx]
            input_point = np.vstack([input_point, new_negative_point])
            input_label = np.append(input_label, 0)
            used_negative_points.add(tuple(new_negative_point))

        # Select the farthest point from the positive sample points that do not satisfy
        if unsatisfied_positive:
            new_positive_idx = select_cluster_center_point(unsatisfied_positive, return_k_points=1)[0]
            new_positive_point = unsatisfied_positive[new_positive_idx]
            input_point = np.vstack([input_point, new_positive_point])
            input_label = np.append(input_label, 1)
            used_positive_points.add(tuple(new_positive_point))
            
    used_positive_vis = np.array(list(used_positive_points))
    used_negative_vis = np.array(list(used_negative_points))
    cur_best_mask = cur_best_mask.astype(np.bool_)
    
    # Here we perform a post-processing on the output mask
    if post_process:
        cur_best_mask = morphology.binary_closing(cur_best_mask, morphology.disk(5))
        cur_best_mask = ndimage.binary_fill_holes(cur_best_mask)   
        
    if visualize:
        viewer.add_labels(cur_best_mask.astype(np.int32), name='cur best mask')
        
        if len(used_positive_vis) > 0:
            viewer.add_points(used_positive_vis[:, [1, 0]], face_color="green", name=f"used positive points {i}")
        if len(used_negative_vis) > 0:
            viewer.add_points(used_negative_vis[:, [1, 0]], face_color="red", name=f"used negative points {i}")
        if positive_bbox is not None:
            u_leftup, v_leftup, u_rightdown, v_rightdown = positive_bbox
            bbox_rect = np.array([
                [u_leftup, v_leftup],
                [u_rightdown, v_leftup],
                [u_rightdown, v_rightdown],
                [u_leftup, v_rightdown],
            ])
            # Let's transpose it here, because the coordinates of napari are (v, u)
            bbox_rect = bbox_rect[:, [1, 0]]
            viewer.add_shapes(
                bbox_rect[None], # should be of shape 1, 4, 2
                face_color="transparent",
                edge_color="green",
                edge_width=5,
                name="positive bbox prompt"
            )
        napari.run()

    return cur_best_mask, used_positive_vis, used_negative_vis