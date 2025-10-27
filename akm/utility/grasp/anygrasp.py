import os
import sys
import torch
import logging
import argparse
import numpy as np
from graspnetAPI.grasp import GraspGroup
from scipy.spatial.transform import Rotation as R

from akm.utility.utils import (
    visualize_pc,
    rotation_matrix_between_vectors,
    find_correspondences
)
from akm.utility.server.client import run_anygrasp_remotely
from akm.utility.constants import ASSET_PATH, RUN_REMOTE_ANYGRASP, PROJECT_ROOT


def crop_grasp(grasp_group, contact_point, radius=0.1):
    """
    contact_point: (3, ). Assuming the transform in grasp_group is Tgrasp2w, contact_point also needs to be in world coordinates.
    Crop grasp_group, retaining points whose distance from contact_point is less than radius. Return None if none are found.
    """
    t_grasp2w = grasp_group.translations # N, 3
    distances = np.linalg.norm(t_grasp2w - contact_point, axis=1) # N
    mask = distances < radius
    
    if mask.sum() == 0:
        return None
    
    grasp_group_ = GraspGroup()
    grasp_group_.grasp_group_array = grasp_group.grasp_group_array[mask]
    return grasp_group_
    
    
def filter_grasp_group(
    grasp_group, 
    degree_thre=30,
    dir_out=None,
):
    '''
    Filter grasp_group so that the negative direction of the retained grasp's appro_vector is as parallel as possible to dir_out.
    grasp_group: Tgrasp2c
    dir_out: (3, )
    contact_region: (N, 3), which is the moving part.
    NOTE: grasp_group, contact_region, and dir_out are all in the camera coordinate system.
    '''
    if grasp_group is None:
        return None
    
    Rgrasp2c = grasp_group.rotation_matrices # N, 3, 3
    neg_x_axis = -Rgrasp2c[:, :, 0] # N, 3
    
    # Make the -x axis of grasp_frame as parallel to dir_out as possible
    product = np.sum(neg_x_axis * dir_out, axis=-1) # N
    product = product / (np.linalg.norm(neg_x_axis, axis=-1) * np.linalg.norm(dir_out))
    index = product > np.cos(np.deg2rad(degree_thre))
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    return grasp_group

def prepare_any_grasp_model(asset_path=ASSET_PATH):
    relative_path = os.path.join(PROJECT_ROOT, "akm", "utility/grasp")
    sys.path.append(relative_path)
    # load model
    from gsnet import AnyGrasp # gsnet.so
    # get a argument namespace
    cfgs = argparse.Namespace()
    cfgs.checkpoint_path = os.path.join(asset_path, 'ckpts/anygrasp/checkpoint_detection.tar')
    cfgs.max_gripper_width = 0.04
    cfgs.gripper_height = 0.03
    cfgs.top_down_grasp = False
    cfgs.debug = False
    model = AnyGrasp(cfgs)
    model.load_net()
    return model

@torch.no_grad()
def detect_grasp_anygrasp(
    points, 
    colors, 
    dir_out, 
    augment=True, 
    visualize=False,
    run_remote=RUN_REMOTE_ANYGRASP,
    logger=None,
):
    '''
    Input points and dir_out in coordinate system a, and output grasp_group detected by anygrasp, which contains information Tgrasp2a.
        points: (N, 3), in coordinate system
        colors: (N, 3), in the range [0-1]
        dir_out: (3, ), in coordinate system, transforms the points in coordinate system a to mimic the format required by the grasp detector network (camera direction pointing into the point cloud).
        augment: If True, performs multiple rotations on dir_out, predicting grasps separately, 
            and then merges the different grasps back into a single coordinate system.
    '''
    dir_out = dir_out / np.linalg.norm(dir_out)
    if augment:
        np.random.seed(666)
        # random_perturb = np.random.randn(20, 3) 
        random_perturb = np.random.randn(5, 3) 
        # Keep the original dir_out
        random_perturb[0] = random_perturb[0] * 0
        dir_outs = dir_out + random_perturb * 0.5 # N, 3
        dir_outs = dir_outs / np.linalg.norm(dir_outs, axis=1, keepdims=True)
    else:
        dir_outs = [dir_out]
    
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)
    lims = np.array([-1, 1, -1, 1, -1, 1]) * 10
    
    ggs = GraspGroup()
    for dir_out_i in dir_outs:
        # Use different dir_out to determine different app coordinate systems, coor_app = Rw2app @ coor_w, that is, (0, 0, 1) = Rw2app @ -dir_out
        Rw2app = rotation_matrix_between_vectors(-dir_out_i, np.array([0, 0, 1]))
        points_input = points @ Rw2app.T # N, 3
        points_input = points_input.astype(np.float32)
        
        # Tgrasp2app
        if not run_remote:
            model = prepare_any_grasp_model(ASSET_PATH)
            gg, _ = model.get_grasp(
                points_input,
                colors, 
                lims,
                apply_object_mask=True,
                dense_grasp=False,
                collision_detection=True
            )
        else:
            gg = run_anygrasp_remotely(
                points_input,
                colors, 
                lims
            )
        
        if gg == None or len(gg) == 0:
            continue
        logger.log(logging.DEBUG, f'grasp num: {len(gg)}')
        
        # Tgrasp2w
        zero_translation = np.array([[0], [0], [0]])
        Rapp2w = Rw2app.T
        Tapp2w = np.hstack((Rapp2w, zero_translation))
        gg.transform(Tapp2w)
        ggs.add(gg)
        torch.cuda.empty_cache()
    
    if visualize:
        visualize_pc(
            points=points,
            colors=colors,
            grasp=ggs,
            contact_point=np.array([0, 0, 0]),
            post_contact_dirs=[dir_outs]
        )
    return ggs


def find_nearest_grasp(grasp_group, contact_point):
    '''
    grasp_group: graspnetAPI 
    contact_point: (3, )
    '''
    # Find the grasp closest to the contact_point in grasp_group and return it.
    # First, sort by grasp score, filtering out the top 20.
    grasp_group = grasp_group.nms().sort_by_score()
    grasp_group = grasp_group[0:50]
    
    # Find the grasp closest to contact_point
    translations = grasp_group.translations # N, 3
    distances = np.linalg.norm(translations - contact_point, axis=1)
    nearest_index = np.argmin(distances)
    nearest_index = int(nearest_index)
    return grasp_group[nearest_index]


def crop_grasp_by_moving(grasp_group, contact_region, crop_thresh=0.1):
    '''
    Find the grasp closest to the midpoint of the contact region.
    grasp_group: Tgrasp2c
    contact_region: (N, 3), which is the point cloud formed by the moving part.
    NOTE: grasp_group, contact_region, and dir_out are all in the camera coordinate system.
    '''
    if grasp_group is None:
        return None
        
    t_grasp2c = grasp_group.translations # N, 3
    _, distances, _ = find_correspondences(t_grasp2c, contact_region) # N
    
    # First, hard-filter it and ignore the grasp that is more than 1dm away from the nearest contact_region
    hard_mask = distances < crop_thresh
    if hard_mask.sum() == 0:
        return None
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[hard_mask]
    
    pred_scores = grasp_group.scores # N
    distance_scores = 1 / (distances[hard_mask] + 1e-6) 
    grasp_scores = pred_scores * distance_scores  # N
    
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group


def sort_grasp_group(grasp_group, contact_region):
    '''
    Find the grasp closest to the midpoint of the contact region.
    grasp_group: Tgrasp2c
    contact_region: (N, 3), which is the point cloud formed by the moving part.
    NOTE: grasp_group, contact_region, and dir_out are all in the camera coordinate system.
    '''
    if grasp_group is None:
        return None
        
    t_grasp2c = grasp_group.translations # N, 3
    _, distances, _ = find_correspondences(t_grasp2c, contact_region) # N
    
    pred_scores = grasp_group.scores # N
    distance_scores = 1 / (distances + 1e-6) 
    grasp_scores = pred_scores * distance_scores  # N
    
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group