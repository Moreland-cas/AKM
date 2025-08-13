import logging
import open3d as o3d
import numpy as np
import torch
import random
import os
import sys
import cv2
from akm.utility.constants import PROJECT_ROOT
relative_path = os.path.join(PROJECT_ROOT, "third_party", "RAM_code")
sys.path.append(relative_path)
from graspness_implementation.gsnet import GSNet, vis_save_grasp
from graspnetAPI import GraspGroup, Grasp
import argparse
from akm.utility.constants import ASSET_PATH


def prepare_gsnet(asset_path=ASSET_PATH):
    config = {
        "gsnet": {
            "save_files": False,
            "dataset_root": "",
            "checkpoint_path": os.path.join(asset_path, "ckpts/gsnet/minkuresunet_kinect.tar"),
            "dump_dir": "./logs/",
            "seed_feat_dim": 512,
            "camera": "kinect",
            "num_point": 15000,
            "batch_size": 1,
            "voxel_size": 0.005,
            "collision_thresh": 0.0,
            "voxel_size_cd": 0.01,
            "infer": True,
            "vis": False
        }
    }
    gsnet = GSNet(config["gsnet"])
    return gsnet

def inference_gsnet(gsnet: GSNet, pcs, keep=1e6, nms=True):
    if gsnet is None:
        gsnet = prepare_gsnet()
    gg: GraspGroup = gsnet.inference(pcs)
    if nms:
        gg = gg.nms()
    gg = gg.sort_by_score()
    if len(gg) > keep:
        gg = gg[:keep]
    return gg
    
def detect_grasp_gsnet(gsnet, points, colors=None, nms=True, keep=1e6, visualize=False, asset_path=None, logger=None):
    '''GSNet'''
    # need to preprocess point cloud
    pcs_input = points.copy()
    gsnet = prepare_gsnet(asset_path)
    gg = inference_gsnet(
        gsnet=gsnet,
        pcs=pcs_input,
        keep=keep,
        nms=nms,
    )
    if gg is not None:
        logger.log(logging.DEBUG, f'grasp num: {len(gg)}')
    if visualize:
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers]) 
    return gg

if __name__ == "__main__":
    test_data = "/home/zby/Programs/Embodied_Analogy/assets_zby/logs/explore_51/44781_1_revolute/obj_repr.npy"
    