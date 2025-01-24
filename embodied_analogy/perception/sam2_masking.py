"""
    给定一个 rgb 视频, 和一个 prompt, 对于视频中的物体和机械臂进行区域跟踪, 输出每一帧的 mask
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def run_sam2(
    rgb_seq, # np.array([T, H, W, 3], uint8)
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/embodied_analogy/tmp/",
    articulated_obj_id=666, 
    articulated_points=None, # np.array([N, 2], float32)
    franka_arm_id=888
    ):
    # load sam2 predictor model
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"
    predictor = build_sam2_video_predictor(
        config_file=sam2_proj_path + "/configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt", 
        device=torch.device("cuda")
    )

    # transform input rgb_seq to jpg images in a tmp folder
    for i, rgb in enumerate(rgb_seq):
        Image.fromarray(rgb).save(os.path.join(tmp_folder, f"{i}.jpg"))

    # initialize the inference state
    inference_state = predictor.init_state(video_path=tmp_folder)
    predictor.reset_state(inference_state)

    # add prompt to the first frame
    # 1) articulated object
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = articulated_obj_id  # give a unique id to each object we interact with (it can be any integers)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.ones(len(articulated_points), dtype=np.uint32) # N
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=articulated_points,
        labels=labels,
    )

    # 2) franka arm
    # ann_frame_idx = 0  # the frame index we interact with
    # ann_obj_id = franka_arm_id  # give a unique id to each object we interact with (it can be any integers)
    # points = np.array([[210, 350]], dtype=np.float32)
    # labels = np.array([1], np.int32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )

    # run propagation throughout the video and collect the results in a dict
    """
        {
            0: {
                    articulated_obj_id: np.array(H, W),
                    franka_arm_id: np.array(H, W),
                },
            }
            1: {
                    articulated_obj_id: np.array(H, W),
                    franka_arm_id: np.array(H, W),
                },
            }
            ...
        }
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i][0] > 0.0).cpu().numpy() # H, W
            for i, out_obj_id in enumerate(out_obj_ids)
        }
