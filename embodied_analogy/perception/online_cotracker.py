# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
from embodied_analogy.visualization.vis_tracks_2d import vis_tracks2d_napari


import cv2
import numpy as np

def mp4_to_numpy_array_list(mp4_path):
    """
    Reads an MP4 file and converts it into a list of numpy arrays.

    Parameters:
        mp4_path (str): Path to the input MP4 file.

    Returns:
        list: A list of numpy arrays, each with shape (H, W, C) and dtype np.uint8.
    """
    # Initialize an empty list to store frames
    frames = []

    # Open the video file using cv2
    cap = cv2.VideoCapture(mp4_path)

    # Check if the file was opened successfully
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open the file: {mp4_path}")

    # Read frames in a loop
    while True:
        ret, frame = cap.read()
        
        # If no frame is returned, we have reached the end of the video
        if not ret:
            break

        # Append the frame to the list (default cv2 frame is np.uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    return frames

def track_any_points(rgb_frames, queries=None, grid_size=30, visualize=False):
    """
    Input:
        rgb_frames: 
            list of np.array([H, W, C], dtype=uint8)
        queries: 
            torch.tensor([N, 2]), 存储初始帧的追踪点坐标 uv, 分辨率与rgb_frame 保持一致
            if queries is not None:
                追踪 queries + supportive_grids, 但不返回后者
            else:
                追踪图像上 uniformly sample 出的 grid_points, 数量由 grid_size 指定
    return:
        pred_tracks: 
            torch.tensor([T, M, 2]) on cpu
        pred_visibility: 
            torch.tensor([T, M]) on cpu
    """
    assert (queries is not None) or (queries is None and grid_size is not None)
    def _process_step(model, queries, window_frames, is_first_step, grid_size, grid_query_frame):
        frames = torch.tensor(np.stack(window_frames[-model.step * 2 :]), device="cuda").float() # T, H, W, 3
        video_chunk = (frames.permute(0, 3, 1, 2)[None])  # (1, T, 3, H, W)
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            add_support_grid=True
        )
        return result
    
    # 调整 queries 的格式为 B, N, 3
    if queries is not None:
        if isinstance(queries, np.ndarray):
            queries = torch.from_numpy(queries).float()
        queries = torch.cat([torch.zeros(queries.shape[0], 1), queries], dim=-1)
        queries = queries[None].cuda()
    
    grid_query_frame = 0
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to("cuda")
    window_frames = []

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(rgb_frames):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                model=model,
                queries=queries,
                window_frames=window_frames,
                is_first_step=is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        model=model,
        queries=queries,
        window_frames=window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
    )

    # 去除 batch 维度, 并转移到 cpu 上去
    pred_tracks, pred_visibility = pred_tracks[0].cpu(), pred_visibility[0].cpu()
    
    if visualize:
        vis_tracks2d_napari(rgb_frames, pred_tracks, viewer_title="Cotracker returned tracks2d")
        
    return pred_tracks, pred_visibility

if __name__ == "__main__":
    rgb_frames = mp4_to_numpy_array_list("./third_party/sam2/notebooks/videos/bedroom.mp4")
    mask = np.load("./embodied_analogy/mask.npy")
    queries = np.column_stack(np.where(mask))
    # import pdb;pdb.set_trace()
    queries = np.flip(queries, axis=-1)
    indices = np.random.choice(len(queries), size=20, replace=False)
    queries = queries[indices] # M, 2
    queries = np.column_stack([np.zeros(len(queries)), queries])[None]
    queries = torch.from_numpy(queries).float()
    track_any_points(rgb_frames, queries=queries.cuda(), always_visible=False, visiualize=True)