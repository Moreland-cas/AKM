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

def track_any_points(rgb_frames, queries=None, grid_size = 30, always_visible=True, visiualize=False):
    """
        rgb_frames: list of RGB frames, in list of numpy arrays of shape [H, W, C], np.uint8
        queries: 大小为 [B, N, 3]
            3的第一位是timestamp, 一般是0, 后两位是像素坐标 [u, v], 大小与 window_frames 中的frame分辨率一致
            如果 queries 不为 None, 则追踪 queries + supportive_grids, 但不返回后者
            否则追踪图像上 uniformly sample 出的 grid_points, 数量由 grid_size 指定
        always_visible: if True, only return the points that are always visible
        visiualize: if True, visualize the tracking results as a saved mp4 video
    """
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
                queries=None,
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
        queries=None,
        window_frames=window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step=is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
    )

    if always_visible:
        always_visible_mask = pred_visibility.all(dim=1)[0] # N
        pred_tracks = pred_tracks[:, :, always_visible_mask, :] # B T N 2 -> B T M 2
        pred_visibility = pred_visibility[:, :, always_visible_mask] # B T N -> B T M
        
    if visiualize:
        # save a video with predicted tracks
        video = torch.tensor(np.stack(window_frames), device="cuda").permute(0, 3, 1, 2)[None]
        vis = Visualizer(save_dir="./", pad_value=120, linewidth=2)
        vis.visualize(video, pred_tracks, pred_visibility, query_frame=grid_query_frame, filename="tracking_results")

    return pred_tracks, pred_visibility

if __name__ == "__main__":
    rgb_frames = mp4_to_numpy_array_list("/home/zby/Datasets/custom_video/0.mp4")
    track_any_points(rgb_frames, queries=None, always_visible=True, visiualize=True)