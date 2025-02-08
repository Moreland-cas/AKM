import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import hydra

from sam2.build_sam import build_sam2_video_predictor
import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/sam2")

def run_sam2_part(
    rgb_folder, 
    moving_tracks_2d, # np.array([T, N1, 2])
    static_tracks_2d, # np.array([T, N2, 2])
    save_mask=True,
    clear_tmp_folder=False,
    tmp_folder = "/home/zby/Programs/Embodied_Analogy/asset/tmp/",
):    
    # load sam2 predictor model
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"
    predictor = build_sam2_video_predictor(
        config_file ="configs/sam2.1/sam2.1_hiera_l.yaml",
        # config_file ="configs/sam2.1/sam2.1_hiera_s.yaml",
        # config_file ="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt", 
        device=torch.device("cuda")
    )

    # initialize the inference state
    inference_state = predictor.init_state(video_path=rgb_folder)
    predictor.reset_state(inference_state)

    # add prompt 
    moving_id = 666
    static_id = 888
    color_map = {
        0: (0, 0, 0), # background
        moving_id: (1, 0, 0), 
        static_id: (0, 1, 0),
    }
    for i in range(len(rgb_seq)):
        ann_frame_idx = i
        
        moving_2d = moving_tracks_2d[i] # N, 2
        static_2d = static_tracks_2d[i] # M, 2
        
        # random sample some of it 
        moving_2d = moving_2d[torch.randperm(len(moving_2d))[:1]]
        static_2d = static_2d[torch.randperm(len(static_2d))[:1]]
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=moving_id,
            points=moving_2d,
            labels=np.ones(len(moving_2d), dtype=np.uint32),
        )
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=moving_id,
            points=static_2d,
            # labels=np.zeros(len(static_2d), dtype=np.uint32),
            labels=np.ones(len(static_2d), dtype=np.uint32),
        )
        
        # predictor.add_new_points_or_box(
        #     inference_state=inference_state,
        #     frame_idx=ann_frame_idx,
        #     obj_id=static_id,
        #     points=static_2d,
        #     labels=np.ones(len(static_2d), dtype=np.uint32),
        # )
        
        break
        
        # predictor.add_new_points_or_box(
        #     inference_state=inference_state,
        #     frame_idx=ann_frame_idx,
        #     obj_id=static_id,
        #     points=moving_2d,
        #     labels=np.zeros(len(moving_2d), dtype=np.uint32),
        # )

    # run propagation throughout the video and collect the results in a dict
    """
        {
            0: {
                    moving_id: np.array(H, W),
                    static_id: np.array(H, W),
                },
            }
            1: {
                    moving_id: np.array(H, W),
                    static_id: np.array(H, W),
                },
            }
            ...
        }
    """
    video_masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = np.zeros_like(out_mask_logits[0][0].cpu().numpy())
        for i, out_obj_id in enumerate(out_obj_ids):
            mask += (out_mask_logits[i][0] > 0.0).cpu().numpy() * out_obj_id # H, W
        video_masks.append(mask)
         
    video_masks = np.stack(video_masks, axis=0) # T, H, W 
    return video_masks

def run_sam2_whole(
    rgb_folder, 
    initial_point_prompt, # np.array([N, 2])
):    
    # load sam2 predictor model
    sam2_proj_path = "/home/zby/Programs/Embodied_Analogy/third_party/sam2"
    predictor = build_sam2_video_predictor(
        config_file ="configs/sam2.1/sam2.1_hiera_l.yaml",
        # config_file ="configs/sam2.1/sam2.1_hiera_s.yaml",
        # config_file ="configs/sam2.1/sam2.1_hiera_t.yaml",
        ckpt_path=sam2_proj_path + "/checkpoints/sam2.1_hiera_large.pt", 
        device=torch.device("cuda")
    )

    # initialize the inference state
    inference_state = predictor.init_state(video_path=rgb_folder)
    predictor.reset_state(inference_state)
    
    # 在第一帧进行 point prompt 的标注
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=666,
        points=initial_point_prompt,
        labels=np.ones(len(initial_point_prompt), dtype=np.uint32),
    )        

    # run propagation throughout the video and collect the results in a dict
    video_masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0][0] > 0.0).cpu().numpy()  # H, W, np.bool
        video_masks.append(mask)
    
    video_masks = np.stack(video_masks, axis=0) # T, H, W 
    return video_masks


if __name__ == "__main__":
    from embodied_analogy.pipeline.process_recorded_data import RecordDataReader
    record_path_prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
    file_name = "/2025-01-07_18-06-10.npz"
    dr = RecordDataReader(record_path_prefix, file_name)
    dr.process_data()

    rgb_seq = dr.rgb # T H W C
    articulated_obj_id = 666
    video_segments = run_sam2_whole(
        rgb_seq, # np.array([T, H, W, 3], uint8)
        tmp_folder = "/home/zby/Programs/Embodied_Analogy/embodied_analogy/tmp/",
        articulated_obj_id=articulated_obj_id, 
        articulated_points=np.array([[380, 360]]), # np.array([N, 2], float32)
        franka_arm_id=888,
        save_intermediate_mask=True
    )
    
    from embodied_analogy.visualization.vis_sam2_mask import visualize_sam2_mask
    masks = [video_segments[i][articulated_obj_id] for i in range(len(video_segments))]
    masks = np.stack(masks, axis=0) # T, H, W
    visualize_sam2_mask(rgb_seq, masks)