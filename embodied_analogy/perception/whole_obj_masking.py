# 给定 rgb_seq, 和 tracks2d（用于 refine mask）, 输出 whole_mask_seq
import os
from PIL import Image
import numpy as np
from embodied_analogy.perception.sam_masking import run_sam_whole
from embodied_analogy.perception.sam2_masking import run_sam2_whole
# 方式1： 通过 sam2 对于视频进行分割
def whole_obj_masking_sam(rgb_seq, positive_tracks2d, negative_tracks2d, visualize=False):
    video_masks = []
    
    if visualize:
        import napari
        viewer = napari.view_image(rgb_seq, rgb=True)
        viewer.title = "whole_obj_masking by sam"
    
    for i in range(len(rgb_seq)):
        rgb_img = rgb_seq[i]
        positive_points = positive_tracks2d[i]
        negative_points = negative_tracks2d[i]
        mask_i, used_pos_i, used_neg_i = run_sam_whole(
            rgb_img=rgb_img, 
            positive_points=positive_points, 
            positive_bbox=None,
            negative_points=negative_points,
            num_iterations=3,
            acceptable_thr=0.9,
            visualize=False
        ) # M, 2
        video_masks.append(mask_i)
        
        if visualize:
            # 把 used points 也存储起来
            pos_pts = used_pos_i[:, [1, 0]]
            neg_pts = used_neg_i[:, [1, 0]]
            
            pos_pts = np.concatenate([np.ones((pos_pts.shape[0], 1)) * i, pos_pts], axis=1) # M, (1+d)
            neg_pts = np.concatenate([np.ones((neg_pts.shape[0], 1)) * i, neg_pts], axis=1) # M, (1+d)
            
            viewer.add_points(pos_pts, face_color="green", name=f"pos_pts {i}")
            viewer.add_points(neg_pts, face_color="red", name=f"neg_pts {i}")
            
    video_masks = np.array(video_masks) # T, H, W
    
    if visualize:
        viewer.add_labels(video_masks.astype(np.int32), name=f'whole obj mask (sam)')
        napari.run()
    return video_masks.astype(np.bool_)

# 方式2： 通过 sam 对单个帧进行分割
# TODO: 给这个函数加一个 iterativly refine 的功能
def whole_obj_masking_sam2(rgb_folder, selected_idx, positive_tracks2d, negative_tracks2d, visualize=False):
    # 首先把 
    initial_point_prompt = positive_tracks2d[0, :2, :].cpu() # N, 2
    video_masks = run_sam2_whole(rgb_folder, initial_point_prompt) # T, H, W
    video_masks = video_masks[selected_idx]
    
    if visualize:
        import napari        
        rgb_seq = []
        image_paths = sorted(
            os.listdir(rgb_folder),
            key=lambda x: int(x.split(".")[0])
        )  # 假设文件夹中是按顺序排列的帧文件
        for i, frame_file in enumerate(image_paths):
            if i not in selected_idx:
                continue
            frame_path = os.path.join(rgb_folder, frame_file)
            frame = Image.open(frame_path)
            frame = np.array(frame)
            rgb_seq.append(frame)
        rgb_seq = np.array(rgb_seq)  # T, H, W, C
        
        viewer = napari.view_image(rgb_seq, rgb=True)
        viewer.title = "whole_obj_masking by sam2"
        viewer.add_labels(video_masks.astype(np.int32), name='whole obj mask (sam2)')
        
        # pos_pts = initial_point_prompt[:, [1, 0]]
        # pos_pts = np.concatenate([np.ones((pos_pts.shape[0], 1)), pos_pts], axis=1) # M, (1+d)
        # viewer.add_points(pos_pts, face_color="green", name=f"pos_pts {0}")
            
        napari.run()
    return video_masks.astype(np.bool_)

