# 给定 rgb_seq, 和 tracks2d（用于 refine mask）, 输出 whole_mask_seq
import numpy as np
# from embodied_analogy.perception.grounding_dino import load_groundingDINO_model, run_groundingDINO
# from embodied_analogy.perception.image_sam2 import sam2_on_image, load_sam2_image_model
from embodied_analogy.perception.grounded_sam import run_grounded_sam, load_groundingDINO_model, load_sam2_image_model
# from embodied_analogy.perception.video_sam2 import run_sam2_whole

# 方式1： 使用 sam2 的 image mode
# TODO: 这个函数应该是可以并行的, 就是一次多个 rgb 图像
def mask_obj_from_video_with_image_sam2(
    rgb_seq,
    obj_description,
    positive_tracks2d,
    negative_tracks2d,
    visualize=False
):
    video_masks = []
    
    if visualize:
        import napari
        viewer = napari.view_image(rgb_seq, rgb=True)
        viewer.title = "mask_obj_from_video_with_image_sam2"
    
    # load dino_model and sam2_image_model
    sam2_image_model = load_sam2_image_model()
    dino_model = load_groundingDINO_model()
    
    for i in range(len(rgb_seq)):
        rgb_img = rgb_seq[i]
        
        positive_points = positive_tracks2d[i] if positive_tracks2d is not None else None
        negative_points = negative_tracks2d[i] if negative_tracks2d is not None else None
        
        obj_bbox, obj_mask = run_grounded_sam(
            rgb_image=rgb_img,
            obj_description=obj_description,
            positive_points=positive_points,  
            negative_points=negative_points,
            num_iterations=3,
            acceptable_thr=0.9,
            dino_model=dino_model,
            sam2_image_model=sam2_image_model,
            visualize=False,
        )
        video_masks.append(obj_mask)
        
        # if visualize:
            # pos_pts = used_pos_i[:, [1, 0]]
            # neg_pts = used_neg_i[:, [1, 0]]
            
            # pos_pts = np.concatenate([np.ones((pos_pts.shape[0], 1)) * i, pos_pts], axis=1) # M, (1+d)
            # neg_pts = np.concatenate([np.ones((neg_pts.shape[0], 1)) * i, neg_pts], axis=1) # M, (1+d)
            
            # viewer.add_points(pos_pts, face_color="green", name=f"pos_pts {i}")
            # viewer.add_points(neg_pts, face_color="red", name=f"neg_pts {i}")
            
    video_masks = np.array(video_masks) # T, H, W
    
    if visualize:
        viewer.add_labels(video_masks.astype(np.int32), name=f'obj mask')
        napari.run()
    return video_masks.astype(np.bool_)

"""
# 方式2: 使用 sam2 的 video mode
# TODO: 给这个函数加一个 iterativly refine 的功能
def mask_obj_from_video_with_video_sam2(rgb_folder, selected_idx, positive_tracks2d, negative_tracks2d, visualize=False):
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
        viewer.title = "mask_obj_from_video_with_video_sam2"
        viewer.add_labels(video_masks.astype(np.int32), name='whole obj mask (sam2)')
        
        # pos_pts = initial_point_prompt[:, [1, 0]]
        # pos_pts = np.concatenate([np.ones((pos_pts.shape[0], 1)), pos_pts], axis=1) # M, (1+d)
        # viewer.add_points(pos_pts, face_color="green", name=f"pos_pts {0}")
            
        napari.run()
    return video_masks.astype(np.bool_)
"""


if __name__ == "__main__":
    data = np.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/explore_data.npz")
    rgb_seq = data["rgb_seq"][::20]
    mask_obj_from_video_with_image_sam2(
        rgb_seq,
        "drawer",
        None,
        None,
        visualize=True
    )
    