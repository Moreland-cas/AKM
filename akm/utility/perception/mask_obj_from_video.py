import numpy as np

from akm.utility.perception.grounded_sam import (
    run_grounded_sam,
    load_groundingDINO_model,
    load_sam2_image_model
)


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
            
    video_masks = np.array(video_masks) # T, H, W
    
    if visualize:
        viewer.add_labels(video_masks.astype(np.int32), name=f'obj mask')
        napari.run()
    return video_masks.astype(np.bool_)
