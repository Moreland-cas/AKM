from skimage import data
import numpy as np
import napari

def visualize_sam2_mask(rgb_seq, mask_seq):
    """
    rgb_seq: T, H, W, C (np.uint8)
    mask_seq: T, H, W (np.int32) 0 for background, 1 for foreground 
    """
    # create the viewer with an image
    viewer = napari.view_image(rgb_seq, rgb=True)
    viewer.add_labels(mask_seq.astype(np.int32), name='articulated objects')
    napari.run()

def visualize_sam2_mask_as_part(rgb_seq, whole_mask_seq, moving_mask_seq, static_mask_seq):
    """
    rgb_seq: T, H, W, C (np.uint8)
    mask_seq: T, H, W (np.int32) 0 for background, 1 for foreground 
    """
    if isinstance(moving_mask_seq, list):
        moving_mask_seq = np.stack(moving_mask_seq, axis=0)
    if isinstance(static_mask_seq, list):
        static_mask_seq = np.stack(static_mask_seq, axis=0)
    # create the viewer with an image
    viewer = napari.view_image(rgb_seq, rgb=True)
    viewer.add_labels(whole_mask_seq.astype(np.int32), name='articulated objects')
    viewer.add_labels(moving_mask_seq.astype(np.int32) * 2, name='moving parts')
    viewer.add_labels(static_mask_seq.astype(np.int32) * 3, name='static parts')
    napari.run()