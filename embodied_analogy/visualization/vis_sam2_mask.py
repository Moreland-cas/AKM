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
