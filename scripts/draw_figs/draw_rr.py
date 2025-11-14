# need to visualize obj_repr and reloc frame


import os
import numpy as np
from PIL import Image
from graspnetAPI.grasp import Grasp
from PIL import Image, ImageDraw, ImageFont

from akm.utility.constants import (
    STATIC_LABEL,
    MOVING_LABEL,
    UNKNOWN_LABEL
)
from akm.utility.utils import (
    draw_points_on_image,
    visualize_pc,
    depth_image_to_pointcloud,
    joint_data_to_transform_np,
    add_text_to_image
)
from akm.representation.obj_repr import Obj_repr
from akm.representation.basic_structure import Frame
from akm.utility.proposal.affordance import Affordance_map_2d

def vis_rr(idx):
    gray_color = np.array([128, 128, 128]) / 255
    apple_green_color = np.array([159, 191, 82]) / 255
    slight_green_color = np.array([149, 187, 114]) / 255
    mobile_pc_color = np.array([147, 205, 245]) / 255
    static_pc_color = np.array([255, 0, 0]) / 255

    render_intrinsic = np.array(
        [[300.,   0., 400.],
        [  0., 300., 300.],
        [  0.,   0.,   1.]], dtype=np.float32)
    render_extrinsic = np.eye(4)
    # z axis
    render_extrinsic[2, -1] = 0.1
    zoom_in_scale = 2

    obj_repr_path = f"/home/zby/Desktop/assets/save_vis2/{idx}/slim_obj_repr.npy"
    obj_repr = Obj_repr.load(obj_repr_path)
    visualize_coarse = True

    # Draw reconstruct
    if visualize_coarse:
        # Coarse stage draws a manipulate first frame + 3D tracks + joint_axis
        obj_pc_s, pc_colors_s = obj_repr.kframes[0].get_obj_pc(
            use_robot_mask=True, 
            use_height_filter=True,
            world_frame=False,
            visualize=False
        )
        pc_colors_s = pc_colors_s / 255

        obj_pc_e, pc_colors_e = obj_repr.kframes[-1].get_env_pc(
            use_robot_mask=True, 
            use_height_filter=True,
            world_frame=False,
            visualize=False
        )
        pc_colors_e = pc_colors_e / 255

        coarse_jonint_dict = obj_repr.get_joint_param(
            resolution="coarse",
            frame="camera"
        )

        moving_track_mask = obj_repr.frames.moving_mask
        coarse_image = visualize_pc(
            points=[obj_pc_s, obj_pc_e],
            point_size=[5, 5],
            voxel_size=None,
            colors=[gray_color, gray_color],
            alpha=[0.6, 0.6],
            tracks_3d=obj_repr.frames.track3d_seq[:, moving_track_mask],
            tracks_3d_colors=apple_green_color,
            pivot_point=coarse_jonint_dict["joint_start"],
            joint_axis=coarse_jonint_dict["joint_dir"],
            tracks_t_step=3, 
            tracks_n_step=None,
            tracks_norm_threshold=0.2e-2,
            camera_intrinsic=render_intrinsic,
            camera_extrinsic=render_extrinsic,
            zoom_in_scale=zoom_in_scale,
            online_viewer=True
        )
        # Image.fromarray(coarse_image).save(os.path.join(save_folder, "coarse_image.png"))


    def mask_on_image(image, mask, overlay_color=(255, 0, 0, 128)):
        """
        Overlays a colored mask on the image and draws the bounding box of the mask.
        
        Args:
            image: PIL.Image, input image
            mask: PIL.Image, binary mask (0 or 255 values)
            overlay_color: tuple, RGBA color for overlay (default: semi-transparent red)
            bbox_color: tuple, RGB color for bounding box (default: green)
        
        Returns:
            PIL.Image, image with overlay and bounding box
        """
        # Convert inputs to appropriate formats
        image = image.convert('RGBA')
        
        # Create overlay image with specified color
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_data = np.array(overlay)
        
        # Get mask array
        mask_array = np.array(mask)
        
        # Apply overlay where mask is non-zero
        overlay_data[mask_array > 0] = overlay_color
        
        # Blend overlay with original image
        overlay = Image.fromarray(overlay_data)
        result = Image.alpha_composite(image, overlay)
        
        return result


    for kframe in obj_repr.kframes:
        rgb = Image.fromarray(kframe.rgb)
        obj_mask = ((kframe.dynamic_mask == STATIC_LABEL) | (kframe.dynamic_mask == MOVING_LABEL))
        result = mask_on_image(rgb, obj_mask)
        result.show()


if __name__ == "__main__":
    while True:
        idx = int(input("Input idx: "))
        try:
            vis_rr(idx)
        except Exception as e:
            print(e)
            continue