"""
读取一个 obj_repr, 然后对其进行绘制

"""
import os
import numpy as np
import open3d as o3d
from PIL import Image
from akm.representation.obj_repr import Obj_repr
from akm.representation.basic_structure import Frame
from akm.utility.proposal.affordance import Affordance_map_2d
from PIL import Image, ImageDraw, ImageFont
from akm.project_config import (
    BACKGROUND_LABEL,
    STATIC_LABEL,
    MOVING_LABEL,
    UNKNOWN_LABEL,
)
from akm.utility.utils import (
    draw_points_on_image,
    visualize_pc,
    depth_image_to_pointcloud,
    joint_data_to_transform_np
)
from graspnetAPI.grasp import Grasp

# mobile_pc_color = np.array([162, 223, 248]) / 255.
# static_pc_color = np.array([179, 173, 235]) / 255.

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
# z 軸
render_extrinsic[2, -1] = 0.1
zoom_in_scale = 2

def draw_points_on_image(image, uv_list, color_list=None, radius=None, normalized_uv=False):
    """
    Args:
        image: PIL.Image 对象, 或是一个 np.array。
        uv_list: 一个包含 (u, v) 坐标的列表,表示要绘制的点。
        返回一个 pil image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # 获取图像的宽度和高度
    width, height = image.size
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    for i, (u, v) in enumerate(uv_list):
        # 将归一化坐标转换为像素坐标
        if normalized_uv:
            x = int(u * width)
            y = int(v * height)
        else:
            x = int(u)
            y = int(v)
        
        # 在 (x, y) 位置画一个红色的点 (填充颜色为红色)
        if color_list is None:
            color = (255, 0, 0)
        else:
            color = tuple(color_list[i])
            
        if radius == None:
            draw.point((x, y), fill=color)  # (255, 0, 0) 表示红色
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            
    return image_draw

def add_text_to_image(image: Image.Image, text: str, font_scale: float = 0.05, text_color: tuple = (255, 255, 255), position_ratio: tuple = (0.05, 0.05)) -> Image.Image:
    """
    在图片上添加文本,字体大小和位置根据图像大小动态确定,使用 Times New Roman 字体
    
    Args:
        image: PIL Image对象
        text: 要添加的文本
        font_scale: 字体大小相对于图像高度的比例(默认0.05,即5%)
        text_color: 文本颜色,RGB元组格式 (R, G, B)
        position_ratio: 文本位置相对于图像宽度和高度的比例 (x_ratio, y_ratio),默认(0.05, 0.05)
    
    Returns:
        PIL Image对象,包含添加的文本
    """
    # 创建图片副本以避免修改原图
    new_image = image.copy()
    new_image = new_image.resize((3200, 2400), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(new_image)
    
    # 获取图像尺寸
    img_width, img_height = new_image.size
    
    # 动态计算字体大小(基于图像高度的比例)
    font_size = int(img_height * font_scale)
    
    # 尝试加载 Times New Roman 字体
    try:
        font_path = "/home/zby/Programs/Embodied_Analogy/scripts/times.ttf"  # 系统字体路径,例如 Windows: "C:/Windows/Fonts/times.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"无法加载 Times New Roman 字体: {e}, 使用默认字体")
        font = ImageFont.load_default()
    
    # 动态计算文字位置(基于图像尺寸的比例)
    position = (int(img_width * position_ratio[0]), int(img_height * position_ratio[1]))
    
    # 在指定位置绘制文本
    draw.text(position, text, font=font, fill=text_color)
    
    return new_image

def overlay_mask_on_image(image, mask, overlay_color=(255, 0, 0, 128), bbox_color=(0, 255, 0)):
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
    
    # Calculate bounding box
    mask_array = np.array(mask)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if rows.any() and cols.any():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Draw bounding box
        draw = ImageDraw.Draw(result)
        draw.rectangle((xmin, ymin, xmax, ymax), outline=bbox_color, width=5)
    
    return result

def farthest_point_sampling_2d(points, M):
    """
    Perform farthest point sampling on a 2D point set.
    
    Args:
        points: np.ndarray of shape (N, 2), input 2D points
        M: int, number of points to retain
    
    Returns:
        mask: np.ndarray of shape (N,), boolean mask where True indicates retained points
    """
    N = points.shape[0]
    if M > N or M <= 0:
        raise ValueError("M must be between 1 and N")
    
    # Initialize mask
    mask = np.zeros(N, dtype=bool)
    
    # Randomly select the first point
    first_idx = np.random.randint(0, N)
    mask[first_idx] = True
    
    # Compute all pairwise distances
    dist_matrix = np.sqrt(np.sum((points[:, np.newaxis] - points) ** 2, axis=2))
    
    # Track minimum distance from each point to selected points
    min_distances = dist_matrix[first_idx].copy()
    
    # Select M-1 more points
    for _ in range(M - 1):
        # Find point with maximum minimum distance to selected points
        next_idx = np.argmax(min_distances)
        mask[next_idx] = True
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, dist_matrix[next_idx])
    
    return mask

# 读取 obj_repr
# vis_idx = [3, 4, 7, 10, 78, 95, 97, 114]
vis_idx = 3
load_path = f"/home/zby/Programs/Embodied_Analogy/assets/logs/6_39/{vis_idx}/obj_repr.npy"
save_folder = f"/home/zby/Programs/Embodied_Analogy/scripts/draw_figs/paper_figs/pipeline_{vis_idx}"
os.makedirs(save_folder, exist_ok=True)
obj_repr: Obj_repr = Obj_repr.load(load_path)

visualize_explore = False
if visualize_explore:
    num_tries = len(obj_repr.save_for_vis["explore_cos_map"])
    # 绘制多次 explore 的首尾帧
    for i in range(1, num_tries):
        explore_first_frame, explore_last_frame = obj_repr.save_for_vis[str(i)]
        Image.fromarray(explore_first_frame.rgb).save(os.path.join(save_folder, f"{i}_first.png"))
        Image.fromarray(explore_last_frame.rgb).save(os.path.join(save_folder, f"{i}_last.png"))
        
    # 绘制 valid explore 的首尾帧
    Image.fromarray(obj_repr.frames[0].rgb).save(os.path.join(save_folder, f"{num_tries}_first.png"))
    Image.fromarray(obj_repr.frames[-1].rgb).save(os.path.join(save_folder, f"{num_tries}_last.png"))

    # 绘制 explore 最开始的 initial_frame
    initial_frame = obj_repr.initial_frame
    Image.fromarray(obj_repr.initial_frame.rgb).save(os.path.join(save_folder, f"initial_frame.png"))

    obj_repr.initial_frame.segment_obj(
        obj_description=obj_repr.obj_description,
        post_process_mask=True,
        filter=True,
        visualize=False,
    )
    overlay_mask_on_image(Image.fromarray(obj_repr.initial_frame.rgb), obj_repr.initial_frame.obj_mask, overlay_color=(230, 100, 110, 128), bbox_color=(10, 120, 40)).save(os.path.join(save_folder, f"initial_frame_gsam.png"))
    # 绘制 cos_map
    cos_map_list = obj_repr.save_for_vis["explore_cos_map"] # list of np.array([H, W])
    cropped_mask = obj_repr.save_for_vis["aff_map"].cropped_mask
    cropped_region = obj_repr.save_for_vis["aff_map"].cropped_region

    for i in range(len(cos_map_list)):
        tmp_cos_map = cos_map_list[i]
        aff_2d = Affordance_map_2d(
            rgb_img=obj_repr.initial_frame.rgb,
            cos_map=tmp_cos_map, 
            cropped_mask=cropped_mask,
            cropped_region=cropped_region
        )
        
        aff_2d.mask_cos_map()
        max_index = np.unravel_index(np.argmax(tmp_cos_map), tmp_cos_map.shape)
        v_cos, u_cos = max_index
        image_cos = aff_2d.get_colored_cos_map()
        image_cos = draw_points_on_image(
            image=image_cos,
            uv_list=[(u_cos, v_cos)],
            radius=5,
            normalized_uv=False
        )
        image_cos.save(os.path.join(save_folder, f"cos_map_{i+1}.png"))

    # 绘制 valid explore 的 tracking result
    num_tracks = obj_repr.frames.track2d_seq.shape[1]
    # import torch
    # random_track_idx = torch.randperm(obj_repr.frames.track2d_seq[0].shape[0])[:200]
    
    random_track_idx = farthest_point_sampling_2d(obj_repr.frames.track2d_seq[0].numpy(), 200)
    track_colors = colors = np.random.randint(0, 256, size=(num_tracks, 3), dtype=np.uint8)
    draw_points_on_image(
        image=obj_repr.frames[0].rgb,
        uv_list=obj_repr.frames.track2d_seq[0][random_track_idx, :],
        color_list=track_colors,
        radius=3,
    ).save(os.path.join(save_folder, f"track_start.png"))

    draw_points_on_image(
        image=obj_repr.frames[-1].rgb,
        uv_list=obj_repr.frames.track2d_seq[-1][random_track_idx, :],
        color_list=track_colors,
        radius=3,
    ).save(os.path.join(save_folder, f"track_end.png"))

# 绘制 reconstruct
# 绘制 kframes 的 dynamic mask
visualize_coarse = True
if visualize_coarse:
    for i, kf in enumerate(obj_repr.kframes):
        kf_rgb = kf.rgb
        dynamic_mask = kf.dynamic_mask
        alpha = 0.4
        kf_rgb[dynamic_mask == MOVING_LABEL] = mobile_pc_color * 255 * alpha + (1-alpha) * np.array([255, 255, 255])
        kf_rgb[dynamic_mask == STATIC_LABEL] = gray_color * 255 * alpha + (1-alpha) * np.array([255, 255, 255])
        kf_rgb[(dynamic_mask!=MOVING_LABEL) & (dynamic_mask!=STATIC_LABEL)] = 255
        kf_pil = Image.fromarray(kf_rgb.astype(np.uint8)) 
        kf_pil = add_text_to_image(
            image=kf_pil,
            text=f"{i}th Keyframe\nJoint State: {kf.joint_state * 100:.2f} cm",
            text_color = (0, 0, 0), 
            font_scale=0.1
        )
        kf_pil.save(os.path.join(save_folder, f"kframe_{i}.png"))
    
# coarse 阶段绘制一个 manipulate first frame + 3D tracks + joint_axis
    obj_pc_s, pc_colors_s = obj_repr.frames[0].get_obj_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=False,
        visualize=False
    )
    pc_colors_s = pc_colors_s / 255

    obj_pc_e, pc_colors_e = obj_repr.frames[-1].get_env_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=False,
        visualize=False
    )
    pc_colors_e = pc_colors_e / 255

    # tracks_3d_colors = obj_repr.frames.moving_mask[:, None] * np.array([[0, 1, 0]]) + ~obj_repr.frames.moving_mask[:, None] * np.array([[1, 0, 0]])

    coarse_jonint_dict = obj_repr.get_joint_param(
        resolution="coarse",
        frame="camera"
    )

    # obj_pc = np.concatenate([obj_pc_s, obj_pc_e])
    # pc_colors = np.concatenate([pc_colors_s, pc_colors_e])
    moving_track_mask = obj_repr.frames.moving_mask
    coarse_image = visualize_pc(
        points=[obj_pc_s, obj_pc_e],
        point_size=[5, 5],
        # voxel_size=0.005,
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
        online_viewer=False
    )
    Image.fromarray(coarse_image).save(os.path.join(save_folder, "coarse_image.png"))
    
    coarse_image_start = visualize_pc(
        points=[obj_pc_s],
        point_size=[5],
        voxel_size=None,
        colors=[gray_color],
        alpha=[0.6],
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
        online_viewer=False
    )
    Image.fromarray(coarse_image_start).save(os.path.join(save_folder, "coarse_image_start.png"))

    coarse_image_end = visualize_pc(
        points=[obj_pc_e],
        point_size=[5],
        voxel_size=None,
        colors=[gray_color],
        alpha=[0.6],
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
        online_viewer=False
    )
    Image.fromarray(coarse_image_end).save(os.path.join(save_folder, "coarse_image_end.png"))

# fine 阶段绘制一个 kframes[0] 和 kframes[-1] 的 mobile part + joint axis
visualize_fine = False
if visualize_fine:
    kframes_start_mobile_pc = depth_image_to_pointcloud(
        obj_repr.kframes[0].depth,
        obj_repr.kframes[0].dynamic_mask == MOVING_LABEL,
        obj_repr.kframes[0].K,
    )
    kframes_start_static_pc = depth_image_to_pointcloud(
        obj_repr.kframes[0].depth,
        obj_repr.kframes[0].dynamic_mask == STATIC_LABEL,
        obj_repr.kframes[0].K,
    )
    kframes_end_mobile_pc = depth_image_to_pointcloud(
        obj_repr.kframes[-1].depth,
        obj_repr.kframes[-1].dynamic_mask == MOVING_LABEL,
        obj_repr.kframes[-1].K,
    )
    kframes_end_static_pc = depth_image_to_pointcloud(
        obj_repr.kframes[-1].depth,
        obj_repr.kframes[-1].dynamic_mask == STATIC_LABEL,
        obj_repr.kframes[-1].K,
    )
    mobile_mask = np.zeros(kframes_start_static_pc.shape[0] + kframes_start_mobile_pc.shape[0] + kframes_end_mobile_pc.shape[0])
    mobile_mask[kframes_start_static_pc.shape[0]:] = 1
    
    fine_jonint_dict = obj_repr.get_joint_param(
        resolution="fine",
        frame="camera"
    )
    
    fine_image = visualize_pc(
        points=[kframes_start_static_pc, kframes_start_mobile_pc, kframes_end_mobile_pc],
        colors=[gray_color, mobile_pc_color, mobile_pc_color],
        voxel_size=None, 
        point_size=[5, 5, 5],
        alpha=[0.6, 0.8, 0.8],
        pivot_point=fine_jonint_dict["joint_start"],
        joint_axis=fine_jonint_dict["joint_dir"],
        camera_intrinsic=render_intrinsic,
        camera_extrinsic=render_extrinsic,
        zoom_in_scale=zoom_in_scale,
        online_viewer=False
    )
    Image.fromarray(fine_image).save(os.path.join(save_folder, "fine_image.png"))
    
    fine_image_start = visualize_pc(
        points=[kframes_start_static_pc, kframes_start_mobile_pc],
        colors=[gray_color, mobile_pc_color],
        voxel_size=None, 
        point_size=[5, 5],
        alpha=[0.6, 0.6],
        pivot_point=fine_jonint_dict["joint_start"],
        joint_axis=fine_jonint_dict["joint_dir"],
        camera_intrinsic=render_intrinsic,
        camera_extrinsic=render_extrinsic,
        zoom_in_scale=zoom_in_scale,
        online_viewer=False
    )
    Image.fromarray(fine_image_start).save(os.path.join(save_folder, "fine_image_start.png"))
    
    fine_image_end = visualize_pc(
        points=[kframes_start_static_pc, kframes_end_mobile_pc],
        colors=[gray_color, mobile_pc_color],
        voxel_size=None, 
        point_size=[5, 5],
        alpha=[0.6, 0.6],
        pivot_point=fine_jonint_dict["joint_start"],
        joint_axis=fine_jonint_dict["joint_dir"],
        camera_intrinsic=render_intrinsic,
        camera_extrinsic=render_extrinsic,
        zoom_in_scale=zoom_in_scale,
        online_viewer=False
    )
    Image.fromarray(fine_image_end).save(os.path.join(save_folder, "fine_image_end.png"))
    
# 绘制 manipulate
visualize_manipulate = True
if visualize_manipulate:
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            task_idx = f"{i}_{j}"
            os.makedirs(os.path.join(save_folder, task_idx), exist_ok=True)
            for k in range(len(obj_repr.save_for_vis[task_idx])):
                joint_type = obj_repr.fine_joint_dict["joint_type"]
                if joint_type == "revolute":
                    joint_state = np.rad2deg(obj_repr.save_for_vis[task_idx][k].gt_joint_state)
                else:
                    joint_state = obj_repr.save_for_vis[task_idx][k].gt_joint_state * 100
                
                prefix = "Init" if k == 0 else "End"
                texted_img = add_text_to_image(
                    image=Image.fromarray(obj_repr.save_for_vis[task_idx][k].rgb),
                    font_scale=0.1,
                    text=f"{prefix} Joint State: {joint_state:.2f} degree", 
                    text_color=(255, 255, 255)
                )
                texted_img.save(os.path.join(save_folder, task_idx, f"manip_{k}.png"))
            # continue
            if i == 1 and j == 3:
                manip_first_frame = obj_repr.save_for_vis[task_idx][0]
                cur_state = manip_first_frame.joint_state
                target_state = manip_first_frame.target_state
                
                manip_first_frame_mobile_pc = depth_image_to_pointcloud(
                    manip_first_frame.depth,
                    manip_first_frame.dynamic_mask == MOVING_LABEL,
                    manip_first_frame.K,
                )
                fine_jonint_dict = obj_repr.get_joint_param(
                    resolution="fine",
                    frame="camera"
                )
                translated_mobile_pc = manip_first_frame_mobile_pc + fine_jonint_dict["joint_dir"] * (target_state - cur_state)
                manip_first_frame: Frame
                manip_first_frame.segment_obj(
                    obj_description=obj_repr.obj_description,
                    post_process_mask=True,
                    filter=True,
                    visualize=False,
                )
                manip_first_frame.obj_mask = manip_first_frame.obj_mask & (~manip_first_frame.robot_mask)
                
                manip_first_frame_static_pc = depth_image_to_pointcloud(
                    manip_first_frame.depth,
                    manip_first_frame.obj_mask & ~(manip_first_frame.dynamic_mask == MOVING_LABEL),
                    manip_first_frame.K,
                )
                
                mobile_mask = np.zeros(manip_first_frame_static_pc.shape[0] + manip_first_frame_mobile_pc.shape[0])
                mobile_mask[manip_first_frame_static_pc.shape[0]:] = 1
                
                fine_jonint_dict = obj_repr.get_joint_param(
                    resolution="fine",
                    frame="camera"
                )
                
                manip_reloc = visualize_pc(
                    points=[manip_first_frame_static_pc, manip_first_frame_mobile_pc],
                    colors=[gray_color, mobile_pc_color],
                    point_size=[5, 5],
                    voxel_size=None,
                    alpha=[0.6, 0.8],
                    camera_intrinsic=render_intrinsic,
                    camera_extrinsic=render_extrinsic,
                    zoom_in_scale=zoom_in_scale,
                    online_viewer=False
                )
                manip_reloc = Image.fromarray(manip_reloc)
                manip_reloc = add_text_to_image(
                    image=manip_reloc,
                    text = f"Relocalized State: {cur_state * 100:.2f} cm", 
                    text_color = (0, 0, 0), 
                    font_scale=0.1,
                )
                manip_reloc.save(os.path.join(save_folder, "manip_reloc.png"))
                
                manip_terget = visualize_pc(
                    points=[manip_first_frame_static_pc, translated_mobile_pc],
                    colors=[gray_color, mobile_pc_color],
                    point_size=[5, 5],
                    alpha=[0.6, 0.8],
                    voxel_size=None,
                    camera_intrinsic=render_intrinsic,
                    camera_extrinsic=render_extrinsic,
                    zoom_in_scale=zoom_in_scale,
                    online_viewer=False
                )
                manip_terget = Image.fromarray(manip_terget)
                manip_terget = add_text_to_image(
                    image=manip_terget,
                    text= f"Target State: {target_state * 100:.2f} cm", 
                    text_color = (0, 0, 0), 
                    font_scale=0.1,
                )
                manip_terget.save(os.path.join(save_folder, "manip_terget.png"))
                
                # 最后绘制 grasp Trajctory
                def transfer_Tph2w(obj_repr, Tph2w_ref, ref_state, tgt_state):
                    """
                    将 Tph2w 从 ref_state 转换到 tgt_state
                    """
                    Tph2c_ref = obj_repr.Tw2c @ Tph2w_ref
                    Tref2tgt_c = joint_data_to_transform_np(
                        joint_type=obj_repr.fine_joint_dict["joint_type"],
                        joint_dir=obj_repr.fine_joint_dict["joint_dir"],
                        joint_start=obj_repr.fine_joint_dict["joint_start"],
                        joint_state_ref2tgt=tgt_state-ref_state
                    )
                    Tph2c_tgt = Tref2tgt_c @ Tph2c_ref
                    Tph2w_tgt = np.linalg.inv(obj_repr.Tw2c) @ Tph2c_tgt
                    return Tph2w_tgt
                
                def inverse_anyGrasp2ph(Tph2w):
                    """
                        Convert Tph2w to Tgrasp2w (inverse of anyGrasp2ph)
                        Tph2w: 4x4 homogeneous transformation matrix from panda_hand to world
                        Returns: Tgrasp2w, 4x4 homogeneous transformation matrix from grasp to world
                    """
                    def Tph2grasp(offset):
                        # Inverse of Tph2grasp from original function
                        T = np.array([
                            [0, 0, 1, -(0.045 + 0.069) + offset],
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 1]
                        ])
                        return np.linalg.inv(T)

                    # Use the same offset as in the original function
                    Tgrasp2ph = Tph2grasp(0.014)
                    Tgrasp2w = Tph2w @ Tgrasp2ph
                    return Tgrasp2w
                
                Initial_Tph2w = obj_repr.frames[0].Tph2w
                Tph2w_draws = [transfer_Tph2w(obj_repr, Initial_Tph2w, 0, alpha * target_state + (1-alpha) * cur_state) for alpha in np.linspace(0, 1, 3)]
                Tgrasp2w_draws = [inverse_anyGrasp2ph(Tph2w) for Tph2w in Tph2w_draws]
                Tgrasp2c_draws = [obj_repr.Tw2c @ Tgrasp2w for Tgrasp2w in Tgrasp2w_draws]
                Grasps = []
                for Tgrasp2c_draw in Tgrasp2c_draws:
                    g_array = [0.1, 0.02, 0.05, 0.00] + Tgrasp2c_draw[:3, :3].reshape(-1).tolist() + Tgrasp2c_draw[:3, -1].tolist() + [0]
                    G = Grasp(np.array(g_array))
                    Grasps.append(G)
                
                from akm.utility.grasp.anygrasp import detect_grasp_anygrasp
                new_grasp = detect_grasp_anygrasp(
                    points=np.concatenate([manip_first_frame_static_pc, manip_first_frame_mobile_pc]), 
                    colors=mobile_pc_color[None] * mobile_mask[:, None] + static_pc_color[None] * (1 - mobile_mask[:, None]),
                    dir_out=np.array([0, 0, -1]), 
                    augment=True,
                    visualize=False, # still have bug visualize this
                    logger=None
                )  
                planned_image = visualize_pc(
                    points=[manip_first_frame_static_pc, manip_first_frame_mobile_pc, translated_mobile_pc],
                    colors=[gray_color, mobile_pc_color, mobile_pc_color],
                    alpha=[0.6, 0.8, 0.8],
                    point_size=[5, 5, 5],
                    voxel_size=None,
                    grasp=Grasps,
                    camera_intrinsic=render_intrinsic,
                    camera_extrinsic=render_extrinsic,
                    zoom_in_scale=zoom_in_scale,
                    online_viewer=False
                )
                planned_image = Image.fromarray(planned_image)
                planned_image = add_text_to_image(
                    image=planned_image,
                    text= f"{cur_state * 100:.2f} cm -> {target_state * 100:.2f} cm", 
                    text_color = (0, 0, 0), 
                    font_scale=0.1,
                )
                planned_image.save(os.path.join(save_folder, "planned_image.png"))
                
                
                planned_image_start = visualize_pc(
                    points=[manip_first_frame_static_pc, manip_first_frame_mobile_pc],
                    colors=[gray_color, mobile_pc_color],
                    point_size=[5, 5],
                    voxel_size=None,
                    alpha=[0.6, 0.8],
                    grasp=Grasps,
                    camera_intrinsic=render_intrinsic,
                    camera_extrinsic=render_extrinsic,
                    zoom_in_scale=zoom_in_scale,
                    online_viewer=False
                )
                planned_image_start = Image.fromarray(planned_image_start)
                planned_image_start = add_text_to_image(
                    image=planned_image_start,
                    text= f"Plan from {cur_state * 100:.2f} cm to {target_state * 100:.2f} cm", 
                    text_color = (0, 0, 0), 
                    font_scale=0.1,
                )
                planned_image_start.save(os.path.join(save_folder, "planned_image_start.png"))
                
                planned_image_end = visualize_pc(
                    points=[manip_first_frame_static_pc, translated_mobile_pc],
                    colors=[gray_color, mobile_pc_color],
                    point_size=[5, 5],
                    voxel_size=None,
                    alpha=[0.6, 0.8],
                    grasp=Grasps,
                    camera_intrinsic=render_intrinsic,
                    camera_extrinsic=render_extrinsic,
                    zoom_in_scale=zoom_in_scale,
                    online_viewer=False
                )
                planned_image_end = Image.fromarray(planned_image_end)
                planned_image_end = add_text_to_image(
                    image=planned_image_end,
                    text= f"Plan from {cur_state * 100:.2f} cm to {target_state * 100:.2f} cm", 
                    text_color = (0, 0, 0), 
                    font_scale=0.1,
                )
                planned_image_end.save(os.path.join(save_folder, "planned_image_end.png"))
                