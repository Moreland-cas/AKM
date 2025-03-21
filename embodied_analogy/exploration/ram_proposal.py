from embodied_analogy.utility.utils import initialize_napari
initialize_napari()

import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/RAM_code")

import cv2
import numpy as np
import torch
import time
from PIL import Image

from vision.featurizer.run_featurizer import match_fts
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline import SubsetRetrievePipeline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use

from embodied_analogy.exploration.affordance import Affordance_map_2d
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    seed_everything,
    image_to_camera,
    depth_image_to_pointcloud,
    # compute_normals,
    visualize_pc,
    draw_points_on_image,
    fit_plane_normal,
    fit_plane_ransac,
    crop_nearby_points,
    get_depth_mask
)
# from embodied_analogy.perception.grounded_sam import run_grounded_sam

from embodied_analogy.grasping.anygrasp import (
    detect_grasp_anygrasp,
    sort_grasp_group,
    crop_grasp
)

def draw_arrows_on_img(img, pixel, normals_2d_directions):
    """
        img: H, W, 3, numpy array
        pixel: x, y coordinate on image plane
        normals_2d_directions: N, 2
    """
    img_ = np.asarray(img).copy() # H, W, C
    num_array = len(normals_2d_directions)
    for i in range(num_array):
        x, y = pixel
        dx, dy = normals_2d_directions[i]
        cv2.arrowedLine(img_, (x, y), (x + int(dx * 50), y + int(dy * 50)), (0, 0, 255 * i / num_array), 2)
        plt.arrow(x, y, dx * 100, dy * 100, color=(0, 0, 1 - i / num_array), linewidth=2.5, head_width=12)
    # revert to RGB
    # img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    return Image.fromarray(img_)
        
def project_normals(img, pixel, normals, visualize=False):
    '''
    project 3D normals into 2D space
        img: PIL Image
        normals: [n, 3], n is the number of normals, in camera frame
    return:
        normals projected in 2d
    '''
    normals = normals[:, :2]  # Ignore Z component for direction because in camera frame
    normals_2d_direction = torch.from_numpy(normals).float().cuda()

    # Optionally, normalize the 2D vectors to have unit length, showing direction only
    norms = torch.norm(normals_2d_direction, dim=1, keepdim=True)
    normals_2d_direction_normalized = normals_2d_direction / norms
    normals_2d_direction_normalized = normals_2d_direction_normalized.cpu().numpy()

    # visualize 2D normal on the image
    if visualize:
        draw_img = draw_arrows_on_img(img, pixel, normals_2d_direction_normalized)
        draw_img.show()

    return normals_2d_direction_normalized
    
def get_ram_affordance_2d(
    query_rgb, # H, W, 3 in numpy
    instruction, # open the drawwer
    data_source, # droid TODO: 把 data_source 扩充一下
    visualize=False
):
    """
        instruction: open the drawer (task description)
        prompt: used to extract dift feature
    """
    seed_everything(SEED)

    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir="/home/zby/Programs/RAM_code/assets/data",
        lang_mode='clip',
        topk=5, 
        crop=True, 
        data_source=data_source,
    )
    
    # use retrieval to get ref_path (or ref image) and ref trajectory in 2d space
    retrieve_start = time.time()
    topk_retrieved_data_dict = subset_retrieve_pipeline.retrieve(
        current_task=instruction,
        current_obs=query_rgb,
        log=True,
        visualize=False
    )
    retrieve_end = time.time()
    print(f"retrieve time: {retrieve_end - retrieve_start}")
    
    ref_trajs = topk_retrieved_data_dict['traj']
    ref_imgs_np = topk_retrieved_data_dict['masked_img']
    ref_imgs_PIL = [Image.fromarray(ref_img_np).convert('RGB') for ref_img_np in ref_imgs_np]
    
    # Note: 这些 query 都是 cropped 之后的结果
    query_mask = topk_retrieved_data_dict["query_mask"][..., 0].astype(np.bool_)
    query_feat = topk_retrieved_data_dict["query_feat"]
    query_region = topk_retrieved_data_dict["query_region"]

    # transfer contact point, cos_map is np.array([cropped_H, cropped_W])    
    cos_map = None
    for i, ref_img_PIL in enumerate(ref_imgs_PIL):
        xy = ref_trajs[i][0]
        # scale cropped_traj to IMG_SIZE, because diff-transfer are done are image of size IMG_SIZE
        ref_pos = (xy[0] * IMG_SIZE / ref_img_PIL.size[0], xy[1] * IMG_SIZE / ref_img_PIL.size[1])
        ref_ft = topk_retrieved_data_dict["feat"][i]
        cos_map = match_fts(ref_ft, query_feat, ref_pos)
        break

    # 进一步将 cos_map 转换为一个概率分布
    affordance_map_2d = Affordance_map_2d(
        rgb_img=query_rgb,
        cos_map=cos_map,
        cropped_mask=query_mask,
        cropped_region=query_region,
    )
    # 保存 affordance map 2d 的输入方便后续 debug
    if False:
        np.savez(
            "/home/zby/Programs/Embodied_Analogy/assets/unit_test/ram_proposal/affordance_map_2d_input.npz",
            rgb_img=query_rgb,
            cos_map=cos_map,
            cropped_mask=query_mask,
            cropped_region=query_region,
        )
    
    if visualize:
        # viewer = napari.Viewer()
        # viewer.add_image(topk_retrieved_data_dict["query_img"], name="query_img")
        # viewer.add_image(topk_retrieved_data_dict["query_mask"] * 255, name="query_mask")
        # viewer.add_image(topk_retrieved_data_dict["masked_query"], name="masked_query")

        # viewer.title = "retrieved reference data by RAM"
        
        # for i in range(len(topk_retrieved_data_dict["img"])):
        #     viewer.add_image(topk_retrieved_data_dict["img"][i], name=f"ref_img_{i}")
        #     masked_img = topk_retrieved_data_dict["masked_img"][i]
        #     masked_img = np.array(draw_points_on_image(masked_img, [topk_retrieved_data_dict["traj"][i][0]], 5))
        #     viewer.add_image(masked_img, name=f"masked_ref_img_{i}")
        #     viewer.add_image(topk_retrieved_data_dict["mask"][i] * 255, name=f"ref_img_mask_{i}")
        #     # viewer.add_image(prob_maps[i], name=f"prob_map_{i}", colormap="viridis")
        # napari.run()
        
        Image.fromarray(topk_retrieved_data_dict["query_img"]).show()
        Image.fromarray(topk_retrieved_data_dict["query_mask"] * 255).show()
        Image.fromarray(topk_retrieved_data_dict["masked_query"]).show()
        
        for i in range(len(topk_retrieved_data_dict["img"])):
            Image.fromarray(topk_retrieved_data_dict["img"][i]).show()
            masked_img = topk_retrieved_data_dict["masked_img"][i]
            masked_img = draw_points_on_image(masked_img, [topk_retrieved_data_dict["traj"][i][0]], 5)
            masked_img.show()
            Image.fromarray(topk_retrieved_data_dict["mask"][i] * 255).show()
            # Image.fromarray((cos_maps[i] + 1) / 2 * 255).show()
            break
        
    return affordance_map_2d

def lift_ram_affordance(
    K, 
    Tw2c,
    query_rgb,
    query_mask,
    query_depth, 
    contact_uv,
    visualize=False
):
    contact_u = int(contact_uv[0])
    contact_v = int(contact_uv[1])
    
    contact_3d_c = image_to_camera(
        uv=np.array(contact_uv)[None], # 1, 3
        depth=np.array(query_depth[contact_v, contact_u])[None], # 1, 1
        K=K,
    )[0] # 3
    
    # 找到 contact_3d_c 附近区域的点云
    query_depth_mask = get_depth_mask(query_depth, K, Tw2c)
    obj_pc_c = depth_image_to_pointcloud(query_depth, query_mask & query_depth_mask, K) # N, 3
    pc_colors = query_rgb[query_mask & query_depth_mask]  # N, 3
    
    # 找到 obj_pc_c 中 contact_3d_c 附近的点, 拟合一个平面, 返回法向量 dir_in 和 dir_out
    cropped_points = crop_nearby_points(
        point_clouds=obj_pc_c,
        contact_3d=contact_3d_c,
        radius=0.1
    )
    # plane_normal = fit_plane_normal(cropped_points)
    plane_normal = fit_plane_ransac(
        points=cropped_points,
        threshold=0.01, 
        max_iterations=100,
        visualize=visualize
    )
    
    if (plane_normal * np.array([0, 0, -1])).sum() > 0:
        dir_out = plane_normal
    else:
        dir_out = -plane_normal
    
    # 这里返回的是 Tgrasp2c 的
    gg = detect_grasp_anygrasp(
        points=obj_pc_c, 
        colors=pc_colors / 255.,
        dir_out=dir_out, 
        augment=True,
        visualize=False
    ) 
    # 使得返回的 grasp 是在一定区间内的, 超出的不算
    cropped_gg = crop_grasp(
        grasp_group=gg,
        contact_point=contact_3d_c,
        radius=0.1,
    )
    if cropped_gg is None:
        return contact_3d_c, None, dir_out

    sorted_grasps, _ = sort_grasp_group(
        grasp_group=cropped_gg,
        contact_region=contact_3d_c[None],
        # axis=np.array([0, 0, -1])
    )
    # best_grasp = sorted_grasps[0]

    if visualize:
        visualize_pc(
            points=obj_pc_c, 
            colors=pc_colors / 255,
            grasp=sorted_grasps, 
            contact_point=contact_3d_c, 
            post_contact_dirs=[dir_out]
        )
        
    return contact_3d_c, sorted_grasps, dir_out


if __name__ == "__main__":
    import sys
    query_rgb = np.asarray(Image.open("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/rgb.png"))
    query_depth = np.load("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/depth.npy")
    # query_mask = np.load("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/mask.npy")
    
    affordance_map_2d = get_ram_affordance_2d(
        query_rgb, # H, W, 3 in numpy
        instruction="open the drawer",
        data_source="droid",
        visualize=False
    )
    
    K = np.array(
        [[300.,   0., 400.],
       [  0., 300., 300.],
       [  0.,   0.,   1.]], dtype=np.float32)
    
    Tw2c = np.array(
        [[-4.8380834e-01, -8.7517393e-01,  5.1781535e-07,  4.5627734e-01],
       [-1.6098598e-01,  8.8994741e-02, -9.8293614e-01,  3.8961503e-01],
       [ 8.6024004e-01, -4.7555280e-01, -1.8394715e-01,  8.8079178e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
    )
    Rw2c = Tw2c[:3, :3]
    
    input_data = np.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/ram_proposal/affordance_map_2d_input.npz")
    # 测试一下 Affordance_map_2d
    # affordance_map_2d = Affordance_map_2d(
    #     rgb_img=input_data["rgb_img"],
    #     cos_map=input_data["cos_map"],
    #     cropped_mask=input_data["cropped_mask"],
    #     cropped_region=input_data["cropped_region"],
    # )
    
    obj_mask = affordance_map_2d.get_obj_mask(visualize=False) # H, W
    contact_uv = affordance_map_2d.sample_highest(visualize=True)
    _, best_grasp, dir_out = lift_ram_affordance(
        K=K, 
        Tw2c=Tw2c,
        query_rgb=query_rgb,
        query_depth=query_depth, 
        query_mask=obj_mask,
        contact_uv=contact_uv,
        visualize=True
    )
    
    affordance_map_2d.update(contact_uv, visualize=False)
    contact_uv = affordance_map_2d.sample_highest(visualize=False)
    _, best_grasp, dir_out = lift_ram_affordance(
        K=K, 
        Tw2c=Tw2c,
        query_rgb=query_rgb,
        query_depth=query_depth, 
        query_mask=obj_mask,
        contact_uv=contact_uv,
        visualize=True
    )
    