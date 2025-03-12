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
from run_realworld.utils import crop_points, cluster_normals
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use

from embodied_analogy.exploration.affordance import Affordance_map_2d
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import (
    seed_everything,
    image_to_camera,
    depth_image_to_pointcloud,
    compute_normals,
    visualize_pc,
    normalize_cos_map_exp,
    draw_points_on_image,
)
# from embodied_analogy.perception.grounded_sam import run_grounded_sam

from embodied_analogy.grasping.anygrasp import (
    detect_grasp_anygrasp,
    sort_grasp_group
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
    save_root="/home/zby/Programs/Embodied_Analogy/assets/tmp/explore",
    visualize=False
):
    """
        instruction: open the drawer (task description)
        prompt: used to extract dift feature
    """
    seed_everything(SEED)

    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir="/home/zby/Programs/RAM_code/assets/data",
        save_root=save_root,
        lang_mode='clip',
        topk=5, 
        crop=True, # 这里记得用 crop=False，如果用 crop=True 的话输出的 contact_point 是相对于 cropped_tgt image 的
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
    
    # 这些 query 都是 cropped 之后的结果
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
    
    if visualize:
        # 用 napari 一直有 bug, 草拟吗
        # import napari
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

def get_ram_affordance_3d():
    pass

def lift_ram_affordance(
    K, 
    query_rgb, 
    query_mask, # H, W
    query_depth, 
    contact_uv, 
    contact_dir_2d, 
    visualize=False
):
    # TODO: 检查坐标系 + 写一个函数注释
    """
        rgb_image: numpy.array, (H, W, 3)
        point_cloud: H*W, 3
        contact_uv: (2, )
        contact_dir_2d: (2, )
        
    """
    H, W = query_rgb.shape[:2]
    contact_3d = image_to_camera(
        uv=contact_uv[None],
        depth=np.array(query_depth[contact_uv[1], contact_uv[0]])[None],
        K=K,
    )[0] # 3
    
    # 找到 contact_uv 附近区域的点云
    crop_radius = 100 # in pixels
    partial_mask = np.zeros((H, W), dtype=np.bool_)
    partial_mask[contact_uv[1] - crop_radius:contact_uv[1] + crop_radius, contact_uv[0] - crop_radius:contact_uv[0] + crop_radius] = True
    partial_mask = partial_mask & query_mask & (query_depth > 0)
    
    partial_points = depth_image_to_pointcloud(query_depth, partial_mask, K) # N, 3
    partial_colors = query_rgb[partial_mask]  # N, 3
    
    # 选取 post-grasp-direction, 对于 partial_point 的区域求一个 normal
    partial_normals = compute_normals(partial_points) # N, 3
    n_clusters = 5
    clustered_centers = cluster_normals(partial_normals, n_clusters=n_clusters) # (2*n_clusters, 3)
    post_contact_dirs_3d = clustered_centers # N, 3
    post_contact_dirs_2d = project_normals(query_rgb, contact_uv, clustered_centers, visualize=False)
    
    # if visualize:
    #     visualize_pc(partial_points, partial_colors / 255., contact_point=contact_3d, post_contact_dirs=post_contact_dirs_3d)
        
    # find clustered_dir that best align with dir
    best_dir_3d, best_score = None, -1
    for i in range(post_contact_dirs_2d.shape[0]):
        score = np.dot(post_contact_dirs_2d[i], contact_dir_2d)
        if score > best_score:
            best_score = score
            best_dir_3d = post_contact_dirs_3d[i]
    
    # if visualize:
    #     visualize_pc(partial_points, partial_colors / 255., contact_point=contact_3d, post_contact_dirs=best_dir_3d[None])
    # TODO: 这里的 partial points是在相机坐标系下的, 需要修改
    gg = detect_grasp_anygrasp(
        points=partial_points, 
        colors=partial_colors / 255.,
        dir_out=best_dir_3d, 
        visualize=True
    ) 
        
    sorted_grasps, _ = sort_grasp_group(
        grasp_group=gg,
        contact_region=contact_3d[None],
        # axis=np.array([0, 0, -1])
    )
    best_grasp = gg[0]

    if visualize:
        visualize_pc(
            points=partial_points, colors=partial_colors / 255., grasp=best_grasp, 
            contact_point=contact_3d, post_contact_dirs=best_dir_3d[None]
        )
        
    return sorted_grasps, best_dir_3d


if __name__ == "__main__":
    import sys
    query_rgb = np.asarray(Image.open("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/rgb.png"))
    query_depth = np.load("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/depth.npy")
    # query_mask = np.load("/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/mask.npy")
    contact_point_2d, post_contact_dir_2d, query_mask = get_ram_proposal(
        query_rgb, # H, W, 3 in numpy
        instruction="open the drawer",
        # prompt="a photo of a drawer", 
        data_source="droid",
        save_root="/home/zby/Programs/Embodied_Analogy/assets/tmp/explore",
        visualize=False
    )
    
    # contact_point_2d = np.array([455, 154])
    # post_contact_dir_2d = np.array([0.64614357, 0.76321588])
    
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
    
    # sys.exit()
    best_grasp, best_dir_3d = lift_ram_affordance(
        K=K,
        query_rgb=query_rgb,
        query_mask=query_mask,
        query_depth=query_depth,
        contact_uv=contact_point_2d,
        contact_dir_2d=post_contact_dir_2d,
        visualize=True
    )