import os
import sys
import cv2
import time
import torch
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

relative_path = os.path.join(PROJECT_ROOT, "third_party", "RAM_code")
sys.path.append(relative_path)
from vision.featurizer.run_featurizer import match_fts
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline import SubsetRetrievePipeline

from akm.utility.constants import *
from akm.utility.utils import draw_points_on_image
from akm.utility.proposal.affordance import Affordance_map_2d


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
    
    
@torch.no_grad()
def get_ram_affordance_2d(
    query_rgb, # H, W, 3 in numpy
    instruction, # open the drawwer
    obj_description, # drawer
    fully_zeroshot=False,
    visualize=False,
    logger=None
):
    """
    instruction: open the drawer (task description)
    prompt: used to extract dift feature
    """
    logger.log(logging.INFO, "Initializing SubsetRetrievePipeline ...")
    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir=os.path.join(ASSET_PATH, "RAM_memory"),
        lang_mode='clip',
        topk=5, 
        crop=True, 
        fully_zeroshot=fully_zeroshot
    )
    
    # use retrieval to get ref_path (or ref image) and ref trajectory in 2d space
    retrieve_start = time.time()
    topk_retrieved_data_dict = subset_retrieve_pipeline.retrieve(
        current_task=instruction,
        current_obs=query_rgb,
        obj_description=obj_description,
        log=True,
        visualize=False
    )
    retrieve_end = time.time()
    logger.log(logging.INFO, f"retrieve time: {retrieve_end - retrieve_start}")
    
    ref_trajs = topk_retrieved_data_dict['traj']
    ref_imgs_np = topk_retrieved_data_dict['masked_img']
    ref_imgs_PIL = [Image.fromarray(ref_img_np).convert('RGB') for ref_img_np in ref_imgs_np]
    
    # Note: These queries are the results after cropping
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

    # Further convert cos_map into a probability distribution
    logger.log(logging.INFO, "Initializing Affordance_map_2d from DIFT similarity ...")
    affordance_map_2d = Affordance_map_2d(
        rgb_img=query_rgb,
        cos_map=cos_map,
        cropped_mask=query_mask,
        cropped_region=query_region,
    )
    
    if visualize:
        for i in range(len(topk_retrieved_data_dict["img"])):
            masked_img = topk_retrieved_data_dict["masked_img"][i]
            masked_img = draw_points_on_image(masked_img, [topk_retrieved_data_dict["traj"][i][0]], 5)
            masked_img.show()
    torch.cuda.empty_cache()
    return affordance_map_2d