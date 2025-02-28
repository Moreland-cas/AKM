# 调用 RAM 的 explore 函数, 输入是一个 rgbd 点云, 输出是抓取位姿，和一小段轨迹
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()

import sys
sys.path.append("/home/zby/Programs/Embodied_Analogy/third_party/RAM_code")

import numpy as np
import torch
from PIL import Image
from vision.featurizer.run_featurizer import transfer_affordance
from vision.featurizer.utils.visualization import IMG_SIZE
from subset_retrieval.subset_retrieve_pipeline import SubsetRetrievePipeline
from run_realworld.utils import crop_points, cluster_normals, visualize_point_directions
import traceback
import matplotlib
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from embodied_analogy.utility.constants import *
from embodied_analogy.utility.utils import seed_everything
# from embodied_analogy.perception.grounded_sam import run_grounded_sam

def lift_affordance(rgb, pcd, pixel, dir, visualize=False):
    H, W = rgb.shape[:2]
    post_contact_dirs_2d, post_contact_dirs_3d = None, None
    partial_points = np.array(pcd.points)
    partial_colors = np.array(pcd.colors)
    position = partial_points[pixel[1] * W + pixel[0]]
    
    # visualization
    # ds_points, _, _ = get_downsampled_pc(partial_points, None, 20000)
    ds_points, _, _ = crop_points(position, partial_points, thres=0.5)
    save_pcd = o3d.geometry.PointCloud()
    save_pcd.points = o3d.utility.Vector3dVector(ds_points)
    # red
    save_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((ds_points.shape[0], 3)))
    # add a point
    save_pcd.points.append(position)
    save_pcd.colors.append(np.array([0, 1, 0]))
    
    if visualize:
        o3d.io.write_point_cloud(f"{cfgs['SAVE_ROOT']}/grasp_point.ply", save_pcd)
    
    MAX_ATTEMPTS = 20 # in case there is no good grasp at one time
    max_dis = 0.05
    best_grasp = None
    max_radius, min_radius = 0.2, 0.1
    gg = None
    for num_attempt in range(MAX_ATTEMPTS):
        try:
            crop_radius = max_radius - (max_radius - min_radius) * num_attempt / MAX_ATTEMPTS
            print('=> crop_radius:', crop_radius)
            cropped_points, cropped_colors, cropped_normals = crop_points(
                position, partial_points, partial_colors, thres=crop_radius, save_root=self.cfgs['SAVE_ROOT']
            )
            try:
                # gg = self.detect_grasp_anygrasp(cropped_points, cropped_colors, save_vis=True) # use AnyGrasp if properly set up
                gg = detect_grasp_gsnet(cropped_points, cropped_colors, save_vis=True)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
            if gg is None:
                continue
            print('=> total grasp:', len(gg))
            if len(gg) == 0:
                continue
            
            best_grasp = get_best_grasp(gg, position, max_dis=max_dis) # original: 0.03
            if best_grasp is None:
                print('==>> no best grasp')
            else:
                break
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
    if best_grasp is None:
        try:
            gg = detect_grasp_gsnet(cropped_points, cropped_colors, False)
        except:
            gg = detect_grasp_gsnet(partial_points, partial_colors, False)
        best_grasp = get_closest_grasp(gg, position)
        print('==>> use GSNet for closest grasp')
    n_clusters = 5
    
    if visualize:
        vis_save_grasp(cropped_points, best_grasp, f"{cfgs['SAVE_ROOT']}/best_grasp.ply")
    clustered_centers = cluster_normals(cropped_normals, n_clusters=n_clusters) # (2*n_clusters, 3)
    visualize_point_directions(cropped_points, position, clustered_centers, cfgs['SAVE_ROOT'])
    post_contact_dirs_3d = clustered_centers
    post_contact_dirs_2d = project_normals(rgb, pixel, clustered_centers)
    grasp_array = best_grasp.grasp_array.tolist()
    
    # post-grasp
    best_dir_3d, best_score = None, -1
    for i in range(post_contact_dirs_2d.shape[0]):
        score = np.dot(post_contact_dirs_2d[i], dir)
        if score > best_score:
            best_score = score
            best_dir_3d = post_contact_dirs_3d[i]
    visualize_point_directions(ds_points, position, [best_dir_3d], cfgs['SAVE_ROOT'], "best_dir_3d")
    post_grasp_dir = best_dir_3d.tolist()
    
    ret_dict = {
        "grasp_array": grasp_array,
        "post_grasp_dir": post_grasp_dir
    }
    return ret_dict

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
    
def test_project_normals():
    # 创建一张白色图像
    img_size = (500, 500)
    img = Image.new('RGB', img_size, (255, 255, 255))

    # 随机生成法线
    num_normals = 10
    normals = np.random.rand(num_normals, 3) * 2 - 1  # 在[-1, 1]范围内生成法线

    # 设置像素位置为图像中心
    pixel = (img_size[0] // 2, img_size[1] // 2)

    # 调用你的函数并可视化
    project_normals(img, pixel, normals, visualize=True)
    
def get_ram_proposal(
    query_rgb, # H, W, 3 in numpy
    instruction, # open the drawwer
    prompt, # a photo of a drawer
    data_source, # droid
    save_root="/home/zby/Programs/Embodied_Analogy/assets/tmp/exploration",
    visualize=False
):
    seed_everything(SEED)

    subset_retrieve_pipeline = SubsetRetrievePipeline(
        subset_dir="/home/zby/Programs/RAM_code/assets/data",
        save_root=save_root,
        lang_mode='clip',
        topk=5, 
        crop=False, # 这里记得用 crop=False，如果用 crop=True 的话输出的 contact_point 是相对于 cropped_tgt image 的
        data_source=data_source,
    )
    
    # get rgb mask from grounded sam
    # _, query_mask = run_grounded_sam(
    #     rgb_image=query_rgb,
    #     text_prompt=obj, # drawer
    #     positive_points=None,
    #     negative_points=None, # 如果有遮挡的话, 在这里需要加入 negative points, 默认在初始时没有遮挡
    #     num_iterations=5,
    #     acceptable_thr=0.9,
    #     visualize=False,
    # )
    
    # query_mask = np.repeat(query_mask[..., None], 3, axis=-1).astype(np.uint8) # H, W, C (0 or 1)
    # query_img_masked = query_rgb * query_mask + 255 * (1 - query_mask) # 0-255
    # query_img_PIL = Image.fromarray(query_img_masked).convert('RGB')
    
    # use retrieval to get ref_path (or ref image) and ref trajectory in 2d space
    _, top1_retrieved_data_dict = subset_retrieve_pipeline.retrieve(instruction, query_rgb)
    traj = top1_retrieved_data_dict['traj']
    ref_img_np = top1_retrieved_data_dict['masked_img']
    ref_img_PIL = Image.fromarray(ref_img_np).convert('RGB')
    
    masked_query_np = top1_retrieved_data_dict["masked_query"]
    query_img_PIL = Image.fromarray(masked_query_np).convert('RGB')

    # scale cropped_traj to IMG_SIZE, because diff-transfer are done are image of size IMG_SIZE
    ref_pos_list = []
    for xy in traj:
        ref_pos_list.append((xy[0] * IMG_SIZE / ref_img_PIL.size[0], xy[1] * IMG_SIZE / ref_img_PIL.size[1]))
    
    # transfer in 2d
    while True:
        try:
            # 这里的 contact_point 的坐标是相对于 masked_query_np 的
            contact_point, post_contact_dir = transfer_affordance(ref_img_PIL, query_img_PIL, prompt, ref_pos_list, save_root=save_root, ftype='sd')
            break
        except Exception as transfer_e:
            traceback.print_exc()
            print('[ERROR] in transfer_affordance:', transfer_e)

    if visualize:
        import napari
        viewer = napari.Viewer(title = "RAM proposal")
        viewer.add_image(query_rgb, name="query_img")
        viewer.add_image(masked_query_np, name="query_img_masked")
        viewer.add_image(ref_img_np, name="reference_img")
        query_img_with_arrow_pil = draw_arrows_on_img(
            img=query_rgb,
            pixel=contact_point,
            normals_2d_directions=[post_contact_dir]
        )
        query_img_with_arrow_np = np.asarray(query_img_with_arrow_pil)
        viewer.add_image(query_img_with_arrow_np, name="query_img + arrow")
        napari.run()
        
    return contact_point, post_contact_dir

    # lift into 3d and generate grasp pose
    # ret_dict = lift_affordance(
    #     rgb,
    #     pcd,
    #     contact_point,
    #     post_contact_dir
    # )
    
    # print("3D Affordance:\n", ret_dict)
            

if __name__ == "__main__":
    query_rgb = np.asarray(Image.open("/home/zby/Programs/RAM_code/run_realworld/sapien_data/input/rgb.png"))
    contact_point, post_contact_dir = get_ram_proposal(
        query_rgb, # H, W, 3 in numpy
        instruction="open the drawer",
        prompt="a photo of a drawer", 
        data_source="droid",
        save_root="/home/zby/Programs/Embodied_Analogy/assets/tmp/exploration",
        visualize=True
    )