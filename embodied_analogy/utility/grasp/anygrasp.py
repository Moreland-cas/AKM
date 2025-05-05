import os
import torch
import numpy as np
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from graspnetAPI.grasp import GraspGroup
from embodied_analogy.utility.utils import (
    visualize_pc,
    rotation_matrix_between_vectors,
    find_correspondences
)
from embodied_analogy.utility.constants import ASSET_PATH

def crop_grasp(grasp_group, contact_point, radius=0.1):
    """
    contact_point: (3, ), 假设 grasp_group 中的 transform 是 Tgrasp2w, 那么 contact_point 也需要在 world 坐标系下
    对于 grasp_group 进行 crop, 保留那些距离 contact_point 的距离小于 radius 的点, 如果没有返回 None
    """
    t_grasp2w = grasp_group.translations # N, 3
    
    distances = np.linalg.norm(t_grasp2w - contact_point, axis=1) # N
    mask = distances < radius
    
    if mask.sum() == 0:
        return None
    
    grasp_group_ =GraspGroup()
    grasp_group_.grasp_group_array = grasp_group.grasp_group_array[mask]
    
    return grasp_group_
    
    
def filter_grasp_group(
    grasp_group, 
    degree_thre=30,
    dir_out=None,
):
    '''
        filter grasp_group, 使得保留的 grasp 的 appro_vector 的负方向与 dir_out 尽可能平行
        grasp_group: Tgrasp2c
        dir_out: (3, )
        contact_region: (N, 3), 也即是 moving part
        NOTE: grasp_group, contact_region 和 dir_out 均在相机坐标系下
    '''
    if grasp_group is None:
        return None
    
    Rgrasp2c = grasp_group.rotation_matrices # N, 3, 3
    neg_x_axis = -Rgrasp2c[:, :, 0] # N, 3
    
    # 让 grasp_frame 的 -x 轴尽可能平行于 dir_out
    product = np.sum(neg_x_axis * dir_out, axis=-1) # N
    product = product / (np.linalg.norm(neg_x_axis, axis=-1) * np.linalg.norm(dir_out))
    index = product > np.cos(np.deg2rad(degree_thre))
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group

def prepare_any_grasp_model(asset_path=ASSET_PATH):
    # load model
    from gsnet import AnyGrasp # gsnet.so
    # get a argument namespace
    cfgs = argparse.Namespace()
    cfgs.checkpoint_path = os.path.join(asset_path, 'ckpts/anygrasp/checkpoint_detection.tar')
    cfgs.max_gripper_width = 0.04
    cfgs.gripper_height = 0.03
    cfgs.top_down_grasp = False
    cfgs.debug = False
    model = AnyGrasp(cfgs)
    model.load_net()
    return model

@torch.no_grad()
def detect_grasp_anygrasp(
    points, 
    colors, 
    dir_out, 
    augment=True, 
    visualize=False,
):
    '''
    输入 a 坐标系下的 points 和 dir_out, 输出用 anygrasp 检测出的 grasp_group, 其中包含信息 Tgrasp2a
        points: (N, 3), in a coordinate
        colors: (N, 3), in range [0-1]
        dir_out: (3, ), in a coordinate, 用于将 a 坐标系下的点进行变换, 以模拟 grasp detector network 所需的格式 (相机方向指向点云内部)
        augment: if True, 对 dir_out 进行多个绕动, 分别预测 grasp, 然后将不同 grasp 在合并到一个坐标系下返回
    '''
    dir_out = dir_out / np.linalg.norm(dir_out)
    # 接下来生成多个 dir_out
    if augment:
        random_perturb = np.random.randn(20, 3) 
        dir_outs = dir_out + random_perturb * 0.5 # N, 3
        dir_outs = dir_outs / np.linalg.norm(dir_outs, axis=1, keepdims=True)
    else:
        dir_outs = [dir_out]
    
    # 改变输入的类型
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)
    lims = np.array([-1, 1, -1, 1, -1, 1]) * 10
    
    ggs = GraspGroup()
    for dir_out_i in dir_outs:
        # 用不同的 dir_out 确定不同的 app 坐标系, coor_app = Rw2app @ coor_w, 也即 (0, 0, 1) = Rw2app @ -dir_out
        Rw2app = rotation_matrix_between_vectors(-dir_out_i, np.array([0, 0, 1]))
        points_input = points @ Rw2app.T # N, 3
        points_input = points_input.astype(np.float32)
        
        # Tgrasp2app
        try:
            model = prepare_any_grasp_model()
            gg, _ = model.get_grasp(
                points_input,
                colors, 
                lims,
                apply_object_mask=True,
                dense_grasp=False,
                collision_detection=True
            )
        except Exception as e:
            print(f"run anygrasp locally failed:{e}")
            try:
                pass
            except Exception as f:
                print(f"run anygrasp remotely failed:{f}")
                assert False, "run anygrasp failed"  # 如果 function_b 也
        
        # print('grasp num:', len(gg))
        if gg == None or len(gg) == 0:
            continue
        
        # Tgrasp2w
        zero_translation = np.array([[0], [0], [0]])
        Rapp2w = Rw2app.T
        Tapp2w = np.hstack((Rapp2w, zero_translation))
        gg.transform(Tapp2w)
        ggs.add(gg)
        torch.cuda.empty_cache()
    
    if visualize:
        visualize_pc(
            points=points,
            colors=colors,
            grasp=ggs,
            contact_point=np.array([0, 0, 0]),
            post_contact_dirs=[dir_outs]
        )
    import time
    time.sleep(0.5)
    return ggs


def find_nearest_grasp(grasp_group, contact_point):
    '''
        grasp_group: graspnetAPI 
        contact_point: (3, )
    '''
    # 找到 grasp_group 中距离 contact_point 最近的 grasp 并返回
    # 首先根据 grasp 的 score 排序, 筛选出前20
    grasp_group = grasp_group.nms().sort_by_score()
    grasp_group = grasp_group[0:50]
    
    # 找到距离 contact_point 最近的 grasp
    translations = grasp_group.translations # N, 3
    distances = np.linalg.norm(translations - contact_point, axis=1)
    nearest_index = np.argmin(distances)
    nearest_index = int(nearest_index)
    return grasp_group[nearest_index]


def crop_grasp_by_moving(grasp_group, contact_region, crop_thresh=0.1):
    '''
        找到离 contact region 中点最近的 grasp
        grasp_group: Tgrasp2c
        contact_region: (N, 3), 也即是 moving part 所构成的点云
        NOTE: grasp_group, contact_region 和 dir_out 均在相机坐标系下
    '''
    if grasp_group is None:
        return None
        
    t_grasp2c = grasp_group.translations # N, 3
    _, distances, _ = find_correspondences(t_grasp2c, contact_region) # N
    
    # 首先 hard-filter 一下, 距离最近 contact_region 也超过 1dm 的 grasp 直接不考虑了
    hard_mask = distances < crop_thresh
    if hard_mask.sum() == 0:
        return None
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[hard_mask]
    
    pred_scores = grasp_group.scores # N
    distance_scores = 1 / (distances[hard_mask] + 1e-6) 
    grasp_scores = pred_scores * distance_scores  # N
    
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group


def sort_grasp_group(grasp_group, contact_region):
    '''
        找到离 contact region 中点最近的 grasp
        grasp_group: Tgrasp2c
        contact_region: (N, 3), 也即是 moving part 所构成的点云
        NOTE: grasp_group, contact_region 和 dir_out 均在相机坐标系下
    '''
    if grasp_group is None:
        return None
        
    t_grasp2c = grasp_group.translations # N, 3
    _, distances, _ = find_correspondences(t_grasp2c, contact_region) # N
    
    pred_scores = grasp_group.scores # N
    distance_scores = 1 / (distances + 1e-6) 
    grasp_scores = pred_scores * distance_scores  # N
    
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group


if __name__ == '__main__':
    import os
    from PIL import Image
    from embodied_analogy.utility.utils import depth_image_to_pointcloud
    path = "/home/zby/Programs/Embodied_Analogy/embodied_analogy/dev/ram_proposal/"
    colors = np.array(Image.open(os.path.join(path, 'rgb.png')), dtype=np.float32) / 255.0
    depths = np.load(os.path.join(path, 'depth.npy'))
    masks = np.load(os.path.join(path, 'mask.npy'))
    
    fx, fy = 300, 300
    cx, cy = 400, 300
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    points_camera = depth_image_to_pointcloud(depths, masks, K)
    colors = colors[masks].astype(np.float32)
    
    Tw2c = np.array(
        [[-4.8380834e-01, -8.7517393e-01,  5.1781535e-07,  4.5627734e-01],
       [-1.6098598e-01,  8.8994741e-02, -9.8293614e-01,  3.8961503e-01],
       [ 8.6024004e-01, -4.7555280e-01, -1.8394715e-01,  8.8079178e-01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
    )
    Rw2c = Tw2c[:3, :3]
    Rc2w = Rw2c.T
    points_world = points_camera @ Rc2w.T
    
    # visualize_pc(points_world, colors)
    import time
    start_time = time.time()
    gg = detect_grasp_anygrasp(
        points=points_world, 
        colors=colors,
        dir_out=np.array([-1, 0, 0]),
        augment=True,
        visualize=False
    )
    end_time = time.time()
    print("time used:", end_time - start_time)
    
    ggs_filtered = filter_grasp_group(
        grasp_group=gg,
        dir_out=np.array([-1, 0, 0]),
    )
    
    visualize_pc(
        points=points_world,
        colors=colors,
        grasp=ggs_filtered,
        contact_point=None,
        post_contact_dirs=np.array([-1, 0, 0])
    )