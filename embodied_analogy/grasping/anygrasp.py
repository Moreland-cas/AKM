import numpy as np
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from graspnetAPI.grasp import GraspGroup
from embodied_analogy.utility.utils import (
    visualize_pc,
    world_to_image,
    rotation_matrix_between_vectors,
    find_correspondences
)

def detect_grasp_anygrasp(points, colors, dir_out, visualize=False):
    '''
    输入世界坐标系下的点云和颜色, 返回 grasp_group (存储了信息 Tgrasp2w)
        函数输入的 points 是世界坐标系下的
        网络输入的点是 app 坐标系下的, app 坐标系需要根据 dir_out 来确定 (app 的 z 轴指向 dir_out 的反方向)
        dir_out: 需要在 points 的坐标系(一般是世界坐标系)下指定, 从物体内部指向物体外部
        网络的输出 grasp 的坐标系由 graspnetAPI 定义, grasp 存储了信息 Tgrasp2app
        
        可视化是在 world 坐标系下的
    '''
    # 首先确定 app 坐标系
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)
    
    # coor_app = Rw2app @ coor_w, 也即 (0, 0, 1) = Rw2app @ -dir_out
    Rw2app = rotation_matrix_between_vectors(-dir_out, np.array([0, 0, 1]))
    points_input = points @ Rw2app.T # N, 3
    points_input = points_input.astype(np.float32)
    
    from gsnet import AnyGrasp # gsnet.so
    # get a argument namespace
    cfgs = argparse.Namespace()
    cfgs.checkpoint_path = '/home/zby/Programs/Embodied_Analogy/assets/ckpts/checkpoint_detection.tar'
    cfgs.max_gripper_width = 0.04
    cfgs.gripper_height = 0.03
    cfgs.top_down_grasp = False
    cfgs.debug = visualize
    model = AnyGrasp(cfgs)
    model.load_net()
    
    lims = np.array([-1, 1, -1, 1, -1, 1]) * 10
    # Tgrasp2app
    gg, _ = model.get_grasp(
        points_input,
        colors, 
        lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )
    print('grasp num:', len(gg))
    
    # Tgrasp2w
    zero_translation = np.array([[0], [0], [0]])
    Rapp2w = Rw2app.T
    Tapp2w = np.hstack((Rapp2w, zero_translation))
    gg.transform(Tapp2w)
    
    if visualize:
        visualize_pc(points, colors, gg)
    return gg

def detect_grasp_anygrasp(points, colors, dir_out, augment=True, visualize=False):
    '''
    输入世界坐标系下的点云和颜色, 返回 grasp_group (存储了信息 Tgrasp2w)
        函数输入的 points 是世界坐标系下的
        网络输入的点是 app 坐标系下的, app 坐标系需要根据 dir_out 来确定 (app 的 z 轴指向 dir_out 的反方向)
        dir_out: 需要在 points 的坐标系(一般是世界坐标系)下指定, 从物体内部指向物体外部
        网络的输出 grasp 的坐标系由 graspnetAPI 定义, grasp 存储了信息 Tgrasp2app
        
        可视化是在 world 坐标系下的
        
        NOTE: augment 为 True 时, 会对 dir_out 进行多个绕动, 分别预测 grasp, 然后将不同 grasp 在合并到一个坐标系下
    '''
    # load model
    from gsnet import AnyGrasp # gsnet.so
    # get a argument namespace
    cfgs = argparse.Namespace()
    cfgs.checkpoint_path = '/home/zby/Programs/Embodied_Analogy/assets/ckpts/checkpoint_detection.tar'
    cfgs.max_gripper_width = 0.04
    cfgs.gripper_height = 0.03
    cfgs.top_down_grasp = False
    cfgs.debug = visualize
    model = AnyGrasp(cfgs)
    model.load_net()
    
    
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
        gg, _ = model.get_grasp(
            points_input,
            colors, 
            lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )
        # print('grasp num:', len(gg))
        if gg == None:
            continue
        
        # Tgrasp2w
        zero_translation = np.array([[0], [0], [0]])
        Rapp2w = Rw2app.T
        Tapp2w = np.hstack((Rapp2w, zero_translation))
        gg.transform(Tapp2w)
        ggs.add(gg)
    
    if visualize:
        visualize_pc(
            points=points,
            colors=colors,
            grasp=ggs,
            contact_point=np.array([0, 0, 0]),
            post_contact_dirs=dir_outs
        )
        
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

def sort_grasp_group(grasp_group, contact_region, axis=None, grasp_pre_filter=False):
    '''
        找到离 contact region 中点最近的 grasp, 且越是垂直于 axis 越好
        grasp_group: from graspnetAPI 
        contact_region: (N, 3), 也即是 moving part
    '''
    if grasp_pre_filter: # 保留前 50 的 grasp
        grasp_group = grasp_group.nms().sort_by_score()
        grasp_group = grasp_group[0:50]
        
    t_grasp2w = grasp_group.translations # N, 3
    pred_scores = grasp_group.scores # N
    
    _, distances, _ = find_correspondences(t_grasp2w, contact_region) # N
    distance_scores = np.exp(-2 * distances) 
    # distance_scores = (distances < 0.05).astype(np.float32)
    
    R_grasp2w = grasp_group.rotation_matrices # N, 3, 3
    def R2unitAxis(rotation_matrix):
        rotation = R.from_matrix(rotation_matrix)
        axis_angle = rotation.as_rotvec()
        rotation_axis = axis_angle / np.linalg.norm(axis_angle)
        return rotation_axis
    
    pred_axis = np.array([R2unitAxis(R_grasp2w[i]) for i in range(len(R_grasp2w))]) # N, 3
    
    angle_scores = 1.
    if axis is not None:
        # TODO: 这里需要 debug, 似乎需要的不是 grasp 的 rotation matrix 对应的 axis 平行于 axis， 而是 z 轴也平行于 axis
        angle_scores = np.abs(np.sum(pred_axis * axis, axis=-1)) # N
    
    # grasp_scores = pred_scores * distance_scores * angle_scores # N
    # grasp_scores = pred_scores * distance_scores  # N
    grasp_scores = distance_scores  # N
    
    # 找到距离 contact_point 最近的 grasp
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group, grasp_scores[index]


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
    detect_grasp_anygrasp(
        points=points_world, 
        colors=colors,
        dir_out=np.array([-1, 0, 0]),
        augment=True,
        visualize=False
    )
    end_time = time.time()
    print("time used:", end_time - start_time)
    
    # augment = False, consume 0.734
    # augment = True, consume 1.79
    # x + y = 0.73 x + 15y = 1.79(15) -> 2.5(20)
    # y = 0.075
    # x = 0.657
    