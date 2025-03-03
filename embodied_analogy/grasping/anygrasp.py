import numpy as np
import argparse
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from embodied_analogy.utility.utils import (
    visualize_pc,
    world_to_image,
    rotation_matrix_between_vectors,
    find_correspondences
)

def detect_grasp_anygrasp(points, colors, joint_axis_out, visualize=False):
    '''
    输入世界坐标系下的点云和颜色, 返回 grasp_group
        定义 approach 坐标系为 xy 轴平行物体表面, z 轴指向物体内部 (joint axis 的反方向)
        定义 grasp 坐标系为 x 轴指向物体内部, y 轴指向物体的宽度
        
        points: N, 3
        colors: N, 3
        joint_axis: (3, ), 指向 outward 方向
    
    '''
    # 传入的点是在世界坐标系下的(xy 轴平行地面, z 轴指向重力反方向)
    # 因此首先将世界坐标系下的点转换到 app 坐标系下
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)
    points_input = points.copy() # N, 3
    colors_input = colors.copy()
    
    # coor_app = Rw2app @ coor_w, 也即 -joint_axis = Rw2app @ (0, 0, 1)
    Rw2app = rotation_matrix_between_vectors(np.array([0, 0, 1]), -joint_axis_out)
    points_input = points_input @ Rw2app.T # N, 3
    points_input = points_input.astype(np.float32)
    
    from gsnet import AnyGrasp # gsnet.so
    # get a argument namespace
    cfgs = argparse.Namespace()
    cfgs.checkpoint_path = 'assets/ckpts/checkpoint_detection.tar'
    cfgs.max_gripper_width = 0.04
    cfgs.gripper_height = 0.03
    cfgs.top_down_grasp = False
    cfgs.debug = visualize
    model = AnyGrasp(cfgs)
    model.load_net()
    
    lims = np.array([-1, 1, -1, 1, -1, 1]) * 10
    gg, cloud = model.get_grasp(
        points_input,
        colors_input, 
        lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )
    print('grasp num:', len(gg))
    
    if visualize:
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([*grippers, cloud])
        
    # 此时的 gg 中的 rotation 和 translation 对应 Tgrasp2app
    # 将预测的 app pose 从 app 坐标系转换回世界坐标系
    zero_translation = np.array([[0], [0], [0]])
    Rapp2w = Rw2app.T
    Tapp2w = np.hstack((Rapp2w, zero_translation))
    gg.transform(Tapp2w)
    # 此时的 gg 中的 rotation 和 translation 对应 Tgrasp2w
    return gg

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
    # distance_scores = np.exp(-2 * distances) 
    distance_scores = (distances < 0.05).astype(np.float32)
    
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
    grasp_scores = pred_scores * distance_scores  # N
    
    # 找到距离 contact_point 最近的 grasp
    index = np.argsort(grasp_scores)
    index = index[::-1]
    grasp_group.grasp_group_array = grasp_group.grasp_group_array[index]
    
    return grasp_group, grasp_scores[index]


if __name__ == '__main__':
    pass
    # TODO: 给 anygrasp 写一个测试函数