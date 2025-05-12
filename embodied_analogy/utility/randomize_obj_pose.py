import os
import json
import numpy as np
import sapien.core as sapien
from embodied_analogy.utility.constants import ASSET_PATH

def axis_angle_to_quat(axis, angle):
    '''
    axis: [[x, y, z]] or [x, y, z]
    angle: rad
    return: a quat that rotates angle around axis
    '''
    axis = np.array(axis)
    shape = axis.shape
    assert(shape[-1] == 3)
    axis = axis.reshape(-1, 3)

    angle = np.array(angle)
    angle = angle.reshape(-1, 1)

    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9)
    quat1 = np.concatenate([np.cos(angle/2), axis[:, 0:1]*np.sin(angle/2), axis[:, 1:2]*np.sin(angle/2), axis[:, 2:3]*np.sin(angle/2)], axis=-1)
    return quat1.reshape(*shape[:-1], 4)

def randomize_pose(ang_low, ang_high, zrot_low, zrot_high, xrot_low, xrot_high, dis_low, dis_high, height_low, height_high) :

    ang = np.random.uniform(ang_low, ang_high)
    zrot = np.random.uniform(zrot_low, zrot_high)
    xrot = np.random.uniform(xrot_low, xrot_high)
    dis = np.random.uniform(dis_low, dis_high)
    height = np.random.uniform(height_low, height_high)

    p0 = sapien.Pose(p=[dis, 0, height])
    r0 = sapien.Pose(q=axis_angle_to_quat([0,0,1], ang)) # 绕原点旋转
    r1 = sapien.Pose(q=axis_angle_to_quat([0,0,1], zrot)) # 原地旋转
    r2 = sapien.Pose(q=axis_angle_to_quat([1,0,0], xrot)) # 原地旋转
    
    p1 = r0 * p0 * r1 * r2
    return p1
    
def randomize_obj(cfg: dict):
    """
    对于 obj_cfg 中的 pose 进行随机化
    根据 tack_cfg 中的 open/close 以及 delta 值, 计算物体的 active_link 的初始状态的范围, 并随机选取一个值进行初始化
    """
    # TODO: 如果改为随机的话, 需要保证 ManipuleEnv 调用的这个函数完全从 cfg 中读取位姿势, 而不是随机生成
    # obj_init_pos_angle_low = -0.4
    # obj_init_pos_angle_high = -0.4
    # obj_init_rot_low = -0.2
    # obj_init_rot_high = 0.2
    # obj_init_dis_low = 0.5
    # obj_init_dis_high = 0.6
    # obj_init_height_low = 0.0
    # obj_init_height_high = 0.0
    # obj_init_dof_low = 0.0
    # obj_init_dof_high = 0.0
    
    obj_init_pos_angle_low = -0.
    obj_init_pos_angle_high = 0.
    obj_init_zrot_low = -0.1
    obj_init_zrot_high = -0.
    
    obj_init_xrot_low = -0.05
    obj_init_xrot_high = 0.05
    
    obj_init_dis_low = 0.5
    obj_init_dis_high = 0.6
    obj_init_height_low = 0.0
    obj_init_height_high = 0.0
    # obj_init_dof_low = 0.0
    # obj_init_dof_high = 0.0
    
    path = os.path.join(ASSET_PATH, cfg["data_path"])
    bbox_path = os.path.join(path, "bounding_box.json")
    with open(bbox_path, "r") as f:
        bbox = json.load(f)
    
    sapien_pose = randomize_pose(
        obj_init_pos_angle_low,
        obj_init_pos_angle_high,
        obj_init_zrot_low,
        obj_init_zrot_high,
        obj_init_xrot_low,
        obj_init_xrot_high,
        obj_init_dis_low - bbox["min"][2]*0.75,
        obj_init_dis_high - bbox["min"][2]*0.75,
        obj_init_height_low - bbox["min"][1]*0.75,
        obj_init_height_high - bbox["min"][1]*0.75
    )
    
    cfg.update({
        "load_pose": sapien_pose.p.tolist(),
        "load_quat": sapien_pose.q.tolist(),
        "load_scale": 1,
    })
    # return sapien_pose
    return cfg