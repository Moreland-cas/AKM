import os, sys
sys.path.append('/home/Programs/Ditto/')
sys.path.append('/home/Programs/Ditto/src')
# os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import math
import trimesh
import torch

import open3d as o3d
import json
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import trimesh
from utils3d.mesh.utils import as_mesh
from utils3d.render.pyrender import get_pose, PyRenderer

def plot_3d_point_cloud(
    x,
    y,
    z,
    show=True,
    show_axis=True,
    in_u_sphere=False,
    marker='.',
    s=8,
    alpha=.8,
    figsize=(5, 5),
    elev=10,
    azim=240,
    axis=None,
    title=None,
    lim=None,
    *args,
    **kwargs
):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y),
                   np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig


def read_depth(depth_path):
    depth_img = np.array(Image.open(depth_path))
    depth_img = depth_img.astype(np.float32) * 0.001
    
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.empty_like(depth_img)),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=0.7,
        convert_rgb_to_intensity=False,
    )
    return rgbd

def sum_downsample_points(point_list, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    points = np.concatenate([np.asarray(x.points) for x in point_list], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def voxel_downsample(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}
    
    for idx, voxel in enumerate(voxel_indices):
        voxel_key = tuple(voxel)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(points[idx])
    
    downsampled_points = []
    for voxel_key, voxel_points in voxel_dict.items():
        downsampled_points.append(np.mean(voxel_points, axis=0))
    
    return np.array(downsampled_points)

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

def vector_to_rotation(vector):
    z = np.array(vector)
    z = z / np.linalg.norm(z)
    x = np.array([1, 0, 0])
    x = x - z*(x.dot(z)/z.dot(z))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]

def add_r_joint_to_scene(
    scene,
    axis,
    pivot_point,
    length,
    radius=0.01,
    joint_color=[200, 0, 0, 180],
    recenter=False
):
    if recenter:
        pivot_point = np.cross(axis, np.cross(pivot_point, axis))
    rotation_mat = vector_to_rotation(axis)
    screw_tran = np.eye(4)
    screw_tran[:3, :3] = rotation_mat
    screw_tran[:3, 3] = pivot_point
    
    axis_cylinder = trimesh.creation.cylinder(radius, height=length)
    axis_arrow = trimesh.creation.cone(radius * 2, radius * 4)
    arrow_trans = np.eye(4)
    arrow_trans[2, 3] = length / 2
    axis_arrow.apply_transform(arrow_trans)
    axis_obj = trimesh.Scene((axis_cylinder, axis_arrow))
    screw = as_mesh(axis_obj)
    
    # screw.apply_translation([0, 0, 0.1])
    screw.apply_transform(screw_tran)
    screw.visual.face_colors = np.array(joint_color, dtype=np.uint8)
    scene.add_geometry(screw)
    return screw

def sample_point_cloud(pc, num_point):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(
        np.arange(num_point_all),
        size=(num_point,),
        replace=num_point > num_point_all,
    )
    return pc[idxs], idxs

def get_normalize_param(pcd):
    min_bound = np.min(pcd, 0)
    max_bound = np.max(pcd, 0)
    center = (min_bound + max_bound) / 2
    scale = (max_bound - min_bound).max() * 1.1
    return center, scale

def normalize_point_cloud(pcd, center, scale):
    return (pcd - center) / scale

def get_ditto_model():
    import torch
    from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import hydra

    from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
    from src.utils.misc import sample_point_cloud

    with initialize(config_path='./configs'):
        config = compose(
            config_name='config',
            overrides=[
                'experiment=Ditto_s2m.yaml',
            ], return_hydra_config=True)
    config.datamodule.opt.train.data_dir = '/home/Programs/Ditto/data/'
    config.datamodule.opt.val.data_dir = '/home/Programs/Ditto/data/'
    config.datamodule.opt.test.data_dir = '/home/Programs/Ditto/data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load('/home/Programs/Ditto/data/Ditto_s2m.ckpt')
    device = torch.device(0)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.eval().to(device)

    generator = Generator3D(
        model.model,
        device=device,
        threshold=0.4,
        seg_threshold=0.5,
        input_type='pointcloud',
        refinement_step=0,
        padding=0.1,
        resolution0=32
    )
    return model, generator

def unnormalize_joint_param(joint_axis, pivot_point, center, scale):
    joint_start = pivot_point * scale + center
    joint_dir = joint_axis * scale + center
    joint_dir = joint_dir / np.linalg.norm(joint_dir)
    return joint_dir, joint_start

def run_ditto(pc_start, pc_end, model=None, generator=None, visualize=False):
    if model is None or generator is None:
        model, generator = get_ditto_model()
        
    center, scale = get_normalize_param(np.concatenate([pc_start, pc_end]))

    pc_start_normalized = normalize_point_cloud(pc_start, center, scale)
    pc_end_normalized = normalize_point_cloud(pc_end, center, scale)
    
    pc_start = voxel_downsample(pc_start_normalized, 0.02)
    pc_end = voxel_downsample(pc_end_normalized, 0.02)

    pc_start, _ = sample_point_cloud(pc_start, 8192)
    pc_end, _ = sample_point_cloud(pc_end, 8192)

    ditto_input = {
        'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to("cuda").float(),
        'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to("cuda").float()
    }

    mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(ditto_input)
    with torch.no_grad():
        joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)
        
    # print(joint_type_logits, joint_param_revolute, joint_param_prismatic)
    
    joint_type_prob = joint_type_logits.sigmoid().mean()
    joint_type = None
    if joint_type_prob.item()< 0.5:
        joint_type = "revolute"
        # axis voting, revolute
        joint_r_axis = (
            normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
        joint_r_p2l_vec = (
            normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
        )
        joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
        p_seg = mobile_points_all[0].cpu().numpy()
        from src.utils.joint_estimation import aggregate_dense_prediction_r
        pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
        (
            joint_axis_pred,
            pivot_point_pred,
            config_pred,
        ) = aggregate_dense_prediction_r(
            joint_r_axis, pivot_point, joint_r_t, method="mean"
        )
    # prismatic
    else:
        joint_type = "prismatic"
        # axis voting
        joint_p_axis = (
            normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
        )
        joint_axis_pred = joint_p_axis.mean(0)
        joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
        config_pred = joint_p_t.mean()
        
        pivot_point_pred = mesh_dict[1].bounds.mean(0)
    
    if visualize:
        renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})

        # compute articulation model
        mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 255], dtype=np.uint8)
        scene = trimesh.Scene()
        static_part = mesh_dict[0].copy()
        mobile_part = mesh_dict[1].copy()
        scene.add_geometry(static_part)
        scene.add_geometry(mobile_part)
        add_r_joint_to_scene(scene, joint_axis_pred, pivot_point_pred, 1.0, recenter=True)

        # render result mesh
        camera_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
        light_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
        rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

        Image.fromarray(rgb).show()
    
    joint_dir, joint_start = unnormalize_joint_param(joint_axis_pred, pivot_point_pred, center, scale)
    joint_dict = {
        "joint_type": joint_type,
        "joint_dir": joint_dir,
        "joint_start": joint_start
    }
    print(joint_dict)
    return joint_dict

if __name__ == "__main__":
    data_path = "/home/Programs/Ditto/data/custom_data"
    pc_start = np.load(os.path.join(data_path, "pc_start.npy"))
    pc_end = np.load(os.path.join(data_path, "pc_start.npy"))
    run_ditto(pc_start, pc_end, visualize=True)