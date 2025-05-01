import argparse
import pickle
import os
import numpy as np
from embodied_analogy.representation.basic_structure import Frame, Frames
from embodied_analogy.representation.obj_repr import Obj_repr
import os, sys
sys.path.append('/home/zby/Programs/Embodied_Analogy/third_party/Ditto/src')
sys.path.append('/home/zby/Programs/Embodied_Analogy/third_party/Ditto/')
os.environ["PYOPENGL_PLATFORM"] = "egl"

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
# from utils3d.mesh.utils import as_mesh
# from utils3d.render.pyrender import get_pose, PyRenderer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    # 计算每个点的体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # 使用字典来存储每个体素中的点
    voxel_dict = {}
    
    for idx, voxel in enumerate(voxel_indices):
        voxel_key = tuple(voxel)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(points[idx])
    
    # 计算每个体素的中心点
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
    """
    返回一个最小的 axis_aligned 点云 bbox 以包裹输入点云
    pcd: (N, 3)
    return: 
        center
        scaled
    """
    min_bound = np.min(pcd, 0)
    max_bound = np.max(pcd, 0)
    center = (min_bound + max_bound) / 2
    scale = (max_bound - min_bound).max() * 1.1
    return center, scale

def normalize_point_cloud(pcd, center, scale):
    """
    对于输入 pcd 做归一化处理, 使其处于 unit_cube 中
    """
    return (pcd - center) / scale

def get_ditto_model(model_type="Ditto_s2m"):
    assert model_type in ["Ditto_s2m", "Ditto_syn"]
    import torch
    from hydra import compose, initialize
    # from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import hydra

    from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
    from src.utils.misc import sample_point_cloud

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    os.chdir("/home/zby/Programs/Embodied_Analogy")
    with initialize(config_path='../../third_party/Ditto/configs'):
        config = compose(
            config_name='config',
            overrides=[
                f'experiment={model_type}.yaml',
            ], return_hydra_config=True)
        
    config.datamodule.opt.train.data_dir = '/home/zby/Programs/Ditto/data/'
    config.datamodule.opt.val.data_dir = '/home/zby/Programs/Ditto/data/'
    config.datamodule.opt.test.data_dir = '/home/zby/Programs/Ditto/data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load(f'/home/zby/Programs/Ditto/data/{model_type}.ckpt')
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
    # NOTE jonit_dir 直接 scale 就行了
    """
    start_w, end_w  ->  start_n, end_n
    start_n = (start_w - center) / scale
    end_n = (end_w - center) / scale
    end_n - start_n = (end_w - start_w) / scale
    """
    joint_dir = joint_axis * scale
    joint_dir = joint_dir / np.linalg.norm(joint_dir)
    return joint_dir, joint_start

def run_ditto_on_pc(pc_start, pc_end, model_type="Ditto_s2m", gt_joint_type=None, model=None, generator=None, visualize=False):
    """
        pc_start 和 pc_end 需要在世界坐标系下
    """
    if model is None or generator is None:
        model, generator = get_ditto_model(model_type)
    
    pc_start_v = pc_start.copy()
    pc_end_v = pc_end.copy()
    # 处理输入
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
    # TODO 原本是 < 0.5, 仅仅做测试，记得改回去 !!
    print("***************************")
    print(joint_type_prob.item())
    
    def is_revolute():
        if gt_joint_type is not None:
            return gt_joint_type == "revolute"
        else:
            return joint_type_prob.item() < 0.5
    
    if is_revolute():
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
        # camera_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
        # light_pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=np.pi/2)
        camera_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/4)
        light_pose = get_pose(1.5, ax=0, ay=0, az=np.pi/4)
        rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)

        Image.fromarray(rgb).show()
    
    # 将 normalized 坐标系下的 joint_param 缩放回世界坐标系下
    joint_dir, joint_start = unnormalize_joint_param(joint_axis_pred, pivot_point_pred, center, scale)
    
    # from embodied_analogy.utility.utils import visualize_pc
    # visualize_pc(
    #     # points=pc_start_v,
    #     points=np.concatenate([pc_start, pc_end]),
    #     colors=None,
    #     grasp=None,
    #     contact_point=pivot_point_pred,
    #     post_contact_dirs=[joint_axis_pred]
    # )
    # visualize_pc(
    #     # points=pc_start_v,
    #     points=np.concatenate([pc_start_v, pc_end_v]),
    #     colors=None,
    #     grasp=None,
    #     contact_point=joint_start,
    #     post_contact_dirs=[joint_dir]
    # )
    
    
    # 返回一个 joint_dict
    joint_dict = {
        "joint_type": joint_type,
        "joint_dir": joint_dir,
        "joint_start": joint_start
    }
    print("joint dict estimated by Ditto:\n", joint_dict)
    return joint_dict

def run_ditto(obj_repr: Obj_repr, model_type="Ditto_s2m", gt_joint_type=None, visualize=False):
    # 提取首尾两帧
    frame_start: Frame = obj_repr.frames[0]
    frame_start.segment_obj(
        obj_description=obj_repr.obj_description,
        post_process_mask=True,
        filter=True
    )
    pc_start, _ = frame_start.get_obj_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=True,
    )

    frame_end: Frame = obj_repr.frames[-1]
    frame_end.segment_obj(
        obj_description=obj_repr.obj_description,
        post_process_mask=True,
        filter=True,
    )
    pc_end, _ = frame_end.get_obj_pc(
        use_robot_mask=True, 
        use_height_filter=True,
        world_frame=True,
        # visualize=True,
    )
    # 把 obj_repr 的 frames 只保留始末两帧
    obj_repr.frames.clear()
    obj_repr.frames.frame_list = [frame_start, frame_end]
    
    joint_dict_w = run_ditto_on_pc(pc_start, pc_end, model_type=model_type, gt_joint_type=gt_joint_type, visualize=False)
    # NOTE 一定要将这个 joint_dict 写入到 obj_repr 中, 由于 ditto 返回的是世界坐标系下的, 因此写回时应该转换到 camera 坐标系下
    Tw2c = obj_repr.Tw2c
    # Tw2c = np.eye(4)
    joint_dict_c = {}
    joint_dict_c["joint_type"] = joint_dict_w["joint_type"]
    joint_dict_c["joint_dir"] = Tw2c[:3, :3] @ joint_dict_w["joint_dir"]
    joint_dict_c["joint_start"] = Tw2c[:3, :3] @ joint_dict_w["joint_start"] + Tw2c[:3, 3]
    obj_repr.coarse_joint_dict = joint_dict_c
    obj_repr.fine_joint_dict = joint_dict_c
    
    if visualize:
        obj_repr.visualize_joint()
    
    # run evaluation
    if obj_repr.gt_joint_dict["joint_type"] is None:
        assert "joint_type not in gt_joint_dict" 
    result = obj_repr.compute_joint_error()
    print("Reconstruction Result:")
    for k, v in result.items():
        print(k, v)
    
    return obj_repr, result
    
def update_cfg(explore_cfg, args):
    # 更新 env_folder
    if args.obj_folder_path_explore is not None:
        explore_cfg['obj_folder_path_explore'] = args.obj_folder_path_explore
    if args.obj_folder_path_reconstruct is not None:
        explore_cfg['obj_folder_path_reconstruct'] = args.obj_folder_path_reconstruct
    if args.use_gt_joint_type is not None:
        explore_cfg['use_gt_joint_type'] = args.use_gt_joint_type
    if args.model_type is not None:
        explore_cfg['model_type'] = args.model_type
        
    return explore_cfg
def read_args():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--obj_folder_path_explore', type=str, help='Folder where things are loaded')
    parser.add_argument('--obj_folder_path_reconstruct', type=str, help='Folder where things are stored')
    parser.add_argument('--use_gt_joint_type', type=str2bool, help='whether to use gt joint_type')
    parser.add_argument('--model_type', type=str, help='which pretrained Ditto to use')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/40147_1_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/48271_0_revolute"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/45168_0_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/49042_0_revolute"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/47578_1_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/48258_1_prismatics"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/48258_3_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/47578_2_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/45746_2_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/45677_3_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/45687_0_revolute"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/45841_2_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/47578_1_prismatic"
    # obj_folder = "/home/zby/Programs/Embodied_Analogy/asset_book/logs/explore_424/48258_2_prismatic"
    # obj_repr: Obj_repr = Obj_repr.load(os.path.join(obj_folder, "obj_repr.npy"))
    # obj_repr, result = run_ditto(obj_repr, model_type="Ditto_syn", gt_joint_type="prismatic", visualize=False)
    # sys.exit(0)
    
    # 给定一个路径, 读取其中 explore 下的数据, 对于有成功 explore 的数据, 进行重建, 并且将重建的数据进行保存, 将结果打印
    args = read_args()
    # print(args.obj_folder)
    explore_folder = args.obj_folder_path_explore
    with open(os.path.join(explore_folder, "cfg.json"), 'r', encoding='utf-8') as file:
        explore_cfg = json.load(file)
    with open(os.path.join(explore_folder, "result.pkl"), 'rb') as f:
        explore_result = pickle.load(f)
    
    # 如果没有成功的 explore, 则退出
    if not explore_result["has_valid_explore"]:
        print("No valid explore, thus no need to run reconstruction...")
    else:
        print("Find valid explore, keep going...")
        recon_save_folder = args.obj_folder_path_reconstruct
        
        recon_cfg = update_cfg(explore_cfg, args)
        print("read reconstruction cfg...")
        print(recon_cfg)

        with open(os.path.join(recon_save_folder, "cfg.json"), 'w', encoding='utf-8') as f:
            json.dump(recon_cfg, f, ensure_ascii=False, indent=4)
            
        obj_repr: Obj_repr = Obj_repr.load(os.path.join(explore_folder, "obj_repr.npy"))
        print("load obj_repr from explore folder...")
        try:
            gt_joint_type = obj_repr.gt_joint_dict["joint_type"] if args.use_gt_joint_type else None
            print("gt_joint_type", gt_joint_type)
            obj_repr, result = run_ditto(obj_repr, model_type=args.model_type, gt_joint_type=gt_joint_type, visualize=False)
        except Exception as e:
            print("Reconstruction failed...")
            print(e)
            print("done")
            sys.exit(0)
        
        # 然后保存 rgbd_seq
        obj_repr.save(os.path.join(recon_save_folder, "obj_repr.npy"))
        
        # 然后保存 运行状态文件
        with open(os.path.join(recon_save_folder, 'result.pkl'), 'wb') as f:
            pickle.dump(result, f)
        
    print("done")
    