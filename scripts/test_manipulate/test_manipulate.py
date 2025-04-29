import os
import sys
import json
import pickle
import argparse
import numpy as np
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()

from embodied_analogy.environment.manipulate_env import ManipulateEnv

def update_cfg(recon_cfg, args):
    # 更新 env_folder
    if args.obj_folder_path_reconstruct is not None:
        recon_cfg['obj_folder_path_reconstruct'] = args.obj_folder_path_reconstruct
    if args.scale_dir is not None:
        recon_cfg['scale_dir'] = args.scale_dir
    if args.manipulate_type is not None:
        recon_cfg['manipulate_type'] = args.manipulate_type
    if args.manipulate_distance is not None:
        recon_cfg['manipulate_distance'] = args.manipulate_distance
    if args.reloc_lr is not None:
        recon_cfg['reloc_lr'] = args.reloc_lr
    if args.max_attempts is not None:
        recon_cfg['max_attempts'] = args.max_attempts
    if args.max_distance is not None:
        recon_cfg['max_distance'] = args.max_distance
    if args.prismatic_reloc_interval is not None:
        recon_cfg['prismatic_reloc_interval'] = args.prismatic_reloc_interval
    if args.prismatic_reloc_tolerance is not None:
        recon_cfg['prismatic_reloc_tolerance'] = args.prismatic_reloc_tolerance
    if args.revolute_reloc_interval is not None:
        recon_cfg['revolute_reloc_interval'] = args.revolute_reloc_interval
    if args.revolute_reloc_tolerance is not None:
        recon_cfg['revolute_reloc_tolerance'] = args.revolute_reloc_tolerance
    return recon_cfg

def read_args():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--obj_folder_path_reconstruct', type=str, help='Folder where things are loaded')
    parser.add_argument('--scale_dir', type=str, help='Folder where things are stored')
    parser.add_argument('--manipulate_type', type=str, help='Open or Close')
    parser.add_argument('--manipulate_distance', type=float, help='Manipulate distance')
    parser.add_argument('--reloc_lr', type=float, help='Learning rate for relocization optimization')
    parser.add_argument('--max_attempts', type=int, help='Maximum number of manipulation retries')
    parser.add_argument('--max_distance', type=float, help='Maximum range of manipulation')
    parser.add_argument('--prismatic_reloc_interval', type=float, help='.')
    parser.add_argument('--prismatic_reloc_tolerance', type=float, help='.')
    parser.add_argument('--revolute_reloc_interval', type=float, help='.')
    parser.add_argument('--revolute_reloc_tolerance', type=float, help='.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = read_args()
    
    # obj_folder_path_reconstruct = args.obj_folder_path_reconstruct.replace("MyBook", "MyBook1")
    try:
        with open(os.path.join(args.obj_folder_path_reconstruct, "cfg.json"), 'r', encoding='utf-8') as file:
            recon_cfg = json.load(file)
            print(recon_cfg)
            for k, v in recon_cfg.items():
                if isinstance(v, str):
                    recon_cfg[k] = v.replace("MyBook", "MyBook1")
    except Exception as e:
        print(f"Error reading cfg file and obj_repr: ")
        print("\t", e)
        print("done")
        sys.exit(0)
    
    
    # 首先读取 recon_cfg
    obj_folder_path_reconstruct = recon_cfg["obj_folder_path_reconstruct"]
    # 读取其中的 result.pkl 文件, 如果发现误差太大，则不进行 manipulate
    with open(os.path.join(obj_folder_path_reconstruct, 'result.pkl'), 'rb') as result_file:
        recon_result = pickle.load(result_file)
        joint_type = recon_result["gt_w"]["joint_type"]
        
        fine_loss = recon_result['fine_loss']
        fine_type_loss = fine_loss['type_err']
        fine_angle_err = fine_loss['angle_err']
        fine_pos_err = fine_loss['pos_err']
        
        if joint_type == "prismatic":
            if not(fine_angle_err < np.deg2rad(10) and fine_type_loss == 0):
                print("Skip since the reconstruction is not good enough")
                print("done")
                sys.exit(0)
        else:
            if not(fine_pos_err < 0.05 and fine_angle_err < np.deg2rad(10) and fine_type_loss == 0):
                print("Skip since the reconstruction is not good enough")
                print("done")
                sys.exit(0)
    
    manip_cfg = update_cfg(recon_cfg, args)
    # 在这里计算 init_joint_state, 更新进 manip_cfg
    
    # 首先初始化 obj_init_dof_low 和 obj_init_dof_high
    print("\t Load for manipulation ...")
    manipulate_type = manip_cfg["manipulate_type"]
    print(f"Since manipulate_type is {manipulate_type}")
    if manip_cfg["manipulate_type"] == "open":
        obj_init_dof_low = 0
        obj_init_dof_high = manip_cfg["manipulate_distance"]
    else:
        obj_init_dof_low = manip_cfg["manipulate_distance"]
        obj_init_dof_high = manip_cfg["max_distance"]
    
    if manip_cfg["joint_type"] == "revolute":
        # 由于 test_manipulate.sh 中指定的 revolute 的 distance 是用 degree 表示的, 因此需要转换为弧度
        obj_init_dof_low = np.deg2rad(obj_init_dof_low)
        obj_init_dof_high = np.deg2rad(obj_init_dof_high)
        
    print(f"\t obj_init_dof_low and high are set to {obj_init_dof_low} and {obj_init_dof_high} respectively")
    
    def randomize_dof(dof_low, dof_high) :
        if dof_low == 'None' or dof_high == 'None' :
            return None
        return np.random.uniform(dof_low, dof_high)
    
    dof = randomize_dof(
        obj_init_dof_low,
        obj_init_dof_high
    )
    print(f"\t Obj init joint state is set to {dof}")
    manip_cfg.update({"init_joint_state": dof})
    
    with open(os.path.join(manip_cfg["scale_dir"], "cfg.json"), 'w', encoding='utf-8') as f:
        json.dump(manip_cfg, f, ensure_ascii=False, indent=4)
    
    env = ManipulateEnv(manip_cfg)
    # NOTE 这里改用最新版本的 manipulate_close_loop_intermediate, 而不是 manipulate_close_loop
    result_list = env.manipulate_close_loop_intermediate()
    
    with open(os.path.join(manip_cfg["scale_dir"], 'result.pkl'), 'wb') as f:
        pickle.dump(result_list, f)
        
    print("done")
    