import os
import pickle
from embodied_analogy.environment.explore_env import ExploreEnv
import argparse
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def none_or_float(value):
    if value.lower() in ['none', 'null']:
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")
    
def none_or_list(value):
    if value.lower() in ['none', 'null']:
        return None
    try:
        result = [int(item) for item in value.split(',')]
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid list value: '{value}'")
    
# 首先调用 argparser 来读取命令行参数, 生成 cfg 文件
def update_cfg(base_cfg, args):
    # 更新 env_folder
    if args.obj_folder_path_explore is not None:
        base_cfg['obj_folder_path_explore'] = args.obj_folder_path_explore
    # 更新 base_cfg 字典中的值
    if args.phy_timestep is not None:
        base_cfg['phy_timestep'] = args.phy_timestep
    if args.planner_timestep is not None:
        base_cfg['planner_timestep'] = args.planner_timestep
    if args.use_sapien2 is not None:
        base_cfg['use_sapien2'] = args.use_sapien2
    if args.fully_zeroshot is not None:
        base_cfg['fully_zeroshot'] = args.fully_zeroshot
    if args.record_fps is not None:
        base_cfg['record_fps'] = args.record_fps
    if args.pertubation_distance is not None:
        base_cfg['pertubation_distance'] = args.pertubation_distance
    if args.valid_thresh is not None:
        base_cfg['valid_thresh'] = args.valid_thresh
    if args.max_tries is not None:
        base_cfg['max_tries'] = args.max_tries
    if args.update_sigma is not None:
        base_cfg['update_sigma'] = args.update_sigma
    if args.reserved_distance is not None:
        base_cfg['reserved_distance'] = args.reserved_distance
    if args.num_initial_pts is not None:
        base_cfg['num_initial_pts'] = args.num_initial_pts
    if args.offscreen is not None:
        base_cfg['offscreen'] = args.offscreen
    if args.use_anygrasp is not None:
        base_cfg['use_anygrasp'] = args.use_anygrasp
    return base_cfg


def read_args():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--phy_timestep', type=float, help='Physical timestep')
    parser.add_argument('--planner_timestep', type=float, help='Planner timestep')
    parser.add_argument('--use_sapien2', type=str2bool, default=True, help='Use Sapien2')
    parser.add_argument('--offscreen', type=str2bool, help='Disable screen visualization')

    # explore_cfg arguments
    parser.add_argument('--record_fps', type=int, help='Record FPS')
    parser.add_argument('--fully_zeroshot', type=str2bool, help='Whether to remove cabinet class in RAM')
    parser.add_argument('--pertubation_distance', type=float, help='Perturbation distance')
    parser.add_argument('--valid_thresh', type=float, help='factor of actual Perturbation distance')
    parser.add_argument('--max_tries', type=int, help='Maximum tries')
    parser.add_argument('--update_sigma', type=float, help='Update sigma')
    parser.add_argument('--reserved_distance', type=float, help='Reserved distance')
    parser.add_argument('--num_initial_pts', type=int, help='number of tracking points initialized on first frame')
    parser.add_argument('--use_anygrasp', type=str2bool, help='whether to use anygrasp or graspnet')
    
    # experiment related
    parser.add_argument('--obj_folder_path_cfg', type=str, help='path to cfg object folder')
    parser.add_argument('--obj_folder_path_explore', type=str, help='path to explore object folder')
    args = parser.parse_args()
    # 更新配置
    return args

# 读取 args, 生成 cfg 文件, 输入给 manipulate_env 进行测试, 并返回测试结果 (数值 or logging)

if __name__ == '__main__':
    args = read_args()
    with open(os.path.join(args.obj_folder_path_cfg, "cfg.json"), 'r', encoding='utf-8') as file:
        base_cfg = json.load(file)
    cfg = update_cfg(base_cfg, args)
    
    # 首先保存 cfg 文件
    save_prefix = cfg['obj_folder_path_explore']
    with open(os.path.join(save_prefix, "cfg.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    
    env = ExploreEnv(cfg)
    result = env.explore_stage(visualize=False)
    
    # 然后保存 rgbd_seq
    env.obj_repr.save(os.path.join(save_prefix, "obj_repr.npy"))
    
    # 然后保存 运行状态文件
    with open(os.path.join(save_prefix, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)

    print("done")