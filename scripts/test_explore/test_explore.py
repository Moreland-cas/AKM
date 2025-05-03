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
    if args.instruction is not None:
        base_cfg['instruction'] = args.instruction
    if args.num_initial_pts is not None:
        base_cfg['num_initial_pts'] = args.num_initial_pts
        
    # load obj information
    if args.obj_description is not None:
        base_cfg['obj_description'] = args.obj_description
    if args.joint_type is not None:
        base_cfg['joint_type'] = args.joint_type
    if args.obj_index is not None:
        base_cfg['obj_index'] = args.obj_index
    if args.joint_index is not None:
        base_cfg['joint_index'] = args.joint_index
    if args.init_joint_state is not None:
        base_cfg['init_joint_state'] = args.init_joint_state
    if args.asset_path is not None:
        base_cfg['asset_path'] = args.asset_path
    if args.load_scale is not None:
        base_cfg['load_scale'] = args.load_scale
    if args.load_pose is not None:
        base_cfg['load_pose'] = args.load_pose
    if args.load_quat is not None:
        base_cfg['load_quat'] = args.load_quat
    if args.active_link_name is not None:
        base_cfg['active_link_name'] = args.active_link_name
    if args.active_joint_name is not None:
        base_cfg['active_joint_name'] = args.active_joint_name

    return base_cfg

def read_cfg():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--phy_timestep', type=float, help='Physical timestep')
    parser.add_argument('--planner_timestep', type=float, help='Planner timestep')
    parser.add_argument('--use_sapien2', type=str2bool, default=True, help='Use Sapien2')

    # explore_cfg arguments
    parser.add_argument('--record_fps', type=int, help='Record FPS')
    parser.add_argument('--fully_zeroshot', type=str2bool, help='Whether to remove cabinet class in RAM')
    parser.add_argument('--pertubation_distance', type=float, help='Perturbation distance')
    parser.add_argument('--valid_thresh', type=float, help='factor of actual Perturbation distance')
    parser.add_argument('--max_tries', type=int, help='Maximum tries')
    parser.add_argument('--update_sigma', type=float, help='Update sigma')
    parser.add_argument('--reserved_distance', type=float, help='Reserved distance')
    parser.add_argument('--num_initial_pts', type=int, help='number of tracking points initialized on first frame')

    # task_cfg arguments
    parser.add_argument('--instruction', type=str, help='Task instruction')
    parser.add_argument('--obj_description', type=str, help='Object description')
    
    # obj_cfg arguments
    parser.add_argument('--asset_path', type=str, help='Asset path')
    parser.add_argument('--joint_type', type=str, help='joint type')
    parser.add_argument('--obj_index', type=str, help='obj_index')
    parser.add_argument('--joint_index', type=str, help='joint_index')
    parser.add_argument('--init_joint_state', type=str, help='initial joint state when loading')
    parser.add_argument('--load_scale', type=none_or_float, help='object scale when loading')
    parser.add_argument('--load_pose', type=none_or_list, help='object pose when loading')
    parser.add_argument('--load_quat', type=none_or_list, help='object quat when loading')
    parser.add_argument('--active_link_name', type=str, help='Active link name')
    parser.add_argument('--active_joint_name', type=str, help='Active joint name')
    
    # experiment related
    parser.add_argument('--obj_folder_path_explore', type=str, help='path to explore object folder')

    args = parser.parse_args()

    # 初始的配置
    default_cfg = {}

    # 更新配置
    updated_cfg = update_cfg(default_cfg, args)

    return updated_cfg

# 读取 args, 生成 cfg 文件, 输入给 manipulate_env 进行测试, 并返回测试结果 (数值 or logging)

if __name__ == '__main__':
    # exp folder: /logs/run_xxx/obj_idx_type/explore
    import os
    import pickle
    cfg = read_cfg()
    
    # 首先保存 cfg 文件
    save_prefix = cfg['obj_folder_path_explore']
    env = ExploreEnv(cfg)
    
    with open(os.path.join(save_prefix, "cfg.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
        
    result = env.explore_stage(visualize=False)
    
    # 然后保存 rgbd_seq
    env.obj_repr.save(os.path.join(save_prefix, "obj_repr.npy"))
    
    # 然后保存 运行状态文件
    with open(os.path.join(save_prefix, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)

    print("done")