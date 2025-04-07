import json


# 读取 args, 生成 cfg 文件, 输入给 manipulate_env 进行测试, 并返回测试结果 (数值 or logging)

import argparse
import json

def update_cfg(base_cfg, args):
    # 更新 base_cfg 字典中的值
    if args.phy_timestep is not None:
        base_cfg['base_cfg']['phy_timestep'] = args.phy_timestep
    if args.planner_timestep is not None:
        base_cfg['base_cfg']['planner_timestep'] = args.planner_timestep
    if args.use_sapien2 is not None:
        base_cfg['base_cfg']['use_sapien2'] = args.use_sapien2

    if args.record_fps is not None:
        base_cfg['explore_cfg']['record_fps'] = args.record_fps
    if args.pertubation_distance is not None:
        base_cfg['explore_cfg']['pertubation_distance'] = args.pertubation_distance
    if args.max_tries is not None:
        base_cfg['explore_cfg']['max_tries'] = args.max_tries
    if args.update_sigma is not None:
        base_cfg['explore_cfg']['update_sigma'] = args.update_sigma

    if args.num_initial_pts is not None:
        base_cfg['recon_cfg']['num_initial_pts'] = args.num_initial_pts
    if args.num_kframes is not None:
        base_cfg['recon_cfg']['num_kframes'] = args.num_kframes
    if args.fine_lr is not None:
        base_cfg['recon_cfg']['fine_lr'] = args.fine_lr

    if args.reloc_lr is not None:
        base_cfg['manip_cfg']['reloc_lr'] = args.reloc_lr
    if args.reserved_distance is not None:
        base_cfg['manip_cfg']['reserved_distance'] = args.reserved_distance

    if args.instruction is not None:
        base_cfg['task_cfg']['instruction'] = args.instruction
    if args.obj_description is not None:
        base_cfg['task_cfg']['obj_description'] = args.obj_description
    if args.delta is not None:
        base_cfg['task_cfg']['delta'] = args.delta
    if args.asset_path is not None:
        base_cfg['task_cfg']['obj_cfg']['asset_path'] = args.asset_path
    if args.scale is not None:
        base_cfg['task_cfg']['obj_cfg']['scale'] = args.scale
    if args.active_link_name is not None:
        base_cfg['task_cfg']['obj_cfg']['active_link_name'] = args.active_link_name
    if args.active_joint_name is not None:
        base_cfg['task_cfg']['obj_cfg']['active_joint_name'] = args.active_joint_name

    return base_cfg

def read_cfg():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--phy_timestep', type=float, help='Physical timestep')
    parser.add_argument('--planner_timestep', type=float, help='Planner timestep')
    parser.add_argument('--use_sapien2', type=bool, default=True, help='Use Sapien2')

    # explore_cfg arguments
    parser.add_argument('--record_fps', type=int, help='Record FPS')
    parser.add_argument('--pertubation_distance', type=float, help='Perturbation distance')
    parser.add_argument('--max_tries', type=int, help='Maximum tries')
    parser.add_argument('--update_sigma', type=float, help='Update sigma')

    # recon_cfg arguments
    parser.add_argument('--num_initial_pts', type=int, help='Number of initial points')
    parser.add_argument('--num_kframes', type=int, help='Number of keyframes')
    parser.add_argument('--fine_lr', type=float, help='Fine learning rate')

    # manip_cfg arguments
    parser.add_argument('--reloc_lr', type=float, help='Relocation learning rate')
    parser.add_argument('--reserved_distance', type=float, help='Reserved distance')

    # task_cfg arguments
    parser.add_argument('--instruction', type=str, help='Task instruction')
    parser.add_argument('--obj_description', type=str, help='Object description')
    parser.add_argument('--delta', type=float, help='Delta value')
    parser.add_argument('--asset_path', type=str, help='Asset path')
    parser.add_argument('--scale', type=float, help='Scale')
    parser.add_argument('--active_link_name', type=str, help='Active link name')
    parser.add_argument('--active_joint_name', type=str, help='Active joint name')

    args = parser.parse_args()

    # 初始的配置
    base_cfg = {
        "base_cfg": {
            "phy_timestep": 1 / 250.,
            "planner_timestep": None,
            "use_sapien2": True
        },
        "robot_cfg": {},
        "explore_cfg": {
            "record_fps": 30,
            "pertubation_distance": 0.1,
            "max_tries": 100,
            "update_sigma": 0.05
        },
        "recon_cfg": {
            "num_initial_pts": 1000,
            "num_kframes": 5,
            "fine_lr": 1e-3
        },
        "manip_cfg": {
            "reloc_lr": 3e-3,
            "reserved_distance": 0.05
        },
        "task_cfg": {
            "instruction": None,
            "obj_description": None,
            "delta": None,
            "obj_cfg": {
                "asset_path": None,
                "scale": 1.,
                "active_link_name": None,
                "active_joint_name": None,
            }
        }
    }

    # 更新配置
    updated_cfg = update_cfg(base_cfg, args)

    return updated_cfg

if __name__ == '__main__':
    cfg = read_cfg()

    from embodied_analogy.environment.manipulate_env import ManipulateEnv
    me = ManipulateEnv(
        base_cfg=cfg["base_cfg"],
        robot_cfg=cfg["robot_cfg"],
        explore_cfg=cfg["explore_cfg"],
        recon_cfg=cfg["recon_cfg"],
        manip_cfg=cfg["manip_cfg"],
        task_cfg=cfg["task_cfg"]
    )
    # TODO: 顺便返回 logging, 以及 cfgs
    result = me.main()
    print(cfg)
    print(result)
