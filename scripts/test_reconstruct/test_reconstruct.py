import os
import json
import pickle
import argparse
from embodied_analogy.utility.utils import initialize_napari
initialize_napari()
from embodied_analogy.representation.obj_repr import Obj_repr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def update_cfg(explore_cfg, args):
    # 更新 env_folder
    if args.obj_folder_path_explore is not None:
        explore_cfg['obj_folder_path_explore'] = args.obj_folder_path_explore
    if args.obj_folder_path_reconstruct is not None:
        explore_cfg['obj_folder_path_reconstruct'] = args.obj_folder_path_reconstruct
        
    if args.num_kframes is not None:
        explore_cfg['num_kframes'] = args.num_kframes
    if args.fine_lr is not None:
        explore_cfg['fine_lr'] = args.fine_lr
    if args.save_memory is not None:
        explore_cfg['save_memory'] = args.save_memory
    return explore_cfg

def read_args():
    parser = argparse.ArgumentParser(description='Update configuration for the robot.')
    
    # base_cfg arguments
    parser.add_argument('--obj_folder_path_explore', type=str, help='Folder where things are loaded')
    parser.add_argument('--obj_folder_path_reconstruct', type=str, help='Folder where things are stored')
    parser.add_argument('--num_kframes', type=int, help='Number of kframes')
    parser.add_argument('--fine_lr', type=float, help='fine lr during optimizing ICP loss')
    parser.add_argument('--save_memory', type=str2bool, help='whether to keep the frames in memory')

    args = parser.parse_args()

    return args

    
if __name__ == "__main__":
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
    
        obj_repr = Obj_repr.load(os.path.join(explore_folder, "obj_repr.npy"))
        obj_repr: Obj_repr
        print("load obj_repr from explore folder...")
        
        recon_result = obj_repr.reconstruct(
            num_kframes=recon_cfg["num_kframes"],
            obj_description=recon_cfg["obj_description"],
            fine_lr=recon_cfg["fine_lr"],
            file_path=None,
            evaluate=True,
            save_memory=recon_cfg["save_memory"],
            visualize=False,
        )
        
        with open(os.path.join(recon_save_folder, "cfg.json"), 'w', encoding='utf-8') as f:
            json.dump(recon_cfg, f, ensure_ascii=False, indent=4)
        
        # 然后保存 rgbd_seq
        obj_repr.save(os.path.join(recon_save_folder, "obj_repr.npy"))
        
        # 然后保存 运行状态文件
        with open(os.path.join(recon_save_folder, 'result.pkl'), 'wb') as f:
            pickle.dump(recon_result, f)
        
    print("done")