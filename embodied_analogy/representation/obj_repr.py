import copy
import numpy as np
import napari
from embodied_analogy.representation.basic_structure import Data, Frame, Frames
from embodied_analogy.utility.estimation.coarse_joint_est import coarse_estimation
from embodied_analogy.utility.estimation.fine_joint_est import fine_estimation

from embodied_analogy.utility.utils import (
    initialize_napari,   
    farthest_scale_sampling,
    set_random_seed,
    joint_data_to_transform_np,
    get_depth_mask,
    camera_to_image,
    depth_image_to_pointcloud
)
initialize_napari()
from embodied_analogy.utility.constants import *
set_random_seed(SEED)


class Obj_repr(Data):
    def __init__(self):
        
        self.obj_description = None
        self.initial_frame = Frame()
        self.frames = Frames()
        self.K = None
        self.Tw2c = None
        
        # 通过 reconstruct 恢复出的结果
        self.kframes = Frames()
        self.track_type = None # either "open" or "close"
        """
        NOTE: 这里的 joint_dir 和 joint_start 均在相机坐标系下
        """
        self.coarse_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
        self.fine_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
        
    def clear_frames(self):
        self.frames.clear()
        
    def clear_kframes(self):
        self.kframes.clear()
        
    def initialize_kframes(self, num_kframes, save_memory=True):
        self.kframes.fps = self.frames.fps
        self.kframes.K = self.frames.K
        self.kframes.Tw2c = self.frames.Tw2c
        
        self.kframes.moving_mask = self.frames.moving_mask
        self.kframes.static_mask = self.frames.static_mask
        
        # NOTE: 这里要保证 frames[0] 一定被选进 kframes, 因为 ph_pose 保存在了 frames[0] 中
        kf_idxs = farthest_scale_sampling(
            arr=self.coarse_joint_dict["joint_states"],
            M=num_kframes,
            include_first=True
        )
        self.kf_idxs = kf_idxs
        
        self.kframes.track2d_seq = self.frames.track2d_seq[kf_idxs, ...]
        
        self.clear_kframes()
        for i, kf_idx in enumerate(kf_idxs):
            tmp_frame = copy.deepcopy(self.frames[kf_idx])
            self.kframes.append(tmp_frame)
        
        if save_memory:
            self.clear_frames()
    
    def coarse_joint_estimation(self, visualize=False):
        coarse_joint_dict = coarse_estimation(
            tracks_3d=self.frames.track3d_seq[:, self.frames.moving_mask, :], 
            visualize=visualize
        )
        self.coarse_joint_dict = coarse_joint_dict
        self.frames.write_joint_states(coarse_joint_dict["joint_states"])
    
    def fine_joint_estimation(self, lr=1e-3, visualize=False):
        joint_type = self.coarse_joint_dict["joint_type"]
        fine_joint_dict = fine_estimation(
            K=self.K,
            joint_type=self.coarse_joint_dict["joint_type"],
            joint_dir=self.coarse_joint_dict["joint_dir"],
            joint_start=self.coarse_joint_dict["joint_start"],
            joint_states=self.kframes.get_joint_states(),
            depth_seq=self.kframes.get_depth_seq(),
            dynamic_seq=self.kframes.get_dynamic_seq(),
            opti_joint_dir=True,
            opti_joint_start=(joint_type=="revolute"),
            opti_joint_states_mask=np.arange(self.kframes.num_frames())!=0,
            # update_dynamic_mask=None,
            lr=lr, # 1mm
            visualize=visualize
        )
        # 在这里将更新的 joint_dict 和 joint_states 写回 obj_repr
        self.kframes.write_joint_states(fine_joint_dict["joint_states"])
        self.fine_joint_dict = fine_joint_dict
        
        # TODO: 默认在 explore 阶段是打开的轨迹
        track_type = "open"
        self.track_type = track_type
        
        # 看下当前的 joint_dir 到底对应 open 还是 close, 如果对应 close, 需要将 joint 进行翻转
        # if track_type == "close":
        #     reverse_joint_dict(coarse_state_dict)
        #     reverse_joint_dict(fine_state_dict)
        
    def reconstruct(
        self,
        num_initial_pts=1000,
        num_kframes=5,
        obj_description="drawer",
        fine_lr=1e-3,
        reloc_lr=3e-3,
        file_path=None,
        gt_joint_dir_w=None,
        visualize=True,
    ):
        """
            从 frames 中恢复出 joint state dict, 并对于 initial_frame 进行重定位
        """
        self.frames[0].segment_obj(
            obj_description=obj_description,
            post_process_mask=True,
            filter=True,
            visualize=visualize
        )
        self.frames[0].sample_points(num_points=num_initial_pts, visualize=visualize)
        self.frames.track_points(visualize=visualize)
        self.frames.track2d_to_3d(visualize=visualize)
        self.frames.cluster_track3d(visualize=visualize)
        self.coarse_joint_estimation(visualize=visualize)
        self.initialize_kframes(num_kframes=num_kframes, save_memory=True)
        self.kframes.segment_obj(obj_description=obj_description, visualize=visualize)
        self.kframes.classify_dynamics(
            filter=True,
            joint_dict=self.coarse_joint_dict,
            visualize=visualize
        )
        self.fine_joint_estimation(lr=fine_lr, visualize=True)
        
        # 在这里对于 initial_frame 进行重定位
        self.initial_frame = self.reloc(
            query_frame=self.initial_frame,
            update_query_dynamic=True,
            reloc_lr=reloc_lr,
            visualize=visualize
        )
           
        if file_path is not None:
            self.visualize()
            self.save(file_path)
        
        if gt_joint_dir_w is not None:
            print(f"\tgt axis: {gt_joint_dir_w}")

            coarse_dir_c = self.coarse_joint_dict["joint_dir"] 
            coarse_dir_w = self.Tw2c[:3, :3].T @ coarse_dir_c
            dot_before = np.dot(gt_joint_dir_w, coarse_dir_w)
            print(f"\tbefore: {np.degrees(np.arccos(dot_before))}")
            print("\tjoint axis: ", coarse_dir_w)
            
            print("\tjoint states: ", self.coarse_joint_dict["joint_states"][self.kf_idxs])

            fine_dir_c = self.fine_joint_dict["joint_dir"] 
            fine_dir_w = self.Tw2c[:3, :3].T @ fine_dir_c
            dot_after = np.dot(gt_joint_dir_w, fine_dir_w)
            print(f"\tafter : {np.degrees(np.arccos(dot_after))}")
            print("\tjoint axis: ", fine_dir_w)
            
            print("\tjoint states: ", self.fine_joint_dict["joint_states"])
    
    def reloc(
        self,
        query_frame: Frame,
        update_query_dynamic=False,
        reloc_lr=3e-3,
        visualize=False
    ) -> Frame:
        """
        对 query frame 的 joint_state, dynamic 进行恢复 
        其中 dynamic 的恢复来自 sam2 和 kframes
        query_frame:
            需包含 query_depth, query_dynamic
        """
        if query_frame.obj_mask is None:
            print("obj_mask of query_frame is None, run grounded sam first..")
            query_frame.segment_obj(obj_description=self.obj_description)
        
        if query_frame.dynamic_mask is None:
            print("dynamic_mask of query_frame is None, initializing using obj_mask..")
            query_frame.dynamic_mask = query_frame.obj_mask.astype(np.int32) * MOVING_LABEL
        
        # 从 obj_repr 中读取必要的数据
        K = self.K
        query_dynamic = query_frame.dynamic_mask
        query_depth = query_frame.depth
        ref_depths = self.kframes.get_depth_seq()
        ref_dynamics = self.kframes.get_dynamic_seq()
        ref_joint_states = self.kframes.get_joint_states()
        assert len(ref_joint_states) == len(self.kframes)
            
        # 首先获取当前帧物体的 mask, 是不是也可以不需要 mask
        num_ref = len(self.kframes)
        
        # 初始化状态, 通过 rgb 或者 depth 找到最近的图像, 我觉得可以先通过 depth
        best_err = 1e10
        best_matched_idx = -1
        for i in range(num_ref):
            query_depth_mask = get_depth_mask(query_depth, K, self.Tw2c)
            ref_depth = ref_depths[i]
            ref_depth_mask = get_depth_mask(ref_depth, K, self.Tw2c)
            
            # 计算 mask 的交集, 对交集中点的深度计算一个观测误差
            inter_mask = query_depth_mask & ref_depth_mask
            delta_depth = (ref_depth - query_depth)**2
            cur_mean_err = delta_depth[inter_mask].mean()
            
            if cur_mean_err < best_err:
                best_err = cur_mean_err
                best_matched_idx = i
        query_state = ref_joint_states[best_matched_idx] 
        
        # 将 query_frame 写进 obj_repr.kframes, 然后复用 fine_estimation 对初始帧进行优化
        query_frame.joint_state = query_state
        self.kframes.frame_list.insert(0, query_frame)
        
        fine_joint_dict = fine_estimation(
            K=self.K,
            joint_type=self.fine_joint_dict["joint_type"],
            joint_dir=self.fine_joint_dict["joint_dir"],
            joint_start=self.fine_joint_dict["joint_start"],
            joint_states=self.kframes.get_joint_states(),
            depth_seq=self.kframes.get_depth_seq(),
            dynamic_seq=self.kframes.get_dynamic_seq(),
            opti_joint_dir=False,
            opti_joint_start=False,
            opti_joint_states_mask=np.arange(num_ref+1)==0,
            # 目前不更新 dynamic_mask
            lr=reloc_lr,
            visualize=False
        )
        # 然后在这里把 query_frame 从 keyframes 中吐出来
        query_frame = self.kframes.frame_list.pop(0)
        query_frame.joint_state = fine_joint_dict["joint_states"][0]
        
        # 也就是说把 ref_frame 的 moving part 投影到 query frame 上, 对 query_dynamic 进行一个更新 (其余部分设置为 unknown)
        if update_query_dynamic:
            query_dynamic_zero = np.zeros_like(query_dynamic).astype(np.bool_) # H, W
            for i in range(num_ref):
                ref_moving_pc = depth_image_to_pointcloud(ref_depths[i], ref_dynamics[i]==MOVING_LABEL, K) # N, 3
                Tref2query = joint_data_to_transform_np(
                    joint_type=self.fine_joint_dict["joint_type"],
                    joint_dir=self.fine_joint_dict["joint_dir"],
                    joint_start=self.fine_joint_dict["joint_start"],
                    joint_state_ref2tgt=query_frame.joint_state-ref_joint_states[i]
                )
                ref_moving_pc_aug = np.concatenate([ref_moving_pc, np.ones((len(ref_moving_pc), 1))], axis=1) # N, 4
                moving_pc = (ref_moving_pc_aug @ Tref2query.T)[:, :3] # N, 3
                moving_uv, _ = camera_to_image(moving_pc, K) # N, 2
                moving_uv = moving_uv.astype(np.int32)
                tmp_zero = np.zeros_like(query_dynamic)
                tmp_zero[moving_uv[:, 1], moving_uv[:, 0]] = True # H, W
                query_dynamic_zero = query_dynamic_zero | tmp_zero
            
            # 将投影来的点与 obj_mask 进行取交
            obj_mask = query_frame.obj_mask
            # 先把整个物体区域赋值为 UNKNOWN
            query_dynamic_updated = obj_mask * UNKNOWN_LABEL  
            # 再把其中的移动部分赋值为 MOVING_LABEL
            query_dynamic_updated[(obj_mask & query_dynamic_zero).astype(np.bool_)] = MOVING_LABEL
            query_frame.dynamic_mask = query_dynamic_updated
        
        if visualize:
            # 展示 dynamic query 的变化
            viewer = napari.Viewer()
            viewer.title = "relocalization"
            query_frame._visualize(viewer, prefix="query")
            self._visualize(viewer, prefix="obj_repr")
            print("query frame joint state: ", query_frame.joint_state)
            napari.run()
            
        return query_frame
    
    def _visualize(self, viewer: napari.Viewer, prefix=""):
        pass
    
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "object representation"
        self.initial_frame._visualize(viewer, prefix="initial")
        if len(self.frames) > 0:
            self.frames._visualize_f(viewer, prefix="frames")
        if len(self.kframes) > 0:
            self.kframes._visualize_kf(viewer, prefix="kframes")
        self._visualize(viewer)
        napari.run()


if __name__ == "__main__":
    # drawer
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/reconstruct/recon_data.pkl")
    # microwave
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/explore/explore_data.pkl")
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/reconstruct/recon_data.pkl")
    # obj_repr.visualize()
    
    # array([ 0.47273752,  0.16408749, -0.8657913 ], dtype=float32)
    pass