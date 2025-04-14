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
    depth_image_to_pointcloud,
    line_to_line_distance
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
        
        # 这里的 coarse_joint_dict 和 fine_joint_dict 是在相机坐标系下估计的
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
        # 这里的 gt_joint_dict 是在 load object 阶段保存的, 存储的值是在世界坐标系下的 
        self.gt_joint_dict = {
            "joint_type": None,
            "joint_dir": None,
            "joint_start": None,
            "joint_states": None
        }
    
    def get_joint_param(self, resolution="coarse", frame="world"):
        assert resolution in ["coarse", "fine", "gt"]
        assert frame in ["world", "camera"]
        if resolution == "coarse":
            if self.coarse_joint_dict["joint_type"] is None:
                assert "coarse joint dict is not found"
            joint_dict = copy.deepcopy(self.coarse_joint_dict)
            if frame == "world":
                Tc2w = np.linalg.inv(self.Tw2c)
                joint_dict["joint_dir"] = Tc2w[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tc2w[:3, :3] @ joint_dict["joint_start"] + Tc2w[:3, 3]
        elif resolution == "fine":
            if self.fine_joint_dict["joint_type"] is None:
                assert "fine joint dict is not found"
            joint_dict = copy.deepcopy(self.fine_joint_dict)
            if frame == "world":
                Tc2w = np.linalg.inv(self.Tw2c)
                joint_dict["joint_dir"] = Tc2w[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tc2w[:3, :3] @ joint_dict["joint_start"] + Tc2w[:3, 3]
        elif resolution == "gt":
            # TODO 可能有问题
            if self.gt_joint_dict["joint_type"] is None:
                assert "Ground truth joint dict is not found"
            joint_dict = copy.deepcopy(self.gt_joint_dict)
            if frame == "camera":
                Tw2c = self.Tw2c
                joint_dict["joint_dir"] = Tw2c[:3, :3] @ joint_dict["joint_dir"]
                joint_dict["joint_start"] = Tw2c[:3, :3] @ joint_dict["joint_start"] + Tw2c[:3, 3]
        return joint_dict
    
    def compute_joint_error(self):
        """
        分别计算 coarse_joint_dict, fine_joint_dict 与 gt_joint_dict 的差距, 并打印, 保存
        还要分别计算 frames 和 kframes 的 joint_state_error
        """
        coarse_w = self.get_joint_param(resolution="coarse", frame="world")
        fine_w = self.get_joint_param(resolution="fine", frame="world")
        gt_w = self.get_joint_param(resolution="gt", frame="world")
        result = {
            "coarse_w": coarse_w,
            "fine_w": fine_w,
            "gt_w": gt_w,
            "coarse_loss": {
                "type_err": 0,
                "angle_err": 0,
                "pos_err": 0,
            },
            "fine_loss": {
                "type_err": 0,
                "angle_err": 0,
                "pos_err": 0,
            }
        }
        # TODO: 这里可能要对 angle_err 取一个 minimum
        result["coarse_loss"]["type_err"] = 1 if coarse_w["joint_type"] != gt_w["joint_type"] else 0
        result["coarse_loss"]["angle_err"] = np.arccos(np.dot(coarse_w["joint_dir"], gt_w["joint_dir"]))
        
        result["fine_loss"]["type_err"] = 1 if fine_w["joint_type"] != gt_w["joint_type"] else 0
        result["fine_loss"]["angle_err"] = np.arccos(np.dot(fine_w["joint_dir"], gt_w["joint_dir"]))
        
        if gt_w["joint_type"] == "revolute":
            result["coarse_loss"]["pos_err"] = line_to_line_distance(
                P1=coarse_w["joint_start"],
                d1=coarse_w["joint_dir"],
                P2=gt_w["joint_start"],
                d2=gt_w["joint_dir"]
            )
            result["fine_loss"]["pos_err"] = line_to_line_distance(
                P1=fine_w["joint_start"],
                d1=fine_w["joint_dir"],
                P2=gt_w["joint_start"],
                d2=gt_w["joint_dir"]
            )
            
        return result
    
    def clear_frames(self):
        self.frames.clear()
        
    def clear_kframes(self):
        self.kframes.clear()
    
    def initialize_kframes(self, num_kframes, save_memory=True):
        self.clear_kframes()
        
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
            gt_joint_dict=self.get_joint_param(resolution="gt", frame="camera"),
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
        # num_initial_pts=1000,
        num_kframes=5,
        obj_description="drawer",
        fine_lr=1e-3,
        file_path=None,
        evaluate=False,
        save_memory=True,
        visualize=True,
    ):
        """
            从 frames 中恢复出 joint state dict
        """
        # self.frames[0].segment_obj(
        #     obj_description=obj_description,
        #     post_process_mask=True,
        #     filter=True,
        #     visualize=visualize
        # )
        # self.frames[0].sample_points(num_points=num_initial_pts, visualize=visualize)
        # self.frames.track_points(visualize=visualize)
        # self.frames.track2d_to_3d(filter=True, visualize=visualize)
        # self.frames.cluster_track3d(visualize=visualize)
        
        self.coarse_joint_estimation(visualize=visualize)
        self.initialize_kframes(num_kframes=num_kframes, save_memory=save_memory)
        self.kframes.segment_obj(obj_description=obj_description, visualize=visualize)
        self.kframes.classify_dynamics(
            filter=True,
            joint_dict=self.coarse_joint_dict,
            visualize=visualize
        )
        self.fine_joint_estimation(lr=fine_lr, visualize=visualize)
           
        if file_path is not None:
            # self.visualize()
            self.save(file_path)
        
        result = None
        if evaluate:
            if self.gt_joint_dict["joint_type"] is None:
                return 
            result = self.compute_joint_error()
            print("Reconstruction Result:")
            for k, v in result.items():
                print(k, v)
        return result
            
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
        
        # 初始化 query_frame 的 joint 状态
        """
        大致的意思是说找到一个不错的 init_opti_state 就行
        策略是 sample 多个 joint state, 然后依次得到多个 transformed_moving_pc
        然后找到与 query_frame 最接近的那个 joint state
        NOTE: 由于我们强制了 kframes[0] 其实就是 manipulate first frame, 且 first frame 的 joint state 是 0
        所以之后的 joint_delta 都是直接加在 kframes[0] 的 joint_state 上
        """
        # 根据关节种类选择 sample 的范围
        if self.coarse_joint_dict["joint_type"] == "revolute":
            joint_delta = np.pi / 2.
        elif self.coarse_joint_dict["joint_type"] == "prismatic":
            joint_delta = 0.5
        
        sampled_states = np.linspace(0, joint_delta, 10)
        # 获取 kframes[0] 中的 moving_part
        moving_pc = depth_image_to_pointcloud(
            depth_image=self.kframes[0].depth, 
            mask=self.kframes[0].dynamic_mask == MOVING_LABEL, 
            K=K
        ) # N, 3
        
        best_err = 1e10
        best_matched_idx = -1
        for i, sampled_state in enumerate(sampled_states):
            Tref2tgt = joint_data_to_transform_np(
                joint_type=self.fine_joint_dict["joint_type"], 
                joint_dir=self.fine_joint_dict["joint_dir"],
                joint_start=self.fine_joint_dict["joint_start"],
                joint_state_ref2tgt=sampled_state
            )
            tf_moving_pc = (Tref2tgt[:3, :3] @ moving_pc.T).T + Tref2tgt[:3, 3] # N, 3
            tf_moving_uv, moving_depth = camera_to_image(tf_moving_pc, K) # (N, 2), (N, )
            
            # 对于超出 depth 范围的 tf_moving_uv 进行过滤
            in_img_mask = (tf_moving_uv[:, 0] >= 0) & (tf_moving_uv[:, 0] < query_frame.depth.shape[1]) & \
                          (tf_moving_uv[:, 1] >= 0) & (tf_moving_uv[:, 1] < query_frame.depth.shape[0])
            tf_moving_uv, moving_depth = tf_moving_uv[in_img_mask], moving_depth[in_img_mask]
            
            # 读取 query_frame 在 tf_moving_uv 处的 depth, 并与 moving_depth 做比较
            query_depth = query_frame.depth[tf_moving_uv[:, 1].astype(np.int32), tf_moving_uv[:, 0].astype(np.int32)]
            cur_mean_err = np.abs(query_depth - moving_depth).mean()
            if cur_mean_err < best_err:
                best_err = cur_mean_err
                best_matched_idx = i
        query_state = sampled_states[best_matched_idx] 
        
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
                # TODO: 这里有越界的 bug
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
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/48878/recon_data.pkl")
    # obj_repr.visualize()
    
    # array([ 0.47273752,  0.16408749, -0.8657913 ], dtype=float32)
    # pass
    # obj_repr = Obj_repr()
    # obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore/41083_2_prismatic/explore/obj_repr.npy")
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore/41083_2_prismatic/explore/obj_repr.npy")
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/logs/test_explore_4_11/40147_1_prismatic/explore/obj_repr.npy")
    print(obj_repr.frames.track2d_seq.shape)
    # obj_repr.visualize()