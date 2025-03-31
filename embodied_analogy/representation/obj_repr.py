import copy
import numpy as np
import napari
from embodied_analogy.representation.basic_structure import Data, Frame, Frames
from embodied_analogy.estimation.coarse_joint_est import coarse_estimation
from embodied_analogy.estimation.fine_joint_est import fine_estimation
from embodied_analogy.utility.utils import farthest_scale_sampling


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
        
        kf_idxs = farthest_scale_sampling(self.coarse_joint_dict["joint_states"], num_kframes)
        
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
            obj_repr=self,
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
        
    def reconstruct(self):
        """
            从 initial_frame 和 frames 中恢复出 joint axis 和 joint states
        """
        pass
    
    def reloc(self, frame: Frame):
        pass
    
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
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/reconstruct/recon_data.pkl")
    # microwave
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/explore/explore_data.pkl")
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/reconstruct/recon_data.pkl")
    obj_repr.visualize()
    
    # array([ 0.47273752,  0.16408749, -0.8657913 ], dtype=float32)