import copy
import napari
from embodied_analogy.representation.basic_structure import Data, Frame, Frames
from embodied_analogy.estimation.coarse_joint_est import coarse_estimation
from embodied_analogy.estimation.fine_joint_est import (
    fine_estimation,
    filter_dynamic_seq
)
from embodied_analogy.utility.utils import (
    initialize_napari,   
    set_random_seed,
    farthest_scale_sampling,
    get_dynamic_seq,
    get_depth_mask_seq,
)


class Obj_repr(Data):
    def __init__(
        self,
        num_kframes=5,
    ):
        self.obj_description = None
        self.initial_frame = Frame()
        self.frames = Frames()
        self.K = None
        self.Tw2c = None
        
        self.num_kframes = num_kframes
        
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
        
    def initialize_kframes(self, save_memory=False):
        kf_idxs = farthest_scale_sampling(self.coarse_joint_dict["joint_states"], num_kframes=self.num_kframes)
        self.clear_kframes()
        for i, kf_idx in enumerate(kf_idxs):
            tmp_frame = copy.deepcopy(obj_repr.frames[kf_idx])
            # tmp_frame.obj_mask = obj_mask_seq[i]
            # tmp_frame.dynamic_mask = dynamic_seq_updated[i]
            obj_repr.kframes.append(tmp_frame)
            
        if save_memory:
            self.clear_frames()
    
    def coarse_joint_estimation(self, visualize=False):
        coarse_joint_dict = coarse_estimation(
            tracks_3d=self.frames.track3d_seq[:, self.frames.moving_mask, :], 
            visualize=visualize
        )
        self.coarse_joint_dict = coarse_joint_dict
        obj_repr.frames.write_joint_states(coarse_joint_dict["joint_states"])
        
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