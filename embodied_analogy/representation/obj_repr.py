import napari
from embodied_analogy.representation.basic_structure import Data, Frame, Frames


class Obj_repr(Data):
    def __init__(self):
        self.initial_frame = Frame()
        self.frames = Frames()
        self.K = None
        self.Tw2c = None
        
        # 通过 reconstruct 恢复出的结果
        self.key_frames = Frames()
        self.track_type = None # either "open" or "close"
        self.joint_dict = None
        
    def clear_frames(self):
        self.frames.clear()
        
    def get_key_frames(self):
        pass
    
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
            self.frames._visualize(viewer, prefix="frames")
        if len(self.key_frames) > 0:
            self.key_frames._visualize(viewer, prefix="kframes")
        self._visualize(viewer)
        napari.run()


if __name__ == "__main__":
    # drawer
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/reconstruct/recon_data.pkl")
    # microwave
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/explore/explore_data.pkl")
    obj_repr.visualize()
    