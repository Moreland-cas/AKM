import napari
from embodied_analogy.representation.basic_structure import Data, Frame, Frames
from embodied_analogy.utility.utils import napari_time_series_transform


class Obj_repr(Data):
    def __init__(
        self,
    ):
        self.initial_frame = Frame(rgb=None, depth=None)
        self.frames = Frames()
        self.key_frames = Frames()

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
    
    def _visualize(self, viewer: napari.Viewer):
        pass
    
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "object representation"
        self.initial_frame._visualize(viewer)
        self.frames._visualize(viewer)
        self._visualize(viewer)
        napari.run()


if __name__ == "__main__":
    # drawer
    # obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/44962/explore/explore_data.pkl")
    # microwave
    obj_repr = Obj_repr.load("/home/zby/Programs/Embodied_Analogy/assets/tmp/7221/explore/explore_data.pkl")
    obj_repr.visualize()
    