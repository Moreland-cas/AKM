from embodied_analogy.representation.basic_structure import Data, Frame, Frames


class Obj_repr(Data):
    def __init__(
        self,
    ):
        self.initial_frame = Frame()
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
    
    def visualize(self):
        pass
    