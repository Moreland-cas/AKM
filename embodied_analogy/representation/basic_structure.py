import os
import pickle
import numpy as np
from embodied_analogy.utility.utils import napari_time_series_transform

class Data():
    def save(self, file_path):
        # 保存这个类到 file_path, 如果 file_path 所在的文件夹路径不存在, 则创建
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
        
    def visualize(self):
        pass
        
        
class Frame(Data):
    def __init__(
        self,
        rgb,
        depth,
        K=None,
        Tw2c=None,
        joint_state=None,
        obj_mask=None,
        dynamic_mask=None,
        contact2d=None,
        contact3d=None,
        franka2d=None,
        franka3d=None,
        franka_mask=None,
    ):
        self.rgb = rgb
        self.depth = depth
        self.K = K
        self.Tw2c = Tw2c
        self.joint_state = joint_state
        self.obj_mask = obj_mask
        self.dynamic_mask = dynamic_mask
        self.contact2d = contact2d
        self.contact3d = contact3d
        self.franka2d = franka2d
        self.franka3d = franka3d
        self.franka_mask = franka_mask
    
    
class Frames(Data):
    def __init__(
        self, 
        frame_list=np.array([], dtype=Frame), 
        fps=30, 
        K=None, 
        Tw2c=None
    ):
        """
        frame_list: list of class Frame
        """
        self.frame_list = frame_list
        self.num_frames = len(self.frame_list)
        self.fps = fps
        self.K = K
        self.Tw2c = Tw2c
        
    def clear(self):
        self.frame_list = []
        self.num_frames = 0
        
    def append(self, frame: Frame):
        self.frame_list.append(frame)
        self.num_frames += 1
        
    def get_rgb_seq(self):
        # T, H, W, C
        rgb_seq = np.stack([self.frame_list[i].rgb for i in range(self.num_frames)]) 
        return rgb_seq
        
    def get_depth_seq(self):
        # T, H, W
        depth_seq = np.stack([self.frame_list[i].depth for i in range(self.num_frames)]) 
        return depth_seq
        
    def get_franka2d_seq(self):
        # T, N, 2
        franka2d_seq = np.stack([self.frame_list[i].franka2d for i in range(self.num_frames)]) 
        return franka2d_seq
    
    def get_joint_states(self):
        # T
        joint_states = np.array([self.frame_list[i].joint_state for i in range(self.num_frames)]) 
        return joint_states
    
    def visualize(self):
        rgb_seq = self.get_rgb_seq()
        franka2d_seq = self.get_franka2d_seq()
        link_names = [
            'panda_link0', 'panda_link1', 'panda_link2', 
            'panda_link3', 'panda_link4', 'panda_link5', 
            'panda_link6', 'panda_link7', 'panda_link8', 
            'panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 
            'panda_rightfinger', 'camera_base_link', 'camera_link'
        ]
        
        import napari
        viewer = napari.view_image(rgb_seq, rgb=True)
        
        franka2d_data = napari_time_series_transform(franka2d_seq) # T*M, (1+2)
        franka2d_data = franka2d_data[:, [0, 2, 1]]
        for i in range(len(link_names)):
            T = len(rgb_seq)
            M = len(link_names)
            viewer.add_points(franka2d_data[i::M, :], face_color="red", name=link_names[i])
        napari.run()
        