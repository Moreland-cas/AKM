import os
import napari
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
        
        
class Frame(Data):
    def __init__(
        self,
        rgb=None,
        depth=None,
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
        """
            NOTE: contact3d 在相机坐标系下, franka3d 在世界坐标系下
        """ 
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
    
    def _visualize(self, viewer: napari.Viewer, prefix=""):
        viewer.add_image(self.rgb, rgb=True, name=f"{prefix}_rgb")
        
        if self.franka_mask is not None:
            viewer.add_labels(self.franka_mask, name=f"{prefix}_robot_mask")
            
        if self.obj_mask is not None:
            viewer.add_labels(self.obj_mask, name=f"{prefix}_obj_mask")
        
        if self.dynamic_mask is not None:
            viewer.add_labels(self.dynamic_mask.astype(np.uint32), name=f"{prefix}_dynamic_mask")
            
        if self.contact2d is not None:
            u, v = self.contact2d
            viewer.add_points((v, u), face_color="red", name=f"{prefix}_contact2d")
    
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "frame visualization"
        self._visualize(viewer)
        napari.run()
    
    
class Frames(Data):
    def __init__(
        self, 
        frame_list=[], 
        fps=30, 
        K=None, 
        Tw2c=None
    ):
        """
        frame_list: list of class Frame
        """
        self.frame_list = frame_list
        self.fps = fps
        self.K = K
        self.Tw2c = Tw2c
    
    def num_frames(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        return self.frame_list[idx]
    
    def __len__(self):
        return len(self.frame_list)
    
    def clear(self):
        self.frame_list = []
        
    def append(self, frame: Frame):
        self.frame_list.append(frame)
        
    def get_rgb_seq(self):
        # T, H, W, C
        rgb_seq = np.stack([self.frame_list[i].rgb for i in range(self.num_frames())]) 
        return rgb_seq
        
    def get_depth_seq(self):
        # T, H, W
        depth_seq = np.stack([self.frame_list[i].depth for i in range(self.num_frames())]) 
        return depth_seq
        
    def get_franka2d_seq(self):
        # T, N, 2
        franka2d_seq = np.stack([self.frame_list[i].franka2d for i in range(self.num_frames())]) 
        return franka2d_seq
    
    def get_joint_states(self):
        # T
        joint_states = np.array([self.frame_list[i].joint_state for i in range(self.num_frames())]) 
        return joint_states
    
    def write_joint_states(self, joint_states):
        assert len(joint_states) == self.num_frames()
        for i, joint_state in enumerate(joint_states):
            self.frame_list[i].joint_state = joint_state
        
    def get_obj_mask_seq(self):
        # T, H, W
        obj_mask_seq = np.stack([self.frame_list[i].obj_mask for i in range(self.num_frames())]) 
        return obj_mask_seq
    
    def get_dynamic_seq(self):
        # T, H, W
        dynamic_mask_seq = np.stack([self.frame_list[i].dynamic_mask for i in range(self.num_frames())]) 
        return dynamic_mask_seq
    
    def _visualize(self, viewer: napari.Viewer, prefix="", visualize_franka2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        # 添加绘制 mask
        if self.frame_list[0].obj_mask is not None:
            obj_mask_seq = self.get_obj_mask_seq()
            viewer.add_labels(obj_mask_seq, name=f"{prefix}_obj_mask_seq")
        
        if self.frame_list[0].dynamic_mask is not None:
            dynamic_mask_seq = self.get_dynamic_seq()
            viewer.add_labels(dynamic_mask_seq.astype(np.int32), name=f"{prefix}_dynamic_mask_seq")
        
        if visualize_franka2d:
            franka2d_seq = self.get_franka2d_seq()
            link_names = [
                'panda_link0', 'panda_link1', 'panda_link2', 
                'panda_link3', 'panda_link4', 'panda_link5', 
                'panda_link6', 'panda_link7', 'panda_link8', 
                'panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 
                'panda_rightfinger', 'camera_base_link', 'camera_link'
            ]
            franka2d_data = napari_time_series_transform(franka2d_seq) # T*M, (1+2)
            franka2d_data = franka2d_data[:, [0, 2, 1]]
            for i in range(len(link_names)):
                T = len(rgb_seq)
                M = len(link_names)
                viewer.add_points(franka2d_data[i::M, :], face_color="red", name=f"{prefix}_{link_names[i]}")
        
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "frames visualization"
        self._visualize(viewer)
        napari.run()
        