import os
import napari
import pickle
import numpy as np
from graspnetAPI import GraspGroup
from embodied_analogy.utility.utils import (
    napari_time_series_transform,
    image_to_camera,
    get_depth_mask,
    depth_image_to_pointcloud,
    crop_nearby_points,
    fit_plane_ransac,
    visualize_pc
)
from embodied_analogy.grasping.anygrasp import (
    detect_grasp_anygrasp,
    filter_grasp_group,
    sort_grasp_group,
    crop_grasp
)

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
        dir_out=None,
        grasp_group: GraspGroup = None,
        robot2d=None,
        robot3d=None,
        robot_mask=None,
    ):
        """
            NOTE: contact3d 在相机坐标系下, robot3d 在世界坐标系下
        """ 
        self.rgb = rgb
        self.depth = depth
        self.K = K
        self.Tw2c = Tw2c
        
        self.obj_mask = obj_mask
        self.dynamic_mask = dynamic_mask
        self.joint_state = joint_state
        
        self.contact2d = contact2d
        self.contact3d = contact3d
        self.dir_out = dir_out
        self.grasp_group = grasp_group
        
        self.robot2d = robot2d
        self.robot3d = robot3d
        self.robot_mask = robot_mask
    
    def _visualize(self, viewer: napari.Viewer, prefix=""):
        viewer.add_image(self.rgb, rgb=True, name=f"{prefix}_rgb")
        
        if self.robot_mask is not None:
            viewer.add_labels(self.robot_mask, name=f"{prefix}_robot_mask")
            
        if self.obj_mask is not None:
            viewer.add_labels(self.obj_mask, name=f"{prefix}_obj_mask")
        
        if self.dynamic_mask is not None:
            viewer.add_labels(self.dynamic_mask.astype(np.uint32), name=f"{prefix}_dynamic_mask")
            
        if self.contact2d is not None:
            u, v = self.contact2d
            viewer.add_points((v, u), face_color="red", name=f"{prefix}_contact2d")
    
    def detect_grasp(self, visualize=False) -> GraspGroup:
        """
        返回当前 frame 的点云上 contact2d 附近的一个 graspGroup
        """
        assert (self.contact2d is not None) and (self.obj_mask is not None)
        c2d_int = self.contact2d.astype(np.int32)
        if self.contact3d is None:
            contact3d = image_to_camera(
                uv=self.contact2d[None], # 1, 3
                depth=np.array(self.depth[c2d_int[1], c2d_int[0]])[None], # 1, 1
                K=self.K,
            )[0]
            self.contact3d = contact3d
        contact3d = self.contact3d
        
        # 找到 contact3d 附近的点
        depth_mask = get_depth_mask(self.depth, self.K, self.Tw2c)
        obj_pc = depth_image_to_pointcloud(
            depth_image=self.depth,
            mask = self.obj_mask & depth_mask,
            K=self.K
        )
        pc_colors = self.rgb[self.obj_mask & depth_mask]
        
        crop_mask = crop_nearby_points(
            point_clouds=obj_pc,
            contact3d=self.contact3d,
            radius=0.1
        )
        cropped_pc = obj_pc[crop_mask]
        cropped_colors = pc_colors[crop_mask]
        
        # fit 一个局部的 normal 做为 grasp 的方向
        plane_normal = fit_plane_ransac(
            points=cropped_pc,
            threshold=0.01, 
            max_iterations=100,
        )
        if (plane_normal * np.array([0, 0, -1])).sum() > 0:
            dir_out = plane_normal
        else:
            dir_out = -plane_normal
        self.dir_out = dir_out
        
        # 1) 在 crop 出的点云上检测 grasp
        # gg = detect_grasp_anygrasp(
        #     points=cropped_pc, 
        #     colors=cropped_colors / 255.,
        #     dir_out=dir_out, 
        #     augment=True,
        #     visualize=True
        # )  
        # 2) 在完整的点云上检测 grasp
        gg = detect_grasp_anygrasp(
            points=obj_pc, 
            colors=pc_colors / 255.,
            dir_out=dir_out, 
            augment=True,
            visualize=True
        )  
        gg = crop_grasp(
            grasp_group=gg,
            contact_point=self.contact3d,
            radius=0.1,
        )

        # 先用 dir_out 进行一个 hard filter, 保留角度在 30度 内的 grasp
        gg_filtered = filter_grasp_group(
            grasp_group=gg,
            degree_thre=45,
            dir_out=dir_out,
        )

        # 再用距离 contact3d 的距离和 detector 本身预测的分数做一个排序
        gg_sorted, _ = sort_grasp_group(
            grasp_group=gg_filtered,
            contact_region=self.contact3d[None],
            pre_filter=False
        )
        self.grasp_group = gg_sorted
        
        if visualize:
            visualize_pc(
                points=obj_pc, 
                colors=pc_colors / 255,
                grasp=gg_sorted, 
                contact_point=self.contact3d, 
                post_contact_dirs=[self.dir_out]
            )

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
        
    def get_robot2d_seq(self):
        # T, N, 2
        robot2d_seq = np.stack([self.frame_list[i].robot2d for i in range(self.num_frames())]) 
        return robot2d_seq
    
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
    
    def _visualize(self, viewer: napari.Viewer, prefix="", visualize_robot2d=False):
        rgb_seq = self.get_rgb_seq()
        viewer.add_image(rgb_seq, rgb=True)
        
        # 添加绘制 mask
        if self.frame_list[0].obj_mask is not None:
            obj_mask_seq = self.get_obj_mask_seq()
            viewer.add_labels(obj_mask_seq, name=f"{prefix}_obj_mask_seq")
        
        if self.frame_list[0].dynamic_mask is not None:
            dynamic_mask_seq = self.get_dynamic_seq()
            viewer.add_labels(dynamic_mask_seq.astype(np.int32), name=f"{prefix}_dynamic_mask_seq")
        
        if visualize_robot2d:
            robot2d_seq = self.get_robot2d_seq()
            link_names = [
                'panda_link0', 'panda_link1', 'panda_link2', 
                'panda_link3', 'panda_link4', 'panda_link5', 
                'panda_link6', 'panda_link7', 'panda_link8', 
                'panda_hand', 'panda_hand_tcp', 'panda_leftfinger', 
                'panda_rightfinger', 'camera_base_link', 'camera_link'
            ]
            robot2d_data = napari_time_series_transform(robot2d_seq) # T*M, (1+2)
            robot2d_data = robot2d_data[:, [0, 2, 1]]
            for i in range(len(link_names)):
                T = len(rgb_seq)
                M = len(link_names)
                viewer.add_points(robot2d_data[i::M, :], face_color="red", name=f"{prefix}_{link_names[i]}")
        
    def visualize(self):
        viewer = napari.Viewer()
        viewer.title = "frames visualization"
        self._visualize(viewer)
        napari.run()


if __name__ == "__main__":
    # frame = Frame.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/grasp/init_frame_drawer.npy")
    frame = Frame.load("/home/zby/Programs/Embodied_Analogy/assets/unit_test/grasp/init_frame_micro.npy")
    frame.detect_grasp(True)
    pass