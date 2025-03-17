import os
import math
import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    image_to_camera,
    visualize_pc,
    napari_time_series_transform,
    get_depth_mask
)
initialize_napari()
from embodied_analogy.pipeline.manipulate import ManipulateEnv

class ExploreEnv(ManipulateEnv):
    def __init__(
        self, 
        obj_config, 
        instruction,
        record_fps=30, 
        pertubation_distance=0.1,
        save_dir=None,
    ):
        super().__init__(obj_config=obj_config, instruction=instruction)
        self.record_fps = record_fps
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = pertubation_distance
        
        self.explore_data = {
            "frames": []
        }
        self.has_valid_explore = False
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    def explore_loop_reset(self, allowed_num_tries=10, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
            # TODO 0
        """
        # 首先得到 affordance_map_2d, 然后开始不断的探索和修改 affordance_map_2d
        from embodied_analogy.exploration.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=rgb_np,
            instruction=self.instruction,
            data_source="droid", # TODO
            visualize=False
        )
        
        num_tries_so_far = 0
        while num_tries_so_far < allowed_num_tries:
            # 初始化相关状态
            self.open_gripper()
            self.reset_franka_arm()
            self.explore_data["frames"] = []
            
            self.explore_once(visualize=visualize)
            num_tries_so_far += 1
            
            if self.get_valid_explore():
                break
            else:
                # 更新 affordance map
                self.affordance_map_2d.update()
                
        # save explore data
        if not self.has_valid_explore:
            assert "No valid exploration during explore phase!"
    
    def explore_loop_direct_reuse(self, allowed_num_tries=10, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
        """
        # 首先得到 affordance_map_2d, 然后开始不断的探索和修改 affordance_map_2d
        from embodied_analogy.exploration.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        # 只在第一次进行 contact transfer, 之后直接进行复用
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=rgb_np,
            instruction=self.instruction,
            data_source="droid", # TODO
            visualize=False
        )
        
        num_tries_so_far = 0
        while num_tries_so_far < allowed_num_tries:
            # 初始化相关状态
            self.open_gripper()
            self.reset_franka_arm()
            self.explore_data["frames"] = []
            
            explore_ok, explore_uv = self.explore_once(visualize=visualize)
            if not explore_ok:
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, visualize=visualize)
                continue
            else:
                num_tries_so_far += 1
            
            if self.get_valid_explore():
                break
            else:
                # 更新 affordance map
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, visualize=visualize)
                
        # save explore data
        if not self.has_valid_explore:
            assert "No valid exploration during explore phase!"
            
    def explore_loop_fusion_reuse(self, allowed_num_tries=10, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
            # TODO 2
        """
        # 首先得到 affordance_map_2d, 然后开始不断的探索和修改 affordance_map_2d
        from embodied_analogy.exploration.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        num_tries_so_far = 0
        while num_tries_so_far < allowed_num_tries:
            # 初始化相关状态
            self.open_gripper()
            self.reset_franka_arm()
            self.explore_data["frames"] = []
            
            self.affordance_map_2d = get_ram_affordance_2d(
                query_rgb=rgb_np,
                instruction=self.instruction,
                data_source="droid", # TODO
                visualize=False
            )
            
            self.explore_once(visualize=visualize)
            num_tries_so_far += 1
            
            if self.get_valid_explore():
                break
            else:
                # 更新 affordance map
                self.affordance_map_2d.update()
                
        # save explore data
        if not self.has_valid_explore:
            assert "No valid exploration during explore phase!"
            
    def explore_once(
        self, 
        reserved_distance=0.05,
        pertubation_distance=0.1,
        visualize=False      
    ):
        """
            在当前状态下进行一次探索, 默认此时的 franka arm 处于 rest 状态
            返回 explore_ok, explore_uv:
                explore_ok: bool, 代表 plan 阶段是否成功
                explore_uv: np.array([2,]), 代表本次尝试的 contact point 的 uv
        """        
        from embodied_analogy.exploration.ram_proposal import lift_ram_affordance
        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        
        # 这里 rgb_np, depth_np 可能和 affordance_map_2d 中存储的不太一样, 不过应该不会差太多
        contact_3d_c, grasps_c, dir_out_c = lift_ram_affordance(
            K=self.camera_intrinsic, 
            Tw2c=self.camera_extrinsic,
            query_rgb=rgb_np,
            query_depth=depth_np, 
            query_mask=obj_mask,
            contact_uv=contact_uv,
            visualize=visualize
        )
        if grasps_c is None:
            return False, None
        
        contact_3d_w = camera_to_world(
            point_camera=contact_3d_c[None],
            extrinsic_matrix=Tw2c
        )[0]
        
        grasps_w = grasps_c.transform(Tc2w) # Tgrasp2w
        dir_out_w = camera_to_world(dir_out_c[None], Tw2c)[0] # 3
        
        result_pre = None
        depth_mask = get_depth_mask(depth_np, self.camera_intrinsic, Tw2c, height=0.02)
        pc_collision_c = depth_image_to_pointcloud(depth_np, obj_mask & depth_mask, self.camera_intrinsic) # N, 3
        pc_colors = rgb_np[obj_mask & depth_mask]
        pc_collision_w = camera_to_world(pc_collision_c, Tw2c)
        self.planner.update_point_cloud(pc_collision_w)
            
        for grasp_w in grasps_w:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            
            if result_pre is not None:
                break
        
        if visualize:
            visualize_pc(
                points=pc_collision_w, 
                colors=pc_colors / 255,
                grasp=grasp, 
                contact_point=contact_3d_w, 
                post_contact_dirs=[dir_out_w]
            )
        
        # 实际执行到该 proposal, 并在此过程中录制数据
        if result_pre is None:
            return False, None
        
        self.step = self.explore_step
        
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(reserved_distance)
        self.close_gripper()
        self.move_along_axis(moving_direction=dir_out_w, moving_distance=pertubation_distance)
        
        # 录制完成, 开始处理
        self.step = self.base_step 
        return True, contact_uv
    
    def explore_step(self):
        # 在 base_step 的基础上, 进行数据的录制
        self.base_step()
        
        self.cur_steps = self.cur_steps % self.record_interval
        if self.cur_steps == 0:
            # record rgb image and display to pygame screen
            rgb_np, depth_np, _, _ = self.capture_rgbd(return_pc=False, visualize=False)
            
            # 在这里添加当前帧的 franka_arm 上的点的 franka_tracks3d 和 franka_tracks2d
            franka_tracks2d, franka_tracks3d = self.get_points_on_arm()
            
            cur_frame = {
                "rgb_np": rgb_np, 
                "depth_np": depth_np,
                "franka_tracks2d": franka_tracks2d, 
                "franka_tracks3d": franka_tracks3d
            }
            self.explore_data["frames"].append(cur_frame)
            
    def get_valid_explore(self):
        from embodied_analogy.perception.grounded_sam import run_grounded_sam
        if len(self.explore_data["frames"]) == 0:
            return False
        
        # 判断录制的数据是否有效的使得物体状态发生了改变    
        # 判断首尾两帧的物体点云方差变化
        first_rgb = self.explore_data["frames"][0]["rgb_np"]
        first_depth = self.explore_data["frames"][0]["depth_np"]
        first_tracks_2d = self.explore_data["frames"][0]["franka_tracks2d"]
        _, first_obj_mask = run_grounded_sam(
            rgb_image=first_rgb,
            obj_description=self.obj_description,
            positive_points=None,  
            negative_points=first_tracks_2d,
            num_iterations=3,
            acceptable_thr=0.9,
            visualize=False
        )
        first_depth_mask = get_depth_mask(
            depth=first_depth,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
            height=0.02
        )
        first_mask = first_obj_mask & first_depth_mask
        
        last_rgb = self.explore_data["frames"][-1]["rgb_np"]
        last_depth = self.explore_data["frames"][-1]["depth_np"]
        last_tracks_2d = self.explore_data["frames"][-1]["franka_tracks2d"]
        _, last_obj_mask = run_grounded_sam(
            rgb_image=last_rgb,
            obj_description=self.obj_description,
            positive_points=None,  
            negative_points=last_tracks_2d,
            num_iterations=3,
            acceptable_thr=0.9,
            visualize=False
        )
        last_depth_mask = get_depth_mask(
            depth=last_depth,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
            height=0.02
        )
        last_mask = last_obj_mask & last_depth_mask
        
        # 计算前后两帧 depth_map 的差值的变化程度
        diff = np.abs(last_depth - first_depth) # H, W
        diff_masked = diff[first_mask & last_mask] # N
        
        # 将 diff_masked 进行聚类, 将变化大的一部分的平均值与 pertubation_distance 的 0.5 倍进行比较
        # 聚类为 3, 使得对于噪声有一定的 robustness
        diff_centroid, labels, _ = cluster.k_means(diff_masked[:, None], init="k-means++", n_clusters=3)
        cluster_num = [(labels == i).sum() for i in range(3)]

        top_indices = np.argsort(cluster_num)[-2:]  # 获取最大的两个索引
        # 提取对应的类中心
        top_centroids = diff_centroid[top_indices]
        
        # 这里 0.5 是一个经验值, 因为对于旋转这个值不好解析的计算
        if top_centroids.max() > self.pertubation_distance * 0.5:
            self.has_valid_explore = True
            return True
        else:
            return False
        
    def process_explore_data(self, visualize=False):
        """
            process rgb and depth data in self.explore_data
        """
        assert self.has_valid_explore
        num_frames = len(self.explore_data["frames"])
        self.rgb_seq = np.stack([self.explore_data["frames"][i]["rgb_np"] for i in range(num_frames)]) # T, H, W, C
        self.depth_seq = np.stack([self.explore_data["frames"][i]["depth_np"] for i in range(num_frames)]) # T, H, W
        self.franka_seq = np.stack([self.explore_data["frames"][i]["franka_tracks2d"] for i in range(num_frames)]) # T, N, 2
        
        if visualize:
            self.visualize_explore_data()
            
    def visualize_explore_data(self):
        """
            visualize rgb and depth data in self.explore_data
        """
        import napari
        viewer = napari.view_image(self.rgb_seq, rgb=True)
        franka_visuailze = napari_time_series_transform(self.franka_seq) # T*M, (1+2)
        franka_visuailze = franka_visuailze[:, [0, 2, 1]]
        self.link_names = [link.name for link in self.robot.get_links()]
        
        for i in range(len(self.link_names)):
            T = len(self.rgb_seq)
            M = len(self.link_names)
            viewer.add_points(franka_visuailze[i::M, :], face_color="red", name=self.link_names[i])
        napari.run()
        
    def save_explore_data(self, save_dir="/home/zby/Programs/Embodied_Analogy/assets/tmp/explore/"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        np.savez(
            os.path.join(save_dir, "explore_data.npz"), 
            rgb_seq=self.rgb_seq, 
            depth_seq=self.depth_seq, 
            franka_seq=self.franka_seq,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
            record_fps=self.record_fps,
        )
    
if __name__ == "__main__":
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        # "active_joint": "joint_2"
        "active_joint": "joint_1"
    }
    
    # obj_config = {
    #     "index": 9288,
    #     "scale": 1.0,
    #     "pose": [1.0, 0., 0.7],
    #     "active_link": "link_2",
    #     "active_joint": "joint_0"
    # }
    
    obj_index = obj_config["index"]
    
    exploreEnv = ExploreEnv(
        obj_config=obj_config,
        instruction="open the drawer",
        # instruction="open the door",
        save_dir=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/explore/{obj_index}"
    )
    exploreEnv.explore_loop_direct_reuse(visualize=False)
    exploreEnv.process_explore_data(visualize=True)
    
    exploreEnv.save_explore_data(
        save_dir=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/explore/{obj_index}"
    )
    