import math
import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    image_to_camera,
    visualize_pc,
    get_depth_mask
)
initialize_napari()
from embodied_analogy.pipeline.manipulate import ManipulateEnv
from embodied_analogy.representation.basic_structure import Frame, Frames
from embodied_analogy.representation.obj_repr import Obj_repr


class ExploreEnv(ManipulateEnv):
    def __init__(
        self, 
        obj_config, 
        instruction,
        record_fps=30, 
        pertubation_distance=0.1,
    ):
        super().__init__(obj_config=obj_config, instruction=instruction)
        self.record_fps = record_fps
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = pertubation_distance
        
        self.obj_repr = Obj_repr()
        self.has_valid_explore = False
        
    def explore_loop(self, max_tries=10, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
            NOTE: 目前是 direct reuse, 之后也许需要改为 fusion 的方式
            TODO
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
        # 在这里保存 first frame
        self.obj_repr.obj_description = self.obj_description
        self.obj_repr.K = self.camera_intrinsic
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.initial_frame = Frame(
            rgb=rgb_np,
            depth=depth_np,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
            joint_state=None,
            obj_mask=self.affordance_map_2d.get_obj_mask(),
            dynamic_mask=None,
            contact2d=None,
            contact3d=None,
            franka2d=None,
            franka3d=None,
            franka_mask=None
        )
        # self.obj_repr.initial_frame.visualize()
        
        num_tries = 0
        while num_tries < max_tries:
            # 初始化相关状态, 需要把之前得到的 frames 进行清楚
            self.open_gripper()
            self.reset_robot()
            self.obj_repr.clear_frames()
            
            explore_ok, explore_uv = self.explore_once(visualize=visualize)
            num_tries += 1
            if not explore_ok:
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, visualize=visualize)
                continue
            
            if self.check_valid():
                # 在这里将 explore_uv 保存到 obj_repr 的 initial_frame 中
                self.obj_repr.initial_frame.contact2d = explore_uv
                uv_depth = self.obj_repr.initial_frame.depth[int(explore_uv[1]), int(explore_uv[0])]
                self.obj_repr.initial_frame.contact3d = image_to_camera(
                    uv=explore_uv[None], 
                    depth=np.array([uv_depth]), 
                    K=self.camera_intrinsic, 
                )[0]
                break
            else:
                # 更新 affordance map
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, visualize=visualize)
                
        # save explore data
        if not self.has_valid_explore:
            print("No valid exploration during explore phase!")
            
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
        from embodied_analogy.exploration.ram_proposal import lift_affordance
        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        
        cur_frame = Frame(
            rgb=rgb_np,
            depth=depth_np,
            K=self.camera_intrinsic,
            Tw2c=self.camera_extrinsic,
            obj_mask=obj_mask,
            contact2d=contact_uv,
            franka2d=self.get_points_on_arm()[0],
            franka_mask=None
        )
        
        # 这里 rgb_np, depth_np 可能和 affordance_map_2d 中存储的不太一样, 不过应该不会差太多
        contact3d_c, grasps_c, dir_out_c = lift_affordance(
            cur_frame=cur_frame,
            visualize=visualize
        )
        if grasps_c is None:
            return False, contact_uv
        
        contact3d_w = camera_to_world(
            point_camera=contact3d_c[None],
            extrinsic_matrix=Tw2c
        )[0]
        
        grasps_w = grasps_c.transform(Tc2w) # Tgrasp2w
        dir_out_w = Tc2w[:3, :3] @ dir_out_c # 3
        
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
            # TODO 这里可能需要更改
            if result_pre is not None:
                break
        
        if visualize:
            visualize_pc(
                points=pc_collision_w, 
                colors=pc_colors / 255,
                grasp=grasp, 
                contact_point=contact3d_w, 
                post_contact_dirs=[dir_out_w]
            )
        
        # 实际执行到该 proposal, 并在此过程中录制数据
        if result_pre is None:
            return False, contact_uv
        
        self.follow_path(result_pre)
        self.open_gripper()
        self.clear_planner_pc()
        self.move_forward(reserved_distance)
        self.close_gripper()
        
        # 在 close gripper 之后再开始录制数据
        self.step = self.explore_step
        # NOTE: 在 explore 阶段, 不管是什么关节, 做的扰动都是直线移动
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=dir_out_w,
            joint_start=None,
            moving_distance=pertubation_distance
        )
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
            franka_tracks2d, franka_tracks3d_w = self.get_points_on_arm()
            
            cur_frame = Frame(
                rgb=rgb_np,
                depth=depth_np,
                K=None,
                Tw2c=None,
                joint_state=None,
                obj_mask=None,
                dynamic_mask=None,
                contact2d=None,
                contact3d=None,
                franka2d=franka_tracks2d,
                franka3d=None,
                franka_mask=None,
            )
            self.obj_repr.frames.append(cur_frame)
            
    def check_valid(self):
        from embodied_analogy.perception.grounded_sam import run_grounded_sam
        # self.obj_repr.visualize()
        
        if self.obj_repr.frames.num_frames() == 0:
            return False
        
        # 判断录制的数据是否有效的使得物体状态发生了改变    
        # 判断首尾两帧的物体点云方差变化
        first_rgb = self.obj_repr.frames[0].rgb
        first_depth = self.obj_repr.frames[0].depth
        first_tracks_2d = self.obj_repr.frames[0].franka2d
        
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
        
        last_rgb = self.obj_repr.frames[-1].rgb
        last_depth = self.obj_repr.frames[-1].depth
        last_tracks_2d = self.obj_repr.frames[-1].franka2d
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
        
    def save(self, file_path=None, visualize=True):
        if visualize:
            self.obj_repr.visualize()
        self.obj_repr.save(file_path)
        
    
if __name__ == "__main__":
    # drawer
    # obj_config = {
    #     "index": 44962,
    #     "scale": 0.8,
    #     "pose": [1.0, 0., 0.5],
    #     "active_link": "link_2",
    #     # "active_joint": "joint_2"
    #     "active_joint": "joint_1"
    # }
    
    # door
    # obj_config = {
    #     "index": 9288,
    #     "scale": 1.0,
    #     "pose": [0.7, 0., 0.7],
    #     "active_link": "link_2",
    #     "active_joint": "joint_0"
    # }
    
    # microwave
    obj_config = {
        "index": 7221,
        "scale": 0.4,
        "pose": [0.8, 0.1, 0.6],
        "active_link": "link_0",
        "active_joint": "joint_0"
    }
    
    obj_index = obj_config["index"]
    
    exploreEnv = ExploreEnv(
        obj_config=obj_config,
        # instruction="open the drawer",
        instruction="open the microwave",
        record_fps=30,
        pertubation_distance=0.1
    )
    exploreEnv.explore_loop(visualize=False)
    exploreEnv.save(file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore/explore_data.pkl")
    