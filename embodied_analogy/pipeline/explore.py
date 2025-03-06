import os
import math
import numpy as np
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    image_to_camera,
    visualize_pc,
    napari_time_series_transform
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
        save_dir="/home/zby/Programs/Embodied_Analogy/assets/tmp/explore/"
    ):
        super().__init__(obj_config)
        self.record_fps = record_fps
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = pertubation_distance
        self.instruction = instruction
        self.obj_description = self.instruction.split(" ")[-1] # TODO: 改为从 instruction 中提取
        self.explore_data = {
            "frames": []
        }
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.has_valid_explore = False
    
    def start_explore_loop(self, allowed_num_tries=10):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
        """
        num_tries_so_far = 0
        while (not self.explore_data_isValid()) and (num_tries_so_far < allowed_num_tries):
            self.explore_once()
            num_tries_so_far += 1
            
        # save explore data
        if not self.has_valid_explore:
            assert "No valid exploration during explore phase!"
            import pdb;pdb.set_trace()
    
    def affordance_transfer(self, visualize=False):
        """
            进行一次 affordance transfer, 保留 transfer 的 contact distribution
            默认使用 "droid" 作为 data_source
        """
        pass
    # TODO: 假如实际执行发现失败的 explore_once 不会对物体状态造成什么变化, 那就可以把 affordance transfer 提炼出来, 进行复用
        
    def explore_once(
        self, 
        reserved_distance=0.05,
        pertubation_distance=0.1,
        visualize=False      
    ):
        from embodied_analogy.exploration.ram_proposal import get_ram_proposal, lift_ram_affordance
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        # 先进行一个 proposal 提出
        contact_uv, contact_dir_2d, obj_mask = get_ram_proposal(
            query_rgb=rgb_np,
            instruction=self.instruction,
            data_source="droid", # TODO
            save_root=self.save_dir,
            visualize=visualize
        )
        contact_3d_c = image_to_camera(
            uv=contact_uv[None],
            depth=np.array(depth_np[contact_uv[1], contact_uv[0]])[None],
            K=self.camera_intrinsic,
        )[0] # 3
        contact_3d_w = camera_to_world(contact_3d_c[None], self.camera_extrinsic)[0]
        # 在这一步保存一下中间输出
        
        # 这个输出的是在相机坐标系下的(Tgrasp2c), 需要转换到世界坐标系下
        sorted_grasps_c, best_dir_3d_c = lift_ram_affordance(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np, 
            query_mask=obj_mask, 
            query_depth=depth_np, 
            contact_uv=contact_uv, 
            contact_dir_2d=contact_dir_2d, 
            visualize=visualize
        )
        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        sorted_grasps = sorted_grasps_c.transform(Tc2w) # Tgrasp2w
        best_dir_3d = camera_to_world(best_dir_3d_c[None], Tw2c)[0] # 3
        
        # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
        pc_collision_c = depth_image_to_pointcloud(depth_np, obj_mask, self.camera_intrinsic) # N, 3
        pc_collision_w = camera_to_world(pc_collision_c, Tw2c)
        # visualize_pc(pc_collision_w, contact_point=contact_3d_w, post_contact_dirs=best_dir_3d[None])
        
        for grasp in sorted_grasps:
            # visualize_pc(pc_collision_w, grasp=grasp)
            grasp = self.get_rotated_grasp(grasp, axis_out_w=best_dir_3d)
            # visualize_pc(pc_collision_w, grasp=grasp)
            
            Tph2w = self.anyGrasp2ph(grasp=grasp)
            
            result_test = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if not result_test:
                continue
            
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            
            self.planner.update_point_cloud(pc_collision_w)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if not result_pre:
                continue
            
            # visualize_pc(pc_collision_w, grasp=grasp)
            # 实际执行到该 proposal, 并在此过程中录制数据
            self.step = self.explore_step
            
            self.follow_path(result_pre)
            print("reach pre-grasp pose")
            
            self.open_gripper()
            self.clear_planner_pc()
            self.move_forward(reserved_distance)
            self.close_gripper()
            
            # 之后再根据 best_dir_3d 移动一定距离 (1dm)
            self.move_along_axis(moving_direction=best_dir_3d, moving_distance=pertubation_distance)
            
            # 打开 gripper, 并返回初始位置
            self.open_gripper()
            self.move_forward(-reserved_distance)
            self.reset_franka_arm()
            
            break
    
        # 录制完成, 开始处理
        self.step = self.base_step 
        # while True:
        #     self.base_step()
    
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
            
    def explore_data_isValid(self):
        from embodied_analogy.perception.grounded_sam import run_grounded_sam
        if len(self.explore_data["frames"]) == 0:
            return False
        
        # 判断录制的数据是否有效的使得物体状态发生了改变    
        # 判断首尾两帧的物体点云方差变化
        first_rgb = self.explore_data["frames"][0]["rgb_np"]
        first_depth = self.explore_data["frames"][0]["depth_np"]
        first_tracks_2d = self.explore_data["frames"][0]["franka_tracks2d"]
        _, first_mask = run_grounded_sam(
            rgb_image=first_rgb,
            obj_description=self.obj_description,
            positive_points=None,  
            negative_points=first_tracks_2d,
            num_iterations=3,
            acceptable_thr=0.9,
            visualize=True
        )
        first_mask = first_mask & (first_depth > 0)
        
        last_rgb = self.explore_data["frames"][-1]["rgb_np"]
        last_depth = self.explore_data["frames"][-1]["depth_np"]
        last_tracks_2d = self.explore_data["frames"][-1]["franka_tracks2d"]
        _, last_mask = run_grounded_sam(
            rgb_image=last_rgb,
            obj_description=self.obj_description,
            positive_points=None,  
            negative_points=last_tracks_2d,
            num_iterations=3,
            acceptable_thr=0.9,
            visualize=True
        )
        last_mask = last_mask & (last_depth > 0)
        
        # 计算前后两帧 depth_map 的差值的变化程度
        diff = np.abs(last_depth - first_depth)
        diff_masked = diff * first_mask.astype(np.float32) * last_mask.astype(np.float32)
        
        if diff_masked.max() > self.pertubation_distance * 0.5:
            self.has_valid_explore = True
            return True
        else:
            return False
        
    def process_explore_data(self, visualize=False, save=False):
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
            
        if save:
            np.savez(
                self.save_dir + "explore_data.npz", 
                rgb_seq=self.rgb_seq, 
                depth_seq=self.depth_seq, 
                franka_seq=self.franka_seq,
                K=self.camera_intrinsic,
                Tw2c=self.camera_extrinsic,
                record_fps=self.record_fps,
            )
            
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
        
    
if __name__ == "__main__":
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        "active_joint": "joint_2"
    }
    exploreEnv = ExploreEnv(
        obj_config=obj_config,
        instruction="open the drawer"
    )
    exploreEnv.start_explore_loop()
    exploreEnv.process_explore_data(
        visualize=True,
        save=True
    )
    