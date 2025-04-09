import os
import json
import pickle
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
from embodied_analogy.environment.obj_env import ObjEnv


class ExploreEnv(ObjEnv):
    def __init__(
            self,
            cfg
        ):        
        super().__init__(cfg=cfg)
        # self.cfg = cfg
        print("loading explore env, using cfg:", cfg)
        
        self.record_fps = cfg["record_fps"]
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = cfg["pertubation_distance"]
        self.valid_thresh = cfg["valid_thresh"]
        self.instruction = cfg["instruction"]
        
        self.update_sigma = cfg["update_sigma"]
        self.max_tries = cfg["max_tries"]
        self.obj_description = cfg["obj_description"]
        self.has_valid_explore = False
        
        # 根据 logs_path 和 run_name 和 obj_info 生成 save 的路径
        logs_path = cfg["logs_path"]
        run_name = cfg["run_name"]
        obj_index = cfg["obj_index"]
        active_joint_name = cfg["active_joint_name"]
        joint_index = cfg["joint_index"]
        joint_type = cfg["joint_type"]
        self.joint_type = joint_type
        obj_folder = f"{obj_index}_{joint_index}_{joint_type}"
        self.save_prefix = os.path.join(logs_path, run_name, obj_folder, "explore")
        os.makedirs(self.save_prefix, exist_ok=True)
        
    def explore_stage(self, save_intermediate=False, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
            NOTE: 目前是 direct reuse, 之后也许需要改为 fusion 的方式
        """
        # 首先得到 affordance_map_2d, 然后开始不断的探索和修改 affordance_map_2d
        from embodied_analogy.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        initial_frame = super().capture_frame()
        
        # 只在第一次进行 contact transfer, 之后直接进行复用
        print("Start transfering 2d contact affordance map...")
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=initial_frame.rgb,
            instruction=self.instruction,
            obj_description=self.obj_description,
            visualize=False
        )
        # self.affordance_map_2d.fit_GMM(
        #     data=self.affordance_map_2d.sample_prob(alpha=30, num_samples=3000, return_rgb_frame=False, visualize=False),
        #     visualize=True
        # )
        # 在这里保存 first frame
        self.obj_repr.obj_description = self.obj_description
        self.obj_repr.K = self.camera_intrinsic
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.frames.K = self.camera_intrinsic
        self.obj_repr.frames.Tw2c = self.camera_extrinsic
        self.obj_repr.initial_frame = initial_frame
        
        print(f"Start exploring..., you have {self.max_tries} chances to explore...")
        self.num_tries = 0
        while self.num_tries < self.max_tries:
            # 初始化相关状态, 需要把之前得到的 frames 进行清楚
            self.obj_repr.clear_frames()
            if self.num_tries == 0:
                self.reset_robot()
            else:
                self.reset_robot_safe()
            
            print(f"[{self.num_tries + 1}|{self.max_tries}]Start exploring once...")
            actually_tried, explore_uv = self.explore_once(visualize=visualize)
            self.num_tries += 1
            if not actually_tried:
                print("The planning path is not valid, update affordance map and try again...")
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                continue
            
            if self.check_valid(visualize=visualize):
                print("good, check valid, break explore loop")
                break
            else:
                print("bad, check invalid, update affordance map and try again...")
                self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                
        # save explore data
        if visualize:
            self.obj_repr.visualize()
        
        result_dict = {
            # 算法认为的是否有成功的探索
            "num_tries": self.num_tries,
            "has_valid_explore": self.has_valid_explore,
            "joint_type": self.joint_type,
            "joint_state_start": 0,
            "joint_state_end": self.get_active_joint_state()
        }
        print("exploration stage result: ", result_dict)
        
        if save_intermediate:
            
            # 首先保存 cfg 文件
            with open(os.path.join(self.save_prefix, "cfg.json"), 'w', encoding='utf-8') as f:
                json.dump(self.cfg, f, ensure_ascii=False, indent=4)
            
            # 然后保存 rgbd_seq
            self.obj_repr.save(os.path.join(self.save_prefix, "obj_repr.npy"))
            
            # 然后保存 运行状态文件
            with open(os.path.join(self.save_prefix, 'result.pkl'), 'wb') as f:
                pickle.dump(result_dict, f)
        
        if not self.has_valid_explore:
            print("In summary, no valid exploration during explore phase!")
            # raise Exception("No valid exploration during explore phase!")
        else:
            print("In summary, get valid exploration during explore phase!")
        print("done")
    
    def explore_once(
        self, 
        reserved_distance=0.05,
        pertubation_distance=0.1,
        visualize=False      
    ):
        """
            在当前状态下进行一次探索, 默认此时的 robot arm 处于 reset 状态
            返回 explore_ok, explore_uv:
                explore_ok: bool, 代表 plan 阶段是否成功
                explore_uv: np.array([2,]), 代表本次尝试的 contact point 的 uv
        """        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        self.base_step()
        cur_frame = self.capture_frame()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        
        cur_frame.obj_mask = obj_mask
        cur_frame.contact2d = contact_uv
        
        # 这里 rgb_np, depth_np 可能和 affordance_map_2d 中存储的不太一样, 不过应该不会差太多
        cur_frame.detect_grasp(visualize=visualize)
        
        if cur_frame.grasp_group is None:
            return False, contact_uv
        
        contact3d_w = camera_to_world(
            point_camera=cur_frame.contact3d[None],
            extrinsic_matrix=Tw2c
        )[0]
        
        grasps_w = cur_frame.grasp_group.transform(Tc2w) # Tgrasp2w
        dir_out_w = Tc2w[:3, :3] @ cur_frame.dir_out # 3
        
        result_pre = None
        # NOTE: 这里没有使用 get_obj_pc, 因为每次 explore 都会有新的 cur_frame, 因此并不总有 obj_mask 信息
        pc_collision_w, pc_colors = cur_frame.get_env_pc(
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        for grasp_w in grasps_w:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            # TODO 这里可能需要更改
            if result_pre is not None:
                if visualize:
                    visualize_pc(
                        points=pc_collision_w, 
                        colors=pc_colors / 255,
                        grasp=grasp, 
                        contact_point=contact3d_w, 
                        post_contact_dirs=[dir_out_w]
                    )
                break
        
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
            cur_frame = self.capture_frame()
            self.obj_repr.frames.append(cur_frame)
            
    def check_valid(self, visualize=False):
        if self.obj_repr.frames.num_frames() == 0:
            return False
        
        # 判断录制的数据是否有效的使得物体状态发生了改变    
        # 判断首尾两帧的物体点云方差变化
        first_frame = self.obj_repr.frames[0]
        first_frame.segment_obj(obj_description=self.obj_description)
        
        last_frame = self.obj_repr.frames[-1]
        last_frame.segment_obj(obj_description=self.obj_description)
        
        # 计算首尾两帧的共同可见点云的变化情况
        first_frame.obj_mask = first_frame.obj_mask & last_frame.obj_mask
        last_frame.obj_mask = first_frame.obj_mask & last_frame.obj_mask
        
        first_pc_w, first_pc_color = first_frame.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=True,
            world_frame=True,
        )
        last_pc_w, last_pc_color = last_frame.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=True,
            world_frame=True,
        )
        if visualize:
            visualize_pc(
                first_pc_w, 
                colors=first_pc_color / 255., 
            )
            visualize_pc(
                last_pc_w, 
                colors=last_pc_color / 255., 
            )
        diff_w = last_pc_w - first_pc_w  # N, 3
        norm_w = np.linalg.norm(diff_w, axis=-1) # N
        
        # 将 diff_masked 进行聚类, 将变化大的一部分的平均值与 pertubation_distance 的 0.5 倍进行比较
        centroids, _, _ = cluster.k_means(norm_w[:, None], init="k-means++", n_clusters=2)
        
        # 这里 0.5 是一个经验值, 因为对于旋转这个值不好解析的计算
        if centroids.max() > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
        
    
if __name__ == "__main__":
    
    exploreEnv = ExploreEnv(
        cfg={
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "logs_path": "/home/zby/Programs/Embodied_Analogy/assets/logs_complex",
    "run_name": "test_explore",
    "valid_thresh": 0.5,
    "instruction": "open the cabinet",
    "obj_description": "cabinet",
    "joint_type": "prismatic",
    "obj_index": "40147",
    "joint_index": "1",
    "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_drawer_cabinet/40147_link_1",
    "active_link_name": "link_1",
    "active_joint_name": "joint_1",
    "load_pose": [
        0.8763857483863831,
        0.0,
        0.4520242512226105
    ],
    "load_quat": [
        1.0,
        0.0,
        0.0,
        0.0
    ],
    "load_scale": 1
}
    )
    exploreEnv.explore_stage(visualize=False)
    # exploreEnv.save(file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore/explore_data.pkl")
    