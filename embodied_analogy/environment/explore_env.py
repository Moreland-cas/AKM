import os
import copy
import json
import logging
import math
import numpy as np
import sklearn.cluster as cluster
from embodied_analogy.utility.utils import (
    camera_to_world,
    initialize_napari,
    visualize_pc,
    numpy_to_json
)
initialize_napari()
from embodied_analogy.environment.obj_env import ObjEnv
from embodied_analogy.utility.constants import ASSET_PATH


class ExploreEnv(ObjEnv):
    def __init__(self, cfg):    
        super().__init__(cfg=cfg)
            
        self.explore_env_cfg = cfg["explore_env_cfg"]
        self.algo_cfg = cfg["algo_cfg"]
        
        self.record_fps = self.explore_env_cfg["record_fps"]
        self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = self.explore_env_cfg["pertubation_distance"]
        self.reserved_distance = self.explore_env_cfg["reserved_distance"]
        self.valid_thresh = self.explore_env_cfg["valid_thresh"]
        self.instruction = self.task_cfg["instruction"]
        
        self.update_sigma = self.explore_env_cfg["update_sigma"]
        self.max_tries = self.explore_env_cfg["max_tries"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.has_valid_explore = False
        
        # 读取和保存所用的变量
        self.joint_type = self.obj_env_cfg["joint_type"]
        
    def explore_stage(self, visualize=False):
        """
            explore 多次, 直到找到一个符合要求的操作序列, 或者在尝试足够多次后退出
            NOTE: 目前是 direct reuse, 之后也许需要改为 fusion 的方式
        """
        # 首先得到 affordance_map_2d, 然后开始不断的探索和修改 affordance_map_2d
        from embodied_analogy.utility.proposal.ram_proposal import get_ram_affordance_2d
        
        self.base_step()
        initial_frame = super().capture_frame()
        
        # 只在第一次进行 contact transfer, 之后直接进行复用
        self.logger.log(logging.INFO, "Start transfering 2d contact affordance map...")
        self.affordance_map_2d = get_ram_affordance_2d(
            query_rgb=initial_frame.rgb,
            instruction=self.task_cfg["instruction"],
            obj_description=self.obj_description,
            fully_zeroshot=self.explore_env_cfg["fully_zeroshot"],
            visualize=visualize,
            logger=self.logger
        )
        
        if not self.explore_env_cfg["contact_analogy"]:
            self.affordance_map_2d.uninit_cosmap()
            self.logger.log(logging.INFO, "Detected contact_analogy flag = False, disable Contact Analogy")
        else:
            self.logger.log(logging.INFO, "Detected contact_analogy flag = True, use Contact Analogy")
        
        # 保存第初始化 affordance map 时得到的 cos_map
        if self.exp_cfg["save_obj_repr"]:
            self.obj_repr.save_for_vis.update({
                "explore_cos_map": [np.copy(self.affordance_map_2d.cos_map)]
            })
            
        # 在这里保存 first frame
        self.obj_repr.obj_description = self.obj_description
        self.obj_repr.K = self.camera_intrinsic
        self.obj_repr.Tw2c = self.camera_extrinsic
        self.obj_repr.frames.K = self.camera_intrinsic
        self.obj_repr.frames.Tw2c = self.camera_extrinsic
        self.obj_repr.initial_frame = initial_frame
        
        self.max_tries = self.explore_env_cfg["max_tries"]
        self.logger.log(logging.INFO, f"Start exploring..., you have {self.max_tries} chances to explore...")
        
        if self.max_tries == 1:
            self.logger.log(logging.INFO, "Only try once, Disable Interactive perception")
        else:
            self.logger.log(logging.INFO, "Enable Interactive perception")
            
        self.num_tries = 0
        while self.num_tries < self.max_tries:
            # 初始化相关状态, 需要把之前得到的 frames 进行清楚
            
            if self.num_tries >= 1:
                if self.exp_cfg["save_obj_repr"]:
                    self.obj_repr.save_for_vis[str(self.num_tries)] = [
                        copy.deepcopy(self.obj_repr.frames[0]),
                        copy.deepcopy(self.obj_repr.frames[-1])
                    ]
                    
            self.obj_repr.clear_frames()
            
            if self.num_tries == 0:
                self.reset_robot()
            else:
                self.reset_robot_safe()
            
            self.logger.log(logging.INFO, f"[{self.num_tries + 1}|{self.max_tries}] Start exploring once...")
            actually_tried, explore_uv = self.explore_once(visualize=visualize)
            self.num_tries += 1
            if not actually_tried:
                self.logger.log(logging.INFO, "The planning path is not valid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                    
                    # 保存 update 后 affordance map 的 cos_map
                    if self.exp_cfg["save_obj_repr"]:
                        self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))
                continue
            
            if self.check_valid(visualize=visualize):
                self.logger.log(logging.INFO, "Check valid, break explore loop")
                break
            else:
                self.logger.log(logging.INFO, "Check invalid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=visualize)
                
                # 保存 update 后 affordance map 的 cos_map
                if self.exp_cfg["save_obj_repr"]:
                    self.obj_repr.save_for_vis["explore_cos_map"].append(np.copy(self.affordance_map_2d.cos_map))

        if self.exp_cfg["save_obj_repr"]:
            self.obj_repr.save_for_vis["aff_map"] = copy.deepcopy(self.affordance_map_2d)
                    
        # save explore data
        if visualize:
            self.obj_repr.visualize()
        
        if self.obj_repr.frames.num_frames() == 0:
            joint_state_end = 0
        else:
            joint_state_end = self.get_active_joint_state() - self.obj_repr.frames[0].gt_joint_state
        result_dict = {
            # 算法认为的是否有成功的探索
            "num_tries": self.num_tries,
            "has_valid_explore": self.has_valid_explore,
            "joint_type": self.joint_type,
            "joint_state_start": 0,
            "joint_state_end": joint_state_end
        }
        self.logger.log(logging.INFO, f"exploration stage result: {result_dict}")
        
        if not self.has_valid_explore:
            self.logger.log(logging.INFO, "In summary, no valid exploration during explore phase!")
        else:
            self.logger.log(logging.INFO, "In summary, get valid exploration during explore phase!")
        
        # if not self.has_valid_explore:
        #     raise Exception("No valid explore found!")
        
        return result_dict
    
    def explore_once(self, visualize=False):
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
        
        if self.explore_env_cfg["use_IOR"]:
            self.logger.log(logging.INFO, "Detected use_IOR flag = True, use Inhibition of Return")
            contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        else:
            # sample_prob 返回的是一个 N, 2 的 list, alpha 越大, 采样越密集
            self.logger.log(logging.INFO, "Detected use_IOR flag = False, do not update affoedance map")
            # contact_uv = self.affordance_map_2d.sample_prob(alpha=10, num_samples=1, return_rgb_frame=True, visualize=False)[0]
            contact_uv = self.affordance_map_2d.sample_prob(alpha=1, num_samples=1, return_rgb_frame=True, visualize=False)[0]
        
        cur_frame.obj_mask = obj_mask
        cur_frame.contact2d = contact_uv
        
        # 这里 rgb_np, depth_np 可能和 affordance_map_2d 中存储的不太一样, 不过应该不会差太多
        cur_frame.detect_grasp(
            use_anygrasp=self.algo_cfg["use_anygrasp"],
            world_frame=True,
            visualize=visualize,
            asset_path=ASSET_PATH,
            logger=self.logger
        )
        
        if cur_frame.grasp_group is None:
            return False, contact_uv
        
        contact3d_w = camera_to_world(
            point_camera=cur_frame.contact3d[None],
            extrinsic_matrix=Tw2c
        )[0]
        dir_out_w = Tc2w[:3, :3] @ cur_frame.dir_out # 3
        
        result_pre = None
        # NOTE: 这里没有使用 get_obj_pc, 因为每次 explore 都会有新的 cur_frame, 因此并不总有最新的 obj_mask 信息
        pc_collision_w, pc_colors = cur_frame.get_env_pc(
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_collision_w)
            
        for grasp_w in cur_frame.grasp_group:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w = self.anyGrasp2ph(grasp=grasp)        
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.reserved_distance)
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
        self.move_forward(
            moving_distance=self.reserved_distance,
            drop_large_move=False
        )
        self.close_gripper()
        
        # 在 close gripper 之后再开始录制数据
        self.step = self.explore_step
        # NOTE: 在 explore 阶段, 不管是什么关节, 做的扰动都是直线移动
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=dir_out_w,
            joint_start=None,
            moving_distance=self.pertubation_distance,
            drop_large_move=False
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
            
    def check_valid_deprecated(self, visualize=False):
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
    
    def check_valid(self, visualize=False): 
        # 对于 frames 进行 tracking, 然后根据聚类结果判断 moving_part 动的多不多, 多的话就认为 valid
        # 这个函数相比于 deprecated 版本, 可以更好的处理 "柜子开一个缝隙, joint state 没有大的变化, 但是缝隙的深度有突变" 的情况
        # 这个函数会写入 obj_repr 的 tracks2d, tracks3d 和 moving mask
        if self.obj_repr.frames.num_frames() == 0:
            return False
        
        self.obj_repr.frames[0].segment_obj(
            obj_description=self.obj_env_cfg["obj_description"],
            post_process_mask=True,
            filter=True,
            visualize=visualize
        )
        self.obj_repr.frames[0].sample_points(num_points=self.explore_env_cfg["num_initial_pts"], visualize=visualize)
        self.obj_repr.frames.track_points(visualize=visualize)
        self.obj_repr.frames.track2d_to_3d(filter=True, visualize=visualize)
        self.obj_repr.frames.cluster_track3d(visualize=visualize)
        
        # 根据 moving tracks 的位移来判断, (T, M, 3)
        moving_tracks = self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :]
        
        # 原始的版本 (用首尾帧的)
        mean_delta = np.linalg.norm(moving_tracks[-1] - moving_tracks[0], axis=-1).mean()
        if mean_delta > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
        
        # 更新的版本 ()
        # def calculate_trajectory_length(trajectory):
        #     """
        #     trajectory: T, 3
        #     """
        #     # 计算相邻时间步之间的差值
        #     diff = np.diff(trajectory, axis=0) # T-1, 3
        #     # 计算每个时间步的欧几里得距离
        #     distances = np.linalg.norm(diff, axis=1) # T - 1
        #     # 累加得到轨迹长度
        #     return np.sum(distances)
        
        # max_traj_len = -1
        # for i in range(moving_tracks.shape[1]):
        #     traj_len = calculate_trajectory_length(moving_tracks[:, i])
        #     if traj_len >= max_traj_len:
        #         max_traj_len = traj_len
        # if max_traj_len > self.pertubation_distance * self.valid_thresh:
        #     self.has_valid_explore = True
        #     return True
        # else:
        #     return False

    
    ###############################################
    def main(self):
        try:
        # if True:
            self.explore_result = {}
            self.explore_result = self.explore_stage()
                    
        except Exception as e:
            self.logger.log(logging.ERROR, f"Explore exception occured: {e}", exc_info=True)
            
            self.explore_result["has_valid_explore"] = False
            self.explore_result["joint_type"] = self.obj_env_cfg["joint_type"]
            self.explore_result["exception"] = str(e)

        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "explore_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.explore_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
                    
if __name__ == "__main__":
    
    exploreEnv = ExploreEnv(
        cfg={
    "joint_type": "prismatic",
    "data_path": "dataset/one_drawer_cabinet/40147_link_1",
    "obj_index": "40147",
    "joint_index": "1",
    "obj_description": "cabinet",
    "load_pose": [
        0.9147002696990967,
        0.0,
        0.4520242512226105
    ],
    "load_quat": [
        0.999747097492218,
        0.022409481927752495,
        -4.2135787225561216e-05,
        -0.0018797904485836625
    ],
    "load_scale": 1,
    "active_link_name": "link_1",
    "active_joint_name": "joint_1",
    "instruction": "open the cabinet",
    "load_joint_state": 0.0,
    "obj_folder_path_explore": "/home/zby/Programs/Embodied_Analogy/assets/logs/explore_512/40147_link_1",
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "fully_zeroshot": False,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "valid_thresh": 0.5,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "num_initial_pts": 1000,
    "offscreen": True,
    "use_anygrasp": False
}
    )
    exploreEnv.explore_stage(visualize=False)
    # exploreEnv.save(file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore/explore_data.pkl")
    