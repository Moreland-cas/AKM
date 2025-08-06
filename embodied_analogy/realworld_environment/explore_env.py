import os
import copy
import time
import json
import logging
import numpy as np
import sklearn.cluster as cluster
import threading
from franky import (
    CartesianMotion,
    Affine,
    ReferenceType,
    Reaction,
    JointStopMotion,
    CartesianStopMotion,
    CartesianImpedanceMotion
)
from embodied_analogy.utility.utils import (
    camera_to_world,
    initialize_napari,
    visualize_pc,
    numpy_to_json
)
initialize_napari()
from embodied_analogy.representation.basic_structure import Frame
from embodied_analogy.realworld_environment.obj_env import ObjEnv
from embodied_analogy.realworld_environment.robot_env import clean_pc_np
from embodied_analogy.utility.constants import ASSET_PATH


class ExploreEnv(ObjEnv):
    def __init__(self, cfg):    
        super().__init__(cfg=cfg)
            
        self.explore_env_cfg = cfg["explore_env_cfg"]
        self.algo_cfg = cfg["algo_cfg"]
        
        # self.record_fps = self.explore_env_cfg["record_fps"]
        # self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
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
        
        # self.base_step()
        self.reset_robot()
        initial_frame = self.capture_frame()
        
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
            
            # if self.num_tries >= 1:
            #     if self.exp_cfg["save_obj_repr"]:
            #         self.obj_repr.save_for_vis[str(self.num_tries)] = [
            #             copy.deepcopy(self.obj_repr.frames[0]),
            #             copy.deepcopy(self.obj_repr.frames[-1])
            #         ]
                    
            self.obj_repr.clear_frames()
            if self.num_tries == 0:
                self.reset_robot()
            else:
                self.reset_robot_safe()
            
            self.logger.log(logging.INFO, f"[{self.num_tries + 1}|{self.max_tries}] Start exploring once...")
            actually_tried, explore_uv = self.explore_once(visualize=visualize, idx=self.num_tries)
            self.num_tries += 1
            if not actually_tried:
                self.logger.log(logging.INFO, "The planning path is not valid, update affordance map and try again...")
                if self.explore_env_cfg["use_IOR"]:
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=True)
                    
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
                    self.affordance_map_2d.update(neg_uv_rgb=explore_uv, update_sigma=self.update_sigma, visualize=True)
                
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
            
        result_dict = {
            # 算法认为的是否有成功的探索
            "num_tries": self.num_tries,
            "has_valid_explore": self.has_valid_explore,
            "joint_type": self.joint_type,
            # "joint_state_start": 0,
            # "joint_state_end": joint_state_end
        }
        self.logger.log(logging.INFO, f"exploration stage result: {result_dict}")
        
        if not self.has_valid_explore:
            self.logger.log(logging.INFO, "In summary, no valid exploration during explore phase!")
        else:
            self.logger.log(logging.INFO, "In summary, get valid exploration during explore phase!")
        
        if not self.has_valid_explore:
            raise Exception("No valid explore found!")
        
        return result_dict
    
    def explore_once(self, visualize=False, idx=0):
        """
            在当前状态下进行一次探索, 默认此时的 robot arm 处于 reset 状态
            返回 explore_ok, explore_uv:
                explore_ok: bool, 代表 plan 阶段是否成功
                explore_uv: np.array([2,]), 代表本次尝试的 contact point 的 uv
        """        
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        
        for _ in range(30):
            self.capture_frame()
        cur_frame = self.capture_frame()
        
        obj_mask = self.affordance_map_2d.get_obj_mask(visualize=False) # H, W
        
        if self.explore_env_cfg["use_IOR"]:
            self.logger.log(logging.INFO, "Detected use_IOR flag = True, use Inhibition of Return")
            contact_uv = self.affordance_map_2d.sample_highest(visualize=False)
        else:
            # sample_prob 返回的是一个 N, 2 的 list, alpha 越大, 采样越密集
            self.logger.log(logging.INFO, "Detected use_IOR flag = False, do not update affoedance map")
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
        cur_frame: Frame
        pc_collision_w, pc_colors = cur_frame.get_obj_pc(
            use_robot_mask=False, 
            use_height_filter=True,
            world_frame=True
        )
        # 这里的 pc 仍然包含了 franka 的, 因此需要修改, 另外需要添加墙壁和 camera的 point cloud
        self.update_point_cloud_with_wall(clean_pc_np(pc_collision_w))
        # visualize_pc(pc_collision_w, point_size=5)
        
        for grasp_w in cur_frame.grasp_group:
            # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
            grasp = self.get_rotated_grasp(grasp_w, axis_out_w=dir_out_w)
            Tph2w =self.anyGrasp2ph(grasp=grasp)
            
            # DEMO
            if idx == 0:
                Tph2w = np.array([[-0.99393647, -0.07964764, -0.07580595,  0.49324237],
                [ 0.07665207, -0.00761566, -0.99702882, -0.19305159],
                [ 0.07883368, -0.99679399,  0.01367463,  0.30222573],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])
            else:
                Tph2w = np.array([[-0.99393647, -0.07964764, -0.07580595,  0.49324237],
                [ 0.07665207, -0.00761566, -0.99702882, -0.19305159],
                [ 0.07883368, -0.99679399,  0.01367463,  0.30222573 - 0.154],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])
                
            Tph2w_pre = self.get_translated_ph(Tph2w, -self.reserved_distance)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if result_pre is not None:
                if visualize or True:
                    visualize_pc(
                        points=pc_collision_w, 
                        point_size=5,
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
        self.open_gripper(target=0.06)
        self.clear_planner_pc()
        
        # self.approach_safe(distance=self.reserved_distance)
        self.approach(distance=self.reserved_distance)
        # self.close_gripper_safe(target=0.0, gripper_force=4)
        self.close_gripper(target=0.0, gripper_force=4)
        
        # 异步的执行 move, 且执行完成后需要将 self.recording 置为 False
        self.dir_out_w = dir_out_w
        move_thread = threading.Thread(target=self.async_move_out, daemon=True)
        move_thread.start()
        # 随即立刻开始录制
        self.recording = True
        
        while self.recording:
            cur_frame = self.capture_frame()
            self.obj_repr.frames.append(copy.deepcopy(cur_frame))
    
        return True, contact_uv
    
    def async_move_out(self):
        """
        当移动距离不够的时候就一直动
        """
        init_Tph2w = self.robot.get_ee_pose(as_matrix=True)
        def get_moved_distance():
            cur_Tph2w = self.robot.get_ee_pose(as_matrix=True)
            distance = np.linalg.norm(init_Tph2w[:3, -1] - cur_Tph2w[:3, -1])
            return distance
        
        num_tries = 2
        cur_tries = 0
        # while (get_moved_distance() < self.pertubation_distance / 2.):
        while (cur_tries < num_tries) and (get_moved_distance() < self.pertubation_distance * 0.8):
            # try:
            pull_out = CartesianMotion(
                Affine([0, 0, -self.pertubation_distance / num_tries]), 
                ReferenceType.Relative, 
                relative_dynamics_factor=0.05
            )
            reaction = Reaction(self.get_force() > 20, CartesianStopMotion())
            pull_out.add_reaction(reaction)
            self.franky_robot.move(pull_out, asynchronous=True)
            self.franky_robot.join_motion()
            cur_tries += 1
            # except Exception as e:
            self.open_gripper(0.06)
            self.franky_robot.recover_from_errors()
            # self.close_gripper_safe(target=0.0, gripper_force=4)
            self.close_gripper(target=0.0, gripper_force=4)
                
        self.recording = False
    
    def check_valid(self, visualize=False): 
        # 对于 frames 进行 tracking, 然后根据聚类结果判断 moving_part 动的多不多, 多的话就认为 valid
        # 这个函数相比于 deprecated 版本, 可以更好的处理 "柜子开一个缝隙, joint state 没有大的变化, 但是缝隙的深度有突变" 的情况
        # 这个函数会写入 obj_repr 的 tracks2d, tracks3d 和 moving mask
        if self.obj_repr.frames.num_frames() == 0:
            return False
        
        # 这里其实可以让 frames[0] 用 initial_frame 的 obj_mask
        self.obj_repr.initial_frame.segment_obj(
            obj_description=self.obj_env_cfg["obj_description"],
            post_process_mask=True,
            filter=True,
            visualize=visualize
        )
        self.obj_repr.frames[0].obj_mask = self.obj_repr.initial_frame.obj_mask & (~self.obj_repr.frames[0].robot_mask)
        self.obj_repr.frames[0].obj_bbox = self.obj_repr.initial_frame.obj_bbox
        # self.obj_repr.frames[0].segment_obj(
        #     obj_description=self.obj_env_cfg["obj_description"],
        #     post_process_mask=True,
        #     filter=True,
        #     visualize=visualize
        # ) 
        # self.obj_repr.save("/home/user/Programs/Embodied_Analogy/assets/unit_test/test.npy")
        self.obj_repr.frames[0].sample_points(num_points=self.explore_env_cfg["num_initial_pts"], visualize=visualize)
        self.obj_repr.frames.track_points(visualize=visualize)
        self.obj_repr.frames.track2d_to_3d(filter_robot=True, filter_consis=True)
        self.obj_repr.frames.cluster_track3d(visualize=visualize)
        
        if True:
            visualize_pc(
                points=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[0],
                point_size=5,
                colors=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[1] / 255.,
                tracks_3d=self.obj_repr.frames.track3d_seq,
                tracks_n_step=None,
                tracks_t_step=3,
                tracks_norm_threshold=0.2e-2,
            )
            
            visualize_pc(
                points=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[0],
                point_size=5,
                colors=self.obj_repr.frames[0].get_obj_pc(
                    use_robot_mask=True, 
                    use_height_filter=True,
                    world_frame=False,
                    visualize=False
                )[1] / 255.,
                tracks_3d=self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :],
                tracks_n_step=None,
                tracks_t_step=3,
                tracks_norm_threshold=0.2e-2,
            )
            
        # 根据 moving tracks 的位移来判断, (T, M, 3)
        moving_tracks = self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :]
        
        # 原始的版本 (用首尾帧的)
        mean_delta = np.linalg.norm(moving_tracks[-1] - moving_tracks[0], axis=-1).mean()
        if mean_delta > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
    
    ###############################################
    def main(self):
        try:
        # if True:
            self.explore_result = {}
            self.explore_result = self.explore_stage(visualize=False)
                    
        except Exception as e:
            self.logger.log(logging.ERROR, f"Explore exception occured: {e}", exc_info=True)
            
            self.explore_result["has_valid_explore"] = False
            self.explore_result["joint_type"] = self.obj_env_cfg["joint_type"]
            self.explore_result["exception"] = str(e)

        # 将 franka 复位
        self.reset_robot_safe()
        
        if self.exp_cfg["save_result"]:
            save_json_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "explore_result.json"
            )
            with open(save_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.explore_result, json_file, ensure_ascii=False, indent=4, default=numpy_to_json)
        
        if self.exp_cfg["save_obj_repr"]:
            save_path = os.path.join(
                self.exp_cfg["exp_folder"],
                str(self.task_cfg["task_id"]),
                "obj_repr_explore.npy"
            )
            self.obj_repr.save(save_path)
                    
if __name__ == "__main__":
    import yaml
    # with open("/home/user/Programs/Embodied_Analogy/embodied_analogy/realworld_environment/cabinet.yaml", "r") as f:
    with open("/home/user/Programs/Embodied_Analogy/embodied_analogy/realworld_environment/drawer.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    exploreEnv = ExploreEnv(cfg=cfg)
    exploreEnv.main()
    # exploreEnv.save(file_path=f"/home/zby/Programs/Embodied_Analogy/assets/tmp/{obj_index}/explore/explore_data.pkl")
    