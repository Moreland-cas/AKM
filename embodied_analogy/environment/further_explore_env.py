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
from embodied_analogy.environment.reconstruct_env import ReconEnv
from embodied_analogy.representation.obj_repr import Obj_repr

class FurtherExploreEnv(ReconEnv):
    def __init__(
            self,
            cfg
        ):        
        # TODO: cfg 需要包含 obj_folder_path_further_explore
        
        ObjEnv().__init__(cfg=cfg)
        self.cfg = cfg
        print("loading further explore env, using cfg:", cfg)
        
        # self.record_fps = cfg["record_fps"]
        # self.record_interval = math.ceil(1.0 / self.phy_timestep / self.record_fps)
        self.pertubation_distance = cfg["pertubation_distance"]
        self.valid_thresh = cfg["valid_thresh"]
        
        # self.instruction = cfg["instruction"]
        
        # self.update_sigma = cfg["update_sigma"]
        # self.max_tries = cfg["max_tries"]
        # self.obj_description = cfg["obj_description"]
        # self.has_valid_explore = False
        
        # load reconstructed obj_repr
        self.obj_repr = Obj_repr.load(os.path.join(cfg["obj_folder_path_reconstruct"], "obj_repr.npy"))
        
        # 读取和保存所用的变量
        # self.joint_type = cfg["joint_type"]
        self.save_prefix = cfg["obj_folder_path_further_explore"]
        os.makedirs(self.save_prefix, exist_ok=True)
        
        self.init_grasp_dict()

    def init_grasp_dict(self):
        self.grasp_dict = {}
        # 是不是可以新在 obj_repr 中添加一个 grasp_dict 属性, 用于存储一些 joint_state: grasp 的 pair
        # 目前先单独保存，另外似乎没有必要保存一个 frame, 因为 frame 到 grasp 的映射最终还是通过 joint_state 作为中间变量的
        self.grasp_dict = {}
        # 然后用 obj_repr 的 kframes[0] 的 grasp 对于 grasp_dict 进行初始化

        # 对于不同关节, 将 joint 可动范围分为不同区间，每个区间只需要有一个对应的 grasp 就行了
        # 假设区间范围是 (joint_min, joint_max), 那就让 grasp_dict 的 keys 为 joint_min, joint_min + delta, ..., joint_max, 共计 N 个
        # joint_type = self.obj_repr.coarse_joint_dict["joint_type"]
        joint_min = 0
        joint_max = self.cfg["max_distance"]
        # TODO: 暂时设计需要 5 个
        num_discrete = 5
        grasp_keys = np.linspace(joint_min, joint_max, num_discrete)
        # 一个区间的长度
        grasp_range = (joint_max - joint_min) / (num_discrete - 1)
        # 那么在 grasp_key 处存储的 grasp, 应该能保证在区间 grasp_key +- 1.1 grasp_range/2 的范围内的稳定移动
        # 这样就可以保证 使用 grasp_key[i] 处的 action 可以将关节状态改变到 grasp_key[i+1] 那里
        for grasp_key in grasp_keys:
            self.grasp_dict[grasp_key] = {
                "num_tries": 0,
                "has_stable": False,
                # 用来存储探索过程中遇到的 joint_state: grasp
                # 注意需要保证存储的 joint_state 在 grasp_key 附近 
                "stable_grasp": [],
                # 把 unstable_grasp 也存储起来, 方便拒绝类似的 grasp
                "unstable_grasp": []
            }
        # 最后把 grasp_key = 0 的地方用 obj_repr 在 explore 阶段得到的信息进行填充
        self.obj_repr: Obj_repr
        self.grasp_dict[joint_min]["stable_grasp"] = {
            "num_tries": 0,
            "has_stable": True,
            "stable_grasp": [(0, self.obj_repr.kframes[0].Tph2w)],
            "unstable_grasp": []
        }
    
    def classify_state_to_key(self):
        keys = self.grasp_dict.keys()
        dis = np.array(keys) - self.cur_state
        dis_abs = np.abs(dis)
        min_index = np.argmin(dis_abs)
        self.cur_key = keys[min_index]

    def get_stable_grasp_proposal(self):
        """
        返回基于当前状态的最好的 grasp
        要求在 cur_state 下首先检测可行的 grasp, 然后利用 dynamic_mask 和 历史记录对于 grasp 进行筛选, 返回最好的 grasp proposal
        """
        pass

    def is_state_action_stable(self, state, manip_type, delta, grasp):
        # 首先到 state 附近, 然后在 manip_type 方向用 grasp 姿势移动 delta 距离
        # 最后判断操作后的 state 和 inital_state + delta 是否足够接近
        # 并且需要复现这个足够多次才可以
        pass

    def explore_from_current_state(self):
        """
        TODO 
            在探索的过程中最好保留每个 (state, manip_type + action, state) pair, 方便后续查询。
            另外 explore 阶段也可以修改为把 invalid pair 也进行保存
        
        探索阶段的逻辑:
            从初始的 state: action 开始, 探索其有效边界 (state_low, state_high)
            TODO 其实这个边界应该跟 manip_type, 也就是操作方向也是有关的, 因此应该是 (state, action, manip_type, delta) 的形式
            边界的定义方式为 能使用该 action 鲁棒的在范围内改变物体状态
            
            由于我们是从 state = 0 开始探索, 因此只需要不断向上扩充边界, 即在当前的 state_high 开始
            找到新的 action, 使得该 action 可以稳定的将物体开关到 (state_high, state_higher)

            这个循环会一直执行, 直到 state_high >= state_max
        """
        # TODO 这里先考虑只打开的情况
        # TODO 保存操作的历史信息
        assert self.obj_repr.kframes[0].joint_state == 0
        # 一些超参数
        if self.obj_repr.fine_joint_dict["joint_type"] == "revolute":
            delta_state = np.deg2rad(10) # 10 degree
            stable_thresh = np.deg2rad(2) # 2 deg
        elif self.obj_repr.fine_joint_dict["joint_type"] == "prismatic":
            delta_state = 0.2 # 1 dm
            stable_thresh = 0.01 # 1cm

        # 一些中间变量
        self.num_further_explore = 0
        self.max_further_explore = 20

        # 这个函数会被外层 while (True) 的执行, 因此需要有退出机制
        if self.num_further_explore > self.max_further_explore:
            return 
        
        # 首先恢复到当前探索的最远的稳定状态
        # 获取当前状态
        self.update_cur_frame()
        # 获取离当前状态最近的稳定状态
        nearest_stable_state = 0 # TODO 这里先设置为 0
        # 首先获取当前状态, 即更新 cur_frame 和 cur_state
        while (abs(self.cur_state - nearest_stable_state) > stable_thresh) and self.num_further_explore < self.max_further_explore:
            # 获取可靠的 grasp
            # 当第一次执行时
            # Tph2w = self.transform_grasp(
            #     Tph2w_ref=self.obj_repr.kframes[0].Tph2w,
            #     ref_state=self.obj_repr.kframes[0].joint_state,
            #     tgt_state=self.cur_state
            # )
            Tph2w = self.get_stable_grasp_proposal()

            self.open_gripper()
            self.move_to_pose_safe(
                Tph2w=Tph2w,
                reserved_distance=self.cfg["reserved_distance"]
            )
            self.close_gripper()
            # 进行复位
            joint_dict = self.obj_repr.get_joint_param(
                resolution="fine",
                frame="world"
            )
            self.move_along_axis(
                joint_type=joint_dict["joint_type"],
                joint_axis=joint_dict["joint_dir"],
                joint_start=joint_dict["joint_start"],
                moving_distance=nearest_stable_state-self.cur_state
                # moving_distance=0-self.cur_state
            )
            self.num_further_explore += 1
            self.update_cur_frame()

            # TODO 对于这一次操作也进行记录
        
        # 记录下来当前的 state, manip_type, grasp
        traj = {
            "start_state": self.cur_state,
            "manip_type": "open",
            "start_grasp": self.get_ee_pose(as_matrix=True),
            # open: [(0.1, 0.02), (0.2, 0.03), ...]
            # close: [(-0.1, 0.01), (-0.2, -0.02), ...]
            "range_with_loss": [] 
        }
        # 现在的状态是机械臂 close 在 nearest_stable_state 的地方, 那么从当前位置开始探索
        # !TODO goal 确实写的不对 不应该是 self.cur_state + delta_state 而应该是 state_start + x * delta_state
        goal = self.cur_state + delta_state # TODO 这里目前设置仅仅往更大的方向

        # 假设尝试次数还有, 并且当前的 grasp 一直可以将物体比较准的打开, 那么就一直跑
        while self.num_further_explore < self.max_further_explore and not self.get_all_grasp():
            range_with_loss_tmp = [goal] # TODO: 是不是可以用一个 traj 类进行管理
            # 那么我们不断的执行一小步的 open delta_state 操作, 每次执行完都看一下效果
            self.move_along_axis(
                joint_type=joint_dict["joint_type"],
                joint_axis=joint_dict["joint_dir"],
                joint_start=joint_dict["joint_start"],
                # 由于之前的 while 循环是从 cur_state 很小跳出的, 所以我们不需要更新 cur_state 的估计
                moving_distance=delta_state
            )
            self.num_further_explore += 1
            
            # 再跑一遍 reloc, 然后记录下真实的状态, 如果误差很大那么就退出 while 循环
            self.update_cur_frame()
            range_with_loss_tmp.append(self.cur_state)
            # TODO: 看看是不是只有误差小的咱们说给他保存起来
            traj["range_with_loss"].append(range_with_loss_tmp)

            # 如果误差太大, 那就 break, 然后返回上一个状态进行 explore
            if abs(self.cur_state - goal) > stable_thresh:   # 不应该大于 1cm
                break
        
        # 退出上述循环后, 继续在当前状态下开始探索
        self.explore_from_current_state()


    def get_all_grasp(self):
        """
        判断目前探索的 grasp 所能控制的区间是否已经包含了 (0, max_distance)
        """
    def manip_complex(self, delta_state):
        """
        操作阶段的逻辑：
        state = state_start
        while state != state_end:
            对于当前的 state, 查询包含该 state 的 action 中, 最能接近 state_end 的 action 并进行执行
        """
        pass

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
        # 对于 frames 进行 tracking, 然后根据聚类结果判断 moving_part 动的多不多, 多的话就认为 valid
        # 这个函数相比于 deprecated 版本, 可以更好的处理 "柜子开一个缝隙, joint state 没有大的变化, 但是缝隙的深度有突变" 的情况
        # 这个函数会写入 obj_repr 的 tracks2d, tracks3d 和 moving mask
        self.obj_repr.frames[0].segment_obj(
            obj_description=self.cfg["obj_description"],
            post_process_mask=True,
            filter=True,
            visualize=visualize
        )
        self.obj_repr.frames[0].sample_points(num_points=self.cfg["num_initial_pts"], visualize=visualize)
        self.obj_repr.frames.track_points(visualize=visualize)
        self.obj_repr.frames.track2d_to_3d(filter=True, visualize=visualize)
        self.obj_repr.frames.cluster_track3d(visualize=visualize)
        
        # 根据 moving tracks 的位移来判断
        moving_tracks = self.obj_repr.frames.track3d_seq[:, self.obj_repr.frames.moving_mask, :]
        
        mean_delta = np.linalg.norm(moving_tracks[-1] - moving_tracks[0], axis=-1).mean()
        if mean_delta > self.pertubation_distance * self.valid_thresh:
            self.has_valid_explore = True
            return True
        else:
            return False
        
    
if __name__ == "__main__":
    
    exploreEnv = FurtherExploreEnv(
        cfg={
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "valid_thresh": 0.5,
    "instruction": "open the cabinet",
    "num_initial_pts": 1000,
    "obj_description": "cabinet",
    "joint_type": "revolute",
    "obj_index": "45162",
    "joint_index": "0",
    "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/45162_link_0",
    "active_link_name": "link_0",
    "active_joint_name": "joint_0",
    "load_pose": [
        0.8806502223014832,
        0.0,
        0.6088799834251404
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
    