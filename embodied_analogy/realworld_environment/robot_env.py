import os
import math
import pybullet as p
import pybullet_data
from franky import *
import mplib
import logging
import numpy as np
import sapien.core as sapien
from PIL import Image
from scipy.ndimage import binary_dilation

import transforms3d as t3d
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from embodied_analogy.realworld_environment.base_env import BaseEnv
from embodied_analogy.utility.utils import visualize_pc


from sklearn.cluster import SpectralClustering
import open3d as o3d
def clean_pc_np(
    points: np.ndarray,
    voxel_size=0.01,
    sor_k=20, sor_std=2.0,
    clustering_threshold=0.1,  # 点数较少的类别占总点数的比例阈值
    num_iterations=5,  # 聚类迭代次数
) -> np.ndarray:
    """
    输入: (N,3) np.ndarray
    输出: (M,3) np.ndarray  (M <= N)
    """
    # 1) np.ndarray → o3d.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 2) 下采样
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # 3) 统计离群点移除
    _, ind_sor = pcd.remove_statistical_outlier(
        nb_neighbors=sor_k,
        std_ratio=sor_std
    )
    pcd = pcd.select_by_index(ind_sor)

    # 4) 多次二分聚类过滤
    points_after_sor = np.asarray(pcd.points)
    for _ in range(num_iterations):
        if len(points_after_sor) < 2:
            break  # 如果点数少于2个，无法继续聚类

        # 使用 SpectralClustering 进行二分聚类
        clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
        labels = clustering.fit_predict(points_after_sor)

        # 统计每个聚类的点数
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        # 检查两个聚类的点数
        if len(label_counts) == 2:
            if label_counts[0] / len(points_after_sor) < clustering_threshold:
                # 如果第一个聚类的点数太少，保留第二个聚类
                points_after_sor = points_after_sor[labels == 1]
            elif label_counts[1] / len(points_after_sor) < clustering_threshold:
                # 如果第二个聚类的点数太少，保留第一个聚类
                points_after_sor = points_after_sor[labels == 0]
            else:
                # 如果两个聚类的点数都足够多，保留所有点
                break
        else:
            # 如果只有一个聚类，直接退出循环
            break

    # 5) 返回结果
    return points_after_sor



# 定义一个自己的类, 兼具 planner 和 control, 也即 mplib 和 franky 的功能
class RealworldRobot():
    def __init__(self, mplib_robot, franky_robot, franky_gripper):
        self.mplib_robot = mplib_robot
        self.franky_robot = franky_robot
        self.franky_gripper = franky_gripper

    def set_root_pose(self, pose):
        self.mplib_robot.set_root_pose(pose)

    def get_qlimits(self):
        return self.mplib_robot.get_qlimits()

    def get_active_joints(self):
        return self.mplib_robot.get_active_joints()

    def get_links(self):
        return self.mplib_robot.get_links()

    def get_qpos(self):
        """
        mplib 需要 (9,) 大小的 array, 但是 self.franky_robot 直接得到的是 (7,) 维度的, 需要扩充
        return realworld robot qpos, np.array([9])
        """
        gripper_width = self.franky_gripper.width
        gripper_qpos = gripper_width / 2.
        qpos_7 = self.franky_robot.current_joint_state.position
        qpos_9 = np.concatenate([qpos_7, [gripper_qpos, gripper_qpos]])
        return qpos_9

    def move(self, motion):
        """
        让机器人执行 franky::motion 类所控制的运动
        """
        self.franky_robot.move(motion)

    def get_ee_pose(self, as_matrix=False):
        """
        获取 end-effector (panda_hand) 的 ee_pos 和 ee_quat (Tph2w)
        ee_pos: np.array([3, ])
        ee_quat: np.array([4, ]), scalar_first=False, in (x, y, z, w) order

        """
        # Get the robot's cartesian state
        cartesian_state = self.franky_robot.current_cartesian_state
        robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
        ee_pose = robot_pose.end_effector_pose

        ee_pos = ee_pose.translation
        ee_quat = ee_pose.quaternion # scalar_last

        R_franky = R.from_quat(ee_quat, scalar_first=False).as_matrix()

        if as_matrix:
            T = np.eye(4)  # 创建一个 4x4 单位矩阵
            # NOTE 对于 sapien 来说, get_ee_pose 返回的 quat 是 w-first 的, 但是对于 franky 来说是 w_last 的
            T[:3, :3] = R_franky  # 将修正后的旋转矩阵放入变换矩阵
            T[:3, 3] = ee_pos  # 将位置向量放入变换矩阵
            return T
        else:
            return ee_pos, ee_quat # scalar_last
    
class RobotEnv(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.load_robot()
        self.pybullet_handle = p
        self.pybullet_handle.connect(self.pybullet_handle.DIRECT)
        self.pybullet_handle.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_robot_id = self.pybullet_handle.loadURDF("franka_panda/panda.urdf")  # 加载Panda
        
        self.reset_ee_pose = np.array([0.38938136, 0.09781701, 0.47887616])
        self.reset_qpos = np.array([-0.45536156, -0.43716924,  0.89536322, -2.38142135, -1.05704519,
        1.42962151, -0.31493183,  0.        ,  0.        ])

    def capture_frame(self, visualize=False, robot_mask=True) -> Frame:
        frame = super().capture_frame(visualize=visualize)
        if robot_mask:
            frame.robot_mask = self.capture_robot_mask()
            frame.Tph2w = self.get_ee_pose(as_matrix=True)
        return frame

    def load_robot(self):
        # 首先 load 一个 franka arm 的 asset
        # urdf_config = dict(
        #     _materials=dict(
        #         gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        #     ),
        #     link=dict(
        #         panda_leftfinger=dict(
        #             material="gripper", patch_radius=0.1, min_patch_radius=0.1
        #         ),
        #         panda_rightfinger=dict(
        #             material="gripper", patch_radius=0.1, min_patch_radius=0.1
        #         ),
        #     ),
        # )
        # config = parse_urdf_config(urdf_config, self.scene)
        # check_urdf_config(config)

        # load Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        self.mplib_robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/fr3.urdf")
        # self.mplib_robot: sapien.Articulation = loader.load(self.asset_prefix + "/fr3/fr3.urdf", config)
        self.franky_robot = Robot("172.16.0.2")  # Replace this with your robot's IP
        self.franky_gripper = Gripper("172.16.0.2")
        self.robot = RealworldRobot(
            mplib_robot=self.mplib_robot,
            franky_robot=self.franky_robot,
            franky_gripper=self.franky_gripper
        )

        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        self.arm_qlimit = self.robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        # Setup control properties
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints[:4]:
            joint.set_drive_property(stiffness=400, damping=40, force_limit=100)    # original: 160
        for joint in self.active_joints[4:-2]:
            joint.set_drive_property(stiffness=400, damping=40, force_limit=50)    # original: 160
        for joint in self.active_joints[-2:]:
            joint.set_drive_property(stiffness=160, damping=10, force_limit=50)

        # set id for getting mask
        # for link in self.robot.get_links():
        #     for s in link.get_visual_bodies():
        #         s.set_visual_id(255)

        """
        准备好 self.franky_robot 和 self.franky_gripper
        """
        # Start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
        self.franky_robot.relative_dynamics_factor = 0.05
        self.franky_robot.recover_from_errors()

        self.gripper_speed = 0.04 # m/s
        self.gripper_force = 5 # N

        self.setup_planner()
        # self.reset_robot()

    def capture_robot_mask(self):
        """
        Renders a boolean mask of a robot using PyBullet.

        Returns:
            Boolean mask (H, W) where True indicates robot pixels
        """
        # first prepare some variables
        H, W = self.frame_height, self.frame_width
        near, far = 0.01, 5
        camera_extrinsic, camera_intrinsic = self.camera_extrinsic, self.camera_intrinsic
        qpos = self.robot.get_qpos()
        
        # Set joint positions
        for i in range(7):
            self.pybullet_handle.resetJointState(self.pybullet_robot_id, i, qpos[i])
        self.pybullet_handle.resetJointState(self.pybullet_robot_id, 9, qpos[-2])
        self.pybullet_handle.resetJointState(self.pybullet_robot_id, 10, qpos[-1])

        # Extract camera parameters, Tw2c
        Tw2c = camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        R = Tc2w[:3, :3]
        t = Tc2w[:3, 3]
        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        
        # Calculate view matrix
        eye_pos = t
        forward = R[:, -1]  # Camera's forward direction (looking along -Z)
        target_pos = (t + forward).tolist()  # Point along camera's view direction
        view_matrix = self.pybullet_handle.computeViewMatrix(eye_pos, target_pos, -R[:, 1])

        # Calculate projection matrix from intrinsic parameters
        # less accurate one of computing Projection matrix
        # fov_y = np.degrees(2 * np.arctan2(H, 2 * fy))  # Vertical FOV
        # aspect = W / H
        # proj_matrix = p.computeProjectionMatrixFOV(fov_y, aspect, near, far)
        
        # More accurate one
        # 计算 openGL frustum 中的 l, r, t, b
        ll = -near * cx / fx
        tt = near * cy / fy
        rr = near * (W - cx) / fx
        bb = -near * (H - cy) / fy
        proj_matrix = self.pybullet_handle.computeProjectionMatrix(ll, rr, bb, tt, near, far)
            
        # Render segmentation image
        _, _, _, _, seg = self.pybullet_handle.getCameraImage(
            W, H, view_matrix, proj_matrix, 
            renderer=self.pybullet_handle.ER_BULLET_HARDWARE_OPENGL,
            # flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        robot_mask = (seg == self.pybullet_robot_id)
        
        robot_mask = binary_dilation(robot_mask, iterations=3)
        
        # 由于 robot mask 并不总精准, 因此需要轻微膨胀
        return robot_mask        

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        # active joints
        # ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
        # 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

        # link names
        # ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5',
        # 'panda_link6', 'panda_link7', 'panda_link8', 'panda_hand', 'panda_hand_tcp', 'panda_leftfinger',
        # 'panda_rightfinger', 'camera_base_link', 'camera_link']

        self.planner = mplib.Planner(
            # urdf=self.asset_prefix + "/fr3/fr3.urdf",
            urdf=self.asset_prefix + "/panda/fr3.urdf",
            srdf=self.asset_prefix + "/panda/panda_v3.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            # move_group="fr3_link8",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def open_gripper(self, target=0.06):
        self.franky_gripper.move(target, self.gripper_speed)

    def close_gripper(self, target=0., gripper_force=10):
        if gripper_force is None:
            gripper_force = self.gripper_force
        if target is None:
            target = 0.
        self.franky_gripper.grasp(target, self.gripper_speed, gripper_force, epsilon_outer=1.0)
    
    # def close_gripper_safe(self, target=0.0, gripper_force=10):
    #     # 首先记录最开始的 Tph2w
    #     init_Tph2w = self.robot.get_ee_pose(as_matrix=True)
    #     init_gripper_width = self.franky_gripper.width
    #     safely_closed = False
    #     num_tries = 0
    #     max_tries = 5
        
    #     while (num_tries < max_tries) and not safely_closed:
    #         try:
    #             self.move_to_pose(init_Tph2w, wrt_world=True)
    #             self.open_gripper(init_gripper_width)
                
    #             # 随机在 y 轴做一个微调
    #             if num_tries == 0:
    #                 random_shift_y = 0
    #             else:
    #                 random_shift_y = np.random.uniform(-0.01, 0.01)
    #             shift_motion = CartesianMotion(
    #                 Affine([0.0, random_shift_y, 0.]), 
    #                 ReferenceType.Relative, 
    #                 relative_dynamics_factor=0.05
    #             )
    #             # reaction_motion = CartesianMotion(
    #             #     Affine([0.0, -random_shift_y / 2, 0.0]), 
    #             #     ReferenceType.Relative,
    #             #     relative_dynamics_factor=0.1
    #             # )
    #             # shift_motion.add_reaction(self.get_force() > 7, reaction_motion)
    #             self.franky_robot.move(shift_motion)
    #             self.franky_gripper.grasp(target, self.gripper_speed, gripper_force, epsilon_outer=1.0)
    #             safely_closed = True
                
    #         except Exception as e:
    #             num_tries += 1
    #             self.franky_robot.recover_from_errors()
    #             self.open_gripper(0.06)
    #             self.franky_robot.recover_from_errors()
        
    #     if not safely_closed:
    #         self.open_gripper(0.06)
    #         self.franky_robot.recover_from_errors()
            
    def close_gripper_safe(self, target=0.02, gripper_force=5):
        # obj_width = 0.01
        # self.open_gripper(0.06)
        shift_motion = CartesianMotion(
            Affine([0.0, self.franky_gripper.width, 0.]), 
            ReferenceType.Relative, 
            relative_dynamics_factor=0.05
        )
        reaction_motion = CartesianMotion(
            Affine([0.0, -(self.franky_gripper.width - 0.01) / 2, 0.0]), 
            ReferenceType.Relative,
            relative_dynamics_factor=0.1
        )
        reaction = Reaction(self.get_force() > gripper_force, reaction_motion)
        shift_motion.add_reaction(reaction)
        self.franky_robot.move(shift_motion)
        self.franky_gripper.move(target, self.gripper_speed)
        # self.franky_gripper.grasp(target, self.gripper_speed, gripper_force, epsilon_outer=1.0)


    def get_force(self, dir_w=None):
        """
        deprecated
        获取 dir_w 方向上的 exernal force, 如果是 None 获取所有的
        """
        if dir_w is None:
            normal_force = (Measure.FORCE_X ** 2 + Measure.FORCE_Y ** 2 + Measure.FORCE_Z ** 2) ** 0.5
        else:
            dir_w = dir_w / np.linalg.norm(dir_w)
            normal_force = (Measure.FORCE_X * dir_w[0] + Measure.FORCE_Y * dir_w[1] + Measure.FORCE_Z * dir_w[2]) **2 **0.5
        return normal_force
    
    def reset_robot(self):
        """
        reset_robot 不控制 gripper 的 open/close 状态, 只把其他关节进行 reset
        """
        self.logger.log(logging.INFO, "Reset robot ...")
        self.open_gripper()
        
        cur_ee_pose = self.robot.get_ee_pose(as_matrix=False)[0]
        if np.linalg.norm(self.reset_ee_pose - cur_ee_pose) > 0.2: 
            # 距离很远时, 先向后撤退 1dm
            self.move_forward(
                moving_distance=-0.1,
                drop_large_move=False
            )
        # init_panda_hand = np.array([
        #     [1, 0, 0, 0.2], 
        #     [0, -1, 0, 0], 
        #     [0, 0, -1, 0.7],
        #     [0, 0, 0, 1]
        # ])
        # self.move_to_pose(pose=init_panda_hand, wrt_world=True)
        # self.update_point_cloud_with_wall(None)
        joint_motion = JointMotion(self.reset_qpos[:7])
        self.franky_robot.move(joint_motion)
        
    def reset_robot_safe(self):
        self.logger.log(logging.INFO,  "Call safe robot reset ...")
        # 先打开 gripper, 再撤退一段距离
        self.open_gripper()
        
        # cur_ee_pose = self.robot.get_ee_pose(as_matrix=False)[0]
        # if np.linalg.norm(self.reset_ee_pose - cur_ee_pose) > 0.2: 
            # 距离很远时, 先向后撤退 1dm
        self.move_forward(
            moving_distance=-0.1,
            drop_large_move=False
        )
        tmp_frame = self.capture_frame()
        # 给 tmp_frame 一个 obj_mask
        tmp_frame.segment_obj(
            obj_description="cabinet",
            post_process_mask=True,
            filter=True,
            visualize=False
        )
        
        tmp_pc = tmp_frame.get_obj_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )[0]
        # 仅保留 y 轴上坐标小于 -0.1 的点
        # tmp_pc = tmp_pc[tmp_pc[:, 1] < -0.1]
        
        tmp_pc = clean_pc_np(tmp_pc)
        self.update_point_cloud_with_wall(tmp_pc)
        try:
            result = self.plan_qpos(goal_qpos=self.reset_qpos)
        except Exception as e:
            result = None
            
        if result is not None:
            self.follow_path(result)
        else:
            self.logger.log(logging.WARNING, "Get None result in reset_robot_safe function, executing reset() instead...")
            self.reset_robot()

    def clear_planner_pc(self):
        self.update_point_cloud_with_wall(pc_w=None)

    def plan_path(self, target_pose: np.ndarray, wrt_world: bool = True):
        """
        NOTE: 改为返回 result 最少的那个 target_pose (绕 z 轴旋转 180 degree 又是一条好汉)
            传入的 target_pose 是 Tph2w
        """
        target_pose1 = np.copy(target_pose)
        target_pose2 = np.copy(target_pose)
        target_pose2[:, 0] *= -1
        target_pose2[:, 1] *= -1
        
        result1, result2 = None, None
        try:
            result1 = self.planner.plan_pose(
                goal_pose=mplib.Pose(target_pose1),
                current_qpos=self.robot.get_qpos(),
                time_step=self.planner_timestep,
                rrt_range=0.1,
                planning_time=1,
                wrt_world=wrt_world
            )
        except Exception as e:
            self.logger.log(logging.WARNING, f"Encounter {e} during RobotEnv.plan_path()")
            
        try:
            result2 = self.planner.plan_pose(
                goal_pose=mplib.Pose(target_pose2),
                current_qpos=self.robot.get_qpos(),
                time_step=self.planner_timestep,
                rrt_range=0.1,
                planning_time=1,
                wrt_world=wrt_world
            )
        except Exception as e:
            self.logger.log(logging.WARNING, f"Encounter {e} during RobotEnv.plan_path()")

        def result_valid(result):
            return (result is not None) and (result['status'] == "Success")
        
        if result_valid(result1):
            if not result_valid(result2):
                return result1
            else:
                # 比较两个 result, 返回 planned path 更短的那个
                n_step1 = result1['position'].shape[0]
                n_step2 = result2['position'].shape[0]
                if n_step1 < n_step2:
                    return result1
                else:
                    return result2
        else:
            if result_valid(result2):
                return result2
            return None

    def plan_qpos(self, goal_qpos):
        """
        调用 self.planner 的 plan_qpos 方法, 输入一个 goal_qpos, 返回一个 plan_result
        
        goal_qpos: np.ndarray, shape=(7,), dtype=np.float32
        return:
            {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
            
        NOTE: 
        区别在于 mplib.planner 的 plan_qpos 会输出多个 goal_qpos, 而返回一个 plan_result
        
        我们就只是输入一个 goal_qpos, 不行就换下一个, 因为我们想要多个 plan_result, 他们有不同
        的qpos, 但是相同的 ee_pos
        """
        format_input = [goal_qpos]
        current_qpos = self.robot.get_qpos()
        result = self.planner.plan_qpos(
            goal_qposes=format_input,
            current_qpos=current_qpos,
            time_step=self.planner_timestep,
            rrt_range=0.1,
            planning_time=1
        )
        if result["status"] != "Success":
            return None
        else:
            return result
        
    def follow_path(self, result, reaction_motion=None):
        # 修改为 franky 的版本
        n_step = result['position'].shape[0]
        self.logger.log(logging.INFO, f"n_step: {n_step}")

        if n_step == 0:
            return
        # 改为用 velocity 和不用两个版本
        use_velocity = False
        joint_waypoint_motion_list = []
        for i in range(n_step):
            position_target = result['position'][i].tolist() # 7
            velocity_target = result['velocity'][i].tolist()

            if use_velocity:
                joint_waypoint = JointWaypoint(
                    JointState(
                        position=position_target,
                        velocity=velocity_target
                    )
                )
            else:
                joint_waypoint = JointWaypoint(position_target)
            joint_waypoint_motion_list.append(joint_waypoint)
        joint_waypoint_motion = JointWaypointMotion(joint_waypoint_motion_list)
        
        if reaction_motion is not None: #TODO 添加 default的
            joint_waypoint_motion.add_reaction(reaction_motion)
        self.franky_robot.move(joint_waypoint_motion)

    def update_point_cloud_with_wall(self, pc_w=None):
        """
        不但更新世界坐标系下的物体点云 pc_w, 还更新室内墙壁和相机所在的柱体, 以及桌面信息
        """
        pc_update = []
        
        if pc_w is not None:
            pc_update.append(pc_w)
        
        # 添加墙面, (1000, 3)
        pc_wall_yz = np.random.uniform(-1, 1, (1000, 2))
        pc_wall_x = np.ones((1000, 1)) * -0.3
        pc_wall = np.concatenate([pc_wall_x, pc_wall_yz], axis=1)
        pc_update.append(pc_wall)
        
        # 添加地面 (虽然好像也不需要?)
        pc_ground_xy = np.random.uniform(-1, 1, (1000, 2))
        pc_ground_z = np.ones((1000, 1)) * 0.01
        pc_ground = np.concatenate([pc_ground_xy, pc_ground_z], axis=1)
        # 2. 过滤掉距离原点 < 0.15 的点
        dist2 = np.sum(pc_ground ** 2, axis=1) ** 0.5      # 平方距离
        pc_ground = pc_ground[dist2 >= 0.2]
        pc_update.append(pc_ground)
        
        # 添加相机所在的柱体, 这里先用一个平面代替
        pc_camera_yz = np.random.uniform(-1, 1, (1000, 2))
        pc_camera_x = np.ones((1000, 1)) * 0.8
        pc_camera = np.concatenate([pc_camera_x, pc_camera_yz], axis=1)
        pc_update.append(pc_camera)
        
        self.planner.update_point_cloud(
            points=np.concatenate(pc_update, axis=0)
        )
    
    def move_to_pose(self, pose: mplib.pymp.Pose, wrt_world: bool):
        # pose: Tph2w
        result = self.plan_path(target_pose=pose, wrt_world=wrt_world)
        if result is not None:
            self.follow_path(result)
        else:
            self.logger.log(logging.WARNING, "Get None result in move_to_pose function, not executing...")

    def move_to_pose_safe(self, Tph2w, reserved_distance=0.05):
        """
        控制 panda_hand 移动到 Tph2w 的目标位置
        不同的是, 需考虑移动过程中的环境点云, 和 reserved_distance
        """
        # 加一个局部调整 Tph2w 的函数
        # self.base_step()
        frame = self.capture_frame()
        pc_w, _ = frame.get_env_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_w)

        # 获取 pre_grasp_pose

        Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
        result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)

        if result_pre is None:
            self.logger.log(logging.WARNING, "move_to_pose_safe(): planning to pre_grasp_pose failed")
            return None
        else:
            self.follow_path(result_pre)
            # 在这里获取 robot qpos, 也即 pre_Tph2w 对应的 pre_qpos
            pre_qpos = self.robot.get_qpos()
            # 在往前走之前把 collision pc 去掉
            self.clear_planner_pc()
            self.move_forward(
                moving_distance=reserved_distance,
                drop_large_move=False
            )
            return pre_qpos

    def get_translated_ph(self, Tph2w, distance):
        """
            输出 Tph2w 沿着当前模型向前或者向后一定距离的一个 ph 位姿
            distance 大于 0 为向前移动, 反之为向后移动
        """
        Tph2w_ = np.copy(Tph2w)
        forward_direction = Tph2w[:3, :3] @ np.array([0, 0, 1]) # 3
        Tph2w_[:3, 3] = Tph2w[:3, 3] + distance * forward_direction
        return Tph2w_

    def get_rotated_grasp(self, grasp, axis_out_w):
        """
            绕着 y 轴旋转 grasp 坐标系使得 grasp 坐标系的 x 轴与 axis_out_w 的点乘尽可能小
            grasp: anygrasp Grasp, Tgrasp2w
            axis_w: np.array, 3
        """
        from graspnetAPI import Grasp
        grasp_ = Grasp()
        grasp_.grasp_array = np.copy(grasp.grasp_array)

        Rgrasp2w = grasp_.rotation_matrix
        """
        Rrefine2w = Rgrasp2w @ Rrefine2grasp
        Rrefine2w[:, 0] * axis_out_w 尽可能小
        又因为 Rrefine2grasp 是绕着 y 轴的旋转, 所以有形式:
            cos(a), 0, sin(a)
            0, 1, 0,
            -sin(a), 0, cos(a)
        所以
            Rrefine2w[:, 0]
            = Rgrasp2w @ [cos(a), 0, -sin(a)].t
            = cos(a) * Rgrasp2w[:, 0] - sin(a) * Rgrasp2w[:, 2]
        所以
            Rrefine2w[:, 0] * axis_out_w
            = cos(a) * Rgrasp2w[:, 0] * axis_out_w - sin(a) * Rgrasp2w[:, 2] * axis_out_w
        对 a 求导数, 有:
            derivatives
            = -sin(a) * Rgrasp2w[:, 0] * axis_out_w - cos(a) * Rgrasp2w[:, 2] * axis_out_w
        另导数为 0, 有
            a1 = arctan( - (axis_out_w * Rgrasp2w[:, 2]) / (axis_out_w * Rgrasp2w[:, 0])   )
            a2 = arctan( + (axis_out_w * Rgrasp2w[:, 2]) / (axis_out_w * Rgrasp2w[:, 0])   )
        然后筛选出 a1 和 a2 中更合理的那个
        """
        def rotation_y(theta):
            rotation_matrix = np.array([
                [math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)]
            ])
            return rotation_matrix

        theta1 = np.arctan(-(axis_out_w * Rgrasp2w[:, 2]).sum() / (axis_out_w * Rgrasp2w[:, 0]).sum())
        theta2 = np.arctan(+(axis_out_w * Rgrasp2w[:, 2]).sum() / (axis_out_w * Rgrasp2w[:, 0]).sum())

        Rrefine2grasp1 = rotation_y(theta1)
        Rrefine2grasp2 = rotation_y(theta2)

        Rrefine2w1 = Rgrasp2w @ Rrefine2grasp1
        Rrefine2w2 = Rgrasp2w @ Rrefine2grasp2

        # 前面已有 Rrefine2w 是当前最优旋转
        R_cur = Rrefine2w1 if (Rrefine2w1[:, 0] * axis_out_w).sum() < (Rrefine2w2[:, 0] * axis_out_w).sum() else Rrefine2w2

        # 目标方向（负 axis_out_w）
        target_x = -axis_out_w / np.linalg.norm(axis_out_w)

        # 当前 x 轴
        cur_x = R_cur[:, 0]

        # 计算绕 z 轴的夹角 φ（叉积方向与 z 轴同向/反向决定符号）
        z_axis = R_cur[:, 2]
        sin_phi = np.cross(cur_x, target_x).dot(z_axis)
        cos_phi = cur_x.dot(target_x)
        phi = np.arctan2(sin_phi, cos_phi)

        # 绕 z 轴旋转 φ
        def rotation_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[ c, -s, 0],
                             [ s,  c, 0],
                             [ 0,  0, 1]])

        R_final = R_cur @ rotation_z(phi)

        # 写回 grasp
        grasp_.rotation_matrix = R_final
        return grasp_

    def move_along_axis(self, joint_type, joint_axis, joint_start, moving_distance, drop_large_move, reaction_motion=None):
        """
        控制 panda_hand 沿着某个轴移动一定距离, 或者绕着某个轴移动一定角度, 并保持 panda_hand 与物体的相对位姿保持不变
        joint_axis: 1) NOTE 在世界坐标系下!! 2) 满足右手定则, 沿着 joint_axis 的方向是打开
        """
        self.logger.log(logging.INFO, "Start move_along_axis() ...")
        assert joint_type in ["prismatic", "revolute"]
        if joint_type == "revolute":
            assert joint_start is not None, "joint_start cannot be None when joint_type is revolute"
            # 根据 moving distance 的大小计算出有多少个插值点
            num_interp = max(3, int(moving_distance / np.deg2rad(1)))
        else:
            num_interp = max(3, int(moving_distance / 0.02))

        self.logger.log(logging.INFO, f"move_along_axis(): Need {num_interp} interpolations to execute...")
        ee_pose, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first=False means quat in (x, y, z, w) order
        # Rph2w = R.from_quat(ee_quat, scalar_first=False).as_matrix() # 3, 3
        Rph2w = R.from_quat(ee_quat, scalar_first=False).as_matrix() # 3, 3

        if joint_type == "prismatic":
            def T_with_delta(delta):
                axis = joint_axis / np.linalg.norm(joint_axis)  # 确保轴是单位向量
                Tph2w = np.eye(4)
                Tph2w[:3, :3] = Rph2w
                # 对于 panda_hand 来说, z-axis 的正方向是向前
                Tph2w[:3, 3] = ee_pose + delta * axis
                return Tph2w

        elif joint_type == "revolute":
            def T_with_delta(delta):
                # 计算旋转矩阵，delta为旋转角度
                axis = joint_axis / np.linalg.norm(joint_axis)  # 确保轴是单位向量
                R_delta = R.from_rotvec(delta * axis).as_matrix()  # 计算旋转矩阵
                # 已知 Tphs2w, 要求 T
                Tph2w = np.eye(4)
                # Tph2w[:3, :3] = (R_delta @ Rph2w.T).T (这句代码是错误的!!)
                # NOTE: 注意 R_delta 是在世界坐标系下的一个旋转, 并不是两个坐标系之间的一个对应关系的旋转
                Tph2w[:3, :3] = R_delta @ Rph2w  # 先旋转再应用当前的旋转
                Tph2w[:3, 3] = R_delta @ (ee_pose - joint_start) + joint_start  # 保持末端执行器的位置不变
                return Tph2w

        # 先得到连续的插值位姿
        deltas = np.linspace(0, moving_distance, num_interp)
        T_list = [T_with_delta(delta) for delta in deltas]

        actually_moved = False
        # 然后依次执行这些位姿
        for Tph2w in T_list:
            result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if result is None:
                self.logger.log(logging.WARNING, "move_along_axis(): skip None planning result")
                continue
            elif len(result["time"]) == 0:
                self.logger.log(logging.WARNING, "move_along_axis(): skip len()=0 planning result")
                continue
            # 针对那种需要一个大的动作的操作, 直接不执行, 否则容易大幅度影响物体状态
            else:
                if len(result["time"]) > 300:
                    big_steps = len(result["time"])
                    self.logger.log(logging.WARNING, f"move_along_axis(): encounter large move ({big_steps} steps)")
                    if drop_large_move:
                        self.logger.log(logging.WARNING, f"move_along_axis(): drop large move")
                        continue
                self.follow_path(result, reaction_motion)
                actually_moved = True
        # NOTE: 这里需要判断到底是不是所有的 Tph2w 返回的 plan_result 都是 None, 如果那样不会有 RGBD frame 被录制, 需要额外跑一些 step
        if not actually_moved:
            for _ in range(int(1 / self.base_env_cfg["phy_timestep"])):
                self.step()

    def move_forward(self, moving_distance, drop_large_move=False, reaction_motion=None):
        _, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first=False means quat in (x, y, z, w) order
        Rph2w = R.from_quat(ee_quat, scalar_first=False).as_matrix() # 3, 3
        moving_direction = Rph2w @ np.array([0, 0, 1]) # 3
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=moving_direction,
            joint_start=None,
            moving_distance=moving_distance,
            drop_large_move=drop_large_move,
            reaction_motion=reaction_motion
        )

        # motion = CartesianMotion(Affine([0, 0, moving_distance]), ReferenceType.Relative)
        # self.franky_robot.move(motion)

    def get_ee_pose(self, as_matrix=False):
        """
        获取 end-effector (panda_hand) 的 ee_pos 和 ee_quat (Tph2w)
        """
        return self.robot.get_ee_pose(as_matrix=as_matrix)

    def anyGrasp2ph(self, grasp):
        """
            从 anygrasp 输出的 grasp 中提取出 Tph2w, 也就是做一个 Tgrasp2w 到 Tph2w 的转换
            grasp: Grasp 对象
        """
        # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
        def T_with_offset(offset):
            Tph2grasp = np.array([
                [0, 0, 1, offset],  
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
            return Tph2grasp

        R_grasp2w = grasp.rotation_matrix # 3, 3
        t_grasp2w = grasp.translation # 3
        Tgrasp2w = np.hstack((R_grasp2w, t_grasp2w[..., None])) # 3, 4
        Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
        # offset 越小, gripper 插入的越浅
        Tph2w = Tgrasp2w @ T_with_offset(offset=-0.12)
        return Tph2w

    def approach_safe(self, distance=0.1):
        # first_half = CartesianMotion(
        #     Affine([0, 0, distance * 0.]), 
        #     ReferenceType.Relative, 
        #     relative_dynamics_factor=0.5
        # )
        # self.robot.move(first_half)
        
        drawback_motion = CartesianMotion(
            Affine([0., 0., -0.01]),
            ReferenceType.Relative, 
            relative_dynamics_factor=0.1
        )
        second_half = CartesianMotion(
            Affine([0, 0, distance * 1]), 
            ReferenceType.Relative, 
            relative_dynamics_factor=0.1
        )
        # Tph2w = self.robot.get_ee_pose(as_matrix=True)
        # 取出 z 轴
        # dir_w = Tph2w[:3, 2]
        drawback_reaction = Reaction(self.get_force() > 5, drawback_motion)
        second_half.add_reaction(drawback_reaction)
        self.robot.move(second_half)
    
    def drawback_safe(self, distance=0.1, force=1):
        init_Tph2w = self.robot.get_ee_pose(as_matrix=True)
        
        drawback_motion = CartesianMotion(
            Affine([0., 0., -distance]),
            ReferenceType.Relative, 
            relative_dynamics_factor=0.1
        )
        drawback_reaction = Reaction(self.get_force() > force, CartesianStopMotion())
        drawback_motion.add_reaction(drawback_reaction)
        self.robot.move(drawback_motion)
        
        cur_Tph2w = self.robot.get_ee_pose(as_matrix=True)
        
        z_diff = init_Tph2w[2, -1] - cur_Tph2w[2, -1] # >0
        if abs(z_diff - distance) < 0.01:
            return
        else:
            self.open_gripper(0.06)
            self.move_dz(-abs(z_diff - distance))
        
    def approach(self, distance=0.1):
        second_half = CartesianMotion(
            Affine([0, 0, distance]), 
            ReferenceType.Relative, 
            relative_dynamics_factor=0.1
        )
        self.robot.move(second_half)
        
    def move_dx(self, distance=0.01):
        dx = distance
        dy = 0.0
        dz = 0.0
        motion = CartesianMotion(Affine([dx, dy, dz]), ReferenceType.Relative)
        self.robot.move(motion)
    
    def move_dy(self, distance=0.01):
        dx = 0.0
        dy = distance
        dz = 0.0
        motion = CartesianMotion(Affine([dx, dy, dz]), ReferenceType.Relative)
        self.robot.move(motion)
    
    def move_dz(self, distance=0.01):
        dx = 0.0
        dy = 0.0
        dz = distance
        motion = CartesianMotion(Affine([dx, dy, dz]), ReferenceType.Relative)
        self.robot.move(motion)
    
    def move_dxyz(self, distance=np.array([0, 0, 0])):
        motion = CartesianMotion(Affine(distance.tolist()), ReferenceType.Relative, relative_dynamics_factor=0.5)
        self.robot.move(motion)
            
    def rot_dx(self, deg=10):
        dx = np.deg2rad(deg)
        dy = 0.0
        dz = 0.0
        quat = R.from_euler("xyz", [dx, dy, dz]).as_quat()
        motion = CartesianMotion(Affine([0, 0, 0], quat), ReferenceType.Relative)
        self.robot.move(motion)
    
    def rot_dy(self, deg=10):
        dx = 0.0
        dy = np.deg2rad(deg)
        dz = 0.0
        quat = R.from_euler("xyz", [dx, dy, dz]).as_quat()
        motion = CartesianMotion(Affine([0, 0, 0], quat), ReferenceType.Relative)
        self.robot.move(motion)
    
    def rot_dz(self, deg=10):
        dx = 0.0
        dy = 0.0
        dz = np.deg2rad(deg)
        quat = R.from_euler("xyz", [dx, dy, dz]).as_quat()
        motion = CartesianMotion(Affine([0, 0, 0], quat), ReferenceType.Relative)
        self.robot.move(motion)
    
    def rot_dxyz(self, deg=np.array([0, 0, 0])):
        """
        要求输入的 quat 是 scalar_first=False 的
        """
        rad = np.deg2rad(deg)
        quat = R.from_euler("xyz", rad.tolist()).as_quat()
        motion = CartesianMotion(Affine([0, 0, 0], quat), ReferenceType.Relative, relative_dynamics_factor=0.5)
        self.robot.move(motion)
        
    def calibrate_reset(self, init_qpos=None):
        # 用 franky 对 robot 进行 reset, 无碰撞检测
        reset_motion = JointMotion(init_qpos[:7])
        self.robot.move(reset_motion)
        
    def delete(self):
        super().delete()
        self.pybullet_handle.disconnect()
        

if __name__ == "__main__":
    import yaml
    cfg_path = "/home/user/Programs/Embodied_Analogy/embodied_analogy/realworld_environment/calibration/test.yaml"

    # open
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = RobotEnv(cfg)
    env.reset_robot()
    env.close_gripper_safe(target=0.03, gripper_force=5)
    env.drawback_safe(distance=0.1, force=5)
    env.move_dz(0.1)
    
    env.reset_robot()
    f = env.capture_frame()
    Image.fromarray(f.rgb).show()
    env.move_dz(0.1)
    f = env.capture_frame()
    Image.fromarray(f.rgb).show()
    
    env.open_gripper(0.04)
    env.close_gripper_safe(target=-0.0, gripper_force=5)
    # env.capture_frame()
    # env.reset_robot_safe()
    # env.close_gripper()
    
    env.reset_robot()
    f = env.capture_frame()
    # Image.fromarray(f.robot_mask).show()
    # Image.fromarray(f.rgb).show()
    env.move_forward(0.1, False)
    # 实现一个功能, 就是先到达一个位置, 然后用多种不同的方式再到达这个位置
    # for i in range(100):
    env.move_along_axis(
        joint_type="prismatic",
        joint_axis=[1, 0, -1],
        joint_start=None,
        moving_distance=0.2,
        drop_large_move=False
    )
    env.move_along_axis(
        joint_type="revolute",
        joint_axis=[1, 0, 0],
        joint_start=[0, 0, 0.8],
        moving_distance=np.deg2rad(30),
        drop_large_move=False
    )
    Tph2w = env.get_ee_pose(as_matrix=True)
    goal_qpos_list = env.IK(Tph2w)

    for i, goal_qpos in enumerate(goal_qpos_list):
        plan_result = env.plan_qpos(goal_qpos)
        if plan_result is None:
            continue
        env.follow_path(plan_result)
        env.reset_robot()

    while True:
        env.step()
