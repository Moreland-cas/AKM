import time
import math
import mplib
import logging
import numpy as np
import pybullet as p
import pybullet_data
from franky import (
    Gripper,
    JointMotion
)
from crisp_py.robot import make_robot
from crisp_py.utils.geometry import Pose
import sapien.core as sapien
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R

from akm.utility.utils import clean_pc_np
from akm.realworld_envs.base_env import BaseEnv
    
    
class RobotEnv(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_robot()
        
        self.pybullet_handle = p
        self.pybullet_handle.connect(self.pybullet_handle.DIRECT)
        self.pybullet_handle.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pybullet_robot_id = self.pybullet_handle.loadURDF("franka_panda/panda.urdf") 
        
        self.reset_ee_pose = np.array([0.38938136, 0.09781701, 0.47887616])
        self.reset_qpos = np.array([-0.45536156, -0.43716924,  0.89536322, -2.38142135, -1.05704519,
        1.42962151, -0.31493183,  0.        ,  0.        ])
        
        # warmup camera
        for i in range(30):
            self.capture_frame(robot_mask=False)

    def capture_frame(self, visualize=False, robot_mask=True, Tph2w=True):
        frame = super().capture_frame(visualize=visualize)
        if Tph2w:
            frame.Tph2w = self.get_ee_pose(as_matrix=True)
        if robot_mask:
            frame.robot_mask = self.capture_robot_mask()
            
        return frame

    def load_robot(self):
        # load Robot
        self.franky_gripper = Gripper("172.16.0.2")
        
        self.crisp_robot = make_robot("fr3")
        self.crisp_robot.wait_until_ready()
        
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.mplib_robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/fr3.urdf")
        self.mplib_robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        self.arm_qlimit = self.mplib_robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        # Setup control properties
        self.active_joints = self.mplib_robot.get_active_joints()
        for joint in self.active_joints[:4]:
            joint.set_drive_property(stiffness=400, damping=40, force_limit=100)    # original: 160
        for joint in self.active_joints[4:-2]:
            joint.set_drive_property(stiffness=400, damping=40, force_limit=50)    # original: 160
        for joint in self.active_joints[-2:]:
            joint.set_drive_property(stiffness=160, damping=10, force_limit=50)

        self.setup_planner()

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
        qpos = self.get_qpos()
        
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

        # Calculate l, r, t, b in OpenGL frustum
        ll = -near * cx / fx
        tt = near * cy / fy
        rr = near * (W - cx) / fx
        bb = -near * (H - cy) / fy
        proj_matrix = self.pybullet_handle.computeProjectionMatrix(ll, rr, bb, tt, near, far)
            
        # Render segmentation image
        _, _, _, _, seg = self.pybullet_handle.getCameraImage(
            W, H, view_matrix, proj_matrix, 
            renderer=self.pybullet_handle.ER_BULLET_HARDWARE_OPENGL,
        )
        robot_mask = (seg == self.pybullet_robot_id)
        # Since the robot mask is not always accurate, it needs to be slightly inflated
        robot_mask = binary_dilation(robot_mask, iterations=3)
        return robot_mask        

    def move_to(self, Tph2w, speed=0.05):
        """
        crisp 的默认接口是 crisp 的 Pose, 这个函数负责进行一个转换
        """
        position = Tph2w[:3, -1]
        rotation = Tph2w[:3, :3]
        # NOTE crisp 定义的 ee 跟 franka Desk 不一样
        position += 0.1 * rotation[:, -1]
        tgt_pose = Pose(position=position, orientation=R.from_matrix(rotation))
        self.crisp_robot.move_to(pose=tgt_pose, speed=speed)

    def setup_planner(self):
        link_names = [link.get_name() for link in self.mplib_robot.get_links()]
        joint_names = [joint.get_name() for joint in self.mplib_robot.get_active_joints()]

        self.planner = mplib.Planner(
            urdf=self.asset_prefix + "/panda/fr3.urdf",
            srdf=self.asset_prefix + "/panda/panda_v3.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def anyGrasp2ph(self, grasp):
        """
        Extract Tph2w from grasp output by anygrasp, that is, perform a Tgrasp2w to Tph2w conversion.
        grasp: Grasp object
        """
        # Convert the grasp coordinate system to the panda_hand coordinate system, i.e. Tph2w
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
        # The larger the offset, the deeper the gripper is inserted.
        # Tph2w = Tgrasp2w @ T_with_offset(offset=-0.14)
        Tph2w = Tgrasp2w @ T_with_offset(offset=-0.1)
        return Tph2w
    
    def get_translated_ph(self, Tph2w, distance):
        """
        Outputs Tph2w: A ph pose that is a certain distance forward or backward along the current model. 
        A distance greater than 0 indicates forward movement, while a distance greater than 0 indicates backward movement.
        """
        Tph2w_ = np.copy(Tph2w)
        forward_direction = Tph2w[:3, :3] @ np.array([0, 0, 1]) # 3
        Tph2w_[:3, 3] = Tph2w[:3, 3] + distance * forward_direction
        return Tph2w_

    def get_rotated_grasp(self, grasp, axis_out_w):
        """
        Rotate the grasp coordinate system around the y-axis so that the dot product of the grasp coordinate system's x-axis and axis_out_w is as small as possible.
        grasp: anygrasp Grasp, Tgrasp2w
        axis_w: np.array, 3
        """
        from graspnetAPI import Grasp
        grasp_ = Grasp()
        grasp_.grasp_array = np.copy(grasp.grasp_array)

        Rgrasp2w = grasp_.rotation_matrix
        """
        Rrefine2w = Rgrasp2w @ Rrefine2grasp
        Rrefine2w[:, 0] * axis_out_w is as small as possible.
        Because Rrefine2grasp is a rotation about the y-axis, it has the form:
        cos(a), 0, sin(a)
        0, 1, 0,
        -sin(a), 0, cos(a)
        So
        Rrefine2w[:, 0]
        = Rgrasp2w @ [cos(a), 0, -sin(a)].t
        = cos(a) * Rgrasp2w[:, 0] - sin(a) * Rgrasp2w[:, 2]
        So
        Rrefine2w[:, 0] * axis_out_w
        = cos(a) * Rgrasp2w[:, 0] * axis_out_w - sin(a) * Rgrasp2w[:, 2] * axis_out_w
        Taking the derivative of a, we have:
        derivatives
        = -sin(a) * Rgrasp2w[:, 0] * axis_out_w - cos(a) * Rgrasp2w[:, 2] * axis_out_w
        If the derivative is 0, we have
        a1 = arctan( - (axis_out_w * Rgrasp2w[:, 2]) / (axis_out_w * Rgrasp2w[:, 0]) )
        a2 = arctan( + (axis_out_w * Rgrasp2w[:, 2]) / (axis_out_w * Rgrasp2w[:, 0]) )
        Then, select the more reasonable one between a1 and a2.
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

        # # Rrefine2w is the current optimal rotation.
        R_cur = Rrefine2w1 if (Rrefine2w1[:, 0] * axis_out_w).sum() < (Rrefine2w2[:, 0] * axis_out_w).sum() else Rrefine2w2
        target_x = -axis_out_w / np.linalg.norm(axis_out_w)

        # current x axis
        cur_x = R_cur[:, 0]

        # Calculates the angle φ about the z-axis (the cross product direction is in the same direction as or opposite to the z-axis, which determines the sign)
        z_axis = R_cur[:, 2]
        sin_phi = np.cross(cur_x, target_x).dot(z_axis)
        cos_phi = cur_x.dot(target_x)
        phi = np.arctan2(sin_phi, cos_phi)

        # Rotate φ around the z-axis
        def rotation_z(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[ c, -s, 0],
                             [ s,  c, 0],
                             [ 0,  0, 1]])

        R_final = R_cur @ rotation_z(phi)
        grasp_.rotation_matrix = R_final
        return grasp_

    def open_gripper(self, target=0.06, speed=0.02):
        self.franky_gripper.move(width=target, speed=speed)

    def close_gripper(self, target=0., gripper_force=10, speed=0.02):
        if target is None:
            target = 0.
        self.franky_gripper.grasp(target, speed, gripper_force, epsilon_outer=1.0)

    def close_gripper_safe(self):
        """
        用 impedance control 关闭 gripper
        """
        # change mode to soft impedance
        self.switch_mode(mode="grasp")
        # self.franky_gripper.move(width=0, speed=0.01)
        self.close_gripper(target=0., gripper_force=20, speed=0.01)
        
        # 此时确定了位置
        cur_Tph2w = self.get_ee_pose(as_matrix=True)
        self.franky_gripper.move(width=0.03, speed=0.02)
        
        self.switch_mode(mode="cartesian_impedance")
        self.move_to(cur_Tph2w)
        self.close_gripper(target=0., gripper_force=20, speed=0.01)
    
    def clear_planner_pc(self):
        self.update_point_cloud_with_wall(pc_w=None)

    def plan_path(self, target_pose: np.ndarray, wrt_world: bool = True):
        """
        NOTE: Return the target_pose with the smallest result (rotating 180 degrees around the z-axis is another good move).
        The target_pose passed in is Tph2w
        """
        target_pose1 = np.copy(target_pose)
        target_pose2 = np.copy(target_pose)
        target_pose2[:, 0] *= -1
        target_pose2[:, 1] *= -1
        
        result1, result2 = None, None
        try:
            result1 = self.planner.plan_pose(
                goal_pose=mplib.Pose(target_pose1),
                current_qpos=self.get_qpos(),
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
                current_qpos=self.get_qpos(),
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
                return result1, target_pose1
            else:
                # Compare the two results and return the one with the shorter planned path.
                n_step1 = result1['position'].shape[0]
                n_step2 = result2['position'].shape[0]
                if n_step1 < n_step2:
                    return result1, target_pose1
                else:
                    return result2, target_pose2
        else:
            if result_valid(result2):
                return result2, target_pose2
            return None, None

    def get_qpos(self):
        """
        get np.array([9]) qpos of current robot actual state
        NOTE: This can also be used to generate robot mask
        """
        gripper_width = self.franky_gripper.width
        gripper_qpos = gripper_width / 2.
        qpos_7 = self.crisp_robot.joint_values
        qpos_9 = np.concatenate([qpos_7, [gripper_qpos, gripper_qpos]])
        return qpos_9
    
    def get_ee_pose(self, as_matrix=False):
        """
        Get the ee_pos and ee_quat (Tph2w) of the end-effector (panda_hand)
        ee_pos: np.array([3, ])
        ee_quat: np.array([4, ]), scalar_first=False, in (x, y, z, w) order
        """
        # Get the robot's cartesian state
        ee_pose = self.crisp_robot.end_effector_pose
        ee_pos = ee_pose.position
        ee_Rotation = ee_pose.orientation
        z_dir = R.as_matrix(ee_Rotation)[:, -1]
        
        # NOTE: crisp 返回的 ee_pose 跟 franka Desk 中定义的差 1dm
        ee_pos = ee_pos - 0.1 * z_dir
        
        if as_matrix:
            Tph2w = np.eye(4)
            Tph2w[:3, :3] = ee_Rotation.as_matrix()
            Tph2w[:3, 3] = ee_pos
            return Tph2w
        else:
            # scalar_last
            return ee_pos, ee_Rotation.as_quat(scalar_first=False) 
        
    def switch_mode(self, mode="cartesian_impedance"):
        assert mode in ["grasp", "pull", "approach", "joint_impedance", "cartesian_impedance"]
        if mode == "grasp":
            self.crisp_robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.crisp_robot.cartesian_controller_parameters_client.load_param_config(
                file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/grasp_cartesian_impedance.yaml"
            )
        elif mode == "pull":
            self.crisp_robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.crisp_robot.cartesian_controller_parameters_client.load_param_config(
                file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/pull_cartesian_impedance.yaml"
            )
        elif mode == "approach":
            self.crisp_robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.crisp_robot.cartesian_controller_parameters_client.load_param_config(
                file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/approach_cartesian_impedance.yaml"
            )
        elif mode == "joint_impedance":
            self.crisp_robot.controller_switcher_client.switch_controller("joint_impedance_controller")
        elif mode == "cartesian_impedance":
            self.crisp_robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            self.crisp_robot.cartesian_controller_parameters_client.load_param_config(
                file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/default_cartesian_impedance.yaml"
            )
        
    def follow_path(self, result):
        """
        Follow the planning result
        """
        n_step = result['position'].shape[0]
        self.logger.log(logging.INFO, f"n_step: {n_step}")

        if n_step == 0:
            return
        
        self.switch_mode("joint_impedance")
        for i in range(n_step):
            joint_target = result['position'][i].tolist() # 7
            self.crisp_robot.set_target_joint(np.array(joint_target))
            # position, quat = self.fk(joint_target)
            # self.crisp_robot.move_to(pose=Pose(position=position, orientation=R.from_quat(quat)))
            time.sleep(1)
            
            
    def update_point_cloud_with_wall(self, pc_w=None):
        """
        Not only does it update the object point cloud pc_w in the world coordinate system, 
        it also updates the indoor walls and the cylinder where the camera is located, as well as the desktop information
        """
        pc_update = []
        
        if pc_w is not None:
            pc_update.append(pc_w)
        
        # add wall, (1000, 3)
        pc_wall_yz = np.random.uniform(-1, 1, (1000, 2))
        pc_wall_x = np.ones((1000, 1)) * -0.3
        pc_wall = np.concatenate([pc_wall_x, pc_wall_yz], axis=1)
        pc_update.append(pc_wall)
        
        # add ground
        pc_ground_xy = np.random.uniform(-1, 1, (1000, 2))
        pc_ground_z = np.ones((1000, 1)) * 0.01
        pc_ground = np.concatenate([pc_ground_xy, pc_ground_z], axis=1)
        
        # Filter out points with a distance < 0.15 from the origin
        dist2 = np.sum(pc_ground ** 2, axis=1) ** 0.5     
        pc_ground = pc_ground[dist2 >= 0.2]
        pc_update.append(pc_ground)
        
        # Add the cylinder where the camera is located, here we use a plane instead
        pc_camera_yz = np.random.uniform(-1, 1, (1000, 2))
        pc_camera_x = np.ones((1000, 1)) * 0.8
        pc_camera = np.concatenate([pc_camera_x, pc_camera_yz], axis=1)
        pc_update.append(pc_camera)
        
        self.planner.update_point_cloud(
            points=np.concatenate(pc_update, axis=0)
        )
    
    def move_to_pregrasp(self, Tph2w, reserved_distance=0.05):
        """
        Control panda_hand to move to the target position of Tph2w. 
        The difference is that the environmental point cloud and reserved_distance must be considered during the movement.
        """
        frame = self.capture_frame()
        pc_w, _ = frame.get_env_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_w)

        Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
        result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)

        if result_pre is None:
            self.logger.log(logging.WARNING, "move_to_pose_safe(): planning to pre_grasp_pose failed")
            return None
        else:
            self.switch_mode("joint_impedance")
            self.follow_path(result_pre)
            # Get robot qpos here, which is the pre_qpos corresponding to pre_Tph2w
            pre_qpos = self.get_qpos()
            # Remove the collision pc before moving forward
            self.clear_planner_pc()
            self.switch_mode("cartesian_impedance")
            self.move_dz(distance=reserved_distance, soeed=0.01)
            return pre_qpos

    def move_dz(self, distance=0.05, speed=0.02):
        # 首先获取当前坐标
        cur_position = self.crisp_robot.end_effector_pose.position
        # print(cur_position)
        cur_Rotation = self.crisp_robot.end_effector_pose.orientation
        z_dir = cur_Rotation.as_matrix()[:, -1]
        # print(z_dir)
        tgt_position = cur_position + distance * z_dir
        # print(tgt_position)
        self.crisp_robot.move_to(position=tgt_position, speed=speed)
    
    def approach(self, distance, speed=0.02):
        self.switch_mode("approach")
        self.move_dz(distance=distance, speed=speed)
        
        # 重置弹簧阻力
        cur_Tph2w = self.get_ee_pose(as_matrix=True)
        self.switch_mode(mode="cartesian_impedance")
        self.move_to(cur_Tph2w)
        
    def move_along_axis(self, joint_type, joint_axis, joint_start, moving_distance):
        """
        Controls the panda_hand to move a certain distance along an axis or a certain angle around an axis, 
        while maintaining the relative position of the panda_hand and the object.
        
        joint_axis: 
            1) NOTE: In world coordinates!! 
            2) Satisfying the right-hand rule, the direction along the joint_axis is open.
            
        NOTE: This method needs cartesian control
        """
        self.logger.log(logging.INFO, "Start move_along_axis() ...")
        # switch to cartesian mode
        assert joint_type in ["prismatic", "revolute"]
        if joint_type == "revolute":
            assert joint_start is not None, "joint_start cannot be None when joint_type is revolute"
            # Calculate the number of interpolation points based on the moving distance
            num_interp = max(3, int(moving_distance / np.deg2rad(1)))
        else:
            num_interp = max(3, int(moving_distance / 0.02))
        
        # num_interp = 50

        self.logger.log(logging.INFO, f"move_along_axis(): Need {num_interp} interpolations to execute...")
        Tph2w = self.get_ee_pose(as_matrix=True) # Tph2w
        ee_pose = Tph2w[:3, -1]
        Rph2w = Tph2w[:3, :3]
        # scalar_first=False means quat in (x, y, z, w) order
        # Rph2w = R.from_quat(ee_quat, scalar_first=False).as_matrix() # 3, 3

        if joint_type == "prismatic":
            def T_with_delta(delta):
                axis = joint_axis / np.linalg.norm(joint_axis) 
                Tph2w = np.eye(4)
                Tph2w[:3, :3] = Rph2w
                # For panda_hand, the positive direction of z-axis is forward
                Tph2w[:3, 3] = ee_pose + delta * axis
                return Tph2w

        elif joint_type == "revolute":
            def T_with_delta(delta):
                # Calculate the rotation matrix, delta is the rotation angle
                axis = joint_axis / np.linalg.norm(joint_axis)  
                R_delta = R.from_rotvec(delta * axis).as_matrix()
                Tph2w = np.eye(4)
                # NOTE: Note that R_delta is a rotation in the world coordinate system, not a corresponding rotation between the two coordinate systems.
                Tph2w[:3, :3] = R_delta @ Rph2w  
                Tph2w[:3, 3] = R_delta @ (ee_pose - joint_start) + joint_start 
                return Tph2w

        deltas = np.linspace(0, moving_distance, num_interp)
        T_list = [T_with_delta(delta) for delta in deltas]

        for Tph2w in T_list:
            self.move_to(Tph2w)
    
    def move_dxyz(self, tgt_t, speed):
        cur_pose= self.crisp_robot.end_effector_pose
        cur_position = cur_pose.position
        tgt_position = cur_position + tgt_t
        self.switch_mode("cartesian_impedance")
        self.crisp_robot.move_to(position=tgt_position, speed=speed)
    
    def rot_dxyz(self, tgt_r, speed):
        cur_pose= self.crisp_robot.end_effector_pose
        cur_orientation = cur_pose.orientation
        tgt_r = R.from_euler('xyz', tgt_r, degrees=True)
        tgt_orientation = tgt_r * cur_orientation
        self.switch_mode("cartesian_impedance")
        tgt_pose = Pose(position=cur_pose.position, orientation=tgt_orientation)
        self.crisp_robot.move_to(pose=tgt_pose, speed=speed)
    
    def reset_safe(self, distance=-0.05):
        self.open_gripper(target=0.06)
        self.switch_mode("cartesian_impedance")
        self.move_dz(distance=distance, speed=0.02)
        self.crisp_robot.home()
    
    def reset(self):
        self.crisp_robot.home()
        
    def calibrate_reset(self, init_qpos=None):
        self.switch_mode("joint_impedance")
        self.crisp_robot.set_target_joint(init_qpos[:7])
        
    def delete(self):
        super().delete()
        self.pybullet_handle.disconnect()
        self.crisp_robot.shutdown()
        

if __name__ == "__main__":
    """
    总结一下需要什么功能
    1) move to pre-grasp
    2) approach (降低碰撞的刚度)
    3) safe-grasp (降低 y 方向刚度然后关闭 gripper, 达到稳定状态之后更新 Tph2w, 重新 move_to 一下)
    4) move_along_axis (只能稍微有一点 impedance)
    5) draw_back
    
    解耦一下能力：
    1) 切换 impedance 方式，包括切换到 joint control
    2) 沿着 z 轴的运动
    """
    
    import yaml
    with open("/home/user/Programs/AKM/cfgs/realworld_cfgs/cabinet.yaml") as f:
        cfg = yaml.safe_load(f)
    re = RobotEnv(cfg=cfg)
    print("done")
    
    re.crisp_robot.home()
    re.switch_mode("cartesian_impedance")
    re.move_dz(distance=0.05, speed=0.03)
    re.move_along_axis(
        joint_type="revolute",
        joint_axis=np.array([1, 0 ,0]),
        joint_start=np.array([0, 0, 0]),
        moving_distance=np.deg2rad(-15)
    )

    re.crisp_robot.home()
    re.delete()