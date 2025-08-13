import math
import mplib
import logging
import numpy as np
import pybullet as p
import pybullet_data
from franky import *
from PIL import Image
import sapien.core as sapien
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R

from akm.utility.utils import clean_pc_np
from akm.realworld_envs.base_env import BaseEnv


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
        mplib requires an array of size (9,), but self.franky_robot directly returns an array of dimension (7,). This needs to be expanded.
        return realworld robot qpos, np.array([9])
        """
        gripper_width = self.franky_gripper.width
        gripper_qpos = gripper_width / 2.
        qpos_7 = self.franky_robot.current_joint_state.position
        qpos_9 = np.concatenate([qpos_7, [gripper_qpos, gripper_qpos]])
        return qpos_9

    def move(self, motion):
        self.franky_robot.move(motion)

    def get_ee_pose(self, as_matrix=False):
        """
        Get the ee_pos and ee_quat (Tph2w) of the end-effector (panda_hand)
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
            T = np.eye(4) 
            # NOTE For sapien, the quat returned by get_ee_pose is w-first, but for franky it is w_last
            T[:3, :3] = R_franky
            T[:3, 3] = ee_pos  
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
        self.pybullet_robot_id = self.pybullet_handle.loadURDF("franka_panda/panda.urdf") 
        
        self.reset_ee_pose = np.array([0.38938136, 0.09781701, 0.47887616])
        self.reset_qpos = np.array([-0.45536156, -0.43716924,  0.89536322, -2.38142135, -1.05704519,
        1.42962151, -0.31493183,  0.        ,  0.        ])

    def capture_frame(self, visualize=False, robot_mask=True):
        frame = super().capture_frame(visualize=visualize)
        if robot_mask:
            frame.robot_mask = self.capture_robot_mask()
            frame.Tph2w = self.get_ee_pose(as_matrix=True)
        return frame

    def load_robot(self):
        # load Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        self.mplib_robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/fr3.urdf")
        # Replace this with your robot's IP
        self.franky_robot = Robot("172.16.0.2")  
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

        # Start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
        self.franky_robot.relative_dynamics_factor = 0.05
        self.franky_robot.recover_from_errors()
        self.gripper_speed = 0.04 # m/s
        self.gripper_force = 5 # N

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

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        self.planner = mplib.Planner(
            urdf=self.asset_prefix + "/panda/fr3.urdf",
            srdf=self.asset_prefix + "/panda/panda_v3.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
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
    
    def close_gripper_safe(self, target=0.02, gripper_force=5):
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

    def get_force(self, dir_w=None):
        """
        Get the exernal force in the dir_w direction, if None, get all
        """
        if dir_w is None:
            normal_force = (Measure.FORCE_X ** 2 + Measure.FORCE_Y ** 2 + Measure.FORCE_Z ** 2) ** 0.5
        else:
            dir_w = dir_w / np.linalg.norm(dir_w)
            normal_force = (Measure.FORCE_X * dir_w[0] + Measure.FORCE_Y * dir_w[1] + Measure.FORCE_Z * dir_w[2]) **2 **0.5
        return normal_force
    
    def reset_robot(self):
        self.logger.log(logging.INFO, "Reset robot ...")
        self.open_gripper()
        
        cur_ee_pose = self.robot.get_ee_pose(as_matrix=False)[0]
        if np.linalg.norm(self.reset_ee_pose - cur_ee_pose) > 0.2: 
            # When the distance is far, retreat 1dm first
            self.move_forward(
                moving_distance=-0.1,
                drop_large_move=False
            )
        joint_motion = JointMotion(self.reset_qpos[:7])
        self.franky_robot.move(joint_motion)
        
    def reset_robot_safe(self):
        self.logger.log(logging.INFO,  "Call safe robot reset ...")
        self.open_gripper()
        self.move_forward(
            moving_distance=-0.1,
            drop_large_move=False
        )
        
        tmp_frame = self.capture_frame()
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
                # Compare the two results and return the one with the shorter planned path.
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
        Call the plan_qpos method of self.planner, input a goal_qpos, and return a plan_result
        goal_qpos: np.ndarray, shape=(7,), dtype=np.float32
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
        n_step = result['position'].shape[0]
        self.logger.log(logging.INFO, f"n_step: {n_step}")

        if n_step == 0:
            return
        
        joint_waypoint_motion_list = []
        for i in range(n_step):
            position_target = result['position'][i].tolist() # 7
            joint_waypoint = JointWaypoint(position_target)
            joint_waypoint_motion_list.append(joint_waypoint)
        joint_waypoint_motion = JointWaypointMotion(joint_waypoint_motion_list)
        
        if reaction_motion is not None: #TODO 添加 default的
            joint_waypoint_motion.add_reaction(reaction_motion)
        self.franky_robot.move(joint_waypoint_motion)

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
        dist2 = np.sum(pc_ground ** 2, axis=1) ** 0.5      # 平方距离
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
    
    def move_to_pose(self, pose: mplib.pymp.Pose, wrt_world: bool):
        # pose: Tph2w
        result = self.plan_path(target_pose=pose, wrt_world=wrt_world)
        if result is not None:
            self.follow_path(result)
        else:
            self.logger.log(logging.WARNING, "Get None result in move_to_pose function, not executing...")

    def move_to_pose_safe(self, Tph2w, reserved_distance=0.05):
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
            self.follow_path(result_pre)
            # Get robot qpos here, which is the pre_qpos corresponding to pre_Tph2w
            pre_qpos = self.robot.get_qpos()
            # Remove the collision pc before moving forward
            self.clear_planner_pc()
            self.move_forward(
                moving_distance=reserved_distance,
                drop_large_move=False
            )
            return pre_qpos

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

    def move_along_axis(self, joint_type, joint_axis, joint_start, moving_distance, drop_large_move, reaction_motion=None):
        """
        Controls the panda_hand to move a certain distance along an axis or a certain angle around an axis, 
        while maintaining the relative position of the panda_hand and the object.
        
        joint_axis: 
            1) NOTE: In world coordinates!! 
            2) Satisfying the right-hand rule, the direction along the joint_axis is open.
        """
        self.logger.log(logging.INFO, "Start move_along_axis() ...")
        assert joint_type in ["prismatic", "revolute"]
        if joint_type == "revolute":
            assert joint_start is not None, "joint_start cannot be None when joint_type is revolute"
            # Calculate the number of interpolation points based on the moving distance
            num_interp = max(3, int(moving_distance / np.deg2rad(1)))
        else:
            num_interp = max(3, int(moving_distance / 0.02))

        self.logger.log(logging.INFO, f"move_along_axis(): Need {num_interp} interpolations to execute...")
        ee_pose, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first=False means quat in (x, y, z, w) order
        Rph2w = R.from_quat(ee_quat, scalar_first=False).as_matrix() # 3, 3

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

        actually_moved = False
        for Tph2w in T_list:
            result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if result is None:
                self.logger.log(logging.WARNING, "move_along_axis(): skip None planning result")
                continue
            elif len(result["time"]) == 0:
                self.logger.log(logging.WARNING, "move_along_axis(): skip len()=0 planning result")
                continue
            # For operations that require a large action, do not execute them directly, otherwise it will easily affect the state of the object.
            else:
                if len(result["time"]) > 300:
                    big_steps = len(result["time"])
                    self.logger.log(logging.WARNING, f"move_along_axis(): encounter large move ({big_steps} steps)")
                    if drop_large_move:
                        self.logger.log(logging.WARNING, f"move_along_axis(): drop large move")
                        continue
                self.follow_path(result, reaction_motion)
                actually_moved = True
        # NOTE: Here you need to check whether all plan_results returned by Tph2w are None. 
        # If so, no RGBD frame will be recorded and you need to run some additional steps.
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

    def get_ee_pose(self, as_matrix=False):
        """
        Get ee_pos and ee_quat (Tph2w) of the end-effector (panda_hand)
        """
        return self.robot.get_ee_pose(as_matrix=as_matrix)

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
        # The smaller the offset, the shallower the gripper is inserted.
        Tph2w = Tgrasp2w @ T_with_offset(offset=-0.12)
        return Tph2w

    def approach_safe(self, distance=0.1):
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
        Requires the input quat to have scalar_first=False
        """
        rad = np.deg2rad(deg)
        quat = R.from_euler("xyz", rad.tolist()).as_quat()
        motion = CartesianMotion(Affine([0, 0, 0], quat), ReferenceType.Relative, relative_dynamics_factor=0.5)
        self.robot.move(motion)
        
    def calibrate_reset(self, init_qpos=None):
        reset_motion = JointMotion(init_qpos[:7])
        self.robot.move(reset_motion)
        
    def delete(self):
        super().delete()
        self.pybullet_handle.disconnect()
        