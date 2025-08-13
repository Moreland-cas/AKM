import math
import mplib
import logging
import numpy as np
import transforms3d as t3d
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

from akm.utility.utils import world_to_image
from akm.simulated_envs.base_env import BaseEnv
from akm.representation.basic_structure import Frame
from akm.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)


class RobotEnv(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_robot()

    def capture_frame(self, visualize=False) -> Frame:
        frame = super().capture_frame(visualize=False)

        frame.robot_mask = self.capture_robot_mask()
        frame.robot2d=self.get_points_on_arm()[0]
        frame.Tph2w = self.get_ee_pose(as_matrix=True)

        if visualize:
            frame.visualize()
        return frame

    def load_robot(self):
        # Robot config
        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )
        config = parse_urdf_config(urdf_config, self.scene)
        check_urdf_config(config)

        # load Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/panda_v3.urdf", config)

        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        self.arm_qlimit = self.robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        # Setup control properties
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints[:4]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=100)    # original: 200
        for joint in self.active_joints[4:-2]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=50)    # original: 200
        for joint in self.active_joints[-2:]:
            joint.set_drive_property(stiffness=160, damping=10, force_limit=50)

        # Set initial joint positions
        self.setup_planner()
        self.reset_robot()

        # set id for getting mask
        for link in self.robot.get_links():
            for s in link.get_visual_bodies():
                s.set_visual_id(255)

    def capture_robot_mask(self):
        camera = self.camera
        camera.take_picture()
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
        mesh_np = seg_labels[..., 0].astype(np.uint8)  # mesh-level [H, W]
        return mesh_np == 255

    def load_panda_hand(self, scale=1., pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        self.asset = loader.load(self.asset_prefix + f"/panda/panda_v2_gripper.urdf")
        self.asset.set_root_pose(sapien.Pose(pos, quat))
        return self.asset

    def get_points_on_arm(self):
        # Get the 2D and 3D coordinates of some points on the robot arm (currently link_pose)
        link_poses_3d = []
        for link in self.robot.get_links():
            link_pos = link.get_pose().p # np.array(3)
            link_poses_3d.append(link_pos)
        link_poses_3d = np.array(link_poses_3d) # N, 3

        # Projection onto the 2d camera plane
        link_poses_2d = world_to_image(
            link_poses_3d,
            self.camera_intrinsic,
            self.camera_extrinsic
        ) # N, 2
        link_poses_3d_w = link_poses_3d
        return link_poses_2d, link_poses_3d_w

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
            urdf=self.asset_prefix + "/panda/panda_v3.urdf",
            srdf=self.asset_prefix + "/panda/panda_v3.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))
        pass

    def open_gripper(self, target=0.02):
        """
        Here, self.step() is used to record the open/close action in the record. 
        At that time, self.step actually corresponds to the self.record_step function.
        """
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(target)

        # NOTE: Same as self.step in reset, see detailed explanation there
        self.step()
        count = 0
        while count < 100:
            vel_norm = np.linalg.norm(self.robot.get_qvel())
            if vel_norm < 1e-3:
                break
            self.step()
            count += 1

    def close_gripper(self, target=0.):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(target)

        # NOTE: Same as self.step in reset, see detailed explanation there
        self.step()

        count = 0
        while count < 100:
            vel_norm = np.linalg.norm(self.robot.get_qvel())
            if vel_norm < 1e-3:
                break
            self.step()
            count += 1

    def reset_robot(self):
        """
        reset_robot does not control the open/close state of the gripper, but only resets other joints
        """
        self.logger.log(logging.INFO, "Reset robot ...")
        init_panda_hand = mplib.Pose(p=[-0.5, 0, 1], q=t3d.euler.euler2quat(np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), axes="syxz"))
        
        self.move_to_pose(pose=init_panda_hand, wrt_world=True)

        # NOTE: A step is required here because the difference between cur_qpos and target_qpos is used to determine whether to terminate synchronization.
        # However, after calling self.robot.set_qpos() , cur_qpos is equal to target_qpos during the first step, and only becomes the actual value later.
        # Therefore, not performing a step will cause synchronization to fail and the process will exit immediately.
        self.base_step()

        count = 0
        while count < 400:
            vel = self.robot.get_qvel()
            # Here, vel_norm is used as the reset synchronization termination condition because vel is more stable than qpos and qacc.
            # qpos is not known what value to set, and qacc often changes suddenly.
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 2e-3: 
                self.logger.log(logging.INFO,  f"Break reset since small robot vel norm: {vel_norm}")
                break
            self.base_step()
            count += 1

        if count == 400:
            self.logger.log(logging.INFO,  "Break reset since reach max reset count")

    def reset_robot_safe(self):
        self.logger.log(logging.INFO,  "Call safe robot reset ...")
        self.open_gripper()

        self.base_step()
        self.planner.update_point_cloud(
            self.capture_frame().get_env_pc(
                use_robot_mask=True,
                use_height_filter=False,
                world_frame=True
            )[0]
        )

        self.move_forward(
            moving_distance=-self.cfg["explore_env_cfg"]["reserved_distance"],
            drop_large_move=False
        ) 

        self.base_step()
        self.planner.update_point_cloud(
            self.capture_frame().get_env_pc(
                use_robot_mask=True,
                use_height_filter=False,
                world_frame=True
            )[0]
        )

        self.reset_robot()

    def clear_planner_pc(self):
        self.planner.update_point_cloud(np.array([[0, 0, -1]]))

    def plan_qpos(self, goal_qpos):
        """
        Call the plan_qpos method of self.planner , taking a goal_qpos as input and returning a plan_result.
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
        The difference is that mplib.planner 's plan_qpos method outputs multiple goal_qpos, while returning a single plan_result.

        We simply input a single goal_qpos and switch to another if it doesn't work, because we want multiple plan_results, each with a different
        qpos but the same ee_pos.
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
    
    def IK(self, target_pose, wrt_world: bool = True, n_init_qpos=20):
        """
        Given an end-effector pose, return a feasible robot qpos.
        NOTE: The returned value is a list, and mplib itself already has some nms support.
        """
        if wrt_world:
            goal_pose = self.planner._transform_goal_to_wrt_base(target_pose)
        
        # If there is more than one list of goal_qpos, ik_status is "Success"
        ik_status, goal_qpos = self.planner.IK(
            goal_pose=goal_pose,
            start_qpos=self.robot.get_qpos(),
            n_init_qpos=n_init_qpos
        )
        if ik_status != "Success":
            return None
        return goal_qpos
    
    def plan_path(self, target_pose, wrt_world: bool = True):
        # The target_pose passed in is Tph2w
        if isinstance(target_pose, np.ndarray):
            target_pose = mplib.Pose(target_pose)
        try:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=self.robot.get_qpos(),
                time_step=self.planner_timestep,
                rrt_range=0.1,
                planning_time=1,
                wrt_world=wrt_world
            )
        except Exception as e:
            self.logger.log(logging.WARNING, f"Encounter {e} during RobotEnv.plan_path()")
            return None

        if result['status'] != "Success":
            return None

        return result

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        self.logger.log(logging.INFO, f"n_step: {n_step}")
        for i in range(n_step):
            position_target = result['position'][i]
            velocity_target = result['velocity'][i]
            # num_repeat needs to be calculated based on the time_step used when mplib.planner is initialized.
            # num_repeat = int(self.time_step / self.phy_timestep)
            num_repeat = math.ceil(self.planner_timestep / self.phy_timestep) 
            for _ in range(num_repeat):
                qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False)
                self.robot.set_qf(qf)

                for j in range(7):
                    self.active_joints[j].set_drive_target(position_target[j])
                    self.active_joints[j].set_drive_velocity_target(velocity_target[j])
                self.step()

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
        self.base_step()
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
            pre_qpos = self.robot.get_qpos()
            self.clear_planner_pc()
            self.move_forward(
                moving_distance=reserved_distance,
                drop_large_move=False
            )
            return pre_qpos

    def move_to_qpos_safe(self, qpos):
        """
        Controls Franka's state to qpos without colliding with environmental objects.
        The difference is that the ambient point cloud and reserved_distance must be considered during the movement.
        NOTE: The input qpos for calling this function is usually pre_qpos.
        """
        self.base_step()
        frame = self.capture_frame()
        pc_w, _ = frame.get_env_pc(
            use_robot_mask=True,
            use_height_filter=False,
            world_frame=True
        )
        self.planner.update_point_cloud(pc_w)

        plan_result = self.plan_qpos(goal_qpos=qpos)

        if plan_result is None:
            self.logger.log(logging.WARNING, "move_to_qpos_safe(): planning to goal_qpos failed")
        else:
            self.follow_path(plan_result)
        
    def get_translated_ph(self, Tph2w, distance):
        """
        Outputs Tph2w: A ph pose that is a certain distance forward or backward along the current model.
        If distance is greater than 0, it indicates forward movement; otherwise, it indicates backward movement.
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

        if (Rrefine2w1[:, 0] * axis_out_w).sum() < (Rrefine2w2[:, 0] * axis_out_w).sum():
            grasp_.rotation_matrix = Rrefine2w1.reshape(-1)
        else:
            grasp_.rotation_matrix = Rrefine2w2.reshape(-1)

        return grasp_

    def move_along_axis(self, joint_type, joint_axis, joint_start, moving_distance, drop_large_move):
        """
        Controls the panda_hand to move a certain distance along an axis, or a certain angle around an axis, 
        while maintaining the relative position of the panda_hand and the object.
        joint_axis: 1) NOTE: In world coordinates!! 2) Satisfying the right-hand rule, directions along the joint_axis are open.
        """
        self.logger.log(logging.INFO, "Start move_along_axis() ...")
        assert joint_type in ["prismatic", "revolute"]
        if joint_type == "revolute":
            assert joint_start is not None, "joint_start cannot be None when joint_type is revolute"
            # Calculate how many interpolation points there are based on the size of the moving distance
            num_interp = max(3, int(moving_distance / np.deg2rad(5)))
        else:
            num_interp = max(3, int(moving_distance / 0.02))

        self.logger.log(logging.INFO, f"move_along_axis(): Need {num_interp} interpolations to execute...")
        ee_pose, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3

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
                axis = joint_axis / np.linalg.norm(joint_axis)  
                R_delta = R.from_rotvec(delta * axis).as_matrix()  
                Tph2w = np.eye(4)
                # NOTE: Note that R_delta is a rotation in the world coordinate system, not a corresponding rotation between the two coordinate systems.
                Tph2w[:3, :3] = R_delta @ Rph2w  
                Tph2w[:3, 3] = R_delta @ (ee_pose - joint_start) + joint_start  
                return Tph2w

        # First get the continuous interpolation pose
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
                self.follow_path(result)
                actually_moved = True
        # NOTE: Here you need to check whether all plan_results returned by Tph2w are None. 
        # If so, no RGBD frame will be recorded and you need to run some additional steps.
        if not actually_moved:
            for _ in range(int(1 / self.base_env_cfg["phy_timestep"])):
                self.step()
                
    def move_forward(self, moving_distance, drop_large_move):
        _, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3
        moving_direction = Rph2w @ np.array([0, 0, 1]) # 3
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=moving_direction,
            joint_start=None,
            moving_distance=moving_distance,
            drop_large_move=drop_large_move
        )

    def get_ee_pose(self, as_matrix=False):
        """
        Get ee_pos and ee_quat (Tph2w) of the end-effector (panda_hand)
        ee_pos: np.array([3, ])
        ee_quat: np.array([4, ]), scalar_first=True, in (w, x, y, z) order
        """
        ee_link = self.robot.get_links()[9] # panda_hand
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q

        if as_matrix:
            T = np.eye(4)  
            R_matrix = R.from_quat(ee_quat, scalar_first=True).as_matrix()
            T[:3, :3] = R_matrix
            T[:3, 3] = ee_pos  
            return T
        else:
            return ee_pos, ee_quat 

    def anyGrasp2ph(self, grasp):
        """
        Extract Tph2w from grasp output by anygrasp, that is, perform a Tgrasp2w to Tph2w conversion.
        grasp: Grasp object
        """
        # Convert the grasp coordinate system to the panda_hand coordinate system, i.e. Tph2w
        def T_with_offset(offset):
            Tph2grasp = np.array([
                [0, 0, 1, -(0.045 + 0.069) + offset],  
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
            return Tph2grasp

        R_grasp2w = grasp.rotation_matrix # 3, 3
        t_grasp2w = grasp.translation # 3
        Tgrasp2w = np.hstack((R_grasp2w, t_grasp2w[..., None])) # 3, 4
        Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
        Tph2w = Tgrasp2w @ T_with_offset(0.014)
        return Tph2w