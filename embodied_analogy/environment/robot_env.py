import math
import mplib
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R
from embodied_analogy.environment.base_env import BaseEnv

from embodied_analogy.utility.utils import world_to_image
from embodied_analogy.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)
from embodied_analogy.representation.basic_structure import Frame

class RobotEnv(BaseEnv):
    def __init__(
            self,
            base_cfg={
                "phy_timestep": 1/250.,
                "planner_timestep": None,
                "use_sapien2": True 
            },
            robot_cfg={},
        ):        
        super().__init__(base_cfg)
        self.load_robot(robot_cfg)
    
    def capture_frame(self, visualize=False) -> Frame:
        frame = super().capture_frame(visualize=False)
        
        frame.robot_mask = self.capture_robot_mask()
        frame.robot2d=self.get_points_on_arm()[0]
        frame.Tph2w = self.get_ee_pose(as_matrix=True)
        
        if visualize:
            frame.visualize()
        return frame
    
    def load_robot(self, robot_cfg=None):
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

        # init_qpos = (self.arm_q_higher + self.arm_q_lower) / 2
        # init_qpos[5] = 0.278
        # self.init_qpos = init_qpos
        
        # Setup control properties 
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints[:4]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=100)    # original: 200
        for joint in self.active_joints[4:-2]:
            joint.set_drive_property(stiffness=160, damping=40, force_limit=50)    # original: 200
        for joint in self.active_joints[-2:]:
            # joint.set_drive_property(stiffness=4000, damping=10)
            joint.set_drive_property(stiffness=160, damping=10, force_limit=50)
            
        # Set initial joint positions
        # init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
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
        # 这里之所以是 255 是因为在 load robot arm 的时候设置了其 visual_id 为 255
        return mesh_np == 255
    
    def load_panda_hand(self, scale=1., pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        self.asset = loader.load(self.asset_prefix + f"/panda/panda_v2_gripper.urdf")
        self.asset.set_root_pose(sapien.Pose(pos, quat))
        return self.asset
    
    def get_points_on_arm(self):
        # 获取 robot arm 上的一些点的 2d 和 3d 的坐标（目前是 link_pose）
        link_poses_3d = []
        for link in self.robot.get_links():
            link_pos = link.get_pose().p # np.array(3)
            link_poses_3d.append(link_pos)
        link_poses_3d = np.array(link_poses_3d) # N, 3
        
        # 投影到 2d 相机平面
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
            
    def open_gripper(self, target=0.03):
        """
        这里用 self.step() 是可能在 record 中录制 open/close 的动作, 那时候 self.step 实际对应的是 self.record_step 函数
        """
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(target)
        
        # NOTE: 与 reset 中的 self.step 一致, 可以在那里查看详细的解释
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
            
        # NOTE: 与 reset 中的 self.step 一致, 可以在那里查看详细的解释
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
        reset_robot 不控制 gripper 的 open/close 状态, 只把其他关节进行 reset
        """
        init_panda_hand = mplib.Pose(p=[0.111, 0, 0.92], q=t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(0), axes="syxz"))
        self.move_to_pose(pose=init_panda_hand, wrt_world=True)
        
        # NOTE: 这里一定要进行一步 step, 因为现在是通过 cur_qpos 和 target_qpos 的差值来判断是否要结束同步的
        # 但是在 self.robot.set_qpos() 后的第一个 step 时的 cur_qpos 是等于 target_qpos 的, 后续才会变为实际值
        # 所以如果不 step 一下会导致同步失效, 直接退出
        self.base_step()
        
        count = 0
        while count < 100:
            vel = self.robot.get_qvel()
            # 这里选用 vel_norm 作为 reset 进行同步的终止条件, 因为 vel 相对于 qpos 和 qacc 更加稳定
            # qpos 不知道要设置什么值, qacc 经常会突变
            if np.linalg.norm(vel) < 2e-3: # 基本 vel 在 mm/s 的量级就是可以的
                break
            self.base_step()
            count += 1
    
    def reset_robot_with_pc(self, pc=None):        
        # 先打开 gripper, 再撤退一段距离
        self.open_gripper()
        self.move_forward(-0.05) # 向后撤退 5 cm
        
        # 读取一帧 rgbd， 经过 sam 得到 pc， 对 pc 进行处理
        if pc is None:
            pc = np.array([[0, 0, -1]])
        self.planner.update_point_cloud(pc)
        
        self.reset_robot()
        
    def clear_planner_pc(self):
        self.planner.update_point_cloud(np.array([[0, 0, -1]]))
        
    def plan_path(self, target_pose, wrt_world: bool = True):
        # 传入的 target_pose 是 Tph2w
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
            print(f"An error occurred during plan_path(): {e}")
            return None
        
        if result['status'] != "Success":
            return None
        
        return result
            
    def follow_path(self, result):
        n_step = result['position'].shape[0]
        print("n_step:", n_step)
        for i in range(n_step):  
            position_target = result['position'][i]
            velocity_target = result['velocity'][i]
            # num_repeat 需要根据 mplib.planner 初始化时候的 time_step 进行计算
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
            "plan path failed!"
    
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
        
        if (Rrefine2w1[:, 0] * axis_out_w).sum() < (Rrefine2w2[:, 0] * axis_out_w).sum():
            grasp_.rotation_matrix = Rrefine2w1.reshape(-1)
        else:
            grasp_.rotation_matrix = Rrefine2w2.reshape(-1)
        
        return grasp_
    
    def move_along_axis(self, joint_type, joint_axis, joint_start, moving_distance):
        """
        控制 panda_hand 沿着某个轴移动一定距离, 或者绕着某个轴移动一定角度, 并保持 panda_hand 与物体的相对位姿保持不变
        joint_axis: 1) 在世界坐标系下!! 2) 满足右手定则, 沿着 joint_axis 的方向是打开
        """
        assert joint_type in ["prismatic", "revolute"]
        if joint_type == "revolute":
            assert joint_start is not None, "joint_start cannot be None when joint_type is revolute"
            # 根据 moving distance 的大小计算出有多少个插值点
            # 对于平移关节时每次移动 3 cm
            num_interp = max(3, int(moving_distance / 0.02))
        else:
            # 对于旋转关节时每次移动 5 degree
            num_interp = max(3, int(moving_distance / np.deg2rad(5)))
            
        ee_pose, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3
            
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
        
        # 然后依次执行这些位姿
        for Tph2w in T_list:
            result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if result is None:
                print("excounter None result when moving along axis, skip!")
                continue
            self.follow_path(result)
    
    def move_forward(self, moving_distance):
        # 控制 panda_hand 沿着 moving_direction 行动 moving_distance 的距离
        _, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3
        moving_direction = Rph2w @ np.array([0, 0, 1]) # 3
        self.move_along_axis(
            joint_type="prismatic",
            joint_axis=moving_direction,
            joint_start=None,
            moving_distance=moving_distance
        )
            
    def get_ee_pose(self, as_matrix=False):
        """
        获取 end-effector (panda_hand) 的 ee_pos 和 ee_quat (Tph2w)
        ee_pos: np.array([3, ])
        ee_quat: np.array([4, ]), scalar_first=True, in (w, x, y, z) order
        """
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        
        if as_matrix:
            T = np.eye(4)  # 创建一个 4x4 单位矩阵
            R_matrix = R.from_quat(ee_quat, scalar_first=True).as_matrix() 
            T[:3, :3] = R_matrix  # 将旋转矩阵放入变换矩阵
            T[:3, 3] = ee_pos  # 将位置向量放入变换矩阵
            return T
        else:
            return ee_pos, ee_quat # numpy array

    def anyGrasp2ph(self, grasp):
        """
            从 anygrasp 输出的 grasp 中提取出 Tph2w, 也就是做一个 Tgrasp2w 到 Tph2w 的转换
            grasp: Grasp 对象
        """
        # 将 grasp 坐标系转换到为 panda_hand 坐标系, 即 Tph2w
        def T_with_offset(offset):
            Tph2grasp = np.array([
                [0, 0, 1, -(0.045 + 0.069) + offset],  # TODO: offset 设置为 0.014?
                [0, 1, 0, 0], 
                [-1, 0, 0, 0], 
                [0, 0, 0, 1]
            ])
            return Tph2grasp
        
        R_grasp2w = grasp.rotation_matrix # 3, 3
        t_grasp2w = grasp.translation # 3
        Tgrasp2w = np.hstack((R_grasp2w, t_grasp2w[..., None])) # 3, 4
        Tgrasp2w = np.vstack((Tgrasp2w, np.array([0, 0, 0, 1]))) # 4, 4
        # Tph2w = Tgrasp2w @ T_with_offset(0.03)
        # Tph2w = Tgrasp2w @ T_with_offset(0.02)
        Tph2w = Tgrasp2w @ T_with_offset(0.014)
        return Tph2w
    
    
if __name__ == "__main__":
    env = RobotEnv()
    
    env.step()
    env.capture_frame()
    
    env.open_gripper()
    env.move_forward(0.1)
    env.close_gripper()
    env.move_forward(-0.1)
    env.open_gripper()
    
    
    # for i in range(100):
    while True:
        env.step()
    