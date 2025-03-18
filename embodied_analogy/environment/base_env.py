import math
import mplib
import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from PIL import Image, ImageColor
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R

from embodied_analogy.utility.utils import (
    visualize_pc,
    world_to_image,
)
from embodied_analogy.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)

class BaseEnv():
    def __init__(
            self,
            phy_timestep=1/250.,
            planner_timestep=None,
            use_sapien2=True # otherwise use sapien3
        ):        
        self.asset_prefix = "/home/zby/Programs/Embodied_Analogy/assets"
        self.cur_steps = 0
        
        self.engine = sapien.Engine()  # Create a physical simulation engine
        self.renderer = sapien.SapienRenderer()  # Create a Vulkan renderer
        self.engine.set_renderer(self.renderer)  # Bind the renderer and the engine
        if False:
            from sapien.core import renderer as R
            self.renderer_context: R.Context = self.renderer._internal_context
        
        scene_config = sapien.SceneConfig()
        # follow video axis aligned (or RGBManip?)
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        # scene_config.contact_offset = 0.0001 # 0.1 mm
        scene_config.contact_offset = 0.02 # TODO: RGBManip 设置的为 0.02
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 1
        
        self.scene = self.engine.create_scene(scene_config)
        self.phy_timestep = phy_timestep
        self.scene.set_timestep(phy_timestep)
        if planner_timestep is None:
            self.planner_timestep = self.phy_timestep
        else:
            self.planner_timestep = planner_timestep
        self.scene.add_ground(0)
        
        # add some lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
        
        self.viewer = Viewer(self.renderer)  # Create a viewer (window)
        self.viewer.set_scene(self.scene)  # Bind the viewer and the scene
        self.viewer.set_camera_xyz(x=-1, y=1, z=2)
        self.viewer.set_camera_rpy(r=0, p=-np.arctan2(1, 1), y=np.arctan2(1, 1))
        self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
            
        if use_sapien2:
            self.viewer.toggle_axes(False)
            self.capture_rgb = self.capture_rgb_sapien2
            self.capture_rgbd = self.capture_rgbd_sapien2
        else:
            self.capture_rgb = self.capture_rgb_sapien3
            self.capture_rgbd = self.capture_rgbd_sapien3
            
        self.step = self.base_step
    
    def clear_planner_pc(self):
        self.planner.update_point_cloud(np.array([[0, 0, -1]]))
        
    def load_robot(self, dof_value=None):
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
        
        self.arm_qlimit = self.robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        init_qpos = dof_value
        if dof_value is None :
            init_qpos = (self.arm_q_higher + self.arm_q_lower) / 2
            # init_qpos = self.arm_q_lower
        
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
        self.robot.set_qpos(init_qpos)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        
        self.robot_init_qpos = init_qpos
                    
        # set joints property to enable pd control
        # self.active_joints = self.robot.get_active_joints()
        # for joint in self.active_joints:
        #     joint.set_drive_property(stiffness=1000, damping=200)

        # disable joints gravity
        # for link in self.robot.get_links():
        #     link.disable_gravity = True
        
        # 让机械手臂复原
        # TODO: 这里改为用 cur_qpos 和 init_state 的差值来判断是不是应该停止
        for i in range(200):
            self.step()
            
        self.setup_planner()
        
    def setup_camera(self):
        # camera config
        near, far = 0.1, 100
        width, height = 800, 600
        
        """ 
            intrinsic matrix should be:
            W, 0, W//2
            0, W, H//2
            0, 0, 1
            fx = fy = W
            fovy = 2 * np.arctan(H / (2*W))
        """
        
        camera = self.scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(90), # 1.57
            near=near,
            far=far,
        )
        """
            set pose method 1
            sapien中的相机坐标系为: forward(x), left(y) and up(z)
            Camera to World 也就是相机在世界坐标系下的位姿
            C2W 中的 t 是相机在世界坐标系下的坐标
            C2W 中的 R 的三列从左到右依次是相机坐标系的 x,y,z 轴在世界坐标系下的坐标向量
        """
        # cam_pos = np.array([-1, 1, 2])
        # forward = -cam_pos / np.linalg.norm(cam_pos)
        # left = np.cross([0, 0, 1], forward)
        # left = left / np.linalg.norm(left)
        # up = np.cross(forward, left)
        # mat44 = np.eye(4)
        # mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        # mat44[:3, 3] = cam_pos
        # camera.entity.set_pose(sapien.Pose(mat44)) # C2W
        
        """
            set pose method 2
            这里的 set_local_pose 其实是指定了 Tc2w
        """
        camera.set_local_pose(sapien.Pose([-0.474219, 0.783512, 0.544986], [0.964151, 0.0230758, 0.0894396, -0.248758]))
        self.camera = camera
        
        # 记录相机的内参和外参
        self.camera_intrinsic = self.camera.get_intrinsic_matrix() # [3, 3], K
        Tw2c = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
        self.camera_extrinsic = Tw2c
    
    def capture_rgb_sapien2(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_float_texture('Color')  # [H, W, 4]
        # An alias
        # rgba = camera.get_color_rgba()  
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    
    def capture_rgb_sapien3(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    
    def capture_rgbd_sapien2(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染过程
        # get rgb image
        rgba = camera.get_float_texture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
    
        # get pointcloud
        position = camera.get_float_texture("Position")  # [H, W, 4], 格式为(x, y, z, render_depth), 其中 render_depth < 1 的点是有效的
        points_opengl = position[..., :3][position[..., 3] < 1] # num_valid_points, 3
        points_color = rgba[position[..., 3] < 1] # num_valid_points, 4
        model_matrix = camera.get_model_matrix() # opengl camera to world, must be called after scene.update_render()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3] # N. 3
        points_world = points_world.astype(np.float64)
        points_color = (np.clip(points_color, 0, 1)).astype(np.float64)
        points_color = points_color[..., :3]
        
        if visualize:
            visualize_pc(points_world, points_color, None)
            
        # get depth image
        depth = -position[..., 2] # H, W, in meters
        depth_numpy = np.array(depth)
        # depth_numpy = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_numpy)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
        
    def capture_rgbd_sapien3(self, return_pc=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染过程
        # get rgb image
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
    
        # get pointcloud
        position = camera.get_picture("Position")  # [H, W, 4], 格式为(x, y, z, render_depth), 其中 render_depth < 1 的点是有效的
        points_opengl = position[..., :3][position[..., 3] < 1] # num_valid_points, 3
        points_color = rgba[position[..., 3] < 1] # num_valid_points, 4
        model_matrix = camera.get_model_matrix() # opengl camera to world, must be called after scene.update_render()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3] # N. 3
        points_world = points_world.astype(np.float64)
        points_color = (np.clip(points_color, 0, 1)).astype(np.float64)
        points_color = points_color[..., :3]
        
        if visualize:
            visualize_pc(points_world, points_color, None)
            
        # get depth image
        depth = -position[..., 2] # H, W, in meters
        depth_numpy = np.array(depth)
        # depth_numpy = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_numpy)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        if return_pc:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
    
    def capture_segmentation(self):
        camera = self.camera
        camera.take_picture()
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        mesh_np = seg_labels[..., 0].astype(np.uint8)  # mesh-level [H, W]
        actor_np = seg_labels[..., 1].astype(np.uint8)  # actor-level [H, W]
        
        # Or you can use aliases below
        # label0_image = camera.get_visual_segmentation()
        # label1_image = camera.get_actor_segmentation()
        
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array(
            [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        )
        mesh_pil = Image.fromarray(color_palette[mesh_np])
        actor_pil = Image.fromarray(color_palette[actor_np])
        return actor_np # [H, W]
        
    def load_object(self, obj_config):
        index = obj_config["index"]
        scale = obj_config["scale"]
        pose = obj_config["pose"]
        active_link = obj_config["active_link"]
        
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        
        # 在这里设置 load 时的 config
        urdf_config = {
            "_materials": {
                "gripper" : {
                    "static_friction": 2.0,
                    "dynamic_friction": 2.0,
                    "restitution": 0.0
                }
            },
            "link": {
                active_link: {
                    "material": "gripper",
                    "density": 1.0,
                }
            }
        }
        load_config = parse_urdf_config(urdf_config, self.scene)
        check_urdf_config(load_config)
        
        self.asset = loader.load(
            filename=self.asset_prefix + f"/{index}/mobility.urdf",
            config=load_config
        )
        
        self.asset.set_root_pose(sapien.Pose(pose, [1, 0, 0, 0]))
        # self.asset.set_qpos(dof_value)
        # self.asset.set_qvel(np.zeros_like(dof_value))
        
        # only for pot
        # lift_joint = self.asset.get_joints()[-1]
        # lift_joint.set_limit(np.array([0, 0.3]))
        
        
        # 设置物体关节的参数, 把回弹关掉
        for joint in self.asset.get_active_joints():
            joint.set_drive_property(stiffness=0, damping=0.1)
            
            # 在这里判断当前的 joint 是不是我们关注的需要改变状态的关节, 如果是, 则初始化读取状态的函数, 以及当前状态
            if joint.get_name() == obj_config["active_joint"]:
                self.evaluate_joint = joint
                self.init_joint_transform = joint.get_global_pose().to_transformation_matrix() # 4, 4, Tw2c
        
    def get_joint_state(self):
        cur_transform = self.evaluate_joint.get_global_pose().to_transformation_matrix()
        # Tinit2cur = Tw2cur @ Tw2init.inv
        Tinit2cur = cur_transform @ np.linalg.inv(self.init_joint_transform)
        return Tinit2cur
    
    def load_panda_hand(self, scale=1., pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        self.asset = loader.load(self.asset_prefix + f"/panda/panda_v2_gripper.urdf")
        self.asset.set_root_pose(sapien.Pose(pos, quat))
        return self.asset
    
    def get_points_on_arm(self):
        # 获取 franka arm 上的一些点的 2d 和 3d 的坐标（目前是 link_pose）
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
    
    def base_step(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        self.cur_steps += 1
            
    def open_gripper(self):
        for i in range(50):
            self.step()
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.03)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()

    def close_gripper(self):
        for i in range(50):
            self.step()
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()
        # self.after_try_to_close = 1

    def reset_franka_arm(self):
        # reset实现为让 panda hand移动到最开始的位置，并关闭夹爪
        # self.open_gripper()
        init_panda_hand = mplib.Pose(p=[0.111, 0, 0.92], q=t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz"))
        self.move_to_pose(pose=init_panda_hand, wrt_world=True)
        self.close_gripper()
        
    def plan_path(self, target_pose, wrt_world: bool = True):
        # 传入的 target_pose 是 Tph2w
        if isinstance(target_pose, np.ndarray):
            target_pose = mplib.Pose(target_pose)
        result = self.planner.plan_pose(
            goal_pose=target_pose, 
            current_qpos=self.robot.get_qpos(), 
            time_step=self.planner_timestep, 
            rrt_range=0.1,
            planning_time=1,
            wrt_world=wrt_world
        )
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
                qf = self.robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True
                )
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
    
    
    def move_along_axis(self, moving_direction, moving_distance, num_interpolation=10):
        # moving_direction 是在世界坐标系下的方向
        # 控制 panda_hand 沿着 moving_direction 行动 moving_distance 的距离
        ee_pose, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3
        
        def T_with_delta(delta):
            Tph2w = np.eye(4)
            Tph2w[:3, :3] = Rph2w
            # 对于 panda_hand 来说, z-axis 的正方向是向前
            Tph2w[:3, 3] = ee_pose[:3] + delta * moving_direction
            return Tph2w
        
        # 先得到连续的插值位姿
        deltas = np.linspace(0, moving_distance, num_interpolation)
        T_list = [T_with_delta(delta) for delta in deltas]
        
        # 然后依次执行这些位姿
        for Tph2w in T_list:
            result = self.plan_path(target_pose=Tph2w, wrt_world=True)
            self.follow_path(result)
    
    def move_forward(self, moving_distance):
        # 控制 panda_hand 沿着 moving_direction 行动 moving_distance 的距离
        _, ee_quat = self.get_ee_pose() # Tph2w
        # scalar_first means quat in (w, x, y, z) order
        Rph2w = R.from_quat(ee_quat, scalar_first=True).as_matrix() # 3, 3
        moving_direction = Rph2w @ np.array([0, 0, 1]) # 3
        self.move_along_axis(moving_direction, moving_distance)
            
    def get_ee_pose(self):
        # 获取ee_pos和ee_quat
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        return ee_pos, ee_quat # numpy array


if __name__ == "__main__":
    env = BaseEnv()
    env.load_robot()
    env.setup_camera()
    # env.load_object()
    
    # for i in range(100):
    while True:
        env.step()
        # env.capture_rgbd_sapien2(visualize=True)
        
    pts_on_arm_2d, pts_on_arm_3d = env.get_points_on_arm() # N, 3
    from embodied_analogy.utility.utils import world_to_image
    pts_on_arm_2d = world_to_image(pts_on_arm_3d, env.camera_intrinsic, env.camera_extrinsic)
    from embodied_analogy.utility.utils import draw_points_on_image
    draw_points_on_image(Image.fromarray(env.capture_rgb()), pts_on_arm_2d, radius=3).show()
    
    
    while True:
        env.step()
    