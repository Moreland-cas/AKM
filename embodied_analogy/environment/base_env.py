import mplib
import numpy as np
import argparse
import sapien.core as sapien
from sapien.utils.viewer import Viewer
# from embodied_analogy.utility import *
from embodied_analogy.utility.utils import (
    visualize_pc,
    world_to_image
)
from embodied_analogy.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)
from embodied_analogy.estimation.utils import rotation_matrix_between_vectors
from PIL import Image, ImageColor
import open3d as o3d
import transforms3d as t3d

class BaseEnv():
    def __init__(
            self,
            phy_timestep=1/250.,
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
        scene_config.contact_offset = 0.0001 # 0.1 mm
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 1
        
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(phy_timestep)
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
    
    def load_franka_arm(self, dof_value=None):
        # Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/panda_v3.urdf")
        
        self.arm_qlimit = self.robot.get_qlimits()
        self.arm_q_lower = self.arm_qlimit[:, 0]
        self.arm_q_higher = self.arm_qlimit[:, 1]

        init_qpos = dof_value
        if dof_value is None :
            init_qpos = (self.arm_q_higher + self.arm_q_lower) / 2
            # init_qpos = self.arm_q_lower
        
        # Setup control properties 
        # self.active_joints = self.robot.get_active_joints()
        # for joint in self.active_joints[:4]:
        #     joint.set_drive_property(stiffness=160, damping=40, force_limit=100)    # original: 200
        # for joint in self.active_joints[4:-2]:
        #     joint.set_drive_property(stiffness=160, damping=40, force_limit=50)    # original: 200
        # for joint in self.active_joints[-2:]:
        #     joint.set_drive_property(stiffness=4000, damping=10)
            
        # Set initial joint positions
        # init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.robot.set_qpos(init_qpos)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        
        self.robot_init_qpos = init_qpos
                    
        # set joints property to enable pd control
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        # disable joints gravity
        # for link in self.robot.get_links():
        #     link.disable_gravity = True
        
        # 让机械手臂复原
        for i in range(100):
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
        # self.camera_extrinsic = np.vstack([Tw2c, np.array([0, 0, 0, 1])]) # 4, 4
        self.camera_extrinsic = Tw2c
        
        # 将相机的内参和外参保存到 self.recorded_data 中
        if not hasattr(self, 'recorded_data'):
            self.recorded_data = {}
            
        self.recorded_data["intrinsic"] = self.camera_intrinsic # [3, 3]
        self.recorded_data["extrinsic"] =  self.camera_extrinsic # [4, 4]
    
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
        
    # def load_articulated_object(self, index=100015, scale=0.4, pose=[0.4, 0.4, 0.2]):
    def load_articulated_object(self, obj_config):
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
                self.init_joint_transform = joint.get_global_pose().to_transformation_matrix() # 4, 4, Tw2j
            
        # 在 load asset 之后拍一张物体的照片，作为初始状态
        # self.scene.step()
        # self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        # self.viewer.render()
        
        # initial_rgb, initial_depth, _, _ = self.capture_rgbd(return_pc=False, visualize=False)
        
        # if not hasattr(self, 'recorded_data'):
        #     self.recorded_data = {}
            
        # self.recorded_data["initial_rgb"] = initial_rgb
        # self.recorded_data["initial_depth"] = initial_depth
        
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
        return link_poses_2d, link_poses_3d
    
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=self.asset_prefix + "/panda/panda_v3.urdf",
            srdf=self.asset_prefix + "/panda/panda_v3.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))
    
    def step(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        self.viewer.render()
        self.cur_steps += 1
            
    def open_gripper(self):
        for i in range(50):
            self.step()
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.04)
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
            joint.set_drive_target(-0.01)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()
        # self.after_try_to_close = 1

    def reset_franka_arm(self):
        # reset实现为让 panda hand移动到最开始的位置，并关闭夹爪
        self.open_gripper()
        init_panda_hand = mplib.Pose(p=[0.111, 0, 0.92], q=t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz"))
        self.move_to_pose(pose=init_panda_hand, wrt_world=True)
        self.close_gripper()
        
    def plan_path(self, target_pose, wrt_world: bool):
        # 传入的 target_pose 是 Tph2w
        if isinstance(target_pose, np.ndarray):
            target_pose = mplib.Pose(target_pose)
        result = self.planner.plan_pose(
            goal_pose=target_pose, 
            current_qpos=self.robot.get_qpos(), 
            time_step=0.1, 
            rrt_range=0.1,
            planning_time=1,
            # planning_time=0.5,
            wrt_world=wrt_world
        )
        if result['status'] != "Success":
            return None
        return result
    
    def follow_path(self, result):
        n_step = result['position'].shape[0]
        # print("n_step:", n_step)
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.step()
    
    def move_to_pose(self, pose: mplib.pymp.Pose, wrt_world: bool):
        result = self.plan_path(target_pose=pose, wrt_world=wrt_world)
        if result is not None:
            self.follow_path(result)
        else:
            "plan path failed!"
    
    def get_ee_pose(self):
        # 获取ee_pos和ee_quat
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        return ee_pos, ee_quat # numpy array
    
    def detect_grasp_anygrasp(self, points, colors, joint_axis, visualize=True):
        '''
        输入世界坐标系下的点云和颜色, 返回 grasp_group
        
        定义 approach 坐标系为 xy 轴平行物体表面, z 轴指向物体内部 (joint axis 的反方向)
        定义 grasp 坐标系为 x 轴指向物体内部, y 轴指向物体的宽度
        
        '''
        # 在这里将 grasp depth 设置的小一点
        # from graspnetAPI import graspnet
        # graspnet.GRASP_HEIGHT = 0.
        
        # 传入的点是在世界坐标系下的(xy 轴平行地面, z 轴指向重力反方向)
        # 因此首先将世界坐标系下的点转换到 app 坐标系下
        points = points.astype(np.float32)
        colors = colors.astype(np.float32)
        points_input = points.copy() # N, 3
        colors_input = colors.copy()
        
        # coor_app = Rw2app @ coor_w, 也即 -joint_axis = Rw2app @ (0, 0, 1)
        Rw2app = rotation_matrix_between_vectors(np.array([0, 0, 1]), -joint_axis)
        points_input = points_input @ Rw2app.T # N, 3
        points_input = points_input.astype(np.float32)
        
        from gsnet import AnyGrasp # gsnet.so
        # get a argument namespace
        cfgs = argparse.Namespace()
        cfgs.checkpoint_path = 'assets/ckpts/checkpoint_detection.tar'
        cfgs.max_gripper_width = 0.04
        cfgs.gripper_height = 0.03
        cfgs.top_down_grasp = False
        cfgs.debug = visualize
        model = AnyGrasp(cfgs)
        model.load_net()
        
        lims = np.array([-1, 1, -1, 1, -1, 1]) * 10
        gg, cloud = model.get_grasp(
            points_input,
            colors_input, 
            lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )
        print('grasp num:', len(gg))
        
        if visualize:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([*grippers, cloud])
            
        # 此时的 gg 中的 rotation 和 translation 对应 Tgrasp2app
        # 将预测的 app pose 从 app 坐标系转换回世界坐标系
        zero_translation = np.array([[0], [0], [0]])
        Rapp2w = Rw2app.T
        Tapp2w = np.hstack((Rapp2w, zero_translation))
        gg.transform(Tapp2w)
        # 此时的 gg 中的 rotation 和 translation 对应 Tgrasp2w
        return gg


if __name__ == "__main__":
    env = BaseEnv()
    env.load_franka_arm()
    env.setup_camera()
    # env.load_articulated_object()
    
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
    