import mplib
import numpy as np
import argparse
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from embodied_analogy.utils import *
from PIL import Image, ImageColor
import open3d as o3d
import transforms3d as t3d

class BaseEnv():
    def __init__(
            self,
            phy_timestep=1/250.,
        ):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)

        self.scene.set_timestep(phy_timestep)
        self.scene.add_ground(0)
        # physical_material = self.scene.create_physical_material(1, 1, 0.0)
        # self.scene.default_physical_material = physical_material

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)
        
        self.asset_prefix = "/home/zby/Programs/Embodied_Analogy/assets"
        self.cur_steps = 0
    
    def load_franka_arm(self):
        # Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/panda_v3.urdf")
        # Set initial joint positions
        init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.robot.set_qpos(init_qpos)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
                    
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
        
    def setup_camera(self):
        # camera config
        near, far = 0.1, 100
        width, height = 800, 600
        
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
    
    def capture_rgb(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    
    def capture_rgbd(self, return_point_cloud=False, visualize=False):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染过程
        # get rgb image
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        rgb_pil = Image.fromarray(rgb_numpy)
    
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
        depth = -position[..., 2]
        depth_numpy = np.array(depth)
        # depth_numpy = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_numpy)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        if return_point_cloud:
            return rgb_numpy, depth_numpy, points_world, points_color
        else:    
            return rgb_numpy, depth_numpy, None, None
    
    def capture_segmentation(self):
        camera = self.camera
        camera.take_picture()
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array(
            [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        )
        label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
        label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
        # Or you can use aliases below
        # label0_image = camera.get_visual_segmentation()
        # label1_image = camera.get_actor_segmentation()
        label0_pil = Image.fromarray(color_palette[label0_image])
        label1_pil = Image.fromarray(color_palette[label1_image])
        # label1_pil.show()
        # label0_pil.save("label0.png")
        return label1_image
        
    def load_articulated_object(self, index=100015, scale=0.4, pose=[0.4, 0.4, 0.2]):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        self.asset = loader.load(self.asset_prefix + f"/{index}/mobility.urdf")
        self.asset.set_root_pose(sapien.Pose(pose, [1, 0, 0, 0]))
        
        lift_joint = self.asset.get_joints()[-1]
        lift_joint.set_limit(np.array([0, 0.3]))
        
    def load_panda_hand(self, scale=1., pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = True
        self.asset = loader.load(self.asset_prefix + f"/panda/panda_v2_gripper.urdf")
        self.asset.set_root_pose(sapien.Pose(pos, quat))
        return self.asset
        
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
            

    def open_gripper(self):
        # for i in range(100):
        #     self.step()
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()

    def close_gripper(self):
        for i in range(100):
            self.step()
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(-0.1)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()
        self.after_try_to_close = 1

    def reset_franka_arm(self):
        # reset实现为让 panda hand移动到最开始的位置，并关闭夹爪
        self.open_gripper()
        init_panda_hand = mplib.Pose(p=[0.111, 0, 0.92], q=t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz"))
        self.move_to_pose(pose=init_panda_hand, wrt_world=True)
        self.close_gripper()
        
    def move_to_pose_with_RRTConnect(self, pose: mplib.pymp.Pose, wrt_world: bool):
        result = self.planner.plan_pose(
            goal_pose=pose, 
            current_qpos=self.robot.get_qpos(), 
            time_step=0.1, 
            rrt_range=0.1,
            # planning_time=1,
            planning_time=0.5,
            wrt_world=wrt_world
        )
        if result['status'] != "Success":
            # print(result['status'])
            return -1
        self.follow_path(result)
        return 0
    
    def move_to_pose(self, pose: mplib.pymp.Pose, wrt_world: bool):
        status = self.move_to_pose_with_RRTConnect(pose, wrt_world)
        return status
    
    def get_ee_pose(self):
        # 获取ee_pos和ee_quat
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        return ee_pos, ee_quat # numpy array
    
    def detect_grasp_anygrasp(self, points, colors, visualize=True):
        '''
        输入世界坐标系下的点云和颜色, 返回 grasp_group
        
        定义 grasp 坐标系为 xy 轴平行地面, z 轴指向重力方向
        定义 gripper 坐标系为 x 轴指向物体内部, y 轴指向物体的宽度
        
        '''
        # 传入的点是在世界坐标系下的(xy 轴平行地面, z 轴指向重力反方向)
        # 因此首先将世界坐标系下的点转换到 grasp 坐标系下
        points = points.astype(np.float32)
        colors = colors.astype(np.float32)
        points_input = points.copy() # N, 3
        colors_input = colors.copy()
        
        # 坐标系转换,将世界坐标系绕着 x 轴旋转 180 度得到 grasp 坐标系
        Tw2grasp = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        points_input = points_input @ Tw2grasp.T # N, 3
        points_input = points_input.astype(np.float32)
        
        from gsnet import AnyGrasp # gsnet.so
        # get a argument namespace
        cfgs = argparse.Namespace()
        cfgs.checkpoint_path = 'assets/ckpts/checkpoint_detection.tar'
        cfgs.max_gripper_width = 0.1
        cfgs.gripper_height = 0.03
        cfgs.top_down_grasp = False
        cfgs.debug = visualize
        model = AnyGrasp(cfgs)
        model.load_net()
        
        lims = [-1, 1, -1, 1, -1, 1]
        gg, cloud = model.get_grasp(points_input, colors_input, lims, \
            apply_object_mask=True, dense_grasp=False, collision_detection=True
                                       )
        print('grasp num:', len(gg))
        
        if visualize:
            grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([*grippers, cloud])
            
        # 此时的 gg 中的 rotation 和 translation 对应 Tgripper2grasp
        # 将预测的 grasp pose 从 grasp 坐标系转换回世界坐标系
        zero_translation = np.array([[0], [0], [0]])
        Tgrasp2w = Tw2grasp.T
        Tgrasp2w = np.hstack((Tgrasp2w, zero_translation))
        gg.transform(Tgrasp2w)
        # 此时的 gg 中的 rotation 和 translation 对应 Tgripper2world
        return gg
    