import sapien.core as sapien
from sapien.utils.viewer import Viewer
import mplib
import numpy as np
import transforms3d as t3d
import trimesh
# from trimesh_utils import *

from embodied_analogy.utils import *

    
class PlanningDemo():
    def __init__(
            self,
            phy_timestep=1/250.,
            record_fps=30,
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
        # Robot
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load(self.asset_prefix + "/panda/panda_v3.urdf")
        # Set initial joint positions
        init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.robot.set_qpos(init_qpos)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        
        if True:
            loader: sapien.URDFLoader = self.scene.create_urdf_loader()
            loader.scale = 0.4
            loader.fix_root_link = True
            self.asset = loader.load(self.asset_prefix + "/100015/mobility.urdf")
            # self.asset = loader.load("./100051/mobility.urdf")
            self.asset.set_root_pose(sapien.Pose([0.4, 0.4, 0.2], [1, 0, 0, 0]))
            
            lift_joint = self.asset.get_joints()[-1]
            lift_joint.set_limit(np.array([0, 0.3]))
            
        # set joints property to enable pd control
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        # disable joints gravity
        # for link in self.robot.get_links():
        #     link.disable_gravity = True
        
        # 根据record_fps计算record_interval
        self.record_interval = max(int(1. / phy_timestep / record_fps), 1)
        self.after_try_to_close = 0
        self.cur_steps = 0
        self.recorded_data = {}
        self.setup_planner()
        self.setup_camera()
    
        # initialize pygame for keyboard control
        pygame.init()
        resolution = (self.camera.get_width(), self.camera.get_height())
        self.pygame_screen = pygame.display.set_mode(resolution)
        pygame.display.set_caption("Keyboard Control")
    def save_recoreded_data(self):
        assert isinstance(self.recorded_data, dict)
        prefix = "/home/zby/Programs/Embodied_Analogy/assets/recorded_data"
        np.savez(prefix + '/recorded_data_2.npz', **self.recorded_data)
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
        # print("Intrinsic matrix\n", camera.get_intrinsic_matrix())
        
        # 将相机的内参和外参保存到 self.recorded_data 中
        self.recorded_data["intrinsic"] = camera.get_intrinsic_matrix() # [3, 3]
        extrinsic = camera.get_extrinsic_matrix() # [3, 4] Tw2c
        extrinsic = np.vstack([extrinsic, np.array([0, 0, 0, 1])])
        self.recorded_data["extrinsic"] =  extrinsic
    
    def capture_rgb(self):
        camera = self.camera
        camera.take_picture()  # submit rendering jobs to the GPU
        
        # 渲染rgb图像
        rgba = camera.get_picture("Color")  # 获取RGBA图像，格式为[H, W, 4]
        rgb = rgba[..., :3]
        rgb_numpy = (rgb * 255).clip(0, 255).astype("uint8") # numpy array, 255
        # rgb_pil = Image.fromarray(rgb_numpy)
        return rgb_numpy
    def capture_rgb_depth(self):
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
        points_color = rgba[position[..., 3] < 1] # num_valid_points, 3
        model_matrix = camera.get_model_matrix() # opengl camera to world, must be called after scene.update_render()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        points_color = (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
        
        # get depth image
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_image)
        depth_valid_mask = position[..., 3] < 1 # H, W
        depth_valid_mask_pil = Image.fromarray(depth_valid_mask)
        
        return rgb_pil, depth_pil, depth_valid_mask_pil
    
    def step(self):
        self.scene.step()
        self.scene.update_render() # 记得在 render viewer 或者 camera 之前调用 update_render()
        
        self.cur_steps += 1
        self.cur_steps = self.cur_steps % self.record_interval
        
        if self.cur_steps == 0:
            # has the gripper closed?
            after_close = self.after_try_to_close
            
            # pose of panda_hand link
            ee_pos, ee_quat = self.get_ee_pose() # np array of size 3 and 4
            
            # contact point in 3d
            cp_3d = []
            valid_contact_link = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
            contacts = self.scene.get_contacts()
            for contact in contacts:
                name0, name1 = contact.bodies[0].entity.name, contact.bodies[1].entity.name
                if (name0 not in valid_contact_link) and (name1 not in valid_contact_link): 
                    continue
                for point in contact.points:
                    cp_3d.append(point.position) # numpy array in world frame
            
            # contact point projected in 2d camera (in uv pixel coordinates, normalized to [0, 1])
            cp_2d = []
            for cp in cp_3d:
                K = self.recorded_data["intrinsic"]
                Tw2c = self.recorded_data["extrinsic"]
                w = self.camera.get_width()
                h = self.camera.get_height()
                u, v = world_to_normalized_uv(cp, K, Tw2c, w, h)
                cp_2d.append(np.array([u, v]))
                
            # record rgb image and display to pygame screen
            rgb_np = self.capture_rgb()
            rgb_pil = Image.fromarray(rgb_np)
            update_image(self.pygame_screen, rgb_pil)
            
            cur_dict = {
                "after_close": after_close,
                "panda_hand_pos": ee_pos,
                "panda_hand_quat": ee_quat,
                "rgb_np": rgb_np, 
                "contact_points_3d": np.array(cp_3d), # N x 3
                "contact_points_2d": np.array(cp_2d), # N x 2
            }
            if "traj" not in self.recorded_data.keys():
                self.recorded_data["traj"] = []
            self.recorded_data["traj"].append(cur_dict)
            
    def follow_path(self, result):
        n_step = result['position'].shape[0]
        print("n_step:", n_step)
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.step()
            # self.scene.update_render()
            self.viewer.render()

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()
            # self.scene.update_render()
            self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(-0.1)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.step()
            # self.scene.update_render()
            self.viewer.render()
        self.after_try_to_close = 1

    def move_to_pose_with_RRTConnect(self, pose, wrt_world):
        result = self.planner.plan_pose(
            goal_pose=pose, 
            current_qpos=self.robot.get_qpos(), 
            time_step=0.1, 
            rrt_range=0.1,
            planning_time=1,
            wrt_world=wrt_world
        )
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0
    
    def move_to_pose(self, pose, wrt_world):
        return self.move_to_pose_with_RRTConnect(pose, wrt_world)
    
    def get_ee_pose(self):
        # 获取ee_pos和ee_quat
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        return ee_pos, ee_quat # numpy array
    def manipulate_franka(self):
        pos_scale_factor = 0.1
        
        while not self.viewer.closed:
            target_pos, _ = self.get_ee_pose()
            target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
            delta_pos = np.array([0, 0, 0])
            # delta_rpy = np.array([0, 0, 0])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        delta_pos = np.array([1, 0, 0]) 
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_DOWN:
                        delta_pos = np.array([-1, 0, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_LEFT:
                        delta_pos = np.array([0, 1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_RIGHT:
                        delta_pos = np.array([0, -1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP_PLUS:
                        delta_pos = np.array([0, 0, 1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP_MINUS:
                        delta_pos = np.array([0, 0, -1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, wrt_world=True)
                    elif event.key == pygame.K_KP0: 
                        self.close_gripper()
                    elif event.key == pygame.K_KP1:  
                        self.open_gripper()
                    elif event.key == pygame.K_KP_ENTER:  
                        self.save_recoreded_data()
                    continue
            
            self.step()
            self.viewer.render() 
    
if __name__ == '__main__':
    demo = PlanningDemo()
    demo.manipulate_franka()
