import sapien.core as sapien
from sapien.utils.viewer import Viewer
import mplib
import numpy as np
import transforms3d as t3d
import pygame
# import trimesh
# from trimesh_utils import *

class PlanningDemo():
    def __init__(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        
        self.asset_prefix = "/home/zby/Programs/Sapien/assets"
        
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)

        self.scene.set_timestep(1 / 250.0)
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

        # Load URDF
        # articulated objects
        if True:
            loader: sapien.URDFLoader = self.scene.create_urdf_loader()
            loader.scale = 0.3
            loader.fix_root_link = True
            self.asset = loader.load(self.asset_prefix + "/100015/mobility.urdf")
            # self.asset = loader.load("./100051/mobility.urdf")
            self.asset.set_root_pose(sapien.Pose([0.5, 0., 0.3], [1, 0, 0, 0]))
            
            lift_joint = self.asset.get_joints()[-1]
            lift_joint.set_limit(np.array([0, 0.3]))
        
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
        
        self.setup_planner()
    
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

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(-0.1)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRTConnect(self, pose, wrt_world):
        result = self.planner.plan_pose(pose, self.robot.get_qpos(), time_step=1/250, wrt_world=wrt_world)
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0

    def move_to_pose_with_screw(self, pose, wrt_world):
        result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1/250, wrt_world=wrt_world)
        if result['status'] != "Success":
            result = self.planner.plan_pose(pose, self.robot.get_qpos(), time_step=1/250, wrt_world=wrt_world)
            if result['status'] != "Success":
                print(result['status'])
                return -1 
        self.follow_path(result)
        return 0
    
    def move_to_pose(self, pose, with_screw, wrt_world):
        if with_screw:
            return self.move_to_pose_with_screw(pose, wrt_world)
        else:
            return self.move_to_pose_with_RRTConnect(pose, wrt_world)

    def sample_points(self):
        meshes = get_articulation_meshes(self.asset)
        pcs = []
        for mesh in meshes:
            pcs.append(trimesh.sample.sample_surface(mesh, 100)[0])
        pcs = np.concatenate(pcs, axis=0)
        return pcs
    def visualize_pcs(self, pcs):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs)
        o3d.visualization.draw_geometries([pcd])
    def open_drawer(self):
        for i in range(500):  
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()
        
        # pcs = self.sample_points()
        # self.visualize_pcs(pcs)
        # self.planner.update_point_cloud(pcs)
        
        self.open_gripper()
        # rotate 90 degrees around y axis
        q = t3d.euler.euler2quat(np.deg2rad(180), np.deg2rad(0), np.deg2rad(90), axes="sxyz")
        # q = [0, 1, 0, 0]
        pose = mplib.pymp.Pose(p=[0.54, 0.11, 0.58], q=q)
        self.move_to_pose(pose, with_screw=True, wrt_world=True)
        
        pose = mplib.pymp.Pose(p=[0.54, 0.11, 0.38], q=q)
        self.move_to_pose(pose, with_screw=True, wrt_world=True)
        
        self.close_gripper()
        
        pose = mplib.pymp.Pose(p=[0.54, 0.11, 0.68], q=q)
        self.move_to_pose(pose, with_screw=True, wrt_world=True)
        
        while not self.viewer.closed:
            self.scene.step()
            self.scene.update_render()
            self.viewer.render() 
    
    def get_ee_pose(self):
        # 获取ee_pos和ee_quat
        ee_link = self.robot.get_links()[9] # panda_hand
        # print(ee_link.name)
        ee_pos = ee_link.get_pose().p
        ee_quat = ee_link.get_pose().q
        return ee_pos, ee_quat # numpy array
    def manipulate_franka(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Keyboard Control")
        pos_scale_factor = 0.05
        
        while not self.viewer.closed:
            target_pos, target_quat = self.get_ee_pose()
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
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_DOWN:
                        delta_pos = np.array([-1, 0, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_LEFT:
                        delta_pos = np.array([0, 1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_RIGHT:
                        delta_pos = np.array([0, -1, 0])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_KP_PLUS:
                        delta_pos = np.array([0, 0, 1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_KP_MINUS:
                        delta_pos = np.array([0, 0, -1])
                        target_pos += delta_pos * pos_scale_factor
                        target_pose = mplib.Pose(p=target_pos, q=target_quat)
                        self.move_to_pose(target_pose, with_screw=True, wrt_world=True)
                    elif event.key == pygame.K_KP0: 
                        self.close_gripper()
                    elif event.key == pygame.K_KP1:  
                        self.open_gripper()
                    continue

            self.scene.step()
            self.scene.update_render()
            self.viewer.render() 
    
if __name__ == '__main__':
    demo = PlanningDemo()
    # demo.open_drawer()d
    demo.manipulate_franka()
