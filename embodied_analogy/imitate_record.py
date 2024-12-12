import sapien
import mplib
from embodied_analogy.base_env import BaseEnv
from embodied_analogy.process_record import RecordDataReader
from embodied_analogy.utils import draw_red_dot, uv_to_camera, camera_to_world
from embodied_analogy.image_matching import match_points
from PIL import Image
import numpy as np
import transforms3d as t3d

class ImitateEvc(BaseEnv):
    def __init__(
            self,
            phy_timestep=1/250.,
            record_path_prefix="/home/zby/Programs/Embodied_Analogy/assets/recorded_data",
            record_file_name="/2024-12-12_11-23-38.npz"
        ):
        super().__init__(phy_timestep)
        
        # self.load_articulated_object(index=100015, pose=[0.4, 0.4, 0.2], scale=0.4)
        self.load_articulated_object(index=100051, pose=[0.4, 0.4, 0.2], scale=0.2)
        self.setup_camera()
        
        self.DataReader = RecordDataReader(record_path_prefix, record_file_name)
        self.DataReader.process_data()
        
    def spawn_cube(self, pose):
        # cube
        builder = self.scene.create_actor_builder()
        
        # builder.use_density = False
        builder.add_box_collision(half_size=[0, 0, 0])
        builder.add_box_visual(half_size=[0.01, 0.01, 0.01], material=[1, 0, 0])
        cube = builder.build(name='cube')
        cube.set_pose(sapien.Pose(pose))
        
        # cube.get_links()[0].set_collision_enabled(False)

        # 取消重力
        # cube.set_gravity(False)

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        print("n_step:", n_step)
        for i in range(n_step):  
            for _ in range(9):
                qf = self.robot.compute_passive_force(
                    gravity=True, 
                    coriolis_and_centrifugal=True)
                self.robot.set_qf(qf)
                for j in range(7):
                    self.active_joints[j].set_drive_target(result['position'][i][j])
                    self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
                # for joint in self.active_joints[-2:]:
                #     if self.after_try_to_close:
                #         print("try close")
                #         joint.set_drive_target(-0.1)
                #     else:
                #         print("try open")
                #         joint.set_drive_target(0.4)
                self.step()
            
    def imitate_from_record(self):
        # 先过10个step让机械手臂复原
        for i in range(200):
            self.step()
            
        # 获取target场景初始位置的rgb图
        target_img_np, target_depth_np = self.capture_rgb_depth()
        target_img_pil = Image.fromarray(target_img_np)
        
        source_img_pil = self.DataReader.source_img
        source_u, source_v = self.DataReader.first_cp_2d
        
        target_uvs, target_probs, similarity_map = match_points(
            source_img_pil, 
            target_img_pil, 
            (source_u, source_v), 
            top_k=500, 
            max_return=500,
            resize=798, 
            device="cuda",
            is_pil=True,
            show_matching=False,
            nms_threshold=0.002
        )
        
        # for i in range(len(target_uvs)):
            # if i != 99:
                # continue
            # target_uv = target_uvs[i]
            # img = draw_red_dot(target_img_pil, target_uv[0], target_uv[1], radius=1)
            # img.show()
            # img.save(self.asset_prefix + f"/tmp/{i}.png")
            # break
        
        # 使用 segmentation map 进行过滤
        similarity_map = None # H, W, 0-1之间
        # self.capture_segmentation()
        
        # 根据深度图找到三维空间点
        target_u, target_v = target_uvs[6]
        # target_u, target_v = target_uvs[0]
        depth_h, depth_w = target_depth_np.shape
        depth = target_depth_np[int(target_v * depth_h), int(target_u * depth_w)]
        K = self.camera.get_intrinsic_matrix()
        
        point_camera = uv_to_camera(target_u, target_v, depth, K, depth_w, depth_h)
        extrinsic_matric = self.camera.get_extrinsic_matrix() # [3, 4] Tw2c
        contact_pos = camera_to_world(point_camera, extrinsic_matric)
        self.spawn_cube(contact_pos)
        
        contact_pos += np.array([0, 0, 0.06])
        self.open_gripper()
        
        target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
        target_pose = mplib.Pose(p=contact_pos, q=target_quat)
        self.move_to_pose(target_pose, wrt_world=True)
        
        contact_pos -= np.array([0, 0, 0.06])
        target_quat = t3d.euler.euler2quat(np.deg2rad(0), np.deg2rad(180), np.deg2rad(90), axes="syxz")
        target_pose = mplib.Pose(p=contact_pos, q=target_quat)
        self.move_to_pose(target_pose, wrt_world=True)
        # a = input("")
        self.close_gripper()
        
        ph_pos = self.DataReader.panda_hand_pos
        ph_quat = self.DataReader.panda_hand_quat
        
        cur_pos, cur_quat = self.get_ee_pose()
        for i in range(len(ph_pos)):
            self.move_to_pose(mplib.Pose(p=cur_pos + ph_pos[i], q=ph_quat[i]), wrt_world=True)
        
        while not self.viewer.closed:
            self.step()
        
        # 获取左图 contact point 在右图的对应坐标
        

if __name__ == '__main__':
    demo = ImitateEvc()
    demo.imitate_from_record()
    