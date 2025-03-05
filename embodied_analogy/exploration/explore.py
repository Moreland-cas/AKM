import numpy as np
from embodied_analogy.utility.utils import (
    depth_image_to_pointcloud,
    camera_to_world,
    initialize_napari,
    image_to_camera,
    visualize_pc
)
initialize_napari()
from embodied_analogy.pipeline.manipulate import ManipulateEnv

class ExploreEnv(ManipulateEnv):
    def __init__(self, obj_config):
        super().__init__(obj_config)
    
    def explore_step(self):
        self.base_step()
        # TODO:加上录制数据
    def explore(
        self, 
        instruction,  
        reserved_distance=0.05,
        visualize=False      
    ):
        from embodied_analogy.exploration.ram_proposal import get_ram_proposal, lift_ram_affordance
        self.base_step()
        rgb_np, depth_np, _, _ = self.capture_rgbd()
        
        # 先进行一个 proposal 提出
        contact_uv, contact_dir_2d, obj_mask = get_ram_proposal(
            query_rgb=rgb_np,
            instruction=instruction,
            data_source="droid", # TODO
            visualize=visualize
        )
        contact_3d_c = image_to_camera(
            uv=contact_uv[None],
            depth=np.array(depth_np[contact_uv[1], contact_uv[0]])[None],
            K=self.camera_intrinsic,
        )[0] # 3
        contact_3d_w = camera_to_world(contact_3d_c[None], self.camera_extrinsic)[0]
        # 在这一步保存一下中间输出
        
        # 这个输出的是在相机坐标系下的(Tgrasp2c), 需要转换到世界坐标系下
        sorted_grasps_c, best_dir_3d_c = lift_ram_affordance(
            K=self.camera_intrinsic, 
            query_rgb=rgb_np, 
            query_mask=obj_mask, 
            query_depth=depth_np, 
            contact_uv=contact_uv, 
            contact_dir_2d=contact_dir_2d, 
            visualize=visualize
        )
        Tw2c = self.camera_extrinsic
        Tc2w = np.linalg.inv(Tw2c)
        sorted_grasps = sorted_grasps_c.transform(Tc2w) # Tgrasp2w
        best_dir_3d = camera_to_world(best_dir_3d_c[None], Tw2c)[0] # 3
        
        # 根据 best grasp 得到 pre_ph_grasp 和 ph_grasp 的位姿
        pc_collision_c = depth_image_to_pointcloud(depth_np, obj_mask, self.camera_intrinsic) # N, 3
        pc_collision_w = camera_to_world(pc_collision_c, Tw2c)
        # visualize_pc(pc_collision_w, contact_point=contact_3d_w, post_contact_dirs=best_dir_3d[None])
        
        for grasp in sorted_grasps:
            # visualize_pc(pc_collision_w, grasp=grasp)
            grasp = self.get_rotated_grasp(grasp, axis_out_w=best_dir_3d)
            # visualize_pc(pc_collision_w, grasp=grasp)
            
            Tph2w = self.anyGrasp2ph(grasp=grasp)
            
            result_test = self.plan_path(target_pose=Tph2w, wrt_world=True)
            if not result_test:
                continue
            
            Tph2w_pre = self.get_translated_ph(Tph2w, -reserved_distance)
            
            # array([-1.40497511,  0.79039853,  0.1942093 ])
            # 实际执行到该 proposal, 并在此过程中录制数据
            self.exploration_data = {}
            self.step = self.explore_step
            
            self.planner.update_point_cloud(pc_collision_w)
            result_pre = self.plan_path(target_pose=Tph2w_pre, wrt_world=True)
            if not result_pre:
                continue
            
            # visualize_pc(pc_collision_w, grasp=grasp)
            self.follow_path(result_pre)
            print("reach pre-grasp pose")
            
            self.open_gripper()
            self.planner.update_point_cloud(np.array([[0, 0, -1]]))
            self.move_forward(reserved_distance)
            self.close_gripper()
            
            # 之后再根据 best_dir_3d 移动一定距离 (1dm)
            self.move_along_axis(moving_direction=best_dir_3d, moving_distance=0.05)
            
            # 打开 gripper, 并返回初始位置
            self.open_gripper()
            self.move_forward(-reserved_distance)
            self.reset_franka_arm()
            
            break
    
        # 录制完成, 开始处理
        self.step = self.base_step 
        while True:
            self.base_step()
        

if __name__ == "__main__":
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        "active_joint": "joint_2"
    }
    exploreEnv = ExploreEnv(obj_config=obj_config)
    exploreEnv.explore(
        instruction="open the drawer",  
        reserved_distance=0.05,
        visualize=False
    )