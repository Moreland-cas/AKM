import os
import json
import numpy as np
import sapien.core as sapien
from embodied_analogy.environment.robot_env import RobotEnv
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)
from embodied_analogy.utility.constants import ASSET_PATH

class ObjEnv(RobotEnv):
    def __init__(
            self,
            cfg
        ):        
        super().__init__(cfg)
        self.load_object(cfg)

    def capture_frame(self, visualize=False):
        frame = super().capture_frame(visualize=False)
        # TODO: 在这里获得 gt joint state, 并进行保存到 frame 中
        if visualize:
            frame.visualize()
        return frame
    
    def load_object(self, obj_cfg, visualize=False):
        print("Loading Object ...")
        self.active_joint_idx = int(obj_cfg["joint_index"])
        self.active_joint_name = obj_cfg["active_joint_name"]
        active_link_name = obj_cfg["active_link_name"]
        
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.scale = obj_cfg["load_scale"]
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
                active_link_name: {
                    "material": "gripper",
                    "density": 1.0,
                }
            }
        }
        load_config = parse_urdf_config(urdf_config, self.scene)
        check_urdf_config(load_config)
        
        data_path = obj_cfg["data_path"]
        self.obj = loader.load(
            filename=f"{ASSET_PATH}/{data_path}/mobility.urdf",
            config=load_config
        )
        assert self.obj is not None
        
        # 改为 load obj_cfg 中的 load_pose, load_quat, load_scale
        sapien_pose = sapien.Pose(p=obj_cfg["load_pose"], q=obj_cfg["load_quat"])
        self.obj.set_root_pose(sapien_pose)
        
        # 设置物体关节的参数, 把回弹关掉
        initial_states = []
        for i, joint in enumerate(self.obj.get_active_joints()):
            joint.set_drive_property(stiffness=0, damping=0.1)
            
            # 在这里判断当前的 joint 是不是我们关注的需要改变状态的关节, 如果是, 则初始化读取状态的函数, 以及当前状态
            if joint.get_name() == obj_cfg["active_joint_name"]:
                self.active_joint = joint
                initial_states.append(obj_cfg["init_joint_state"])
            else:
                initial_states.append(0)
                joint.set_limits(np.array([[0, 0]]))
        self.obj.set_qpos(initial_states)
        
        # 在这里调用一个 base step 以实际 load 物体
        self.base_step()
        
        self.obj_repr = Obj_repr()
        # 在这里要顺便把 gt_joint_param 也进行保存
        # 首先获取 parent link 的位置 
        # NOTE: parent link 一般是物体的主体部分, child link 一般是 mobing part 对应的 link
        # active_joint = self.obj.get_active_joints()[self.active_joint_idx]
        if active_link_name != self.active_joint.get_child_link().get_name():
            print("active_link_name is not consistent with active_joint_name, there might be an error!")
        
        Tparent2w = self.active_joint.get_parent_link().get_pose().to_transformation_matrix() # Tparent2w
        joint_in_parent = self.active_joint.get_pose_in_parent().to_transformation_matrix()
        Tjoint2w = Tparent2w @ joint_in_parent
        # 不管是 prismatic joint 还是 revolute joint, joint_dir 都是由 joint 坐标系的 x 轴决定的
        self.obj_repr.gt_joint_dict = {
            "joint_type": self.cfg["joint_type"],
            "joint_dir": Tjoint2w[:3, 0],
            "joint_start": Tjoint2w[:3, 3],
            "joint_states": None
        }
        
        if visualize:
            # 获取一帧的点云
            frame = self.capture_frame(visualize=False)
            frame.obj_mask = np.ones_like(frame.depth).astype(np.bool_)
            pc, colors = frame.get_obj_pc(world_frame=True)
            from embodied_analogy.utility.utils import visualize_pc
            visualize_pc(
                points=pc,
                colors=colors / 255.,
                grasp=None,
                contact_point=self.obj_repr.gt_joint_dict["joint_start"],
                post_contact_dirs=[self.obj_repr.gt_joint_dict["joint_dir"]],
            )
            
    def get_active_joint_state(self):
        """
        获取我们关心的关节的状态值
        """
        # TODO: 如果是对于 RGBManip 的数据集, 只有一个关节不是 fixed, 因此直接读取就行
        # return self.obj.get_qpos()[self.active_joint_idx]
        return self.obj.get_qpos()[0]


if __name__ == "__main__":
    
    cfg = {
    "phy_timestep": 0.004,
    "planner_timestep": 0.01,
    "use_sapien2": True,
    "record_fps": 30,
    "pertubation_distance": 0.1,
    "max_tries": 10,
    "update_sigma": 0.05,
    "reserved_distance": 0.05,
    "logs_path": "/home/zby/Programs/Embodied_Analogy/assets/logs",
    "run_name": "test_explore",
    "valid_thresh": 0.5,
    "instruction": "open the cabinet",
    "obj_description": "cabinet",
    "joint_type": "revolute",
    "obj_index": "44781",
    "joint_index": "0",
    "init_joint_state": 0.2,
    "asset_path": "/home/zby/Programs/Embodied_Analogy/assets/dataset/one_door_cabinet/44781_link_0",
    "active_link_name": "link_0",
    "active_joint_name": "joint_0",
    "load_pose": [
        0.8304692506790161,
        0.0,
        0.5423032641410828
    ],
    "load_quat": [
        1.0,
        0.0,
        0.0,
        0.0
    ],
    "load_scale": 1
}
    objEnv = ObjEnv(cfg)
    while True:
        objEnv.step()
        