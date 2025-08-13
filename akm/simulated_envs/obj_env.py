import os
import logging
import json
import numpy as np
import sapien.core as sapien
from akm.simulated_envs.robot_env import RobotEnv
from akm.representation.obj_repr import Obj_repr
from akm.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)
from akm.utility.constants import ASSET_PATH

class ObjEnv(RobotEnv):
    def __init__(self, cfg):        
        super().__init__(cfg)
        
        self.obj_env_cfg = cfg["obj_env_cfg"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.load_joint_state = self.obj_env_cfg["load_joint_state"]
        
        self.load_object(self.obj_env_cfg)

    def capture_frame(self, visualize=False):
        frame = super().capture_frame(visualize=False)
        # 在这里获得 gt joint state, 并进行保存到 frame 中
        frame.gt_joint_state = self.get_active_joint_state()
        if visualize:
            frame.visualize()
        return frame
    
    def load_object(self, obj_cfg, visualize=False):
        self.logger.log(logging.INFO, f"Loading Object: \n{obj_cfg}")
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
        
        if self.obj is None:
            self.logger.log(logging.ERROR, f'{ASSET_PATH}/{data_path}/mobility.urdf load None')
            raise Exception("obj asset load failed.")
        
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
                initial_states.append(obj_cfg["load_joint_state"])
            else:
                initial_states.append(0)
                joint.set_limits(np.array([[0, 0]]))
        self.obj.set_qpos(initial_states)
        
        # 在这里调用一个 base step 以实际 load 物体
        self.base_step()
        
        self.obj_repr = Obj_repr()
        self.obj_repr.setup_logger(self.logger)
        
        # 在这里要顺便把 gt_joint_param 也进行保存
        # 首先获取 parent link 的位置 
        # NOTE: parent link 一般是物体的主体部分, child link 一般是 mobing part 对应的 link
        # active_joint = self.obj.get_active_joints()[self.active_joint_idx]
        if active_link_name != self.active_joint.get_child_link().get_name():
            self.logger.log(logging.ERROR, "active_link_name is not consistent with active_joint_name!")
            raise Exception("active_link_name is not consistent with active_joint_name!")
        
        Tparent2w = self.active_joint.get_parent_link().get_pose().to_transformation_matrix() # Tparent2w
        joint_in_parent = self.active_joint.get_pose_in_parent().to_transformation_matrix()
        Tjoint2w = Tparent2w @ joint_in_parent
        # 不管是 prismatic joint 还是 revolute joint, joint_dir 都是由 joint 坐标系的 x 轴决定的
        self.obj_repr.gt_joint_dict = {
            "joint_type": obj_cfg["joint_type"],
            "joint_dir": Tjoint2w[:3, 0],
            "joint_start": Tjoint2w[:3, 3],
            "joint_states": None
        }
        
        if visualize:
            # 获取一帧的点云
            frame = self.capture_frame(visualize=False)
            frame.obj_mask = np.ones_like(frame.depth).astype(np.bool_)
            pc, colors = frame.get_obj_pc(world_frame=True)
            from akm.utility.utils import visualize_pc
            visualize_pc(
                points=pc,
                colors=colors / 255.,
                grasp=None,
                contact_point=self.obj_repr.gt_joint_dict["joint_start"],
                post_contact_dirs=[self.obj_repr.gt_joint_dict["joint_dir"]],
            )
    
    def set_active_joint_state(self, joint_state):
        """
        对 active joint state 进行设置
        """
        self.obj.set_qpos(joint_state)
        
    def get_active_joint_state(self):
        """
        获取我们关心的关节的状态值
        """
        # NOTE: 对于 RGBManip 的数据集, 只有一个关节不是 fixed, 因此直接读取就行
        # return self.obj.get_qpos()[self.active_joint_idx]
        return self.obj.get_qpos()[0]


if __name__ == "__main__":
    test_cfgs_path = "/home/zby/Programs/AKM/scripts/test_cfgs.json"
    with open(test_cfgs_path, 'r', encoding='utf-8') as f:
        test_cfgs = json.load(f)
        
    for test_cfg in test_cfgs:
        objEnv = ObjEnv(test_cfg)
        for i in range(200):
            objEnv.step()
        objEnv.delete()
        # 这里写 del 没用, 必须用 objEnv.viewer.close()
        # del objEnv
        