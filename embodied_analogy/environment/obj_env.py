import numpy as np
import sapien.core as sapien
from embodied_analogy.environment.robot_env import RobotEnv
from embodied_analogy.representation.obj_repr import Obj_repr
from embodied_analogy.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)

class ObjEnv(RobotEnv):
    def __init__(
            self,
            base_cfg={
                "phy_timestep": 1/250.,
                "planner_timestep": None,
                "use_sapien2": True 
            },
            robot_cfg={},
            obj_cfg={
                "index": 44962,
                "scale": 0.8,
                "pose": [1.0, 0., 0.5],
                "active_link": "link_2",
                "active_joint": "joint_2"
            }
        ):        
        super().__init__(
            base_cfg=base_cfg,
            robot_cfg=robot_cfg,
        )
        self.load_object(obj_cfg)
        # 随机初始化物体对应 joint 的状态
        cur_joint_state = self.asset.get_qpos()
        active_joint_names = [joint.name for joint in self.asset.get_active_joints()]
        initial_state = []
        for i, joint_name in enumerate(active_joint_names):
            if joint_name == obj_cfg["active_joint"]:
                limit = self.asset.get_active_joints()[i].get_limits() # (2, )
                # initial_state.append(0.1)
                initial_state.append(np.deg2rad(0))
            else:
                initial_state.append(cur_joint_state[i])
        self.asset.set_qpos(initial_state)
        
        self.obj_repr = Obj_repr()
    
    def capture_frame(self, visualize=False):
        frame = self.capture_frame(visualize=False)
        # TODO: 在这里获得 gt joint state, 并进行保存到 frame 中
        if visualize:
            frame.visualize()
        return frame
    
    def load_object(self, obj_cfg):
        index = obj_cfg["index"]
        scale = obj_cfg["scale"]
        pose = obj_cfg["pose"]
        active_link = obj_cfg["active_link"]
        
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
            if joint.get_name() == obj_cfg["active_joint"]:
                self.evaluate_joint = joint
                self.init_joint_transform = joint.get_global_pose().to_transformation_matrix() # 4, 4, Tw2c
        
        # 在这里调用一个 base step 以实际 load 物体
        self.base_step()
    def get_joint_state(self):
        cur_transform = self.evaluate_joint.get_global_pose().to_transformation_matrix()
        # Tinit2cur = Tw2cur @ Tw2init.inv
        Tinit2cur = cur_transform @ np.linalg.inv(self.init_joint_transform)
        return Tinit2cur


if __name__ == "__main__":
    # drawer
    obj_config = {
        "index": 44962,
        "scale": 0.8,
        "pose": [1.0, 0., 0.5],
        "active_link": "link_2",
        "active_joint": "joint_2"
    }
    
    # door
    # obj_config = {
    #     "index": 9288,
    #     "scale": 1.0,
    #     "pose": [0.7, 0., 0.7],
    #     "active_link": "link_2",
    #     "active_joint": "joint_0"
    # }
    
    # microwave
    # obj_config = {
    #     "index": 7221,
    #     "scale": 0.4,
    #     "pose": [0.8, 0.1, 0.6],
    #     "active_link": "link_0",
    #     "active_joint": "joint_0"
    # }
    
    obj_index = obj_config["index"]
    
    objEnv = ObjEnv(
        obj_cfg=obj_config,
    )
    while True:
        objEnv.step()
    