import os
import json
import random
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
            task_cfg={
                "instruction": "open the drawer",
                "obj_description": "drawer",
                "delta": 0.15,
                "obj_cfg": {
                    "index": 44962,
                    "asset_path": None, 
                    "scale": 0.8,
                    "active_link_name": "link_2",
                    "active_joint_name": "joint_2",
                    "active_joint_limit": None,
                },
            }
        ):        
        super().__init__(
            base_cfg=base_cfg,
            robot_cfg=robot_cfg,
        )
        obj_cfg = task_cfg["obj_cfg"]
        self.load_object(obj_cfg)
        
        # 然后在这里记录下 gt_start_joint_state 和 gt_delta
        self.gt_start_joint_state = self.get_active_joint_state()
        self.gt_delta = task_cfg["delta"]
        
    def change_obj(self, obj_cfg):
        '''
        Change the object in the env
        '''

        if self.renderer_type == 'sapien' :
            self.scene.remove_articulation(self.obj)
            if config is None :
                self._add_object(*self._generate_object_config())
            else :
                self._add_object(*self._load_object_config(config))
        elif self.renderer_type == 'client' :
            # remove_articulation not supported in client
            # So only change the randomization params
            path, dof, pose = self._generate_object_config()
            self.obj.set_qpos(dof)
            self.obj.set_root_pose(pose)
            self.obj_root_pose = pose
            self.obj_init_dof = dof
        pass
        
    def capture_frame(self, visualize=False):
        frame = super().capture_frame(visualize=False)
        # TODO: 在这里获得 gt joint state, 并进行保存到 frame 中
        if visualize:
            frame.visualize()
        return frame
    
    def randomize_obj(self, obj_cfg):
        """
        对于 obj_cfg 中的 pose 进行随机化
        根据 tack_cfg 中的 open/close 以及 delta 值, 计算物体的 active_link 的初始状态的范围, 并随机选取一个值进行初始化
        """
        self.obj_init_pos_angle_low = -0.4
        self.obj_init_pos_angle_high = 0.4
        self.obj_init_rot_low = -0.2
        self.obj_init_rot_high = 0.2
        self.obj_init_dis_low = 0.5
        self.obj_init_dis_high = 0.6
        self.obj_init_height_low = 0.0
        self.obj_init_height_high = 0.0
        self.obj_init_dof_low = 0.0
        self.obj_init_dof_high = 0.0
        
        def axis_angle_to_quat(axis, angle):
            '''
            axis: [[x, y, z]] or [x, y, z]
            angle: rad
            return: a quat that rotates angle around axis
            '''
            axis = np.array(axis)
            shape = axis.shape
            assert(shape[-1] == 3)
            axis = axis.reshape(-1, 3)

            angle = np.array(angle)
            angle = angle.reshape(-1, 1)

            axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9)
            quat1 = np.concatenate([np.cos(angle/2), axis[:, 0:1]*np.sin(angle/2), axis[:, 1:2]*np.sin(angle/2), axis[:, 2:3]*np.sin(angle/2)], axis=-1)
            return quat1.reshape(*shape[:-1], 4)

        def randomize_pose(ang_low, ang_high, rot_low, rot_high, dis_low, dis_high, height_low, height_high) :

            ang = np.random.uniform(ang_low, ang_high)
            rot = np.random.uniform(rot_low, rot_high)
            dis = np.random.uniform(dis_low, dis_high)
            height = np.random.uniform(height_low, height_high)

            p0 = sapien.Pose(p=[dis, 0, height])
            r0 = sapien.Pose(q=axis_angle_to_quat([0,0,1], ang))
            r1 = sapien.Pose(q=axis_angle_to_quat([0,0,1], rot))

            p1 = r0 * p0 * r1
            return p1
    
        def randomize_dof(dof_low, dof_high) :
            if dof_low == 'None' or dof_high == 'None' :
                return None
            return np.random.uniform(dof_low, dof_high)
        
        path = obj_cfg["asset_path"]
        bbox_path = os.path.join(path, "bounding_box.json")
        with open(bbox_path, "r") as f:
            bbox = json.load(f)
        
        pose = randomize_pose(
            self.obj_init_pos_angle_low,
            self.obj_init_pos_angle_high,
            self.obj_init_rot_low,
            self.obj_init_rot_high,
            self.obj_init_dis_low - bbox["min"][2]*0.75,
            self.obj_init_dis_high - bbox["min"][2]*0.75,
            self.obj_init_height_low - bbox["min"][1]*0.75,
            self.obj_init_height_high - bbox["min"][1]*0.75
        )
        
        dof = randomize_dof(
            self.obj_init_dof_low,
            self.obj_init_dof_high
        )
        
        obj_cfg.update({
            "pose": pose,
            "init_joint_state": dof
        })
        return path, dof, pose
    
    def load_object(self, obj_cfg):
        # 首先获取 obj 的 pose
        self.randomize_obj(obj_cfg)
        
        # index = obj_cfg["index"]
        scale = obj_cfg["scale"]
        pose = obj_cfg["pose"]
        active_link = obj_cfg["active_link_name"]
        
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
        
        asset_path = obj_cfg["asset_path"]
        self.obj = loader.load(
            filename=f"{asset_path}/mobility.urdf",
            config=load_config
        )
        
        self.obj.set_root_pose(pose)
        
        # 设置物体关节的参数, 把回弹关掉
        initial_states = []
        for i, joint in enumerate(self.obj.get_active_joints()):
            joint.set_drive_property(stiffness=0, damping=0.1)
            
            # 在这里判断当前的 joint 是不是我们关注的需要改变状态的关节, 如果是, 则初始化读取状态的函数, 以及当前状态
            if joint.get_name() == obj_cfg["active_joint_name"]:
                initial_states.append(obj_cfg["init_joint_state"])
                self.active_joint_name = obj_cfg["active_joint_name"]
                self.active_joint_idx = i
            else:
                initial_states.append(0)
                joint.set_limits(np.array([[0, 0]]))
        self.obj.set_qpos(initial_states)
        
        # 在这里调用一个 base step 以实际 load 物体
        self.base_step()
        
        self.obj_repr = Obj_repr()
        
    def get_active_joint_state(self):
        """
        获取我们关心的关节的状态值
        """
        return self.obj.get_qpos()[self.active_joint_idx]


if __name__ == "__main__":
    task_cfg={
        "obj_cfg": {
            "index": 41083,
            "asset_path": "/home/zby/Programs/VideoTracking-For-AxisEst/downloads/dataset/one_drawer_cabinet/41083_link_2", 
            "scale": 1.,
            "active_link_name": "link_2",
            "active_joint_name": "joint_2",
            "active_joint_limit": None,
        },
    }
    
    # drawer
    # obj_config = {
    #     "index": 44962,
    #     "scale": 0.8,
    #     "pose": [1.0, 0., 0.5],
    #     "active_link": "link_2",
    #     "active_joint": "joint_2"
    # }
    
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
    
    # obj_index = obj_config["index"]
    
    objEnv = ObjEnv(
        task_cfg=task_cfg,
    )
    while True:
        objEnv.step()
        