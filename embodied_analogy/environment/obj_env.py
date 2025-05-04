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
    
    def randomize_obj(self, cfg):
        """
        对于 obj_cfg 中的 pose 进行随机化
        根据 tack_cfg 中的 open/close 以及 delta 值, 计算物体的 active_link 的初始状态的范围, 并随机选取一个值进行初始化
        """
        # TODO: 如果改为随机的话, 需要保证 ManipuleEnv 调用的这个函数完全从 cfg 中读取位姿势, 而不是随机生成
        # self.obj_init_pos_angle_low = -0.4
        # self.obj_init_pos_angle_high = -0.4
        # self.obj_init_rot_low = -0.2
        # self.obj_init_rot_high = 0.2
        # self.obj_init_dis_low = 0.5
        # self.obj_init_dis_high = 0.6
        # self.obj_init_height_low = 0.0
        # self.obj_init_height_high = 0.0
        # self.obj_init_dof_low = 0.0
        # self.obj_init_dof_high = 0.0
        
        self.obj_init_pos_angle_low = -0.
        self.obj_init_pos_angle_high = 0.
        self.obj_init_zrot_low = -0.1
        self.obj_init_zrot_high = -0.
        
        self.obj_init_xrot_low = -0.05
        self.obj_init_xrot_high = 0.05
        
        self.obj_init_dis_low = 0.5
        self.obj_init_dis_high = 0.6
        self.obj_init_height_low = 0.0
        self.obj_init_height_high = 0.0
        # self.obj_init_dof_low = 0.0
        # self.obj_init_dof_high = 0.0
        
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

        def randomize_pose(ang_low, ang_high, zrot_low, zrot_high, xrot_low, xrot_high, dis_low, dis_high, height_low, height_high) :

            ang = np.random.uniform(ang_low, ang_high)
            zrot = np.random.uniform(zrot_low, zrot_high)
            xrot = np.random.uniform(xrot_low, xrot_high)
            dis = np.random.uniform(dis_low, dis_high)
            height = np.random.uniform(height_low, height_high)

            p0 = sapien.Pose(p=[dis, 0, height])
            r0 = sapien.Pose(q=axis_angle_to_quat([0,0,1], ang)) # 绕原点旋转
            r1 = sapien.Pose(q=axis_angle_to_quat([0,0,1], zrot)) # 原地旋转
            r2 = sapien.Pose(q=axis_angle_to_quat([1,0,0], xrot)) # 原地旋转
            
            p1 = r0 * p0 * r1 * r2
            return p1
        
        path = os.path.join(ASSET_PATH, cfg["data_path"])
        bbox_path = os.path.join(path, "bounding_box.json")
        with open(bbox_path, "r") as f:
            bbox = json.load(f)
        
        sapien_pose = randomize_pose(
            self.obj_init_pos_angle_low,
            self.obj_init_pos_angle_high,
            self.obj_init_zrot_low,
            self.obj_init_zrot_high,
            self.obj_init_xrot_low,
            self.obj_init_xrot_high,
            self.obj_init_dis_low - bbox["min"][2]*0.75,
            self.obj_init_dis_high - bbox["min"][2]*0.75,
            self.obj_init_height_low - bbox["min"][1]*0.75,
            self.obj_init_height_high - bbox["min"][1]*0.75
        )
        
        cfg.update({
            "load_pose": sapien_pose.p.tolist(),
            "load_quat": sapien_pose.q.tolist(),
            "load_scale": 1,
        })
        return sapien_pose
    
    def load_object(self, obj_cfg, visualize=False):
        print("Loading Object ...")
        # 首先获取 obj 的 pose
        sapien_pose = self.randomize_obj(obj_cfg)
        
        # index = obj_cfg["index"]
        scale = obj_cfg["load_scale"]
        self.active_joint_idx = int(obj_cfg["joint_index"])
        self.active_joint_name = obj_cfg["active_joint_name"]
        active_link_name = obj_cfg["active_link_name"]
        
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
        