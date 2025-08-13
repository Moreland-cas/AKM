import logging
import numpy as np
import sapien.core as sapien

from akm.utility.constants import ASSET_PATH
from akm.representation.obj_repr import Obj_repr
from akm.simulated_envs.robot_env import RobotEnv
from akm.utility.sapien_utils import (
    parse_urdf_config,
    check_urdf_config,
)


class ObjEnv(RobotEnv):
    def __init__(self, cfg):        
        super().__init__(cfg)
        
        self.obj_env_cfg = cfg["obj_env_cfg"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.load_joint_state = self.obj_env_cfg["load_joint_state"]
        
        self.load_object(self.obj_env_cfg)

    def capture_frame(self, visualize=False):
        frame = super().capture_frame(visualize=False)
        # Get the gt joint state here and save it to the frame
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
        
        # Change to load_pose, load_quat, load_scale in load obj_cfg
        sapien_pose = sapien.Pose(p=obj_cfg["load_pose"], q=obj_cfg["load_quat"])
        self.obj.set_root_pose(sapien_pose)
        
        # Set the parameters of the object joint and turn off the rebound
        initial_states = []
        for i, joint in enumerate(self.obj.get_active_joints()):
            joint.set_drive_property(stiffness=0, damping=0.1)
            
            # Here we determine whether the current joint is the joint we are concerned about and need to change the state. 
            # If so, initialize the function to read the state and the current state
            if joint.get_name() == obj_cfg["active_joint_name"]:
                self.active_joint = joint
                initial_states.append(obj_cfg["load_joint_state"])
            else:
                initial_states.append(0)
                joint.set_limits(np.array([[0, 0]]))
        self.obj.set_qpos(initial_states)
        
        # Call a base step here to actually load the object
        self.base_step()
        
        self.obj_repr = Obj_repr()
        self.obj_repr.setup_logger(self.logger)
        
        # First, get the location of the parent link.
        # NOTE: The parent link is usually the main part of the object, and the child link is usually the link corresponding to the mobing part.
        if active_link_name != self.active_joint.get_child_link().get_name():
            self.logger.log(logging.ERROR, "active_link_name is not consistent with active_joint_name!")
            raise Exception("active_link_name is not consistent with active_joint_name!")
        
        Tparent2w = self.active_joint.get_parent_link().get_pose().to_transformation_matrix() # Tparent2w
        joint_in_parent = self.active_joint.get_pose_in_parent().to_transformation_matrix()
        Tjoint2w = Tparent2w @ joint_in_parent
        # Whether it is a prismatic joint or a revolute joint, joint_dir is determined by the x-axis of the joint coordinate system
        self.obj_repr.gt_joint_dict = {
            "joint_type": obj_cfg["joint_type"],
            "joint_dir": Tjoint2w[:3, 0],
            "joint_start": Tjoint2w[:3, 3],
            "joint_states": None
        }
        
        if visualize:
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
        setup active joint state
        """
        self.obj.set_qpos(joint_state)
        
    def get_active_joint_state(self):
        """
        Get the status value of the joint we care about
        """
        # NOTE: For the RGBManip dataset, only one joint is not fixed, so just read it directly
        return self.obj.get_qpos()[0]
