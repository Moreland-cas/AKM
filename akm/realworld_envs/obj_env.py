import logging

from akm.realworld_envs.robot_env import RobotEnv
from akm.representation.obj_repr import Obj_repr


class ObjEnv(RobotEnv):
    def __init__(self, cfg):        
        super().__init__(cfg)
        
        self.obj_env_cfg = cfg["obj_env_cfg"]
        self.obj_description = self.obj_env_cfg["obj_description"]
        self.load_joint_state = self.obj_env_cfg["load_joint_state"]
        self.load_object(self.obj_env_cfg)

    def capture_frame(self, visualize=False, robot_mask=True):
        frame = super().capture_frame(visualize=False, robot_mask=robot_mask)
        if visualize:
            frame.visualize()
        return frame
    
    def load_object(self, obj_cfg):
        self.logger.log(logging.INFO, f"Loading Object: \n{obj_cfg}")
        self.obj_repr = Obj_repr()
        self.obj_repr.gt_joint_dict = None
        self.obj_repr.setup_logger(self.logger)
        
    def set_active_joint_state(self, joint_state):
        input(f'Please set joint state to {joint_state} before type in anything: ')
        
    def get_active_joint_state(self):
        joint_state = float(input(f'Please measure the actual joint state and type in: '))
        return joint_state
