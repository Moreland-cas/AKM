import os
import logging
import json
import numpy as np
import sapien.core as sapien
from akm.realworld_environment.robot_env import RobotEnv
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

    def capture_frame(self, visualize=False, robot_mask=True):
        frame = super().capture_frame(visualize=False, robot_mask=robot_mask)
        # 在这里获得 gt joint state, 并进行保存到 frame 中
        # frame.gt_joint_state = self.get_active_joint_state()
        if visualize:
            frame.visualize()
        return frame
    
    def load_object(self, obj_cfg):
        self.logger.log(logging.INFO, f"Loading Object: \n{obj_cfg}")
        self.obj_repr = Obj_repr()
        self.obj_repr.gt_joint_dict = None
        self.obj_repr.setup_logger(self.logger)
        
    def set_active_joint_state(self, joint_state):
        """
        对 active joint state 进行设置
        """
        input(f'Please set joint state to {joint_state} before type in anything: ')
        
    def get_active_joint_state(self):
        """
        获取我们关心的关节的状态值
        """
        joint_state = float(input(f'Please measure the actual joint state and type in: '))
        return joint_state


if __name__ == "__main__":
    test_cfgs_path = "/home/zby/Programs/Embodied_Analogy/scripts/test_cfgs.json"
    with open(test_cfgs_path, 'r', encoding='utf-8') as f:
        test_cfgs = json.load(f)
        
    for test_cfg in test_cfgs:
        objEnv = ObjEnv(test_cfg)
        for i in range(200):
            objEnv.step()
        objEnv.delete()
        # 这里写 del 没用, 必须用 objEnv.viewer.close()
        # del objEnv
        