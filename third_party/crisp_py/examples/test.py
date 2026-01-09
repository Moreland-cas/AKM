"""A simple example to demonstrate gravity compensation mode."""

# %%
from crisp_py.robot import make_robot
from crisp_py.robot import Robot
import time
import numpy as np


from scipy.spatial.transform import Rotation
from crisp_py.utils.geometry import Pose



# robot = Robot()
robot = make_robot("fr3")
robot.wait_until_ready()

# robot.home()
# %%
# robot.cartesian_controller_parameters_client.load_param_config(
#     file_path="config/control/gravity_compensation.yaml"
# )
robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
# robot.controller_switcher_client.switch_controller("joint_impedance_controller")

robot.cartesian_controller_parameters_client.load_param_config(
    # file_path="/home/zby/Programs/crisp_py/crisp_py/config/control/default_cartesian_impedance.yaml"
    file_path="/home/zby/Programs/crisp_py/crisp_py/config/control/soft_cartesian_impedance.yaml"
)

init_ee_pose = robot.end_effector_pose

# 方向：四元数 (x, y, z, w) -> 单位旋转（无旋转）
orientation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
target_ee_pose = init_ee_pose + Pose(position=np.array([0, 0, -0.1]), orientation=orientation)

# import franky 
from franky import Gripper
gripper = Gripper("172.16.0.2")

# gripper.move(width=0.0, speed=0.01)

i = 0
while True:
    if i % 2 == 0:
        # robot.move_to(pose=target_ee_pose)
        gripper.move(width=0.08, speed=0.01)
    else:
        # robot.move_to(pose=init_ee_pose)
        gripper.move(width=0.0, speed=0.01)
    time.sleep(1)
    i += 1
    
# robot.home()
# or if available
# robot.controller_switcher_client.switch_controller("gravity_compensation")
