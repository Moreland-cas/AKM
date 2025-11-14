"""A simple example to demonstrate gravity compensation mode."""

# %%
from crisp_py.robot import make_robot
from crisp_py.robot import Robot
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


from scipy.spatial.transform import Rotation
from crisp_py.utils.geometry import Pose



# robot = Robot()
robot = make_robot("fr3")
robot.wait_until_ready()


from franky import Gripper
gripper = Gripper("172.16.0.2")

# gripper.move(width=0.04, speed=0.01)
# import sys
# sys.exit()
# i = 0
# while i < 3:
#     if i % 2 == 0:
#         # robot.move_to(pose=target_ee_pose)
#         gripper.move(width=0.08, speed=0.01)
#     else:
#         # robot.move_to(pose=init_ee_pose)
#         gripper.move(width=0.0, speed=0.01)
#     time.sleep(1)
#     i += 1

def switch_mode(mode="cartesian_impedance_soft"):
    assert mode in ["cartesian_impedance_soft", "cartesian_impedance_hard", "joint_impedance"]
    if mode == "cartesian_impedance_soft":
        robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        robot.cartesian_controller_parameters_client.load_param_config(
            file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/soft_cartesian_impedance.yaml"
        )
    elif mode == "cartesian_impedance_hard":
        robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        robot.cartesian_controller_parameters_client.load_param_config(
            file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/default_cartesian_impedance.yaml"
        )
    elif mode == "joint_impedance":
        robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        

def move_dz(delta=0.1, speed=0.05, mode=None):
    # 首先获取当前坐标
    cur_position = robot.end_effector_pose.position
    # print(cur_position)
    cur_Rotation = robot.end_effector_pose.orientation
    z_dir = cur_Rotation.as_matrix()[:, -1]
    # print(z_dir)
    tgt_position = cur_position + delta * z_dir
    # print(tgt_position)
    robot.move_to(position=tgt_position, speed=speed)


def safe_grasp():
    # change mode to soft impedance
    switch_mode(mode="cartesian_impedance_soft")
    
    gripper.move(width=0, speed=0.01)
    
    # 此时确定了位置
    cur_pose = robot.end_effector_pose
    
    switch_mode(mode="cartesian_impedance_hard")
    gripper.move(width=0.04, speed=0.01)
    robot.move_to(pose=cur_pose)
    gripper.move(width=0, speed=0.01)
    

def move_along_axis():
    pass


def follow_path(result):
    switch_mode("joint_impedance")
    n_step = result['position'].shape[0]
    # self.logger.log(logging.INFO, f"n_step: {n_step}")

    if n_step == 0:
        return
    
    for i in range(n_step):
        joint_target = result['position'][i].tolist() # 7
        robot.set_target_joint(joint_target)
        

def get_ee_pose(as_matrix=False):
    """
    Get the ee_pos and ee_quat (Tph2w) of the end-effector (panda_hand)
    ee_pos: np.array([3, ])
    ee_quat: np.array([4, ]), scalar_first=False, in (x, y, z, w) order
    """
    # Get the robot's cartesian state
    ee_pose = robot.end_effector_pose
    ee_pos = ee_pose.position
    ee_Rotation = ee_pose.orientation
    z_dir = R.as_matrix(ee_Rotation)[:, -1]
    
    # NOTE: crisp 返回的 ee_pose 跟 franka Desk 中定义的差 1dm
    ee_pos = ee_pos - 0.1 * z_dir
    
    # if as_matrix:
    Tph2w = np.eye(4)
    Tph2w[:3, :3] = ee_Rotation.as_matrix()
    Tph2w[:3, 3] = ee_pos
    return Tph2w

 
tgt_pose = Pose(
    position=np.array([ 0.44, -0.27,  0.44]),
    orientation=R.from_matrix([
        [ 0.58, -0.81 , 0.08],
        [-0.82 ,-0.57 , 0.10],
        [-0.04, -0.12 ,-0.99]]
    )
)

# robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
# robot.cartesian_controller_parameters_client.load_param_config(
#     file_path="/home/user/Programs/AKM/third_party/crisp_py/crisp_py/config/control/pull_cartesian_impedance.yaml"
# )
# while True:
#     time.sleep(1)
gripper.move(width=0.08, speed=0.01)
robot.home()
# input("type anything: ") 
robot.shutdown()
