import math
from franky import *
import numpy as np
from scipy.spatial.transform import Rotation as R

robot = Robot("172.16.0.2")  # Replace this with your robot's IP

def get_eepose():
    cartesian_state = robot.current_cartesian_state
    robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
    ee_pose = robot_pose.end_effector_pose
    quat = ee_pose.quaternion
    R_matrix = R.from_quat(quat, scalar_first=False).as_matrix()
    return ee_pose.translation, R_matrix

def get_qpos():
    qpos_7 = robot.current_joint_state.position
    qpos_9 = np.concatenate([qpos_7, [0, 0]])
    return qpos_9
# array([-0.75872018, -0.67371757,  1.13502274, -2.49747217, -0.87193153,
        # 1.75960374,  1.32661956,  0.        ,  0.        ])
# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
# robot.relative_dynamics_factor = 0.05
robot.relative_dynamics_factor = 0.5
robot.recover_from_errors()
# Move the robot 20cm along the relative X-axis of its end-effector
dx = 0.0
dy = 0.0
dz = 0.1
quat = R.from_euler("xyz", [0*math.pi, 0*math.pi, 0*math.pi]).as_quat()
motion = CartesianMotion(Affine([dx, dy, dz], quat), ReferenceType.Relative)
robot.move(motion)

motion_inv = CartesianMotion(Affine([-dx, -dy, -dz]), ReferenceType.Relative)
robot.move(motion_inv)


m_jp1 = JointMotion([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])
m_jp2 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint([0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8]),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])
robot.move(m_jp1)

m_jp3 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint(
        JointState(
            position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
            velocity=[0.1, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0])),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])


"""
抓取箱子的 qpos
array([-0.79153061, -0.05432596,  0.36339963, -2.78340408, -1.95813945,
        1.75718034,  1.98793949,  0.        ,  0.        ])
"""