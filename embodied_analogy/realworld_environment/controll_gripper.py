from franky import *
import math
from scipy.spatial.transform import Rotation as R


robot = Robot("172.16.0.2")  # Replace this with your robot's IP
gripper = Gripper("172.16.0.2")
gripper.open(speed=0.04)

robot.set_cartesian_impedance([100, 100, 100, 20, 20, 20])

robot.set_joint_impedance([600.0, 600.0, 600.0, 200.0, 150.0, 80.0, 20.0])
robot.recover_from_errors()
# Move the fingers to a specific width (5cm)
gripper.open(speed=0.04)
# Grasp an object of unknown width
success = gripper.grasp(0.0, speed=0.02, force=1, epsilon_outer=1.0)
robot.join_motion()


dx = 0.0
dy = 0.0
dz = -0.05
quat = R.from_euler("xyz", [0*math.pi, 0*math.pi, 0*math.pi]).as_quat()
motion = CartesianMotion(Affine([dx, dy, dz], quat), ReferenceType.Relative)
robot.move(motion, asynchronous=True)
try:
    robot.join_motion()
except Exception as e:
    print(e)