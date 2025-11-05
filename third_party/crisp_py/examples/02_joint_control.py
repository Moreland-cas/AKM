"""Example controlling the joints."""
import time

from crisp_py.robot import make_robot

robot = make_robot("fr3")
robot.wait_until_ready()

robot.home()
# %%
robot.controller_switcher_client.switch_controller("joint_impedance_controller")

# %%
q = robot.joint_values


for i in range(100):
    q[0] += 0.2 / 100
    robot.set_target_joint(q)
    time.sleep(0.01)

time.sleep(1.0)

robot.shutdown()
