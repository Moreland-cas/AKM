"""A simple example to demonstrate gravity compensation mode."""

# %%
from crisp_py.robot import make_robot
from crisp_py.robot import Robot
import time

# robot = Robot()
robot = make_robot("fr3")
robot.wait_until_ready()

robot.home()
# %%
# robot.cartesian_controller_parameters_client.load_param_config(
#     file_path="config/control/gravity_compensation.yaml"
# )
# robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
robot.controller_switcher_client.switch_controller("joint_impedance_controller")
prefix = "/home/user/Programs/crisp_py/crisp_py/"
robot.cartesian_controller_parameters_client.load_param_config(
    file_path=prefix + "config/control/soft_cartesian_impedance.yaml"
    # file_path=prefix + "config/control/joint_cartesian_impedance.yaml"
    # file_path=prefix + "config/control/default_cartesian_impedance.yaml"
)

while True:
    print("Im here")
    time.sleep(1)
    
robot.home()
# or if available
# robot.controller_switcher_client.switch_controller("gravity_compensation")
