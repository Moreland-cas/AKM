"""Simple example to control the gripper."""

# %%
# import time


# from crisp_py.gripper.gripper import make_gripper

# gripper = make_gripper("gripper_franka")
# gripper.wait_until_ready()


# # %%
# freq = 1.0
# rate = gripper.node.create_rate(freq)
# t = 0.0
# while t < 10.0:
#     print(gripper.value)
#     print(gripper.torque)
#     rate.sleep()
#     t += 1.0 / freq

# # %%
# gripper.value

# gripper.set_target(1.0)
# time.sleep(3.0)
# gripper.set_target(0.0)
# time.sleep(3.0)

# for _ in range(6):
#     gripper.set_target(1.0)
#     time.sleep(3.0)
#     gripper.set_target(0.0)
#     time.sleep(3.0)



from crisp_py.gripper import Gripper, GripperConfig

# config = GripperConfig.from_yaml(path="...")  
config = GripperConfig(
    min_value=0.0,
    max_value=1.0,
    command_topic="/gripper_position_controller/commands",
    joint_state_topic="/joint_states",
)  
gripper = Gripper(gripper_config=config, namespace="/gripper")  
gripper.wait_until_ready()  

print(gripper.value)

gripper.open()
gripper.close()
gripper.set_target(0.5)