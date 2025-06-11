# dynamic label
BACKGROUND_LABEL = 0
STATIC_LABEL = 1
MOVING_LABEL = 2
UNKNOWN_LABEL = 3

# visualization
ACTIVATE_NAPARI=False

# project path
ASSET_PATH="/home/zby/Programs/Embodied_Analogy/assets"
PROJECT_ROOT="/home/zby/Programs/Embodied_Analogy"

# anygrasp
RUN_REMOTE_ANYGRASP=False

# SEED
SEED=666

# test cfg generation
# NUM_EXP_PER_SETTING=1
PRISMATIC_JOINT_MAX_RANGE=0.4   # m
REVOLUTE_JOINT_MAX_RANGE=40     # degree

# PRISMATIC_TEST_JOINT_DELTAS=(0.1, 0.15, 0.2)
# REVOLUTE_TEST_JOINT_DELTAS=(10, 20, 30)

"""
以下这个先不要，因为我们现在是对于所有的数据进行一个跑实验，有异常直接退出的方式
那些直接退出的数据, 我们就按照 goal_delta 计算误差？或者按照退出前的最后状态计算
# explore (for filtering valid explore traj)
EXPLORE_PRISMATIC_VALID=0.09    # m
EXPLORE_REVOLUTE_VALID=5.7  # degree

# reconstruct (Reconstruct error under this threshs is considered valid)
RECON_PRISMATIC_VALID=0.05    # m
RECON_REVOLUTE_VALID=10  # degree
"""
