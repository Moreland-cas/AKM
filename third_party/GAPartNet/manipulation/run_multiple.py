import sys, os
import argparse

# add args
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
#parser.add_argument('--f', type=str, default="python reconstruction/mesh_reconstruction.py")
parser.add_argument('--f', type=str, default="python rotation_clip.py")
#parser.add_argument('--f', type=str, default="python overall_clip.py")



args = parser.parse_args()

for i in range(args.n):
    os.system(args.f)
    # os.system("rm /home/haoran/Projects/ObjectPlacement/output/gym_outputs_task_gen_ycb_0229/*/*/*/*.ply")