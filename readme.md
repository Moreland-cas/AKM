<div align="center" style="font-size: 3em">
  AKM
</div>

Here’s the complete code for our paper “Active Kinematic Modelling for Precise Manipulation of Unseen Articulated Objects,” currently under review at RAL.

<!-- TODO video, paper link -->

## Quick Start
### 1. Prerequisites
- Python 3.10  
- CUDA 12.1  
- PyTorch 2.5.1  

### 2. Install Main Environment
Clone the repo and create the `AKM` Conda environment in one shot:
```bash
git clone https://github.com/Moreland-cas/AKM
cd AKM
bash scripts/prepare/create_akm_env.sh
```

(Optional) To run baseline methods, simply execute the corresponding script. They will automatically clone the `AKM` environment and create dedicated Conda environments.
```bash
# GeneralFlow
bash scripts/prepare/create_gflow_env.sh
# GAPartNet
bash scripts/prepare/create_gpnet_env.sh
```

**Important**
Make sure to update `CUDA_HOME` in the scripts to your actual CUDA path:
```bash
export CUDA_HOME=YOUR_ACTUAL_CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 
```

## Data and Checkpoints
We provide script to download the assets for simulation environment and model checkpoints of third-party methods.

The pretrained model checkpoints are stored in `/assets/ckpts` folder, as,
```
/assets/ckpts
├── anygrasp
│   └── checkpoint_detection.tar
├── fastSAM
│   ├── FastSAM-s.pt
│   ├── FastSAM-x.pt
│   └── FastSAM-X.pt
├── gapartnet
│   ├── all_best.ckpt
│   └── release.ckpt
├── generalFlow
│   └── kpst_hoi4d
│       ├── ScaleGFlow-B
│       ├── ScaleGFlow-B.tar.gz
│       ├── ScaleGFlow-L
│       ├── ScaleGFlow-L.tar.gz
│       ├── ScaleGFlow-S
│       └── ScaleGFlow-S.tar.gz
├── grounding_dino
│   └── groundingdino_swint_ogc.pth
├── gsnet
│   ├── minkuresunet_kinect.tar
│   └── minkuresunet_realsense.tar
└── sam2
    └── sam2.1_hiera_large.pt
```
the download link can be refered to the respective links.
[AnyGrasp](https://github.com/graspnet/anygrasp_sdk)
[gapartnet](https://github.com/PKU-EPIC/GAPartNet)
[generalFlow](https://github.com/michaelyuancb/general_flow)
[grounding_dino](https://github.com/IDEA-Research/GroundingDINO)
[gsnet](https://github.com/graspnet/graspness_unofficial)
[sam2](https://github.com/facebookresearch/sam2)
[fastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

The dataset used in our simulation environment is from [RGBManip](https://github.com/hyperplane-lab/RGBManip) and can be downloaded through [this_link](https://drive.google.com/file/d/154g8SzGFWOcLLjes40aTXFoREX47-ZPk/view), unzip the downloaded dataset.zip in `/assets/dataset` folder, and structure as:
```
/assets/dataset
├── chair
├── mugs
├── one_door_cabinet
├── one_drawer_cabinet
└── pots
```

The [RAM](https://github.com/yxKryptonite/RAM_code) affordance memory used in our method can be download through [this_link](https://drive.google.com/file/d/16cEIj8JHZ8KkGiRRub_qxdZoKB45ZXsa/view) and unzip as:
```
/assets/RAM_memory
├── customize
├── droid
├── HOI4D
└── memory_data.zip
```

Then run the `pre_extract_dift_feat.py` scripts to pre-extract [DIFT](https://github.com/Tsingularity/dift) feature for the reference images in the RAM affordance memory to accelerate the exploration stage in our method:
```
cd /AKM/scripts/prepare
conda activate akm
python pre_extract_dift_feat.py
```
After running this script, there should be an additional `*_new_sd.pkl` file in each memory directory:
```
assets/RAM_memory/customize/open_the_dishwasher
├── open_the_dishwasher.json
├── open_the_dishwasher_new.pkl
├── open_the_dishwasher_new_sd.pkl  (pre-extracted sd feature)
└── vis
```

To use the AnyGrasp Detector, TODO

## Test our method in Simulation
### Test script
We use [SAPIEN](https://github.com/haosulab/SAPIEN) simulator as our testbed, to run the code, use the `test_batch.py` in `scripts`:
```
cd /AKM/scripts/test_whole_pipeline
python test_batch.py --ts n_ts --cs cs_idx --method_cfg method_cfg_path --task_cfgs_folder path_to_task_cfgs
```
This script will test all the preset precise manipulation task as shown in,
```
/AKM/cfgs/simulation_cfgs/tasks
├── 0.yaml
├── 1.yaml
├── 2.yaml
├── 3.yaml
├── 4.yaml
```
each yaml is somthing like,
```
obj_env_cfg:
  joint_type: revolute
  data_path: dataset/one_door_cabinet/49133_link_0
  obj_index: 49133
  joint_index: 0
  obj_description: cabinet
  active_link_name: link_0
  active_joint_name: joint_0
  load_joint_state: 0
  load_pose:
  - 1.0405160188674927
  - 0.06454005837440491
  - 0.5545517802238464
  load_quat:
  - 0.9983863830566406
  - -0.018462015315890312
  - 0.0009928725194185972
  - -0.05369243025779724
  load_scale: 1.0904898116312924
task_cfg:
  instruction: open the cabinet
  task_id: 115
manip_env_cfg:
  tasks:
    '0_1':
      manip_start_state: 0.012847773999452361
      manip_end_state: 0.1873806991988853
    '0_2':
      manip_start_state: 0.025628238519227322
      manip_end_state: 0.3746940889180932
    '0_3':
      manip_start_state: 0.06156751386099259
      manip_end_state: 0.5851662894592914
    '1_0':
      manip_start_state: 0.2613419665715769
      manip_end_state: 0.08680904137214392
    '1_2':
      manip_start_state: 0.27585443727347464
      manip_end_state: 0.4503873624729076
    '1_3':
      manip_start_state: 0.25948967462449785
      manip_end_state: 0.6085555250233637
    '2_0':
      manip_start_state: 0.5216431412548702
      manip_end_state: 0.17257729085600435
    '2_1':
      manip_start_state: 0.4829762594353156
      manip_end_state: 0.3084433342358826
    '2_3':
      manip_start_state: 0.49116302222682306
      manip_end_state: 0.665695947426256
    '3_0':
      manip_start_state: 0.5681079659768422
      manip_end_state: 0.04450919037854335
    '3_1':
      manip_start_state: 0.596698494617329
      manip_end_state: 0.24763264421846318
    '3_2':
      manip_start_state: 0.6418019245439464
      manip_end_state: 0.4672689993445134
```
which defines a unique way of loading an articulated object, and upon this, with 12 different task that alter the joint state in a different way.

`--ts n_ts` means to seperate the 116 objects into n_ts total split, and `--cs cs_idx` means to run the `cs_idx`th split. The result will be saved to `/AKM/assets/logs_batch` as,
```
/AKM/assets/logs_batch/ours_1
├── 0
│   ├── 0.yaml
│   ├── explore_result.json
│   ├── log.txt
│   ├── manip_result.json
│   └── recon_result.json
├── 1
│   ├── 1.yaml
│   ├── explore_result.json
│   ├── log.txt
│   ├── manip_result.json
│   └── recon_result.json
```
The saved path is indicated by the cfg file of a run.

`--method_cfg method_cfg_path` indicate where to find the cfg for this run, where method to be tested and hyperparameters are set, e.g. in `/AKM/cfgs/ours_1.yaml`, hyperparameters for each stage are listed as,
```
exp_cfg:
# one experiment has multiple task, each has a unique task id
# the related files of task id are saved under exp_folder/task_id
  exp_folder: "/home/zby/Programs/AKM/assets/logs_batch/ours_1"
  method_name: "ours"
  save_cfg: True
  save_obj_repr: False
  save_result: True

logging:
  level: INFO

base_env_cfg:
  offscreen: True
  phy_timestep: 0.004
  use_sapien2: True
  planner_timestep: 0.01

task_cfg:
  task_id: null
  instruction: null

obj_env_cfg:
  data_path: null
  joint_type: null
  joint_index: null
  obj_index: null
  init_joint_state: null
  active_joint_name: null
  active_link_name: null
  load_scale: null
  load_pose: null
  load_quat: null
  obj_description: null

algo_cfg:
  use_anygrasp: True

explore_env_cfg:
  record_fps: 30
  pertubation_distance: 0.1
  reserved_distance: 0.05
  valid_thresh: 0.5
  update_sigma: 0.05
  max_tries: 25
  fully_zeroshot: False
  num_initial_pts: 1000
  use_IOR: True
  contact_analogy: True

recon_env_cfg:
  num_kframes: 5
  fine_lr: 0.001
  reloc_lr: 0.003
  num_R_augmented: 1000

manip_env_cfg:
  # max number of manipulations used for closed-loop control
  max_manip: 5
  # for whole traj closed-loop
  prismatic_whole_traj_success_thresh: 0.01
  revolute_whole_traj_success_thresh: 3
```

### Evaluate Result 

## Test our method in Real-World 
We also modify our `Env` class to support real-world deployment, the code are in `/AKM/akm/realworld_envs`.
We use a Franka Robot Arm and RealSense D455 camera for real-world deployment.
To callibrate the intrinsic and extrinsic of the D455 camera, use the `/scripts/collect_and_calibrate.py`.
To test the three stages of our framework, use scripts in `/AKM/scripts/test_rw`.