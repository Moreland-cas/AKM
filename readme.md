<p align="center">
  <img src="./assets/github_pics/realworld.jpg" alt="banner" style="width: 100%" />
</p>

<h1 align="center"> 
  AKM
</h1>

Here’s the complete code for our paper “Active Kinematic Modeling for Precise Manipulation of Unseen Articulated Objects,” accepted at RAL-2026.

<!-- TODO video, paper link -->

## Setup Environment
This section provides instructions for setting up conda environments to run our code.
### 1. Prerequisites
- Python 3.10  
- CUDA 12.1  
- PyTorch 2.5.1  

### 2. Install AKM Environment
Clone the repo and create the `AKM` conda environment in one shot:
```bash
git clone https://github.com/Moreland-cas/AKM
cd AKM
bash scripts/prepare/create_akm_env.sh
```

### 3. (Optional) Install Baseline Environment
To run baseline methods, simply execute the corresponding script. They will automatically clone the `AKM` environment and create dedicated conda environments.
```bash
# GeneralFlow
bash scripts/prepare/create_gflow_env.sh
# GAPartNet
bash scripts/prepare/create_gpnet_env.sh
```

**NOTE:** Make sure to update `CUDA_HOME` in all the scripts to your actual CUDA path:
```bash
export CUDA_HOME=YOUR_ACTUAL_CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 
```

## Datasets and Checkpoints
This section provides instructions for downloading the necessary data and model checkpoints to run the simulation experiments.

### 1. Downloading Articulated Object Assets
To download the articulated object assets for simulation, run the following command:
```bash
bash scripts/prepare/download_data.sh
```

### 2. Downloading Third-Party Model Checkpoints
To download the checkpoints for third-party models, execute:
```bash
bash scripts/prepare/download_ckpts.sh
```

### 3. Downloading RAM Affordance Memory
To download the RAM affordance memory, use:
```bash
bash scripts/prepare/download_ram.sh
```

### 4. (Optional) Pre-Extracting DIFT Features for RAM Memory
To significantly improve efficiency, we strongly recommend pre-extracting DIFT features for the reference images in RAM memory. Activate the environment and run the following commands:
```bash
conda activate akm
python scripts/prepare/pre_extract_dift_feat.py
```

### 5. (Optional) Setting Up the AnyGrasp Detector
To use the [AnyGrasp](https://github.com/graspnet/anygrasp_sdk) Detector, you need to prepare a license file. Follow the [instructions](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration) in the license registration guide. Once obtained, place the license file in the `AKM/akm/utility/grasp` directory with the following structure:
```
├── anygrasp.py
├── gsnet.py
├── gsnet.so
├── lib_cxx.so
├── license
```
Otherwise, a third-party implementation of [GSNet](https://github.com/graspnet/graspness_unofficial) is used.

## Running Simulation Experiments

### 1. Tasks/Methods Configurations 
Our simulation tasks utilize the 116 objects from [RGBManip](https://github.com/hyperplane-lab/RGBManip). For each object, random initial poses are generated using `scripts/prepare/generate_tasks.py` with a fixed random seed of 666. The resulting task configurations are stored in `cfgs/simulation_cfgs/tasks`.

Configuration files for the evaluated methods are located in `cfgs/simulation_cfgs/methods`, including two baselines, our full method, and five ablations:
```
├── gflow.yaml
├── gpnet.yaml
├── ours.yaml
├── ours_wo_AT.yaml
├── ours_wo_CA.yaml
├── ours_wo_FS.yaml
├── ours_wo_CL.yaml
└── ours_zs.yaml
```

### 2. Running Experiments
To evaluate a method on the tasks, run the following command:
```bash
python scripts/test_whole_pipeline/test_batch.py --method_cfg ours.yaml --task_cfgs_folder cfgs/simulation_cfgs/tasks
```

Results are saved to `/AKM/assets/logs_batch` in the following structure:
```
assets/logs_batch/ours
├── 0
│   ├── 0.yaml
│   ├── explore_result.json
│   ├── log.txt
│   ├── manip_result.json
│   └── recon_result.json
├── 1
├── 2
├── ...
```

### 3. (Optional) Parallel Execution
Each of the 116 objects has 12 manipulation tasks, resulting in a large number of experiments. To distribute tasks across multiple GPUs for faster execution, use the `--ts` (total splits) and `--cs` (current split index) flags. For example, to split tasks across 4 GPUs, run the following on each GPU:
```bash
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 0
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 1
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 2
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 3
```

### 4. Evaluation Statistics
To summarize key statistics for a run, execute:
```bash
python scripts/test_sim/summarize_run.py --run_name ours
```
This script aggregates results across all 12 manipulation tasks for the 116 objects and saves the output to `assets/analysis/{run_name}`.

To aggregate summarized outputs across multiple runs, execute:
```bash
python scripts/test_sim/summarize_runs.py
```

## Running Real-World Experiments 

### 1. Hardware
- Robot: Franka Emika Panda  
- RGB-D Camera: Intel RealSense D455

### 2. Camera Calibration
To obtain the intrinsic and extrinsic parameters of the D455:
```bash
python scripts/collect_and_calibrate.py
```
This script will command the robot to grasp a checkerboard, move the checkerboard to multiple poses while capturing images, then perform hand-eye calibration using OpenCV.

### 3. Running Three-Stages
To evaluate the three stages of our pipeline directly on the real robot, you shoule first prepare a real-world cfg file in `AKM/cfgs/realworld_cfgs`, refer to `cfgs/realworld_cfgs/p1_demo.yaml` for an example.

Then use the following commands to run the three stages:
```bash
cd AKM
# 1. Exploration
python akm/realworld_envs/explore_env.py

# 2. Unsupervised Articulation Modeling
python akm/realworld_envs/reconstruct_env.py

# 3. Precise Manipulation
python akm/realworld_envs/manipulate_env.py
```

## Acknowledgement
This project builds on solid earlier works, thank you to everyone who contributed!

- AnyGrasp: https://github.com/graspnet/anygrasp_sdk
- gapartnet: https://github.com/PKU-EPIC/GAPartNet
- generalFlow: https://github.com/michaelyuancb/general_flow
- gsnet: https://github.com/graspnet/graspness_unofficial
- sam2: https://github.com/facebookresearch/sam2
- fastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
- RGBManip: https://github.com/hyperplane-lab/RGBManip
- RAM: https://github.com/yxKryptonite/RAM_code