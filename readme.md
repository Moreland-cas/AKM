<div align="center">
  # AKM
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
This section provides instructions for downloading the necessary data and model checkpoints to set up the simulation environment and third-party methods.

### Downloading Articulated Object Assets
To download the articulated object assets for simulation, run the following command:
```bash
bash scripts/prepare/download_data.sh
```

### Downloading Third-Party Model Checkpoints
To download the checkpoints for third-party models, execute:
```bash
bash scripts/prepare/download_ckpts.sh
```

### Downloading RAM Affordance Memory
To download the RAM affordance memory, use:
```bash
bash scripts/prepare/download_ram.sh
```

### (Optional) Pre-Extracting DIFT Features for RAM Memory
To significantly improve efficiency, we strongly recommend pre-extracting DIFT features for the reference images in RAM memory. Activate the environment and run the following commands:
```bash
conda activate akm
python scripts/prepare/pre_extract_dift_feat.py
```

### Setting Up the AnyGrasp Detector
To use the [AnyGrasp](https://github.com/graspnet/anygrasp_sdk) Detector, you need to prepare a license file. Follow the [instructions](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration) in the license registration guide. Once obtained, place the license file in the `AKM/akm/utility/grasp` directory with the following structure:
```
├── anygrasp.py
├── gsnet.py
├── gsnet.so
├── lib_cxx.so
├── license
```

## Running Simulation Experiments

### Tasks/Methods Configurations 
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

### Running Experiments
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

### Parallel Execution
Each of the 116 objects has 12 manipulation tasks, resulting in a large number of experiments. To distribute tasks across multiple GPUs for faster execution, use the `--ts` (total splits) and `--cs` (current split index) flags. For example, to split tasks across 4 GPUs, run the following on each GPU:
```bash
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 0
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 1
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 2
python test_batch.py --method_cfg xxx --task_cfgs_folder xxx --ts 4 --cs 3
```

### Evaluation Statistics
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
We also modify our `Env` class to support real-world deployment, the code are in `/AKM/akm/realworld_envs`.
We use a Franka Robot Arm and RealSense D455 camera for real-world deployment.
To callibrate the intrinsic and extrinsic of the D455 camera, use the `/scripts/collect_and_calibrate.py`.
To test the three stages of our framework, use scripts in `/AKM/scripts/test_rw`.

## Acknowledgement
This project builds on solid earlier work—thank you to everyone who contributed.

- AnyGrasp: https://github.com/graspnet/anygrasp_sdk
- gapartnet: https://github.com/PKU-EPIC/GAPartNet
- generalFlow: https://github.com/michaelyuancb/general_flow
- gsnet: https://github.com/graspnet/graspness_unofficial
- sam2: https://github.com/facebookresearch/sam2
- fastSAM: https://github.com/CASIA-IVA-Lab/FastSAM
- RGBManip: https://github.com/hyperplane-lab/RGBManip
- RAM: https://github.com/yxKryptonite/RAM_code