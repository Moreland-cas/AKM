#!/usr/bin/env zsh

WS_PATH="/home/zby/Programs/AKM"
CLONED_ENV_NAME="general_flow"
SOURCE_ENV_NAME="akm"

export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if the conda environment exists
if ! conda info --envs | grep -q $CLONED_ENV_NAME; then
    # Clone the environment if it doesn't exist
    echo "Conda environment $CLONED_ENV_NAME not found, cloned from existing environment $SOURCE_ENV_NAME."
    conda create --name $CLONED_ENV_NAME --clone $SOURCE_ENV_NAME
fi

source $(conda info --base)/bin/activate $CLONED_ENV_NAME

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install open3d transformers multimethod termcolor shortuuid h5py easydict tabulate ultralytics==8.0.100 matplotlib==3.9.0

cd $WS_PATH/baselines/GeneralFlow
# install the pointnet++ library
./install.sh

# install fastSAM
cd $WS_PATH/baselines/GeneralFlow/tool_repos/FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

cd $WS_PATH

# Optional: This will cause an error when running aff_exec.py
# Change `device = select_device(self.args.device, verbose=verbose)` in `ultralytics.yolo.engine.predictor.BasePredictor.setup_model` to `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`
# Add `gripper_2d_pos = gripper_2d_pos.astype(np.int32)` to `KPSTExecutor.find_corresponding_3d_point` in aff_exec.py .