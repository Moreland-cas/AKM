#!/usr/bin/env zsh

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")

CLONED_ENV_NAME="gflow"
SOURCE_ENV_NAME="akm"

export CUDA_HOME=/home/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if the conda environment exists
if ! conda info --envs | grep -q $SOURCE_ENV_NAME; then
    # Clone the environment if it doesn't exist
    echo "Conda environment $SOURCE_ENV_NAME not found, please run create_akm_env.sh first!"    
fi

conda create --name $CLONED_ENV_NAME --clone $SOURCE_ENV_NAME
source $(conda info --base)/bin/activate $CLONED_ENV_NAME

echo "Install Dependencies..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install open3d transformers multimethod termcolor shortuuid h5py easydict tabulate ultralytics==8.0.100 matplotlib==3.9.0

echo "Install GeneralFlow..."
cd $PROJECT_PATH/baselines/GeneralFlow
# install the pointnet++ library
./install.sh

echo "Install FastSAM..."
# install fastSAM
cd $PROJECT_PATH/baselines/GeneralFlow/tool_repos/FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

cd $PROJECT_PATH
echo "Setup complete! Activate command: conda activate gflow"