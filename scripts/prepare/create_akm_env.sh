#!/usr/bin/env zsh

# 设置环境变量
set -e
set -o pipefail

EXPECTED_PATH="/home/zby/Programs/Embodied_Analogy"
CURRENT_PATH=$(pwd)

if [[ "$CURRENT_PATH" != "$EXPECTED_PATH" ]]; then
    echo "Error: This script must be run from the directory: $EXPECTED_PATH"
    echo "Current path is: $CURRENT_PATH"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Error: No environment name provided."
    echo "Usage: $0 <environment_name>"
    exit 1
fi
ENV_NAME="$1"

# 检查环境是否已存在
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Removing it..."
    conda remove -n $ENV_NAME -y
fi

# 创建 Conda 环境
echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME


# 安装依赖
echo "Installing dependencies..."
conda install openblas-devel -c anaconda -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ninja scipy numpy==1.24.4
pip install pytorch-lightning==2.5.0.post0
pip install flask requests
# 升级，否则可能找不到torch
pip install --upgrade pip setuptools

# 设置 CUDA 相关变量
export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 

# 安装 MinkowskiEngine
echo "Installing MinkowskiEngine..."
cp $CONDA_PREFIX/lib/libopenblas.so* $CONDA_PREFIX/lib/python3.10/site-packages/torch/lib/.
cd $EXPECTED_PATH/third_party/MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# 安装 graspnetAPI
echo "Installing graspnetAPI..."
# git clone git@github.com:graspnet/graspnetAPI.git
cd $EXPECTED_PATH/third_party/graspnetAPI
pip install -e .

# 安装 pointnet2
cd $EXPECTED_PATH/third_party/pointnet2
python setup.py install

# 安装 co-tracker
echo "Installing co-tracker..."
# git clone https://github.com/facebookresearch/co-tracker.git
cd $EXPECTED_PATH/third_party/co-tracker
pip install -e .

# 安装 SAM 2
echo "Installing SAM 2..."
# git clone https://github.com/facebookresearch/sam2.git 
cd $EXPECTED_PATH/third_party/sam2
pip install -e ".[notebooks]"

# 安装 Grounding DINO
echo "Installing Grounding DINO..."
# git clone git@github.com:IDEA-Research/GroundingDINO.git
cd $EXPECTED_PATH/third_party/GroundingDINO
# pip install -e .
python3 setup.py install

# 安装 RAM 相关包
pip install diffusers==0.15.0 
pip install transformers==4.49.0 
pip install ipympl==0.9.6 
pip install xformers==0.0.29.post1 
pip install accelerate==1.4.0 
pip install urllib3==2.3.0 
pip install open_clip_torch==2.31.0 einops openai

# 安装 Embodied Analogy
echo "Installing Embodied Analogy..."
cd $EXPECTED_PATH/
pip install -e .

echo "Setup complete!"
