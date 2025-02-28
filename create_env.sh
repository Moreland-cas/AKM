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

# 设置 CUDA 相关变量
export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 安装 MinkowskiEngine
echo "Installing MinkowskiEngine..."
cp /home/zby/ProgramFiles/anaconda3/envs/$ENV_NAME/lib/libopenblas.so* /home/zby/ProgramFiles/anaconda3/envs/$ENV_NAME/lib/python3.10/site-packages/torch/lib/.
cd third_party
# git clone git@github.com:NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
cd ../

# 安装 graspnetAPI
echo "Installing graspnetAPI..."
# git clone git@github.com:graspnet/graspnetAPI.git
cd graspnetAPI
pip install -e .
cd ..

# 安装 pointnet2
cd pointnet2
python setup.py install
cd ../

# 安装 FeatUp
echo "Installing FeatUp..."
# git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .
cd ../

# 安装 DIFT
echo "Installing DIFT related support..."
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install jupyterlab ipympl triton transformers matplotlib diffusers==0.15.0 accelerate
pip install huggingface_hub==0.26 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 co-tracker
echo "Installing co-tracker..."
# git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .
# pip install matplotlib flow_vis tqdm tensorboard 'imageio[ffmpeg]'
cd ../

# 安装 SAM 2
echo "Installing SAM 2..."
# git clone https://github.com/facebookresearch/sam2.git 
cd sam2
pip install -e ".[notebooks]"
cd ../

# 安装 Grounding DINO
echo "Installing Grounding DINO..."
# git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# 安装 Embodied Analogy
echo "Installing Embodied Analogy..."
cd ../
pip install -e .

# 修复 opencv-python qt5 bug
echo "Replace opencv-python with its headless version..."
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless

# 安装 napari
echo "Installing napari..."
python -m pip install "napari[pyqt5]" numpy==1.24.4

# 删除 opencv 中的 qt5 插件, 防止与 cv2 冲突
# rm /home/zby/ProgramFiles/anaconda3/envs/$ENV_NAME/lib/python3.10/site-packages/cv2/qt/plugins/platforms/libqxcb.so

# 安装 RAM 相关支持
echo "Installing RAM..."
pip install open_clip_torch einops openai

echo "Setup complete!"

# TROUBLE SHOOTING
# 1) 要确保 initialize_napari 函数和 sapien 的 renderer 函数在 import grounding_dino  之前执行 !!!