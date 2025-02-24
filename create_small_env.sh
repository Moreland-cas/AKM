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
    conda remove -n $ENV_NAME --all -y
fi

# 创建 Conda 环境
echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 安装 Grounding DINO
# echo "Installing Grounding DINO..."
# cd third_party/GroundingDINO
# pip install -e .

# 修复 opencv-python qt5 bug
# pip uninstall opencv-python -y
# pip install opencv-python-headless

# 安装 napari
echo "Installing napari..."
# pip install opencv-python-headless
# conda install numpy=1.24 -y
python -m pip install "napari[pyqt5]" numpy==1.24 opencv-python
# pip install opencv-python

echo "Setup complete!"