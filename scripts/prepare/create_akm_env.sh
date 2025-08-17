#!/usr/bin/env zsh

set -e
set -o pipefail

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")
ENV_NAME="akm"

export CUDA_HOME=/home/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 

echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing dependencies..."
conda install openblas-devel -c anaconda -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ninja scipy numpy==1.24.4
pip install pytorch-lightning==2.5.0.post0
pip install flask requests
pip install --upgrade pip setuptools

# install MinkowskiEngine
echo "Installing MinkowskiEngine..."
cp $CONDA_PREFIX/lib/libopenblas.so* $CONDA_PREFIX/lib/python3.10/site-packages/torch/lib/.
cd $PROJECT_PATH/third_party/MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# install graspnetAPI
echo "Installing graspnetAPI..."
cd $PROJECT_PATH/third_party/graspnetAPI
pip install -e .

# install pointnet2
cd $PROJECT_PATH/third_party/pointnet2
python setup.py install

# install co-tracker
echo "Installing co-tracker..."
# git clone https://github.com/facebookresearch/co-tracker.git
cd $PROJECT_PATH/third_party/co-tracker
pip install -e .

# install SAM 2
echo "Installing SAM 2..."
# git clone https://github.com/facebookresearch/sam2.git 
cd $PROJECT_PATH/third_party/sam2
pip install -e ".[notebooks]"

# install Grounding DINO
echo "Installing GroundingDINO..."
# git clone git@github.com:IDEA-Research/GroundingDINO.git
cd $PROJECT_PATH/third_party/GroundingDINO
# pip install -e .
python3 setup.py install

# install RAM
echo "Installing RAM dependencies..."
pip install diffusers==0.15.0 
pip install transformers==4.49.0 
pip install ipympl==0.9.6 
pip install xformers==0.0.29.post1 
pip install accelerate==1.4.0 
pip install urllib3==2.3.0 
pip install open_clip_torch==2.31.0 einops openai

# install AKM
echo "Installing AKM..."
cd $PROJECT_PATH/
pip install -e .

echo "Setup complete! Activate command: conda activate akm"