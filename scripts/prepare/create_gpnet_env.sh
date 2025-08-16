#!/usr/bin/env zsh

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")

CLONED_ENV_NAME="gpnet"
SOURCE_ENV_NAME="akm"

export CUDA_HOME=/home/zby/Cudas/cuda-12.1
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
pip install ninja jupyter_packaging jupyterlab

echo "Install wayland and xkbcommon packages..."
# Install wayland and xkbcommon packages
conda install conda-forge::libxkbcommon
conda install conda-forge::wayland

# install GapartNet_Env
echo "Install GapartNet_Env..."
# please refer to https://github.com/geng-haoran/GAPartNet_env

# Requires installation of some third-party libraries in the GapartNet environment, including:
# open3d (needs to be compiled based on the actual torch version used)
cd $PROJECT_PATH
git clone https://github.com/isl-org/Open3D
cd Open3D
# Only needed for Ubuntu
bash util/install_deps_ubuntu.sh
mkdir build
cd build
cmake \
-DPython3_ROOT="$CONDA_PREFIX/bin" \
-DWITH_OPENMP=ON \
-DWITH_SIMD=ON \
-DBUILD_PYTORCH_OPS=ON \
-DBUILD_CUDA_MODULE=ON \
-DBUILD_COMMON_CUDA_ARCHS=ON \
-DGLIBCXX_USE_CXX11_ABI=OFF \
-DBUILD_JUPYTER_EXTENSION=ON \
-DCMAKE_CUDA_COMPILER=$(which nvcc) \
-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE \
-DCMAKE_INSTALL_PREFIX="$PROJECT_PATH/Open3D/open3d_install" \
..

make -j12
make install

# update node.js version
echo "update node.js version..."
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
source ~/.zshrc
nvm install 14
nvm use 14
npm install -g yarn

make install-pip-package

# epic_ops
echo "Install epic_ops..."
# git clone git@github.com:geng-haoran/epic_ops.git
cd $PROJECT_PATH
cd third_party/epic_ops
# python setup.py develop
# Use the following command to solve the torch not found problem
pip install -e . --no-build-isolation --config-settings editable_mode=compat

# spconv NOTE: xxx here needs to be replaced with the actual cuda version
echo "Install spconv..."
pip install spconv-cu120
pip install wandb tensorboard ipdb gym tqdm rich opencv_python pyparsing lightning==1.9.0 addict yapf h5py sorcery  pynvml torchdata==0.9.0 einops
conda install -c iopath iopath
# If you have problems with the following step, you can try export MAX_JOBS=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -U 'jsonargparse[signatures]>=4.27.7'

# In addition, you need to add a link from lib to lib64 in cuda-12.1 because the cumm library is looking for the library in lib
echo "If you have errors regarding cumm library, try:"
echo "ln -s xxx/cuda-12.1/lib64 xxx/Cudas/cuda-12.1/lib"

cd $PROJECT_PATH
echo "Setup complete! Activate command: conda activate gpnet"