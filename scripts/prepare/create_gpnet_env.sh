#!/usr/bin/env zsh

WS_PATH="/home/zby/Programs/AKM"
CLONED_ENV_NAME="gpnet"
SOURCE_ENV_NAME="ea_v2"

export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if the conda environment exists
if ! conda info --envs | grep -q $CLONED_ENV_NAME; then
    # If it does not exist, clone the environment
    echo "Conda environment $CLONED_ENV_NAME not found, cloned from existing environment $SOURCE_ENV_NAME."
    conda create --name $CLONED_ENV_NAME --clone $SOURCE_ENV_NAME
fi

pip install ninja jupyter_packaging jupyterlab

# Update gcc and g++ to version 11 to support CUDA 12.1
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11

# Install wayland and xkbcommon packages
conda install conda-forge::libxkbcommon
conda install conda-forge::wayland

# install GapartNet_Env
# please refer to https://github.com/geng-haoran/GAPartNet_env

# Requires installation of some third-party libraries in the GapartNet environment, including:
# open3d (needs to be compiled based on the actual torch version used)
git clone https://github.com/isl-org/Open3D
cd Open3D
# Only needed for Ubuntu
util/install_deps_ubuntu.sh
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
-DCMAKE_INSTALL_PREFIX="/home/zby/Programs/Open3D/open3d_install" \
..

make -j12
make install

# update node.js version
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
source ~/.zshrc
nvm install 14
nvm use 14
npm install -g yarn

make install-pip-package

# epic_ops
git clone git@github.com:geng-haoran/epic_ops.git
cd epic_ops
# python setup.py develop
# Use the following command to solve the torch not found problem
pip install -e . --no-build-isolation --config-settings editable_mode=compat

# spconv NOTE: xxx here needs to be replaced with the actual cuda version
pip install spconv-cu120

pip install wandb tensorboard ipdb gym tqdm rich opencv_python pyparsing lightning==1.9.0 addict yapf h5py sorcery  pynvml torchdata==0.9.0 einops
conda install -c iopath iopath
# If you have problems with the following step, you can try export MAX_JOBS=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

pip install -U 'jsonargparse[signatures]>=4.27.7'

# In addition, you need to add a link from lib to lib64 in cuda-12.1 because the cumm library is looking for the library in lib
ln -s /home/zby/Cudas/cuda-12.1/lib64 /home/zby/Cudas/cuda-12.1/lib