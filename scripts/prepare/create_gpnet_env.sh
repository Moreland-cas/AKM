#!/usr/bin/env zsh

WS_PATH="/home/zby/Programs/AKM"
CLONED_ENV_NAME="gpnet"
SOURCE_ENV_NAME="ea_v2"

export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 检查conda环境是否存在
if ! conda info --envs | grep -q $CLONED_ENV_NAME; then
    # 如果不存在，克隆环境
    echo "Conda environment $CLONED_ENV_NAME not found, cloned from existing environment $SOURCE_ENV_NAME."
    conda create --name $CLONED_ENV_NAME --clone $SOURCE_ENV_NAME
fi

pip install ninja jupyter_packaging jupyterlab

# 更新 gcc 和 g++ 为 11 版本以支持 CUDA 12.1
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11

# 安装 wayland 以及 xkbcommon 包
conda install conda-forge::libxkbcommon
conda install conda-forge::wayland

# 安装 GapartNet_Env
TODO

# 需要在 GapartNet 环境下安装一些 third_party, 包括:
# open3d (需要根据实际使用的torch版本进行编译)
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

# 更新 node.js 版本
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
# 使用以下命令解决 torch not found 的问题
pip install -e . --no-build-isolation --config-settings editable_mode=compat

# spconv NOTE: 这里的 xxx 需要替换为实际的 cuda 版本
pip install spconv-cu120

# 安装一些其他的包
pip install wandb tensorboard ipdb gym tqdm rich opencv_python pyparsing lightning==1.9.0 addict yapf h5py sorcery  pynvml torchdata==0.9.0 einops
# 单独安装 pytorch3d
conda install -c iopath iopath
# 如果底下这步有问题的话可以尝试 export MAX_JOBS=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 安装一些其他的包
pip install -U 'jsonargparse[signatures]>=4.27.7'

# 另外需要在 cuda-12.1中加一个 lib 到 lib64 的链接因为 cumm 库是在 lib 中去找库
ln -s /home/zby/Cudas/cuda-12.1/lib64 /home/zby/Cudas/cuda-12.1/lib