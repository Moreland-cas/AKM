conda create -n ea_v2 python=3.10

# 1)首先安装最难安装的 MinkowskiEngine
# openblas-devel需要在torch之前安装，否则会使得torch-gpu降级为torch-cpu
conda install openblas-devel -c anaconda  
# 安装 2.5.1 版本torch、0.20.1 版本 torchvision 和 2.5.1 版本 torchaudio
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.24 ninja scipy tqdm open3d 

# 测试torch是否安装成功
python -c 'import torch;print(torch.cuda.is_available())'
# True
python -c 'import torchvision'

# 设置 CUDA 相关变量，需要本地安装 CUDA 12.1
export CUDA_HOME=/home/zby/Cudas/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd Embodied_Analogy/third_party
git clone git@github.com:NVIDIA/MinkowskiEngine.git
# Note：为解决 error: namespace "thrust" has no member "device"
# 需修改 MinkowskiEngine 仓库中的四个文件，主要是使用cuda 12而不是11造成的问题
https://axi404.top/posts/Tech-Talk/MISCs/anygrasp

# Note：为解决 ld: cannot find -lopenblas: No such file or directory
cp /home/zby/ProgramFiles/anaconda3/envs/ea_v2/lib/libopenblas.so* /home/zby/ProgramFiles/anaconda3/envs/ea_v2/lib/python3.10/site-packages/torch/lib/.

# 编译安装 MinkowskiEngine
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

# 测试 MinkowskiEngine
python -c 'import MinkowskiEngine as ME;print(ME.__version__)' 
# 0.5.4

# 2) 安装 graspnetAPI
cd Embodied_Analogy/third_party
git clone git@github.com:graspnet/graspnetAPI.git
cd graspnetAPI
# Note：需要把setup.py中的所有数字版本注释掉
pip install -e .

cd Embodied_Analogy/third_party/pointnet2
python setup.py install

# 将 anygrasp 所需的文件放到项目根目录 Embodied_Analogy 下
# 包含 license、gsnet.so 和 lib_cxx.so

# 3) 安装 Featup
cd Embodied_Analogy/third_party
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .

# 4) 安装 DIFT
# 安装pre-built的xformers, 0.0.29
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install jupyterlab ipympl triton transformers matplotlib diffusers==0.15.0
# 为解决 cannot import name 'cached_download' from 'huggingface_hub'
pip install huggingface_hub==0.25.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5) 安装 joint estimation、Grounded SAM、SAM 2
# TODO

# 6) 安装 ea
cd Embodied_Analogy
pip install -e .