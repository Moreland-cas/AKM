#!/usr/bin/env zsh

WS_PATH="/home/zby/Programs/AKM"
CLONED_ENV_NAME="general_flow"
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

# optional:在跑 aff_exec.py 的时候会报错 
# 将 ultralytics.yolo.engine.predictor.BasePredictor.setup_model 的
device = select_device(self.args.device, verbose=verbose)
改为
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 将 aff_exec.py 中的 KPSTExecutor.find_corresponding_3d_point 加一个 gripper_2d_pos = gripper_2d_pos.astype(np.int32)

# 安装 embodied analogy