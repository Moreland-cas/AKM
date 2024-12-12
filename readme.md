conda create -n embodied_analogy python=3.10

# conda install torch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# for compilation of curobo
conda install nvidia/label/cuda-12.1.0::cuda-toolkit

# install g++ 
# make sure you have gxx installed in your system

# install ninja
pip install ninja

# add symbolic link to libcudart.so under lib64 folder
cd /home/zby/ProgramFiles/anaconda3/envs/embodied_analogy
mkdir lib64
cp lib/*cudart* ./lib64 
cd lib64 && ln -s libcudart.so.12 libcudart.so

# install Featup
cd Embodied_Analogy/third_party
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .

# install embodied analogy
cd Embodied_Analogy
pip install -e .