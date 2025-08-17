#!/usr/bin/env zsh

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")

cd $PROJECT_PATH/assets/ckpts
python3 -m pip install --upgrade gdown

echo "Start downloading Anygrasp Detection ckpts..."
mkdir -p anygrasp
cd anygrasp
FILE_ID="1jNvqOOf_fR3SWkXuz8TAzcHH9x8gE8Et"
gdown "https://drive.google.com/uc?id=${FILE_ID}"


echo "Start downloading FastSAM ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p fastSAM
cd fastSAM
# FastSAM-x.pt
FILE_ID="1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
# FastSAM-s.pt
FILE_ID="10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV"
gdown "https://drive.google.com/uc?id=${FILE_ID}"


echo "Start downloading GAPartNet ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p gapartnet
cd gapartnet
# release.ckpt
FILE_ID="1D1PwfXPYPtxadthKAJdehhIBbPEyBB6X"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
# all_best.ckpt
FILE_ID="1TzsVKVlbqRg3fd3XEutQ2jgTH07Q8Lad"
gdown "https://drive.google.com/uc?id=${FILE_ID}"


echo "Start downloading GeneralFlow ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p generalFlow/kpst_hoi4d
cd generalFlow/kpst_hoi4d
# S
FILE_ID="1jUW86qDrl8iEkxClE3KjQMpxBwIMRcrZ"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
# B
FILE_ID="1U8_TpjKg6ycy-URMa5e1ABq_6Xvd9LJ4"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
# L
FILE_ID="1vJ_mwNfGC8P8WHuLQBdL28e_MywUJeZX"
gdown "https://drive.google.com/uc?id=${FILE_ID}"


echo "Start downloading grounding_dino ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p grounding_dino
cd grounding_dino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


echo "Start downloading gsnet ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p gsnet
cd gsnet
# realsense ckpt
FILE_ID="1RfdpEM2y0x98rV28d7B2Dg8LLFKnBkfL"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
# kinect ckpt
FILE_ID="10o5fc8LQsbI8H0pIC2RTJMNapW9eczqF"
gdown "https://drive.google.com/uc?id=${FILE_ID}"


echo "Start downloading sam2 ckpts..."
cd $PROJECT_PATH/assets/ckpts
mkdir -p sam2
cd sam2
FILE_ID="10o5fc8LQsbI8H0pIC2RTJMNapW9eczqF"
gdown "https://drive.google.com/uc?id=${FILE_ID}"
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt


cd $PROJECT_PATH
echo "Download finished."