#!/usr/bin/env zsh

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")

cd $PROJECT_PATH/assets/
python3 -m pip install --upgrade gdown

echo "Start downloading RAM Affordance Memory..."
mkdir -p RAM_memory
cd RAM_memory
FILE_ID="16cEIj8JHZ8KkGiRRub_qxdZoKB45ZXsa"
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O memory_data.zip
unzip memory_data.zip

cd $PROJECT_PATH
echo "RAM download finishs."