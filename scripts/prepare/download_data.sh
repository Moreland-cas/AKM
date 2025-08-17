#!/usr/bin/env zsh

SCRIPT_PATH=$(realpath "$0")     
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PROJECT_PATH=$(dirname "$(dirname "$SCRIPT_DIR")")

cd $PROJECT_PATH/assets/dataset

python3 -m pip install --upgrade gdown

echo "Start downloading data..."
FILE_ID="154g8SzGFWOcLLjes40aTXFoREX47-ZPk"
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O dataset.zip
unzip dataset.zip

cd $PROJECT_PATH
echo "Download finished."