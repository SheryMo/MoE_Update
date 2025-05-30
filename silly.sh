#!/bin/bash
#
# LLM environment initialization script
# Author: <Your Name or Email>
#
# This script sets up a Python/Conda environment for LLM workloads, installs
# required packages, and downloads the model from Hugging Face.
#

LOGF="llm_setup.log"

function update_system()
{
    echo "==> Updating system..." | tee -a $LOGF
    sudo apt-get update && sudo apt-get upgrade -y
    sudo apt-get install -y python3-pip wget git
}

function install_miniconda()
{
    echo "==> Installing Miniconda..." | tee -a $LOGF
    local INSTALLER="Anaconda3-2024.10-1-Linux-x86_64.sh"
    wget -c https://repo.anaconda.com/archive/$INSTALLER
    bash $INSTALLER -b -p $HOME/anaconda3

    export PATH=$HOME/anaconda3/bin:$PATH
    eval "$(~/anaconda3/bin/conda shell.bash hook)"
    conda init
}

function setup_conda_env()
{
    echo "==> Creating and activating conda environment 'silly_env'..." | tee -a $LOGF
    conda create -y -n silly_env python=3.10
    conda activate silly_env
}

function install_dependencies()
{
    echo "==> Installing conda and pip dependencies..." | tee -a $LOGF
    conda install -y cudatoolkit
    pip install datasets==2.16.0 flask huggingface_hub
}

function install_local_repository()
{
    local REPO_DIR="/local/repository"
    echo "==> Installing local repository at $REPO_DIR..." | tee -a $LOGF

    if [[ ! -d $REPO_DIR ]]; then
        echo "ERROR: $REPO_DIR does not exist" | tee -a $LOGF
        exit 1
    fi

    cd $REPO_DIR
    pip install -e .
}

function download_model()
{
    echo "==> Downloading model from Hugging Face..." | tee -a $LOGF

    local MODEL_NAME="LLaMA-MoE-v1-3_5B-2_8"
    local WORK_DIR="/local/repository/ll_test"
    local DEST_DIR="/local/repository/llama-moe/$MODEL_NAME"

    mkdir -p $WORK_DIR
    cd $WORK_DIR

    python3 <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="llama-moe/LLaMA-MoE-v1-3_5B-2_8",
    local_dir="./",
    local_dir_use_symlinks=False
)
EOF

    mkdir -p $DEST_DIR
    mv pytorch_model-00001-of-00002.bin $DEST_DIR/
    mv pytorch_model-00002-of-00002.bin $DEST_DIR/
    mv tokenizer.model $DEST_DIR/
}

function main()
{
    update_system
    install_miniconda
    setup_conda_env
    install_dependencies
    install_local_repository
    download_model

    echo "==> [$(date)] Setup complete. Activate with: conda activate silly_env" | tee -a $LOGF
}

main > $LOGF 2>&1
