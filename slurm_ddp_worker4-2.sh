#!/bin/bash
#SBATCH --job-name=uci-W2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --constraint=h100
#SBATCH --error=z_err-worker-%J.out
#SBATCH --output=z_out-worker-%J.out

# Load necessary modules (like Python, CUDA, etc.)
module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd
nvidia-smi

# Set master address and port
read -r MASTER_INFO < .master_info.txt
export MASTER_ADDR=$(echo $MASTER_INFO | awk -F: '{print $1}')
export MASTER_PORT=$(echo $MASTER_INFO | awk -F: '{print $2}')

echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

script_args="config/train_lichess_uci.py"

# Run the training script on the worker node
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $script_args
