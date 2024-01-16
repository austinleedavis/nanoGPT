#!/bin/bash
#SBATCH --job-name=uci-M0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --constraint=h100
#SBATCH --error=z_err-master-%J.out
#SBATCH --output=z_out-master-%J.out

# Load necessary modules (like Python, CUDA, etc.)
module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd

# Set master address and port
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=$(id -u)


echo $MASTER_ADDR:$MASTER_PORT > .master_info.txt

script_args="config/train_lichess_uci.py"

# Run the training script on the master node
torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $script_args
