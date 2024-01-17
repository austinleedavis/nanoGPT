#!/bin/bash
#SBATCH --job-name=uciBatch
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --time=48:00:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4


# Load necessary modules (like Python, CUDA, etc.)
module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd
nvidia-smi



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
rdzv_port=$(id -u)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

script_args="config/train_lichess_uci.py"

srun torchrun \
--nnodes 10 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:$rdzv_port \
train.py $script_args




#S---BATCH --job-name=uci-M0
#S---BATCH --nodes=1
#S---BATCH --ntasks-per-node=8
#S---BATCH --time=3:00
#S---BATCH --gpus=1
#S---BATCH --mem=100G

#S---BATCH --error=z_err-master-%J.out
#S---BATCH --output=z_out-master-%J.out

#S---BATCH --constraint=h100