#!/bin/bash

#SBATCH --job-name=lqr
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:A100:1
#SBATCH --time=5:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=./logs/task.out
#SBATCH --mail-type=NONE
#SBATCH --mem-per-gpu=40G

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate <your_env_name>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --mem=0 python3 <your_file>.py