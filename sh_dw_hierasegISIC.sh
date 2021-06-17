#!/bin/bash
#SBATCH -J ISIC_DW_HieraSeg
#SBATCH -o ISIC_DW_HieraSeg.out               
#SBATCH -e error.err
#SBATCH --gres=gpu:1
#SBATCH -w hpc-gpu006
#SBATCH --partition=gpu_7d1g 
#SBATCH --ntasks-per-node=2 
#SBATCH --nodes=1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

source /home/xiaoqiguo2/.bashrc

module load cuda/10.2.89
conda activate torch

cd ~/experiment/DW_HieraSeg_ISIC/
python ./train.py
