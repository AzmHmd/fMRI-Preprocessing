#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=flip_detection_gpu
#SBATCH --out=flip_detection_gpu.out.%J
#SBATCH --err=flip_detection_gpu.err.%J
#SBATCH -p gpu
#SBATCH --mem-per-cpu=30G
#SBATCH -n 1
#SBATCH --gres=gpu:1

module purge 
module load tensorflow
python test_on_gpu_save_files.py
