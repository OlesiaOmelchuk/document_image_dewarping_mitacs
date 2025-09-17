#!/bin/bash
#SBATCH --account=def-saadi
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1    
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --job-name=supervised_dewarping_doc3d_v2_continue_25_ep
#SBATCH --output=supervised_dewarping_doc3d_v2_continue_25_ep_%j.out

module load python/3.11
module load gcc opencv/4.12.0  # Load OpenCV system module

source ~/kaggle_env/bin/activate
pip install opencv-python
srun python /home/olesiao/projects/def-saadi/olesiao/train_supervised_dewarper_v2.py
