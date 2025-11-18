#!/bin/bash
#SBATCH --job-name=RUN_LOG
#SBATCH --output=RUN_LOG_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gp100gl:2
#SBATCH --time=00:02:00

. /etc/profile.d/modules.sh

module load cuda/10.2 python/3/pycuda/2019.1.1

python3.4 foundationModel.py
