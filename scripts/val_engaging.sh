#!/bin/bash
#SBATCH -n 4           # 1 core
#SBATCH -t 0-04:00:00   # 4 hours
#SBATCH -J valid_flower_no_oxd # sensible name for the job
#SBATCH --output=opt.log   # Standard output and error log.
#SBATCH -p mit_normal_gpu # partition
#SBATCH --mem-per-cpu=4G # 4 gb

##SBATCH --gres=gpu:h200:2 #1 gpu

conda activate flower

sh run_FlowER_large_newData.sh