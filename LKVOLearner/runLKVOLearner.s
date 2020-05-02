#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=23:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=LKVO
#SBATCH --mail-type=END
#SBATCH --mail-user=ww1351@nyu.edu
#SBATCH --output=slurm_%j.out

cd /home/ww1351/LKVOLearner
python train_main_posenet.py --dataroot ../data --checkpoints_dir ./checkpoints --which_epoch -1 --save_latest_freq 1000 --batchSize 2 --display_freq 50 --name posenet --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009
