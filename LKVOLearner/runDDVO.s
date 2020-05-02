#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=DDVO
#SBATCH --mail-type=END
#SBATCH --mail-user=ww1351@nyu.edu
#SBATCH --output=slurm_%j.out

cd /home/ww1351/LKVOLearner
CUDA_LAUNCH_BLOCKING=1 python train_main_finetune.py --dataroot ../data/ --checkpoints_dir ./checkpointsDDVO --which_epoch -1 --save_latest_freq 1000 --batchSize 1 --display_freq 200 --name finetune --lk_level 1 --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009 --epoch_num 10 --lr 0.00001
