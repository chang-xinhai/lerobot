#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=dp_1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=50
#SBATCH --mem=500G  
#SBATCH --time 5-00:00:00

echo "Job started at $(date)"
echo "Memory limit: $SLURM_MEM_PER_NODE"
echo "CPU cores: $SLURM_CPUS_PER_TASK"

exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=512 \
  --steps=1000000 \
  --log_freq=50 \
  --eval_freq=1000 \
  --save_freq=100000 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_$exp_name \
  --policy.device=cuda \
  --wandb.enable=true \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true