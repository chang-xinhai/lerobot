#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=act_1
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
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=512 \
  --steps=1000000 \
  --log_freq=50 \
  --eval_freq=1000 \
  --save_freq=100000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true