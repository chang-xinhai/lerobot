exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name 

exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=64 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name 

exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=1000 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp_$exp_name


# DEBUG RUNS

exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=10 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/act_$exp_name 

exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --dataset.preload=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name 

lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --dataset.preload=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name \
  --resume=true \
  --config_path=outputs/train/dp3_multi_object_open_7221_scene_0_seed_0/checkpoints/last/pretrained_model/train_config.json

exp_name="multi_object_open_7221_scene_0_seed_0"
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=1000 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=data/$exp_name \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/dp_$exp_name



# Convert data
rm -rf /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0_test_image
python -m lerobot.scripts.lerobot_edit_dataset \
      --repo_id /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0_test \
      --operation.type convert_to_image \
      --operation.output_dir /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0_test_image

python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0_test \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.ego_topdown', 'observation.images.ego_wrist', 'observation.images.fix_local', 'observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"

python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0 \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.ego_topdown', 'observation.images.ego_wrist', 'observation.images.fix_local', 'observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"


python -m lerobot.scripts.lerobot_edit_dataset \
      --repo_id /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0 \
      --operation.type convert_to_image \
      --operation.output_dir /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0_image


python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id /home/xinhai/projects/automoma/third_party/lerobot/data/multi_object_open_7221_scene_0_seed_0 \
        --operation.type split \
        --operation.splits '{"test": 0.1, "val": 0.1, "train": 0.8}'