#################################################################################
# IsaacLab-Arena HDF5 to LeRobot Dataset 
#################################################################################

# Arena-G1-Loco-Manipulation-Task
python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot.py \
  --yaml_file isaaclab_arena_gr00t/config/g1_locomanip_config.yaml

python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/data/nvidia/Arena-G1-Loco-Manipulation-Task/arena_g1_loco_manipulation_dataset_generated_small/lerobot

exp_name="Arena-G1-Loco-Manipulation-Task"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=32 \
  --steps=100 \
  --log_freq=5 \
  --eval_freq=50 \
  --save_freq=100 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_$exp_name \
  --policy.device=cuda \
  --wandb.enable=true 

# Evaluation

lerobot-eval \
    --policy.path=nvidia/smolvla-arena-gr1-microwave \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1

lerobot-eval \
    --policy.path=outputs/train/act_Arena-GR1-Manipulation-Task-v3/checkpoints/005000/pretrained_model \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1 

lerobot-eval \
    --policy.path=outputs/train/dp_Arena-G1-Loco-Manipulation-Task/checkpoints/000100/pretrained_model \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=g1_locomanip_pnp \
    --env.embodiment=g1_wbc_pink \
    --env.object=brown_box \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1 
    
lerobot-eval \
    --policy.path=outputs/train/dp_Arena-GR1-Manipulation-Task-v3/checkpoints/005000/pretrained_model \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1

TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 lerobot-eval \
    --policy.path=nvidia/pi05-arena-gr1-microwave \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=15 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1
























################################################################################
# AutoMoMa HDF5 to LeRobot Dataset 
#################################################################################
exp_name="multi_object_open_7221_scene_0_seed_0"

python scripts/automoma/automoma_hdf5_to_lerobot.py \
  --input-dir data/raw_data/$exp_name \
  --repo-id automoma/$exp_name \
  --root data/lerobot/$exp_name \
  --fps 10 \
  --mobile-base-mode relative \
  --use-videos

lerobot-dataset-viz \
    --repo-id automoma/$exp_name \
    --root data/lerobot/$exp_name \
    --episode-index 0 \
    --video-backend pyav \
    --save 1 \
    --output-dir ./viz_results

# DP for debug
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/dp_$exp_name
lerobot-train \
  --policy.type=diffusion \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=100 \
  --save_freq=10000 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_$exp_name \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true

# DP for train
exp_name="test_lerobot"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/dp_$exp_name
CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=diffusion \
  --batch_size=512 \
  --steps=1000 \
  --log_freq=50 \
  --eval_freq=1000 \
  --save_freq=100 \
  --job_name=dp_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_$exp_name \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.preload=true \
  --dataset.preload_cache=true \
  --dataset.filter_features_by_policy=true \
  --num_workers=4

# ACT for debug
exp_name="test_lerobot"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
CUDA_VISIBLE_DEVICES=1  lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=10000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true \
  --dataset.preload_cache=true \
  --dataset.filter_features_by_policy=true 

# DP3 for debug
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/dp3_$exp_name \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true

# Pi05 for debug
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/pi05_$exp_name
lerobot-train \
    --policy.type=pi05 \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --job_name=pi05_training \
    --dataset.repo_id=$exp_name \
    --dataset.root=$dataset_root \
    --output_dir=outputs/train/pi05_$exp_name \
    --policy.repo_id=lerobot/pi05_base \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 \
    --dataset.preload=true \
    --dataset.filter_features_by_policy=true

# Debug for multigpu

exp_name="test_lerobot"
dataset_root=data/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --policy.type=act \
  --batch_size=128 \
  --steps=1000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=100 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=false \
  --dataset.filter_features_by_policy=true \
  --num_workers=8






###############################################################
# Example usage: [EXP] Multi-object open task
###############################################################

# Step 1: Generate plans
python scripts/pipeline/1_generate_plans.py --exp multi_object_open --scene scene_0_seed_0 --object 7221

# Step 2: Render dataset
python scripts/pipeline/2_render_dataset.py --exp multi_object_open --scene scene_0_seed_0 --object 7221 --max-episodes 10
python scripts/pipeline/2_render_dataset.py --exp multi_object_open --scene scene_0_seed_0 --object 7221 --headless --max-episodes 10

# Step 3: Train policies
# (1) ACT
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
CUDA_VISIBLE_DEVICES=1  lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=10000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true

# (2) DP3
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=5000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --output_dir=outputs/train/dp3_$exp_name \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true


# (3) Diffusion Policy
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
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
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/dp_$exp_name \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.preload=true \
  --dataset.filter_features_by_policy=true

# (4) Pi05
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root=data/multi_object_open/lerobot/$exp_name
rm -rf outputs/train/pi05_$exp_name
lerobot-train \
    --policy.type=pi05 \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --job_name=pi05_training \
    --dataset.repo_id=$exp_name \
    --dataset.root=$dataset_root \
    --output_dir=outputs/train/pi05_$exp_name \
    --policy.repo_id=lerobot/pi05_base \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 \
    --dataset.preload=true \
    --dataset.filter_features_by_policy=true

# Step 4: Evaluate policies
# (1) ACT
python scripts/pipeline/4_evaluate.py \
    --exp multi_object_open \
    --policy-type act \
    --checkpoint-dir outputs/train/act_multi_object_open_7221_scene_0_seed_0 \
    --initial-state-path data/multi_object_open/traj/summit_franka/scene_0_seed_0/7221 \
    --scene scene_0_seed_0 \
    --object 7221 \
    --headless

python scripts/pipeline/4_evaluate.py \
    --exp multi_object_open \
    --policy-type dp3 \
    --checkpoint-dir outputs/train/dp3_multi_object_open_7221_scene_0_seed_0 \
    --initial-state-path data/multi_object_open/traj/summit_franka/scene_0_seed_0/7221 \
    --scene scene_0_seed_0 \
    --object 7221 \
    --headless

python scripts/pipeline/4_evaluate.py \
    --exp multi_object_open \
    --policy-type act \
    --checkpoint-dir outputs/train/act_multi_object_open_11622_scene_25_seed_0 \
    --initial-state-path data/multi_object_open/traj/summit_franka/scene_0_seed_0/7221 \
    --scene scene_0_seed_0 \
    --object 7221 \
    --headless

# (2) DP3
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp3_multi_object_open_7221_scene_0_seed_0 \
    --dataset_root data/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0

# (3) Diffusion Policy
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp_multi_object_open_7221_scene_0_seed_0 \
    --dataset_root data/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0



###############################################################
# Example usage: [EXP] Single-object reach task
###############################################################

# Step 1: Generate plans
python scripts/pipeline/1_generate_plans.py --exp single_object_reach --scene scene_0_seed_0 --object 7221

# Step 2: Render dataset
python scripts/pipeline/2_render_dataset.py --exp single_object_reach --scene scene_0_seed_0 --object 7221 --max-episodes 100 --headless

# Step 3: Train policies
# (1) ACT
CUDA_VISIBLE_DEVICES=0
exp_name="single_object_reach_7221_scene_0_seed_0_1000"
dataset_root=data/single_object_reach/lerobot/$exp_name
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=10000 \
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
  --dataset.preload=true

# (2) DP3
CUDA_VISIBLE_DEVICES=1
exp_name="single_object_reach_7221_scene_0_seed_0_1000_dp3"
dataset_root=data/single_object_reach/lerobot/$exp_name
rm -rf outputs/train/dp3_$exp_name
lerobot-train \
  --policy.type=dp3 \
  --batch_size=128 \
  --steps=300000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=100000 \
  --job_name=dp3_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/dp3_$exp_name \
  --dataset.preload=true

# Step 4: Evaluate policies
python scripts/pipeline/4_evaluate.py \
    --run-dir outputs/train/dp3_single_object_reach_7221_scene_0_seed_0 \
    --dataset_root data/single_object_reach/lerobot/single_object_reach_7221_scene_0_seed_0

###############################################################
# Example usage: Utils of lerobot-dataset
###############################################################

# Local Viusalization
lerobot-dataset-viz \
    --repo-id single_object_open_test \
    --root data/single_object_open_test/lerobot/single_object_open_test \
    --episode-index 0 \
    --video-backend pyav

# Remote Viusalization
lerobot-dataset-viz \
    --repo-id multi_object_open_7221_scene_0_seed_0 \
    --root data/automoma-docker-1/multi_object_open/lerobot/multi_object_open_7221_scene_0_seed_0 \
    --episode-index 0 \
    --video-backend pyav \
    --save 1 \
    --output-dir ./viz_results

# Merge Dataset
dataset_root="$(pwd)/data/multi_object_open/lerobot_test"
exp_name_new="multi_object_open_merged"
repo_ids_str="[
    '$dataset_root/multi_object_open_11622_scene_18_seed_18',
    '$dataset_root/multi_object_open_7221_scene_0_seed_0',
    '$dataset_root/multi_object_open_46197_scene_38_seed_38'
]"
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id "$dataset_root/$exp_name_new" \
    --operation.type merge \
    --operation.repo_ids "$repo_ids_str"


# Split Dataset
exp_name="multi_object_open_7221_scene_0_seed_0"
dataset_root="$(pwd)/data/multi_object_open/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id $dataset_root \
  --operation.type split \
  --operation.splits '{"50": 0.1, "val": 0.1, "train": 0.8}'


# Remove Feature (for dp3)
# p.s. repo_id needs to be absolute path
exp_name="single_object_reach_7221_scene_0_seed_0_dp3"
dataset_root="$(pwd)/data/single_object_reach/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id $dataset_root \
        --operation.type remove_feature \
        --operation.feature_names "['observation.images.ego_topdown', 'observation.images.ego_wrist', 'observation.images.fix_local', 'observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"


exp_name="single_object_reach_7221_scene_0_seed_0"
dataset_root="$(pwd)/data/multi_object_open/lerobot/$exp_name"
python -m lerobot.scripts.lerobot_edit_dataset \
        --repo_id $dataset_root \
        --operation.type remove_feature \
        --operation.backup false \
        --operation.ignore_invalid true \
        --operation.feature_names "['observation.depth.ego_topdown', 'observation.depth.ego_wrist', 'observation.depth.fix_local', 'observation.eef']"

