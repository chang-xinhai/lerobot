#!/usr/bin/env python3
"""
Hyperparameter optimizer for memory-safe training
Analyzes system resources and suggests optimal settings
"""
import psutil
import subprocess
import os

def get_gpu_info():
    """Get GPU memory information"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                gpus.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'total_mb': int(parts[2]),
                    'used_mb': int(parts[3]),
                    'free_mb': int(parts[4])
                })
            return gpus
        return None
    except:
        return None

def get_cpu_info():
    """Get CPU and memory information"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    cpu_count_logical = psutil.cpu_count(logical=True)  # Logical cores
    memory = psutil.virtual_memory()
    
    return {
        'physical_cores': cpu_count,
        'logical_cores': cpu_count_logical,
        'total_memory_gb': memory.total / 1024**3,
        'available_memory_gb': memory.available / 1024**3,
        'used_memory_gb': memory.used / 1024**3
    }

def suggest_hyperparameters():
    """Suggest optimal hyperparameters based on system specs"""
    print("🔧 Hyperparameter Optimization for Memory-Safe Training")
    print("=" * 60)
    
    # Get system info
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    
    print("\n📊 System Analysis:")
    print(f"CPU: {cpu_info['physical_cores']} physical cores, {cpu_info['logical_cores']} logical cores")
    print(f"RAM: {cpu_info['total_memory_gb']:.1f}GB total, {cpu_info['available_memory_gb']:.1f}GB available")
    
    if gpu_info:
        for gpu in gpu_info:
            print(f"GPU {gpu['id']}: {gpu['name']} - {gpu['total_mb']/1024:.1f}GB total, {gpu['free_mb']/1024:.1f}GB free")
    
    print("\n🎯 Optimized Hyperparameter Suggestions:")
    print("-" * 60)
    
    # Memory-based suggestions
    available_memory = cpu_info['available_memory_gb']
    
    # Training batch size suggestions
    if gpu_info:
        max_gpu_free = max(gpu['free_mb']/1024 for gpu in gpu_info)
        
        if max_gpu_free > 40:
            train_batch = 64
            speed_level = "🚀 High Speed"
        elif max_gpu_free > 20:
            train_batch = 32
            speed_level = "⚡ Medium-High Speed"
        elif max_gpu_free > 10:
            train_batch = 16
            speed_level = "🏃 Medium Speed"
        else:
            train_batch = 8
            speed_level = "🚶 Conservative Speed"
    else:
        train_batch = 16
        speed_level = "🤷 Unknown GPU"
    
    # Data processing batch size
    if available_memory > 100:
        data_batch = 200
        cache_size = 100
    elif available_memory > 50:
        data_batch = 100
        cache_size = 50
    elif available_memory > 20:
        data_batch = 50
        cache_size = 25
    else:
        data_batch = 25
        cache_size = 10
    
    # Workers based on CPU cores
    max_workers = min(cpu_info['physical_cores'], 8)  # Cap at 8 for stability
    train_workers = max(2, max_workers // 2)
    
    # Gradient accumulation to maintain effective batch size
    target_effective_batch = 128  # Standard effective batch size
    grad_accumulate = max(1, target_effective_batch // train_batch)
    
    print(f"🎯 **{speed_level}** Configuration:")
    print(f"   Training batch size: {train_batch}")
    print(f"   Data processing batch: {data_batch}")
    print(f"   Cache size: {cache_size} episodes")
    print(f"   Workers: {train_workers}")
    print(f"   Gradient accumulation: {grad_accumulate}")
    print(f"   Effective batch size: {train_batch * grad_accumulate}")
    
    print(f"\n🚀 **Command to run:**")
    print(f"./train_safe.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 {train_batch} {data_batch}")
    
    print(f"\n⚡ **Speed Optimization Tips:**")
    print("1. **Pin Memory**: Enabled by default (faster GPU transfers)")
    print("2. **Persistent Workers**: Disabled by default (saves memory)")
    print("3. **Mixed Precision**: Consider adding for 2x speed boost")
    print("4. **Compile Model**: PyTorch 2.0 compile for 10-20% speedup")
    
    print(f"\n🛡️ **Memory Safety Settings:**")
    print(f"   Estimated GPU memory usage: ~{train_batch * 0.3:.1f}GB")
    print(f"   Estimated CPU memory usage: ~{cache_size * 0.06:.1f}GB for cache")
    print(f"   Data processing peak: ~{data_batch * 0.06:.1f}GB")
    
    # Advanced suggestions
    print(f"\n🔧 **Advanced Optimizations:**")
    print("1. **Increase cache_size** if you have more RAM available")
    print("2. **Use larger data_batch** for faster data processing")
    print("3. **Enable mixed precision** with: training.use_amp=True")
    print("4. **Reduce validation frequency** with: training.val_every=100")
    
    # Conservative and aggressive options
    print(f"\n📈 **Alternative Configurations:**")
    
    # Conservative (minimum memory)
    conservative_train = max(4, train_batch // 4)
    conservative_data = max(10, data_batch // 4)
    conservative_cache = max(5, cache_size // 4)
    print(f"   🐌 **Ultra Conservative** (minimum memory):")
    print(f"      ./train_safe.sh ... {conservative_train} {conservative_data}")
    print(f"      Cache size: {conservative_cache}")
    
    # Aggressive (maximum speed)
    if available_memory > 50 and gpu_info and max(gpu['free_mb']/1024 for gpu in gpu_info) > 20:
        aggressive_train = min(128, train_batch * 2)
        aggressive_data = min(500, data_batch * 2)
        aggressive_cache = min(200, cache_size * 2)
        print(f"   🚀 **Aggressive** (maximum speed, higher memory):")
        print(f"      ./train_safe.sh ... {aggressive_train} {aggressive_data}")
        print(f"      Cache size: {aggressive_cache}")
    
    return {
        'train_batch': train_batch,
        'data_batch': data_batch,
        'cache_size': cache_size,
        'workers': train_workers,
        'grad_accumulate': grad_accumulate
    }

def create_optimized_config(params):
    """Create an optimized configuration file"""
    config_content = f"""# Optimized configuration based on system analysis
defaults:
  - _self_
  - task: automoma_manip_summit_franka_memory_safe

name: dp3_optimized

task_name: null
shape_meta: ${{task.shape_meta}}
exp_name: "optimized_training"

horizon: 8
n_obs_steps: 3
n_action_steps: 6
n_latency_steps: 0
dataset_obs_steps: ${{n_obs_steps}}
keypoint_visible_rate: 1.0
obs_as_global_cond: True

policy:
  _target_: diffusion_policy_3d.policy.dp3.DP3
  use_point_crop: true
  condition_type: film
  use_down_condition: true
  use_mid_condition: true
  use_up_condition: true
  
  diffusion_step_embed_dim: 64
  down_dims:
  - 512
  - 1024
  kernel_size: 5
  n_groups: 8
  use_film_scale_modulation: true
  encoder_output_dim: 256
  cond_predict_scale: True
  
  n_action_steps: ${{n_action_steps}}
  n_obs_steps: ${{n_obs_steps}}
  horizon: ${{horizon}}
  obs_as_global_cond: ${{obs_as_global_cond}}
  pred_action_steps_only: False
  
  obs_encoder:
    _target_: diffusion_policy_3d.model.vision.multi_view_rgb_encoder.MultiViewRgbEncoder
    backbone: resnet18
    pretrained: False
    global_pool: False
    num_views: 1
    
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddpm.DDPMScheduler
    num_train_timesteps: 100
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    variance_type: fixed_small
    clip_sample: True
    prediction_type: epsilon
    
  use_pc_color: false
  pointcloud_encoder_cfg:
    in_channels: 3
    out_channels: ${{policy.encoder_output_dim}}
    use_layernorm: true
    final_norm: layernorm
    normal_channel: false

ema:
  _target_: diffusion_policy_3d.model.diffusion.ema_model.EMAModel
  update_after_step: 0
  inv_gamma: 1.0
  power: 0.75
  min_value: 0.0
  max_value: 0.9999

# Optimized dataloader settings
dataloader:
  batch_size: {params['train_batch']}
  num_workers: {params['workers']}
  shuffle: True
  pin_memory: True
  persistent_workers: False
  prefetch_factor: 2
  drop_last: True

val_dataloader:
  batch_size: {params['train_batch']}
  num_workers: {params['workers']}
  shuffle: False
  pin_memory: True
  persistent_workers: False
  prefetch_factor: 2

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  betas: [0.95, 0.999]
  eps: 1.0e-8
  weight_decay: 1.0e-6

training:
  device: "cuda:0"
  seed: 42
  debug: False
  resume: True
  lr_scheduler: cosine
  lr_warmup_steps: 500
  num_epochs: 500
  gradient_accumulate_every: {params['grad_accumulate']}
  use_ema: True
  rollout_every: 200
  checkpoint_every: 50  # More frequent checkpoints
  val_every: 25        # Less frequent validation for speed
  sample_every: 20
  max_train_steps: null
  max_val_steps: 50    # Limit validation steps for speed
  tqdm_interval_sec: 1.0

logging:
  group: ${{exp_name}}
  id: null
  mode: online
  name: ${{exp_name}}
  project: RoboTwin
  resume: never
  tags:
  - RoboTwin
  - Optimized

checkpoint:
  save_ckpt: True
  topk:
    monitor_key: test_mean_score
    mode: max
    k: 3
    format_str: 'epoch={{epoch:04d}}-score={{test_mean_score:.3f}}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False
  
hydra:
  job:
    override_dirname: ${{name}}
  run:
    dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
  sweep:
    dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
    subdir: ${{hydra.job.num}}

multi_run:
  run_dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
  wandb_name_base: ${{now:%Y.%m.%d-%H.%M.%S}}_${{name}}_${{task_name}}

checkpoint_num: 3000
expert_data_num: 100
raw_task_name: none
setting: none
"""
    
    return config_content

if __name__ == "__main__":
    params = suggest_hyperparameters()
    
    print(f"\n💾 Creating optimized configuration file...")
    config_content = create_optimized_config(params)
    
    # Save to file
    config_path = "/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_optimized.yaml"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Saved optimized config to: robot_dp3_optimized.yaml")
    
    # Update task config with optimized cache size
    task_config_content = f"""name: ${{task_name}}-${{setting}}-${{expert_data_num}}

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [1024, 6]
      type: point_cloud
    agent_pos:
      shape: [11]
      type: low_dim
  action:
    shape: [11]

env_runner:
  _target_: diffusion_policy_3d.env_runner.robot_runner.RobotRunner
  max_steps: 300
  n_obs_steps: ${{n_obs_steps}}
  n_action_steps: ${{n_action_steps}}
  task_name: robot

dataset:
  _target_: diffusion_policy_3d.dataset.memory_efficient_robot_dataset.MemoryEfficientRobotDataset
  zarr_path: ../../../data/${{task.name}}.zarr
  horizon: ${{horizon}}
  pad_before: ${{eval:'${{n_obs_steps}}-1'}}
  pad_after: ${{eval:'${{n_action_steps}}-1'}}
  seed: 0
  val_ratio: 0.02
  max_train_episodes: null
  cache_size: {params['cache_size']}  # Optimized cache size
"""
    
    task_config_path = "/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/automoma_manip_summit_franka_optimized.yaml"
    
    with open(task_config_path, 'w') as f:
        f.write(task_config_content)
    
    print(f"✅ Saved optimized task config with cache_size={params['cache_size']}")
    
    print(f"\n🎯 **Quick Start Commands:**")
    print(f"1. Test the configuration:")
    print(f"   python scripts/test_memory_dataset.py")
    print(f"")
    print(f"2. Run optimized training:")
    print(f"   ./train_safe.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 {params['train_batch']} {params['data_batch']}")
    print(f"")
    print(f"3. Monitor during training:")
    print(f"   python scripts/monitor_gpu_memory.py")