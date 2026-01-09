# 3D Diffusion Policy (DP3)

This is the LeRobot implementation of **3D Diffusion Policy (DP3)** from the paper:

> [**3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations**](https://arxiv.org/abs/2403.03954)  
> Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, Huazhe Xu  
> *Robotics: Science and Systems (RSS) 2024*

## Overview

DP3 is a visuomotor policy that leverages 3D point cloud observations for generalizable robot learning. Unlike image-based policies, DP3 uses point clouds as input, which provides:

- Better generalization to novel viewpoints and scenes
- More robust handling of occlusions
- Efficient representation of 3D spatial information

The policy uses a PointNet-based encoder to extract features from point clouds, combined with a diffusion-based action generation process.

## Architecture

```
Point Cloud + State → PointNet Encoder → Conditional U-Net → Action Sequence
```

Key components:
- **PointNet Encoder**: Extracts global features from point cloud observations
- **State MLP**: Encodes robot proprioceptive state
- **Conditional U-Net**: Denoises action sequences conditioned on observations
- **Diffusion Process**: Iteratively refines random noise into coherent action trajectories

## Configuration

The policy can be configured with the following key parameters:

```python
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.configs.types import FeatureType, PolicyFeature

config = DP3Config(
    # Input/output structure
    n_obs_steps=2,           # Number of observation steps
    horizon=16,              # Action prediction horizon
    n_action_steps=8,        # Number of actions to execute

    # U-Net architecture
    condition_type="film",   # Type of conditioning (film, add, cross_attention_*)
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,

    # Point cloud encoder
    encoder_output_dim=256,
    use_pc_color=False,      # Whether to use RGB color from point clouds
    pointnet_type="pointnet",
    pointnet_use_layernorm=False,
    pointnet_final_norm="none",

    # State encoder
    state_mlp_size=(64, 64),

    # Diffusion settings
    noise_scheduler_type="DDPM",
    num_train_timesteps=100,
    prediction_type="epsilon",

    # Input/output features
    input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        'observation.pointcloud': PolicyFeature(type=FeatureType.VISUAL, shape=(4096, 6)),
    },
    output_features={
        'action': PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
    },
)
```

## Training

Train DP3 using the LeRobot training script:

```bash
lerobot-train \
    --policy.type dp3 \
    --dataset.repo_id your-dataset-with-pointcloud \
    --steps 200000
```

## Dataset Requirements

DP3 requires datasets with point cloud observations. The expected data format:

- `observation.state`: Robot proprioceptive state (e.g., joint positions, end-effector pose)
- `observation.pointcloud`: Point cloud data with shape `(num_points, channels)` where:
  - `channels=3` for XYZ only
  - `channels=6` for XYZRGB (set `use_pc_color=True`)
- `action`: Target actions

## Condition Types

DP3 supports multiple conditioning mechanisms:

- `"film"`: FiLM (Feature-wise Linear Modulation) conditioning
- `"add"`: Additive conditioning
- `"cross_attention_add"`: Cross-attention with additive output
- `"cross_attention_film"`: Cross-attention with FiLM modulation
- `"mlp_film"`: MLP-based FiLM conditioning

## References

- [Original DP3 Repository](https://github.com/YanjieZe/3D-Diffusion-Policy)
- [DP3 Project Page](https://3d-diffusion-policy.github.io/)
- [Paper on arXiv](https://arxiv.org/abs/2403.03954)

## Citation

```bibtex
@inproceedings{ze20243d,
  title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
  author={Ze, Yanjie and Zhang, Gu and Zhang, Kangning and Hu, Chenyuan and Wang, Muhan and Xu, Huazhe},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```
