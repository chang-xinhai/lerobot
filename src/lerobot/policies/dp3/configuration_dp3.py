#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration class for 3D Diffusion Policy (DP3).

Based on "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations"
(paper: https://arxiv.org/abs/2403.03954, code: https://github.com/YanjieZe/3D-Diffusion-Policy)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("dp3")
@dataclass
class DP3Config(PreTrainedConfig):
    """Configuration class for 3D Diffusion Policy (DP3).

    DP3 uses point cloud observations as input and generates action sequences through
    a diffusion process. It combines PointNet-based encoding with a conditional U-Net
    for action generation.

    Notes on the inputs and outputs:
        - "observation.state" (agent_pos) is required as an input key.
        - "observation.pointcloud" is required as an input key containing point cloud data.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model action prediction size.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        obs_as_global_cond: Whether to use observations as global conditioning.
        condition_type: Type of conditioning for the U-Net ('film', 'add', 'cross_attention_add',
            'cross_attention_film', 'mlp_film').
        diffusion_step_embed_dim: Embedding dimension for diffusion timestep.
        down_dims: Feature dimensions for each stage of temporal downsampling in the U-Net.
        kernel_size: Convolutional kernel size for the U-Net.
        n_groups: Number of groups for GroupNorm in the U-Net.
        use_down_condition: Whether to use conditioning in the downsampling path.
        use_mid_condition: Whether to use conditioning in the middle block.
        use_up_condition: Whether to use conditioning in the upsampling path.
        encoder_output_dim: Output dimension of the point cloud encoder.
        use_pc_color: Whether to use RGB color information from point clouds.
        pointnet_type: Type of PointNet encoder ('pointnet').
        pointnet_use_layernorm: Whether to use LayerNorm in PointNet.
        pointnet_final_norm: Type of final normalization in PointNet ('none', 'layernorm').
        state_mlp_size: Hidden layer sizes for the state MLP.
        noise_scheduler_type: Type of noise scheduler ('DDPM', 'DDIM').
        num_train_timesteps: Number of diffusion steps for training.
        beta_schedule: Beta schedule type for diffusion.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        prediction_type: Prediction type ('epsilon', 'sample', 'v_prediction').
        clip_sample: Whether to clip samples during denoising.
        clip_sample_range: Range for sample clipping.
        num_inference_steps: Number of diffusion steps for inference.
        do_mask_loss_for_padding: Whether to mask loss for padded actions.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "POINTCLOUD": NormalizationMode.IDENTITY,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Observation conditioning
    obs_as_global_cond: bool = True

    # U-Net architecture
    condition_type: str = "film"
    diffusion_step_embed_dim: int = 256
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    use_down_condition: bool = True
    use_mid_condition: bool = True
    use_up_condition: bool = True

    # Point cloud encoder
    encoder_output_dim: int = 256
    use_pc_color: bool = False
    pointnet_type: str = "pointnet"
    pointnet_use_layernorm: bool = False
    pointnet_final_norm: str = "none"

    # State encoder
    state_mlp_size: tuple[int, ...] = (64, 64)

    # Noise scheduler
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        supported_condition_types = ["film", "add", "cross_attention_add", "cross_attention_film", "mlp_film"]
        if self.condition_type not in supported_condition_types:
            raise ValueError(
                f"`condition_type` must be one of {supported_condition_types}. Got {self.condition_type}."
            )

        supported_prediction_types = ["epsilon", "sample", "v_prediction"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )

        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        supported_pointnet_types = ["pointnet"]
        if self.pointnet_type not in supported_pointnet_types:
            raise ValueError(
                f"`pointnet_type` must be one of {supported_pointnet_types}. Got {self.pointnet_type}."
            )

        supported_final_norms = ["none", "layernorm"]
        if self.pointnet_final_norm not in supported_final_norms:
            raise ValueError(
                f"`pointnet_final_norm` must be one of {supported_final_norms}. "
                f"Got {self.pointnet_final_norm}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Validate that required features are present."""
        if self.robot_state_feature is None:
            raise ValueError("DP3 requires 'observation.state' (agent_pos) in input features.")

        if self.pointcloud_feature is None:
            raise ValueError("DP3 requires 'observation.pointcloud' in input features.")

        if self.action_feature is None:
            raise ValueError("DP3 requires 'action' in output features.")

    @property
    def pointcloud_feature(self):
        """Get the point cloud feature if present."""
        if not self.input_features:
            return None
        for key, ft in self.input_features.items():
            if "pointcloud" in key.lower():
                return ft
        return None

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
