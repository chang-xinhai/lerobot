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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_POINTCLOUD, OBS_PREFIX, OBS_STATE, REWARD
from lerobot.datasets.utils import dataset_to_policy_features

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata, allowed_keys: set[str] | None = None
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def resolve_required_keys(
    cfg: PreTrainedConfig | None, ds_meta: LeRobotDatasetMetadata
) -> set[str] | None:
    if cfg is None:
        return None

    policy_features = dataset_to_policy_features(ds_meta.features)
    action_keys = {key for key, ft in policy_features.items() if ft.type is FeatureType.ACTION}

    def _keys_by_types(types: set[FeatureType]) -> set[str]:
        return {key for key, ft in policy_features.items() if ft.type in types}

    # If the user explicitly configured input/output features, respect those.
    if cfg.input_features:
        input_keys = set(cfg.input_features.keys())
    else:
        if cfg.type == "dp3":
            input_keys = {
                key
                for key in policy_features
                if key in {OBS_STATE, OBS_POINTCLOUD} or "pointcloud" in key.lower()
            }
            logging.info(f"Resolved input keys for DP3 policy: {input_keys}")
        elif cfg.type in {
            "act",
            "diffusion",
            "vqbet",
            "tdmpc",
            "groot",
            "sac",
            "pi0",
            "pi0_fast",
            "pi05",
            "smolvla",
            "wall_x",
            "xvla",
            "rtc",
            "sarm",
        }:
            input_keys = _keys_by_types(
                {FeatureType.VISUAL, FeatureType.STATE, FeatureType.ENV, FeatureType.LANGUAGE}
            )
            input_keys = {
                key
                for key in input_keys
                if key != OBS_POINTCLOUD and "pointcloud" not in key.lower()
            }
            logging.info(
                f"Resolved input keys for {cfg.type} policy (typed selection): {input_keys}"
            )
        else:
            input_keys = {key for key in policy_features if key not in action_keys}

    if cfg.output_features:
        output_keys = set(cfg.output_features.keys())
    else:
        output_keys = set(action_keys)

    required_keys = input_keys | output_keys
    if cfg.reward_delta_indices is not None and REWARD in ds_meta.features:
        required_keys.add(REWARD)

    return required_keys


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        required_keys = (
            resolve_required_keys(cfg.policy, ds_meta) if cfg.dataset.filter_features_by_policy else None
        )
        logging.info(f"Required keys: {pformat(required_keys)}")
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta, allowed_keys=required_keys)

        # If policy features are not provided, set them based on the resolved required keys.
        if cfg.dataset.filter_features_by_policy and cfg.policy is not None and required_keys is not None:
            policy_features = dataset_to_policy_features(ds_meta.features)
            action_keys = {key for key, ft in policy_features.items() if ft.type is FeatureType.ACTION}
            if not cfg.policy.input_features:
                cfg.policy.input_features = {
                    key: ft for key, ft in policy_features.items() if key in required_keys
                    and key not in action_keys
                }
            if not cfg.policy.output_features:
                cfg.policy.output_features = {
                    key: ft for key, ft in policy_features.items() if key in action_keys
                }
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
                preload=cfg.dataset.preload,
                requested_keys=required_keys,
                video_decode_dtype=cfg.dataset.video_decode_dtype,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
                requested_keys=required_keys,
                video_decode_dtype=cfg.dataset.video_decode_dtype,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
