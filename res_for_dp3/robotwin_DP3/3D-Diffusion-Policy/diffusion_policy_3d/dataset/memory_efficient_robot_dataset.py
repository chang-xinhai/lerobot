import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
import numpy as np
import copy
import zarr
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb


class MemoryEfficientReplayBuffer:
    """Memory-efficient replay buffer that loads data on-demand"""
    
    def __init__(self, zarr_path, keys=None):
        self.zarr_path = os.path.expanduser(zarr_path)
        self.root = zarr.open(self.zarr_path, "r")
        self.keys = keys if keys is not None else list(self.root["data"].keys())
        
        # Load only metadata
        self.episode_ends = self.root["meta"]["episode_ends"][:]
        self.n_episodes = len(self.episode_ends)
        
        # Cache data shapes and dtypes without loading the data
        self._data_info = {}
        for key in self.keys:
            data_array = self.root["data"][key]
            self._data_info[key] = {
                'shape': data_array.shape,
                'dtype': data_array.dtype,
                'chunks': data_array.chunks
            }
    
    def __getitem__(self, key):
        """Return a lazy data accessor"""
        if key in self.keys:
            return LazyDataAccessor(self.root["data"][key])
        else:
            raise KeyError(f"Key {key} not found in replay buffer")
    
    def get_episode_range(self, episode_idx):
        """Get start and end indices for an episode"""
        if episode_idx == 0:
            start_idx = 0
        else:
            start_idx = self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        return start_idx, end_idx
    
    def load_episode_data(self, episode_idx, keys=None):
        """Load data for a specific episode"""
        keys = keys if keys is not None else self.keys
        start_idx, end_idx = self.get_episode_range(episode_idx)
        
        episode_data = {}
        for key in keys:
            episode_data[key] = self.root["data"][key][start_idx:end_idx]
        
        return episode_data


class LazyDataAccessor:
    """Lazy accessor for zarr arrays"""
    
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
    
    def __getitem__(self, idx):
        return self.zarr_array[idx]
    
    def __len__(self):
        return len(self.zarr_array)


class MemoryEfficientRobotDataset(BaseDataset):
    """Memory-efficient version of RobotDataset that loads data on-demand"""

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        cache_size=100,  # Number of episodes to cache in memory
    ):
        super().__init__()
        self.task_name = task_name
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for episodes
        self.cache_order = []  # For LRU eviction
        
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        
        # Use memory-efficient replay buffer
        self.replay_buffer = MemoryEfficientReplayBuffer(zarr_path, keys=["state", "action", "point_cloud"])
        
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        
        # Create a custom sampler that works with lazy loading
        self.sampler = MemoryEfficientSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # Load a small sample for normalizer computation
        self._prepare_normalizer_data()

    def _prepare_normalizer_data(self):
        """Load a subset of data for computing normalizer statistics"""
        # Load data from first 50 episodes (or all if less than 50)
        n_episodes_for_norm = min(50, self.replay_buffer.n_episodes)
        
        all_actions = []
        all_states = []
        all_point_clouds = []
        
        for ep_idx in range(n_episodes_for_norm):
            episode_data = self.replay_buffer.load_episode_data(ep_idx)
            all_actions.append(episode_data["action"])
            all_states.append(episode_data["state"])
            all_point_clouds.append(episode_data["point_cloud"])
        
        self.normalizer_data = {
            "action": np.concatenate(all_actions, axis=0),
            "agent_pos": np.concatenate(all_states, axis=0),
            "point_cloud": np.concatenate(all_point_clouds, axis=0),
        }

    def get_episode_data(self, episode_idx):
        """Get episode data with caching"""
        if episode_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(episode_idx)
            self.cache_order.append(episode_idx)
            return self.cache[episode_idx]
        
        # Load episode data
        episode_data = self.replay_buffer.load_episode_data(episode_idx)
        
        # Add to cache
        self.cache[episode_idx] = episode_data
        self.cache_order.append(episode_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest_episode = self.cache_order.pop(0)
            del self.cache[oldest_episode]
        
        return episode_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = MemoryEfficientSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(data=self.normalizer_data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"][
            :,
        ].astype(np.float32)
        point_cloud = sample["point_cloud"][
            :,
        ].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class MemoryEfficientSequenceSampler:
    """Memory-efficient sequence sampler"""
    
    def __init__(
        self,
        replay_buffer,
        sequence_length,
        pad_before=0,
        pad_after=0,
        episode_mask=None,
        key_first_k=dict(),
    ):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.sequence_length = sequence_length
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.key_first_k = key_first_k

        if episode_mask is None:
            episode_mask = np.ones(replay_buffer.n_episodes, dtype=bool)
        
        episode_starts = np.array([0] + list(replay_buffer.episode_ends[:-1]))
        episode_ends = replay_buffer.episode_ends
        episode_lengths = episode_ends - episode_starts
        
        # Filter episodes
        valid_episodes = np.where(episode_mask)[0]
        valid_starts = episode_starts[valid_episodes]
        valid_ends = episode_ends[valid_episodes]
        valid_lengths = episode_lengths[valid_episodes]
        
        # Calculate valid indices for each episode
        self.episode_to_indices = []
        self.indices_to_episode = []
        
        total_indices = 0
        for i, (ep_idx, start, end, length) in enumerate(zip(valid_episodes, valid_starts, valid_ends, valid_lengths)):
            # Number of valid starting positions in this episode
            n_valid = length - sequence_length + 1
            if n_valid > 0:
                self.episode_to_indices.append((ep_idx, start, n_valid, total_indices))
                self.indices_to_episode.extend([i] * n_valid)
                total_indices += n_valid
        
        self.indices_to_episode = np.array(self.indices_to_episode)
        self.total_indices = total_indices

    def sample_sequence(self, idx):
        # Find which episode this index belongs to
        episode_info_idx = self.indices_to_episode[idx]
        ep_idx, ep_start, n_valid, ep_indices_start = self.episode_to_indices[episode_info_idx]
        
        # Find position within episode
        pos_in_episode = idx - ep_indices_start
        
        # Load episode data (this will use caching if dataset supports it)
        if hasattr(self.replay_buffer, 'load_episode_data'):
            episode_data = self.replay_buffer.load_episode_data(ep_idx)
        else:
            # Fallback for regular replay buffer
            start_idx, end_idx = self.replay_buffer.get_episode_range(ep_idx)
            episode_data = {}
            for key in self.replay_buffer.keys:
                episode_data[key] = self.replay_buffer[key][start_idx:end_idx]
        
        # Extract sequence
        seq_start = pos_in_episode
        seq_end = seq_start + self.sequence_length
        
        sample = {}
        for key, data in episode_data.items():
            sample[key] = data[seq_start:seq_end]
        
        return sample

    def __len__(self):
        return self.total_indices