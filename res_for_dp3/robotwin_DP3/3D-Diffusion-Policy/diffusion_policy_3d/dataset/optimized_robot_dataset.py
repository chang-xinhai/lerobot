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
import gc
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
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from termcolor import cprint
import psutil


class OptimizedReplayBuffer:
    """Optimized replay buffer that loads all data efficiently into memory"""
    
    def __init__(self, data_dict):
        self.data = data_dict["data"]
        self.meta = data_dict["meta"]
        self.episode_ends = self.meta["episode_ends"]
        self.n_episodes = len(self.episode_ends)
        
        # Print memory usage info
        total_size = 0
        for key, value in self.data.items():
            size_gb = value.nbytes / 1024**3
            total_size += size_gb
            cprint(f"Replay Buffer: {key}, shape {value.shape}, dtype {value.dtype}, size {size_gb:.2f}GB", "green")
        cprint(f"Total data size: {total_size:.2f}GB", "yellow")
        cprint("--------------------------", "green")
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __contains__(self, key):
        return key in self.data
    
    def keys(self):
        return self.data.keys()
    
    def items(self):
        return self.data.items()
    
    @classmethod
    def copy_from_path_optimized(cls, zarr_path, keys=None, progress_callback=None):
        """
        Optimized loading with memory monitoring and progress tracking
        """
        print(f"Loading data from: {zarr_path}")
        zarr_path = os.path.expanduser(zarr_path)
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024**3
        print(f"Available memory: {available_gb:.1f}GB")
        
        # Open zarr file
        group = zarr.open(zarr_path, "r")
        
        # Load metadata first (small)
        print("Loading metadata...")
        meta = {}
        for key, value in group["meta"].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]
        
        # Estimate data size
        if keys is None:
            keys = list(group["data"].keys())
        
        total_size_gb = 0
        for key in keys:
            arr = group["data"][key]
            size_gb = arr.nbytes / 1024**3
            total_size_gb += size_gb
            print(f"  {key}: {arr.shape}, {arr.dtype}, {size_gb:.2f}GB")
        
        print(f"Total estimated size: {total_size_gb:.2f}GB")
        
        if total_size_gb > available_gb * 0.8:  # Leave 20% buffer
            raise MemoryError(f"Not enough memory! Need {total_size_gb:.1f}GB, have {available_gb:.1f}GB")
        
        # Load data efficiently
        print("Loading data arrays...")
        data = {}
        
        for i, key in enumerate(keys):
            print(f"Loading {key} ({i+1}/{len(keys)})...")
            
            # Get current memory usage
            process = psutil.Process()
            current_memory_gb = process.memory_info().rss / 1024**3
            
            arr = group["data"][key]
            
            # Load in chunks if array is very large (>10GB)
            if arr.nbytes > 10 * 1024**3:
                print(f"  Large array detected, loading in chunks...")
                data[key] = cls._load_large_array_chunked(arr)
            else:
                # Direct load for smaller arrays
                data[key] = np.array(arr[:])
            
            # Memory cleanup
            gc.collect()
            
            new_memory_gb = process.memory_info().rss / 1024**3
            print(f"  Memory usage: {new_memory_gb:.1f}GB (Δ +{new_memory_gb - current_memory_gb:.1f}GB)")
            
            if progress_callback:
                progress_callback(i + 1, len(keys), key)
        
        print("Data loading completed successfully!")
        
        return cls({"data": data, "meta": meta})
    
    @staticmethod
    def _load_large_array_chunked(zarr_array, chunk_size_gb=2):
        """Load large arrays in chunks to avoid memory spikes"""
        print(f"    Loading {zarr_array.shape} array in chunks...")
        
        # Calculate chunk size in the first dimension
        element_size = zarr_array.dtype.itemsize
        elements_per_gb = (1024**3) // element_size
        
        if len(zarr_array.shape) > 1:
            elements_per_row = np.prod(zarr_array.shape[1:])
            rows_per_chunk = max(1, (chunk_size_gb * elements_per_gb) // elements_per_row)
        else:
            rows_per_chunk = chunk_size_gb * elements_per_gb
        
        rows_per_chunk = min(rows_per_chunk, zarr_array.shape[0])
        
        # Pre-allocate output array
        result = np.empty(zarr_array.shape, dtype=zarr_array.dtype)
        
        # Load in chunks
        for start_idx in range(0, zarr_array.shape[0], rows_per_chunk):
            end_idx = min(start_idx + rows_per_chunk, zarr_array.shape[0])
            print(f"    Loading chunk {start_idx}:{end_idx}")
            
            chunk = zarr_array[start_idx:end_idx]
            result[start_idx:end_idx] = chunk
            
            # Clean up chunk reference
            del chunk
            gc.collect()
        
        return result


class OptimizedRobotDataset(BaseDataset):
    """Fast robot dataset that loads all data into memory efficiently"""

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
    ):
        super().__init__()
        self.task_name = task_name
        
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)
        
        print("=" * 60)
        print("🚀 OPTIMIZED DATA LOADING")
        print("=" * 60)
        
        # Load all data into memory efficiently
        def progress_callback(current, total, key_name):
            progress = (current / total) * 100
            print(f"    Progress: {progress:.1f}% ({current}/{total}) - Loaded {key_name}")
        
        try:
            self.replay_buffer = OptimizedReplayBuffer.copy_from_path_optimized(
                zarr_path, 
                keys=["state", "action", "point_cloud"],
                progress_callback=progress_callback
            )
            print("✅ All data loaded successfully into memory!")
        except MemoryError as e:
            print(f"❌ Memory error: {e}")
            print("💡 Falling back to lazy loading...")
            # Fall back to lazy loading if needed
            from diffusion_policy_3d.dataset.memory_efficient_robot_dataset import MemoryEfficientRobotDataset
            fallback = MemoryEfficientRobotDataset(
                zarr_path=zarr_path.replace(os.path.join(parent_directory, ""), ""),
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                task_name=task_name,
                cache_size=200  # Large cache as fallback
            )
            # Copy fallback attributes
            self.__dict__.update(fallback.__dict__)
            return
        
        print("=" * 60)
        
        # Set up samplers (same as original)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)
        
        self.sampler = SequenceSampler(
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

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"][..., :],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
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