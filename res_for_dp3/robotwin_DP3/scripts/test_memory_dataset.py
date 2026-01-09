#!/usr/bin/env python3
"""
Test script to verify memory-efficient dataset loading
"""
import sys
import os
import pathlib

# Add paths for imports
DP3_ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(DP3_ROOT)
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy', 'diffusion_policy_3d'))

import psutil
import torch
from torch.utils.data import DataLoader
from diffusion_policy_3d.dataset.memory_efficient_robot_dataset import MemoryEfficientRobotDataset

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def test_memory_efficient_dataset():
    """Test the memory-efficient dataset"""
    print("Testing Memory-Efficient Dataset")
    print("=" * 50)
    
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    # Test dataset loading
    try:
        dataset = MemoryEfficientRobotDataset(
            zarr_path="../../../data/automoma_manip_summit_franka-task_1object_15scene_20pose-15000.zarr",
            horizon=8,
            pad_before=2,
            pad_after=5,
            seed=0,
            val_ratio=0.02,
            cache_size=10  # Small cache for testing
        )
        
        after_init_memory = get_memory_usage()
        print(f"Memory after dataset init: {after_init_memory:.2f} GB (Δ: {after_init_memory - initial_memory:.2f} GB)")
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Number of episodes: {dataset.replay_buffer.n_episodes}")
        
        # Test data loading
        print("\nTesting data loading...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        # Load a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test 3 batches
                break
            
            current_memory = get_memory_usage()
            print(f"Batch {i+1} - Memory: {current_memory:.2f} GB (Δ: {current_memory - initial_memory:.2f} GB)")
            print(f"  Point cloud shape: {batch['obs']['point_cloud'].shape}")
            print(f"  Agent pos shape: {batch['obs']['agent_pos'].shape}")
            print(f"  Action shape: {batch['action'].shape}")
        
        final_memory = get_memory_usage()
        print(f"\nFinal memory usage: {final_memory:.2f} GB (Total Δ: {final_memory - initial_memory:.2f} GB)")
        
        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"Validation dataset length: {len(val_dataset)}")
        
        print("\n✅ Memory-efficient dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d')
    success = test_memory_efficient_dataset()
    
    if success:
        print("\nDataset is ready for training with reduced memory usage!")
    else:
        print("\nPlease check the dataset configuration.")