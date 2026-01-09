#!/usr/bin/env python3
"""
Test script to verify optimized fast dataset loading
"""
import sys
import os
import pathlib
import time

# Add paths for imports
DP3_ROOT = str(pathlib.Path(__file__).parent.parent)
sys.path.append(DP3_ROOT)
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy', 'diffusion_policy_3d'))

import psutil
import torch
from torch.utils.data import DataLoader

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def test_optimized_dataset():
    """Test the optimized fast dataset"""
    print("🚀 TESTING OPTIMIZED FAST DATASET")
    print("=" * 60)
    
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / 1024**3
    print(f"Available memory: {available_gb:.1f} GB")
    
    if available_gb < 90:
        print("⚠️  WARNING: You might not have enough memory for fast loading!")
        print("   Recommended: 90GB+ available memory")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Test aborted.")
            return False
    
    try:
        from diffusion_policy_3d.dataset.optimized_robot_dataset import OptimizedRobotDataset
        
        print("\n🏁 Starting optimized dataset initialization...")
        start_time = time.time()
        
        dataset = OptimizedRobotDataset(
            zarr_path="../../../data/automoma_manip_summit_franka-task_1object_15scene_20pose-15000.zarr",
            horizon=8,
            pad_before=2,
            pad_after=5,
            seed=0,
            val_ratio=0.02,
        )
        
        load_time = time.time() - start_time
        after_init_memory = get_memory_usage()
        
        print(f"\n✅ Dataset initialization completed!")
        print(f"Load time: {load_time:.1f} seconds")
        print(f"Memory after loading: {after_init_memory:.2f} GB (Δ: {after_init_memory - initial_memory:.2f} GB)")
        print(f"Dataset length: {len(dataset)}")
        
        # Test data access speed
        print("\n⚡ Testing data access speed...")
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Test 10 batches and measure speed
        start_time = time.time()
        batch_count = 0
        
        for i, batch in enumerate(dataloader):
            if i >= 10:  # Test 10 batches
                break
            
            batch_count += 1
            
            if i == 0:
                print(f"  Batch shapes:")
                print(f"    Point cloud: {batch['obs']['point_cloud'].shape}")
                print(f"    Agent pos: {batch['obs']['agent_pos'].shape}")
                print(f"    Action: {batch['action'].shape}")
        
        total_time = time.time() - start_time
        avg_time_per_batch = total_time / batch_count
        batches_per_second = batch_count / total_time
        
        print(f"\n📊 Performance Results:")
        print(f"  Batches processed: {batch_count}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per batch: {avg_time_per_batch:.3f} seconds")
        print(f"  Batches per second: {batches_per_second:.1f}")
        print(f"  Samples per second: {batches_per_second * 32:.0f}")
        
        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"\n✅ Validation dataset length: {len(val_dataset)}")
        
        final_memory = get_memory_usage()
        print(f"\nFinal memory usage: {final_memory:.2f} GB (Total Δ: {final_memory - initial_memory:.2f} GB)")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if batches_per_second > 10:
            print("  🚀 EXCELLENT: Very fast data loading!")
        elif batches_per_second > 5:
            print("  ⚡ GOOD: Fast data loading")
        elif batches_per_second > 2:
            print("  🏃 MODERATE: Acceptable speed")
        else:
            print("  🐌 SLOW: Consider memory-safe version")
        
        if final_memory - initial_memory < 80:
            print("  💚 Memory usage within expected range")
        else:
            print("  ⚠️  High memory usage - monitor during training")
        
        print("\n✅ Fast dataset test completed successfully!")
        return True
        
    except MemoryError as e:
        print(f"❌ Memory Error: {e}")
        print("💡 Your system doesn't have enough memory for fast loading.")
        print("   Use ./train_safe.sh instead for memory-efficient training.")
        return False
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d')
    success = test_optimized_dataset()
    
    if success:
        print("\n🎉 READY FOR FAST TRAINING!")
        print("Use: ./train_fast.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 64")
    else:
        print("\n💡 Consider using memory-safe training instead:")
        print("Use: ./train_safe.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 32 50")