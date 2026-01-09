#!/usr/bin/env python
"""
Merge two zarr datasets into one using batch processing.

Usage:
    python merge_scenes.py --input_zarr1 <path_to_zarr1> --input_zarr2 <path_to_zarr2>
    python merge_scenes.py --input_zarr1 <path1> --input_zarr2 <path2> --output_zarr <output_path>
    
Example:
    python merge_scenes.py \
        --input_zarr1 ./data/automoma_manip_summit_franka_fixed_base-task_1object_1scene_20pose-6400.zarr \
        --input_zarr2 ./data/automoma_manip_summit_franka_fixed_base-task_1object_1scene_20pose_new-6400.zarr
    
    python merge_scenes.py \
        --input_zarr1 ./data/dataset1.zarr \
        --input_zarr2 ./data/dataset2.zarr \
        --output_zarr ./data/merged-12800.zarr \
        --batch_size 500
"""

import os
import argparse
import zarr
import numpy as np
import shutil


def merge_zarr_datasets(input_zarr1_path, input_zarr2_path, output_zarr_path=None, batch_size=1000):
    """
    Merge two zarr datasets into one using batch processing to handle large datasets.
    
    Args:
        input_zarr1_path: Path to first input zarr file
        input_zarr2_path: Path to second input zarr file
        output_zarr_path: Path to output zarr file (default: auto-generated based on total trajectories)
        batch_size: Number of episodes to process in each batch (default: 1000)
    """
    # Convert to absolute paths and validate
    input_zarr1_path = os.path.abspath(input_zarr1_path)
    input_zarr2_path = os.path.abspath(input_zarr2_path)
    
    if not os.path.exists(input_zarr1_path):
        raise FileNotFoundError(f"Input zarr file 1 not found: {input_zarr1_path}")
    
    if not os.path.exists(input_zarr2_path):
        raise FileNotFoundError(f"Input zarr file 2 not found: {input_zarr2_path}")
    
    # Check for nested zarr structure and fix if needed
    def get_actual_zarr_path(path):
        """Handle cases where zarr is nested in a directory with the same name."""
        # Check if the path contains data and meta groups
        try:
            test_root = zarr.open(path, 'r')
            if 'data' in test_root and 'meta' in test_root:
                return path
        except:
            pass
        
        # Check if there's a nested directory with the same name
        basename = os.path.basename(path)
        nested_path = os.path.join(path, basename)
        if os.path.exists(nested_path):
            try:
                test_root = zarr.open(nested_path, 'r')
                if 'data' in test_root and 'meta' in test_root:
                    print(f"  Note: Using nested path: {nested_path}")
                    return nested_path
            except:
                pass
        
        return path
    
    input_zarr1_path = get_actual_zarr_path(input_zarr1_path)
    input_zarr2_path = get_actual_zarr_path(input_zarr2_path)
    
    # Open input zarr files
    print(f"Opening input zarr 1: {input_zarr1_path}")
    input_root1 = zarr.open(input_zarr1_path, 'r')
    
    print(f"Opening input zarr 2: {input_zarr2_path}")
    input_root2 = zarr.open(input_zarr2_path, 'r')
    
    # Get episode information from both datasets
    episode_ends1 = input_root1['meta/episode_ends'][:]
    episode_ends2 = input_root2['meta/episode_ends'][:]
    
    total_episodes1 = len(episode_ends1)
    total_episodes2 = len(episode_ends2)
    total_episodes = total_episodes1 + total_episodes2
    
    total_timesteps1 = episode_ends1[-1] if len(episode_ends1) > 0 else 0
    total_timesteps2 = episode_ends2[-1] if len(episode_ends2) > 0 else 0
    total_timesteps = total_timesteps1 + total_timesteps2
    
    print(f"\nDataset 1: {total_episodes1} episodes, {total_timesteps1} timesteps")
    print(f"Dataset 2: {total_episodes2} episodes, {total_timesteps2} timesteps")
    print(f"Total after merge: {total_episodes} episodes, {total_timesteps} timesteps")
    
    # Verify robot names match
    robot_name1 = input_root1['meta'].attrs.get('robot_name', 'unknown')
    robot_name2 = input_root2['meta'].attrs.get('robot_name', 'unknown')
    
    if robot_name1 != robot_name2:
        print(f"WARNING: Robot names do not match! Dataset 1: {robot_name1}, Dataset 2: {robot_name2}")
        print("Proceeding with merge anyway, using robot name from dataset 1.")
    
    robot_name = robot_name1
    
    # Verify data shapes are compatible
    pc_shape1 = input_root1['data/point_cloud'].shape
    pc_shape2 = input_root2['data/point_cloud'].shape
    
    state_shape1 = input_root1['data/state'].shape
    state_shape2 = input_root2['data/state'].shape
    
    action_shape1 = input_root1['data/action'].shape
    action_shape2 = input_root2['data/action'].shape
    
    if pc_shape1[1:] != pc_shape2[1:]:
        raise ValueError(f"Point cloud shapes incompatible: {pc_shape1[1:]} vs {pc_shape2[1:]}")
    
    if state_shape1[1:] != state_shape2[1:]:
        raise ValueError(f"State shapes incompatible: {state_shape1[1:]} vs {state_shape2[1:]}")
    
    if action_shape1[1:] != action_shape2[1:]:
        raise ValueError(f"Action shapes incompatible: {action_shape1[1:]} vs {action_shape2[1:]}")
    
    print(f"\nData shapes verified:")
    print(f"  Point cloud: {pc_shape1[1:]}")
    print(f"  State: {state_shape1[1:]}")
    print(f"  Action: {action_shape1[1:]}")
    
    # Determine output path
    if output_zarr_path is None:
        output_dir = os.path.dirname(input_zarr1_path)
        output_name = f"merge-{total_episodes}.zarr"
        output_zarr_path = os.path.join(output_dir, output_name)
    else:
        output_zarr_path = os.path.abspath(output_zarr_path)
    
    # Remove existing output if it exists
    if os.path.exists(output_zarr_path):
        print(f"\nRemoving existing output: {output_zarr_path}")
        shutil.rmtree(output_zarr_path)
    
    # Create output zarr
    print(f"Creating output zarr: {output_zarr_path}")
    output_root = zarr.group(output_zarr_path)
    output_data = output_root.create_group("data")
    output_meta = output_root.create_group("meta")
    
    # Copy compression settings
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Initialize zarr datasets with proper dimensions
    point_cloud_shape = (0,) + pc_shape1[1:]
    state_shape = (0,) + state_shape1[1:]
    action_shape = (0,) + action_shape1[1:]
    
    zarr_point_cloud = output_data.create_dataset(
        "point_cloud",
        shape=point_cloud_shape,
        chunks=input_root1['data/point_cloud'].chunks,
        dtype=input_root1['data/point_cloud'].dtype,
        compressor=compressor,
    )
    
    zarr_state = output_data.create_dataset(
        "state",
        shape=state_shape,
        chunks=input_root1['data/state'].chunks,
        dtype=input_root1['data/state'].dtype,
        compressor=compressor,
    )
    
    zarr_action = output_data.create_dataset(
        "action",
        shape=action_shape,
        chunks=input_root1['data/action'].chunks,
        dtype=input_root1['data/action'].dtype,
        compressor=compressor,
    )
    
    # Process dataset 1 in batches
    print(f"\nMerging dataset 1 in batches (batch_size={batch_size} episodes)...")
    current_ep = 0
    
    while current_ep < total_episodes1:
        batch_end = min(current_ep + batch_size, total_episodes1)
        print(f"  Processing episodes {current_ep + 1}-{batch_end} / {total_episodes1}")
        
        # Get timestep range for this batch
        start_ts = 0 if current_ep == 0 else episode_ends1[current_ep - 1]
        end_ts = episode_ends1[batch_end - 1]
        
        # Load batch data
        batch_point_cloud = input_root1['data/point_cloud'][start_ts:end_ts]
        batch_state = input_root1['data/state'][start_ts:end_ts]
        batch_action = input_root1['data/action'][start_ts:end_ts]
        
        # Resize and append to zarr arrays
        old_size = zarr_point_cloud.shape[0]
        new_size = old_size + len(batch_point_cloud)
        
        zarr_point_cloud.resize(new_size, *zarr_point_cloud.shape[1:])
        zarr_state.resize(new_size, *zarr_state.shape[1:])
        zarr_action.resize(new_size, *zarr_action.shape[1:])
        
        zarr_point_cloud[old_size:new_size] = batch_point_cloud
        zarr_state[old_size:new_size] = batch_state
        zarr_action[old_size:new_size] = batch_action
        
        # Clear batch data to free memory
        del batch_point_cloud, batch_state, batch_action
        
        current_ep = batch_end
    
    # Process dataset 2 in batches
    print(f"\nMerging dataset 2 in batches (batch_size={batch_size} episodes)...")
    current_ep = 0
    
    while current_ep < total_episodes2:
        batch_end = min(current_ep + batch_size, total_episodes2)
        print(f"  Processing episodes {current_ep + 1}-{batch_end} / {total_episodes2}")
        
        # Get timestep range for this batch
        start_ts = 0 if current_ep == 0 else episode_ends2[current_ep - 1]
        end_ts = episode_ends2[batch_end - 1]
        
        # Load batch data
        batch_point_cloud = input_root2['data/point_cloud'][start_ts:end_ts]
        batch_state = input_root2['data/state'][start_ts:end_ts]
        batch_action = input_root2['data/action'][start_ts:end_ts]
        
        # Resize and append to zarr arrays
        old_size = zarr_point_cloud.shape[0]
        new_size = old_size + len(batch_point_cloud)
        
        zarr_point_cloud.resize(new_size, *zarr_point_cloud.shape[1:])
        zarr_state.resize(new_size, *zarr_state.shape[1:])
        zarr_action.resize(new_size, *zarr_action.shape[1:])
        
        zarr_point_cloud[old_size:new_size] = batch_point_cloud
        zarr_state[old_size:new_size] = batch_state
        zarr_action[old_size:new_size] = batch_action
        
        # Clear batch data to free memory
        del batch_point_cloud, batch_state, batch_action
        
        current_ep = batch_end
    
    # Merge episode ends
    print("\nMerging episode ends...")
    # episode_ends2 needs to be offset by the last timestep of dataset 1
    episode_ends2_offset = episode_ends2 + total_timesteps1
    episode_ends = np.concatenate([episode_ends1, episode_ends2_offset], axis=0)
    
    output_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype=input_root1['meta/episode_ends'].dtype,
        compressor=compressor,
    )
    
    # Copy robot name metadata
    output_meta.attrs['robot_name'] = robot_name
    
    # Print summary
    print("\n" + "="*60)
    print("Merge complete!")
    print("="*60)
    print(f"Output saved to: {output_zarr_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"  - From dataset 1: {total_episodes1}")
    print(f"  - From dataset 2: {total_episodes2}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"  - From dataset 1: {total_timesteps1}")
    print(f"  - From dataset 2: {total_timesteps2}")
    print(f"Point cloud shape: {zarr_point_cloud.shape}")
    print(f"State shape: {zarr_state.shape}")
    print(f"Action shape: {zarr_action.shape}")
    print(f"Robot name: {robot_name}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge two zarr datasets into one using batch processing"
    )
    parser.add_argument(
        "--input_zarr1",
        type=str,
        required=True,
        help="Path to first input zarr file",
    )
    parser.add_argument(
        "--input_zarr2",
        type=str,
        required=True,
        help="Path to second input zarr file",
    )
    parser.add_argument(
        "--output_zarr",
        type=str,
        default=None,
        help="Path to output zarr file (default: ./data/merge-{total_episodes}.zarr)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of episodes to process in each batch (default: 1000)",
    )
    
    args = parser.parse_args()
    
    merge_zarr_datasets(args.input_zarr1, args.input_zarr2, args.output_zarr, args.batch_size)


if __name__ == "__main__":
    main()
