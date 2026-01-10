#!/usr/bin/env python
"""
Downsample trajectories from zarr dataset.

Two modes:
1. Random: Randomly sample N trajectories from the dataset
2. Uniform: Sample every (total/N) trajectory uniformly

Usage:
    python downsample_trajectories.py --input_zarr <path_to_zarr> --num_trajectories <N> --mode <random|uniform>
    
Example:
    python downsample_trajectories.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --num_trajectories 10000 --mode uniform
    python downsample_trajectories.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --num_trajectories 5000 --mode random --seed 42
"""

import os
import argparse
import zarr
import numpy as np
import shutil


def downsample_trajectories(input_zarr_path, num_trajectories, mode='uniform', seed=None, output_dir=None):
    """
    Downsample trajectories from a zarr dataset.
    
    Args:
        input_zarr_path: Path to input zarr file
        num_trajectories: Number of trajectories to sample
        mode: 'random' or 'uniform'
        seed: Random seed (only used for random mode)
        output_dir: Output directory (default: same directory as input)
    """
    # Validate input
    if not os.path.exists(input_zarr_path):
        raise FileNotFoundError(f"Input zarr file not found: {input_zarr_path}")
    
    if mode not in ['random', 'uniform']:
        raise ValueError(f"Mode must be 'random' or 'uniform', got: {mode}")
    
    # Open input zarr
    print(f"Opening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, 'r')
    
    # Get total episodes
    episode_ends = input_root['meta/episode_ends'][:]
    total_episodes = len(episode_ends)
    
    print(f"Total episodes in dataset: {total_episodes}")
    print(f"Downsampling to {num_trajectories} trajectories using {mode} mode")
    
    if num_trajectories > total_episodes:
        raise ValueError(f"Requested {num_trajectories} trajectories but only {total_episodes} available")
    
    # Select episodes based on mode
    if mode == 'random':
        if seed is not None:
            np.random.seed(seed)
            print(f"Using random seed: {seed}")
        
        selected_episodes = np.sort(np.random.choice(total_episodes, num_trajectories, replace=False))
        print(f"Randomly selected {num_trajectories} episodes")
        
    else:  # uniform
        step = total_episodes / num_trajectories
        if not step.is_integer():
            print(f"Warning: {total_episodes} / {num_trajectories} = {step} is not an integer")
            print(f"Using uniform sampling with step size {step}")
        
        selected_episodes = np.array([int(i * step) for i in range(num_trajectories)])
        print(f"Uniformly sampled every {step:.2f} episodes")
    
    print(f"Selected episodes: first 10 = {selected_episodes[:10]}, last 10 = {selected_episodes[-10:]}")
    
    # Calculate timestep ranges for each selected episode
    # Episode starts at episode_ends[i-1] (or 0 for first episode)
    # Episode ends at episode_ends[i]
    timestep_indices = []
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    for ep_idx in selected_episodes:
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        timestep_indices.extend(range(start, end))
    
    timestep_indices = np.array(timestep_indices)
    print(f"Total timesteps to extract: {len(timestep_indices)}")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_zarr_path)
    
    # Create output name based on input name
    # Extract base name and replace the trajectory count with new count
    input_name = os.path.basename(input_zarr_path)
    # Remove .zarr extension
    input_base = input_name.replace('.zarr', '')
    # Find the last '-' which typically precedes the trajectory count
    last_dash_idx = input_base.rfind('-')
    if last_dash_idx != -1:
        base_prefix = input_base[:last_dash_idx]
        output_name = f"{base_prefix}-{num_trajectories}.zarr"
    else:
        # Fallback if no dash found
        output_name = f"{input_base}-{num_trajectories}.zarr"
    
    output_path = os.path.join(output_dir, output_name)
    
    # Remove existing output if it exists
    if os.path.exists(output_path):
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)
    
    # Create output zarr
    print(f"Creating output zarr: {output_path}")
    output_root = zarr.group(output_path)
    output_data = output_root.create_group("data")
    output_meta = output_root.create_group("meta")
    
    # Copy compression settings
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Extract and save data with batching to handle large datasets
    print("Extracting point cloud data...")
    point_cloud_shape = input_root['data/point_cloud'].shape
    point_cloud_dtype = input_root['data/point_cloud'].dtype
    point_cloud_chunks = input_root['data/point_cloud'].chunks
    
    output_point_cloud = output_data.create_dataset(
        "point_cloud",
        shape=(len(timestep_indices),) + point_cloud_shape[1:],
        chunks=point_cloud_chunks,
        dtype=point_cloud_dtype,
        compressor=compressor,
    )
    
    # Process in batches to avoid memory issues
    batch_size = 10000
    for i in range(0, len(timestep_indices), batch_size):
        end_idx = min(i + batch_size, len(timestep_indices))
        batch_indices = timestep_indices[i:end_idx]
        output_point_cloud[i:end_idx] = input_root['data/point_cloud'].oindex[batch_indices]
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{len(timestep_indices)} timesteps...")
    
    print("Extracting state data...")
    state_shape = input_root['data/state'].shape
    state_dtype = input_root['data/state'].dtype
    state_chunks = input_root['data/state'].chunks
    
    output_state = output_data.create_dataset(
        "state",
        shape=(len(timestep_indices),) + state_shape[1:],
        chunks=state_chunks,
        dtype=state_dtype,
        compressor=compressor,
    )
    
    for i in range(0, len(timestep_indices), batch_size):
        end_idx = min(i + batch_size, len(timestep_indices))
        batch_indices = timestep_indices[i:end_idx]
        output_state[i:end_idx] = input_root['data/state'].oindex[batch_indices]
    
    print("Extracting action data...")
    action_shape = input_root['data/action'].shape
    action_dtype = input_root['data/action'].dtype
    action_chunks = input_root['data/action'].chunks
    
    output_action = output_data.create_dataset(
        "action",
        shape=(len(timestep_indices),) + action_shape[1:],
        chunks=action_chunks,
        dtype=action_dtype,
        compressor=compressor,
    )
    
    for i in range(0, len(timestep_indices), batch_size):
        end_idx = min(i + batch_size, len(timestep_indices))
        batch_indices = timestep_indices[i:end_idx]
        output_action[i:end_idx] = input_root['data/action'].oindex[batch_indices]
    
    print("Creating new episode ends...")
    # Recalculate episode ends for the new dataset
    # Each episode should maintain its original length
    new_episode_ends = []
    current_end = 0
    
    for ep_idx in selected_episodes:
        episode_length = episode_ends[ep_idx] - episode_starts[ep_idx]
        current_end += episode_length
        new_episode_ends.append(current_end)
    
    new_episode_ends = np.array(new_episode_ends)
    
    output_meta.create_dataset(
        "episode_ends",
        data=new_episode_ends,
        dtype=input_root['meta/episode_ends'].dtype,
        compressor=compressor,
    )
    
    # Copy robot name metadata
    if 'robot_name' in input_root['meta'].attrs:
        output_meta.attrs['robot_name'] = input_root['meta'].attrs['robot_name']
        print(f"Robot name: {input_root['meta'].attrs['robot_name']}")
    
    # Print summary
    print("\n" + "="*60)
    print("Downsampling complete!")
    print("="*60)
    print(f"Output saved to: {output_path}")
    print(f"Mode: {mode}")
    if mode == 'random' and seed is not None:
        print(f"Random seed: {seed}")
    print(f"Episodes sampled: {num_trajectories} (from {total_episodes})")
    print(f"Timesteps extracted: {len(timestep_indices)}")
    print(f"Point cloud shape: {output_point_cloud.shape}")
    print(f"State shape: {output_state.shape}")
    print(f"Action shape: {output_action.shape}")
    print(f"Episode ends shape: {new_episode_ends.shape}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Downsample trajectories from zarr dataset"
    )
    parser.add_argument(
        "--input_zarr",
        type=str,
        required=True,
        help="Path to input zarr file",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        required=True,
        help="Number of trajectories to sample",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['random', 'uniform'],
        default='uniform',
        help="Sampling mode: 'random' or 'uniform' (default: uniform)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (only used for random mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input directory)",
    )
    
    args = parser.parse_args()
    
    downsample_trajectories(
        args.input_zarr,
        args.num_trajectories,
        args.mode,
        args.seed,
        args.output_dir
    )


if __name__ == "__main__":
    main()
