#!/usr/bin/env python
"""
Downsample trajectories from zarr dataset based on unique start IK configurations.

Two modes:
1. collect: Analyze dataset and collect unique start IK configurations
2. downsample: Downsample trajectories based on unique start IKs and total trajectory count

Usage:
    # Step 1: Analyze dataset to get unique start IKs
    python downsample_trajectories_start_ik.py --input_zarr <path> --mode collect
    
    # Step 2: Downsample based on unique start IKs and total trajectories
    python downsample_trajectories_start_ik.py --input_zarr <path> --mode downsample --num_trajectories <N> --num_start_ik <K>
    
Example:
    python downsample_trajectories_start_ik.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --mode collect
    python downsample_trajectories_start_ik.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --mode downsample --num_trajectories 10000 --num_start_ik 50
"""

import os
import argparse
import zarr
import numpy as np
import shutil
import json
from collections import defaultdict


def calculate_start_ik_hash(state_vector, precision=3):
    """
    Calculate a hash for the start IK configuration.
    
    Args:
        state_vector: numpy array of shape (11,) representing robot joint state
        precision: number of decimal places to round to
    
    Returns:
        tuple: rounded state values as a hashable tuple
    """
    rounded = np.round(state_vector, precision)
    return tuple(rounded)


def collect_unique_start_iks(input_zarr_path):
    """
    Analyze dataset and collect unique start IK configurations.
    
    Args:
        input_zarr_path: Path to input zarr file
    
    Returns:
        dict: Mapping from start IK hash to list of trajectory indices
    """
    # Validate input
    if not os.path.exists(input_zarr_path):
        raise FileNotFoundError(f"Input zarr file not found: {input_zarr_path}")
    
    # Open input zarr
    print(f"Opening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, 'r')
    
    # Get episode information
    episode_ends = input_root['meta/episode_ends'][:]
    total_episodes = len(episode_ends)
    state_data = input_root['data/state']
    
    print(f"Total episodes in dataset: {total_episodes}")
    print(f"State shape: {state_data.shape}")
    
    # Verify state dimensions
    if len(state_data.shape) != 2 or state_data.shape[1] != 11:
        print(f"Warning: Expected state shape (N, 11), got {state_data.shape}")
    
    # Collect start IK configurations
    start_ik_to_episodes = defaultdict(list)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    print("Analyzing start IK configurations...")
    for ep_idx in range(total_episodes):
        start_idx = episode_starts[ep_idx]
        start_state = state_data[start_idx]
        
        # Calculate hash for start IK
        start_ik_hash = calculate_start_ik_hash(start_state)
        start_ik_to_episodes[start_ik_hash].append(int(ep_idx))
        
        if (ep_idx + 1) % 1000 == 0:
            print(f"  Processed {ep_idx + 1}/{total_episodes} episodes...")
    
    # Convert to regular dict with string keys for JSON serialization
    result = {}
    for ik_hash, episodes in start_ik_to_episodes.items():
        # Convert tuple to string for JSON
        key = ",".join([f"{v:.4f}" for v in ik_hash])
        result[key] = episodes
    
    # Sort by number of episodes (most common first)
    sorted_iks = sorted(result.items(), key=lambda x: len(x[1]), reverse=True)
    result_sorted = {k: v for k, v in sorted_iks}
    
    print(f"\nFound {len(result_sorted)} unique start IK configurations")
    print(f"Top 10 most common start IKs:")
    for i, (ik_hash, episodes) in enumerate(list(result_sorted.items())[:10]):
        print(f"  {i+1}. IK={ik_hash[:50]}... : {len(episodes)} episodes")
    
    return result_sorted, total_episodes


def save_start_ik_analysis(input_zarr_path, start_ik_data, total_episodes):
    """
    Save start IK analysis to JSON file.
    
    Args:
        input_zarr_path: Path to input zarr file
        start_ik_data: Dictionary mapping start IK to episode indices
        total_episodes: Total number of episodes
    """
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_name = os.path.basename(input_zarr_path).replace('.zarr', '')
    output_path = os.path.join(script_dir, f"{input_name}_start_ik_analysis.json")
    
    # Prepare output data
    output = {
        "input_zarr": input_zarr_path,
        "total_episodes": total_episodes,
        "num_unique_start_iks": len(start_ik_data),
        "start_ik_to_episodes": start_ik_data
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nStart IK analysis saved to: {output_path}")
    print(f"Total episodes: {total_episodes}")
    print(f"Unique start IKs: {len(start_ik_data)}")


def load_start_ik_analysis(input_zarr_path):
    """
    Load start IK analysis from JSON file.
    
    Args:
        input_zarr_path: Path to input zarr file
    
    Returns:
        dict: Start IK analysis data
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_name = os.path.basename(input_zarr_path).replace('.zarr', '')
    json_path = os.path.join(script_dir, f"{input_name}_start_ik_analysis.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Start IK analysis not found: {json_path}\n"
            f"Please run with --mode collect first"
        )
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def downsample_trajectories_by_start_ik(input_zarr_path, num_trajectories, num_start_ik, output_dir=None):
    """
    Downsample trajectories based on unique start IK configurations and total trajectory count.
    
    Args:
        input_zarr_path: Path to input zarr file
        num_trajectories: Target number of trajectories
        num_start_ik: Target number of unique start IKs
        output_dir: Output directory (default: same directory as input)
    """
    # Load start IK analysis
    print("Loading start IK analysis...")
    analysis = load_start_ik_analysis(input_zarr_path)
    
    total_episodes = analysis['total_episodes']
    start_ik_to_episodes = analysis['start_ik_to_episodes']
    num_unique_iks = len(start_ik_to_episodes)
    
    print(f"Total episodes: {total_episodes}")
    print(f"Unique start IKs: {num_unique_iks}")
    print(f"Target episodes: {num_trajectories}")
    print(f"Target unique start IKs: {num_start_ik}")
    
    # Validate inputs
    if num_trajectories > total_episodes:
        raise ValueError(f"Requested {num_trajectories} trajectories but only {total_episodes} available")
    
    if num_start_ik > num_unique_iks:
        raise ValueError(f"Requested {num_start_ik} unique start IKs but only {num_unique_iks} available")
    
    # Select start IKs (use most common ones)
    selected_start_iks = list(start_ik_to_episodes.keys())[:num_start_ik]
    
    print(f"\nSelected {num_start_ik} start IKs (from {len(start_ik_to_episodes)} available)")
    
    # Collect all available episodes from selected start IKs
    all_available_episodes = []
    for i, ik_hash in enumerate(selected_start_iks):
        available_episodes = start_ik_to_episodes[ik_hash]
        all_available_episodes.extend(available_episodes)
        if i < 10 or i == len(selected_start_iks) - 1:
            print(f"  Start IK {i+1}: {len(available_episodes)} episodes available")
    
    print(f"\nTotal available episodes from selected start IKs: {len(all_available_episodes)}")
    
    # Ensure each start IK is represented at least once, then fill the rest randomly
    print(f"\nSampling {num_trajectories} episodes ensuring all {num_start_ik} start IKs are covered...")
    
    selected_episodes = []
    
    # Step 1: Take at least one episode from each selected start IK
    for ik_hash in selected_start_iks:
        available_episodes = start_ik_to_episodes[ik_hash]
        sampled = np.random.choice(available_episodes, 1, replace=False)[0]
        selected_episodes.append(sampled)
    
    print(f"Selected 1 episode from each of {num_start_ik} start IKs")
    
    # Step 2: Sample the remaining episodes from all available episodes (with replacement)
    num_remaining = num_trajectories - num_start_ik
    if num_remaining > 0:
        additional_episodes = np.random.choice(all_available_episodes, num_remaining, replace=True)
        selected_episodes.extend(additional_episodes)
        print(f"Randomly sampled {num_remaining} additional episodes (with replacement)")
    
    selected_episodes = np.array(selected_episodes)
    
    # Sort selected episodes (in case of duplicates from replacement sampling)
    selected_episodes = np.sort(selected_episodes)
    print(f"Total selected episodes: {len(selected_episodes)}")
    print(f"Selected episodes: first 10 = {selected_episodes[:10]}, last 10 = {selected_episodes[-10:]}")
    print(f"Unique selected episodes: {len(np.unique(selected_episodes))}")
    
    # Open input zarr
    print(f"\nOpening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, 'r')
    
    # Get episode boundaries
    episode_ends = input_root['meta/episode_ends'][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    # Calculate timestep ranges for each selected episode
    timestep_indices = []
    for ep_idx in selected_episodes:
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        timestep_indices.extend(range(start, end))
    
    timestep_indices = np.array(timestep_indices)
    print(f"Total timesteps to extract: {len(timestep_indices)}")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_zarr_path)
    
    # Create output name with both num_trajectories and num_start_ik
    input_name = os.path.basename(input_zarr_path)
    input_base = input_name.replace('.zarr', '')
    
    # Find the last '-' which typically precedes the trajectory count
    last_dash_idx = input_base.rfind('-')
    if last_dash_idx != -1:
        base_prefix = input_base[:last_dash_idx]
        output_name = f"{base_prefix}_{num_start_ik}-{num_trajectories}.zarr"
    else:
        output_name = f"{input_base}_{num_start_ik}-{num_trajectories}.zarr"
    
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
    print(f"Episodes sampled: {len(selected_episodes)} (from {total_episodes})")
    print(f"Unique start IKs: {num_start_ik} (from {num_unique_iks})")
    print(f"Timesteps extracted: {len(timestep_indices)}")
    print(f"Point cloud shape: {output_point_cloud.shape}")
    print(f"State shape: {output_state.shape}")
    print(f"Action shape: {output_action.shape}")
    print(f"Episode ends shape: {new_episode_ends.shape}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Downsample trajectories based on unique start IK configurations"
    )
    parser.add_argument(
        "--input_zarr",
        type=str,
        required=True,
        help="Path to input zarr file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['collect', 'downsample'],
        required=True,
        help="Mode: 'collect' to analyze start IKs, 'downsample' to create downsampled dataset",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Number of trajectories to sample (required for downsample mode)",
    )
    parser.add_argument(
        "--num_start_ik",
        type=int,
        default=None,
        help="Number of unique start IKs to include (required for downsample mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    if args.mode == 'collect':
        # Collect and analyze unique start IKs
        start_ik_data, total_episodes = collect_unique_start_iks(args.input_zarr)
        save_start_ik_analysis(args.input_zarr, start_ik_data, total_episodes)
        
    elif args.mode == 'downsample':
        # Validate required arguments for downsample mode
        if args.num_trajectories is None:
            parser.error("--num_trajectories is required for downsample mode")
        if args.num_start_ik is None:
            parser.error("--num_start_ik is required for downsample mode")
        
        # Downsample trajectories
        downsample_trajectories_by_start_ik(
            args.input_zarr,
            args.num_trajectories,
            args.num_start_ik,
            args.output_dir
        )


if __name__ == "__main__":
    main()
