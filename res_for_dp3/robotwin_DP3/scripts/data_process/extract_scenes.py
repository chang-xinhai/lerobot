#!/usr/bin/env python
"""
Extract specified scenes from zarr dataset.
Each scene contains 1000 trajectories.

Usage:
    python extract_scenes.py --input_zarr <path_to_zarr> --num_scenes <N>
    python extract_scenes.py --input_zarr <path_to_zarr> --scenes <scene_indices>
    
Example:
    python extract_scenes.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --num_scenes 10
    python extract_scenes.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --scenes 0 1
    python extract_scenes.py --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr --scenes 0 4
"""

import os
import argparse
import zarr
import numpy as np
import shutil


def extract_scenes(input_zarr_path, scene_indices, output_dir=None):
    """
    Extract specified scenes from a zarr dataset.
    
    Args:
        input_zarr_path: Path to input zarr file
        scene_indices: List of scene indices to extract (each scene has 1000 trajectories)
        output_dir: Output directory (default: same directory as input)
    """
    # Validate input
    if not os.path.exists(input_zarr_path):
        raise FileNotFoundError(f"Input zarr file not found: {input_zarr_path}")
    
    # Open input zarr
    print(f"Opening input zarr: {input_zarr_path}")
    input_root = zarr.open(input_zarr_path, 'r')
    
    # Get total episodes
    episode_ends = input_root['meta/episode_ends'][:]
    total_episodes = len(episode_ends)
    total_scenes = total_episodes // 1000
    
    print(f"Total scenes in dataset: {total_scenes} (total episodes: {total_episodes})")
    print(f"Extracting {len(scene_indices)} scenes with indices: {scene_indices}")
    
    # Validate scene indices
    for scene_idx in scene_indices:
        if scene_idx < 0 or scene_idx >= total_scenes:
            raise ValueError(f"Scene index {scene_idx} out of range [0, {total_scenes-1}]")
    
    # Calculate episode ranges for each scene
    episode_ranges = []
    start_timestep = 0
    end_timestep = 0
    
    for scene_idx in sorted(scene_indices):
        # Scene i contains episodes from i*1000 to (i+1)*1000-1
        start_episode = scene_idx * 1000
        end_episode = start_episode + 1000 - 1
        
        if end_episode >= total_episodes:
            raise ValueError(f"Scene {scene_idx} is out of range")
        
        episode_ranges.append((start_episode, end_episode))
        end_timestep = episode_ends[end_episode]
    
    start_timestep = 0 if not episode_ranges else episode_ends[episode_ranges[0][0] - 1] if episode_ranges[0][0] > 0 else 0
    
    num_episodes_to_extract = len(scene_indices) * 1000
    
    print(f"Extracting {num_episodes_to_extract} episodes from {len(scene_indices)} scenes")
    print(f"Extracting data up to timestep {end_timestep} (episode {num_episodes_to_extract})")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(input_zarr_path)
    
    # Create output name based on scenes
    scenes_str = "_".join(map(str, sorted(scene_indices)))
    output_name = f"automoma_manip_summit_franka-task_1object_{len(scene_indices)}scene_20pose-{num_episodes_to_extract}_scenes_{scenes_str}.zarr"
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
    
    # Extract and save data - concatenate data from all specified scenes
    print("Extracting point cloud data...")
    point_cloud_data_list = []
    state_data_list = []
    action_data_list = []
    new_episode_ends = []
    
    current_timestep = 0
    for scene_idx in sorted(scene_indices):
        start_episode = scene_idx * 1000
        end_episode = start_episode + 1000 - 1
        
        start_ts = 0 if start_episode == 0 else episode_ends[start_episode - 1]
        end_ts = episode_ends[end_episode]
        
        point_cloud_data_list.append(input_root['data/point_cloud'][start_ts:end_ts])
        state_data_list.append(input_root['data/state'][start_ts:end_ts])
        action_data_list.append(input_root['data/action'][start_ts:end_ts])
        
        # Adjust episode ends to be relative to the combined dataset
        for i in range(start_episode, end_episode + 1):
            ep_end = episode_ends[i]
            if start_episode > 0:
                ep_end = ep_end - start_ts
            new_episode_ends.append(ep_end + current_timestep)
        
        current_timestep = new_episode_ends[-1]
    
    point_cloud_data = np.concatenate(point_cloud_data_list, axis=0)
    state_data = np.concatenate(state_data_list, axis=0)
    action_data = np.concatenate(action_data_list, axis=0)
    
    output_data.create_dataset(
        "point_cloud",
        data=point_cloud_data,
        chunks=input_root['data/point_cloud'].chunks,
        dtype=input_root['data/point_cloud'].dtype,
        compressor=compressor,
    )
    
    print("Extracting state data...")
    output_data.create_dataset(
        "state",
        data=state_data,
        chunks=input_root['data/state'].chunks,
        dtype=input_root['data/state'].dtype,
        compressor=compressor,
    )
    
    print("Extracting action data...")
    output_data.create_dataset(
        "action",
        data=action_data,
        chunks=input_root['data/action'].chunks,
        dtype=input_root['data/action'].dtype,
        compressor=compressor,
    )
    
    print("Extracting episode ends...")
    output_meta.create_dataset(
        "episode_ends",
        data=np.array(new_episode_ends),
        dtype=input_root['meta/episode_ends'].dtype,
        compressor=compressor,
    )
    
    # Copy robot name metadata
    if 'robot_name' in input_root['meta'].attrs:
        output_meta.attrs['robot_name'] = input_root['meta'].attrs['robot_name']
        print(f"Robot name: {input_root['meta'].attrs['robot_name']}")
    
    # Print summary
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)
    print(f"Output saved to: {output_path}")
    print(f"Scenes extracted: {len(scene_indices)} (indices: {scene_indices})")
    print(f"Episodes extracted: {num_episodes_to_extract}")
    print(f"Timesteps extracted: {current_timestep}")
    print(f"Point cloud shape: {point_cloud_data.shape}")
    print(f"State shape: {state_data.shape}")
    print(f"Action shape: {action_data.shape}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract specified scenes from zarr dataset (1000 trajectories per scene)"
    )
    parser.add_argument(
        "--input_zarr",
        type=str,
        required=True,
        help="Path to input zarr file",
    )
    
    # Create a mutually exclusive group for num_scenes and scenes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_scenes",
        type=int,
        help="Number of scenes to extract from the beginning (each scene = 1000 trajectories). "
             "If specified, extracts scenes [0, 1, ..., num_scenes-1]",
    )
    group.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        help="Space-separated scene indices to extract, e.g., '0 1' or '0 4'. "
             "If specified, extracts only these specific scenes",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input directory)",
    )
    
    args = parser.parse_args()
    
    # Determine scene indices
    if args.num_scenes is not None:
        scene_indices = list(range(args.num_scenes))
    else:
        scene_indices = args.scenes
    
    extract_scenes(args.input_zarr, scene_indices, args.output_dir)


if __name__ == "__main__":
    main()
