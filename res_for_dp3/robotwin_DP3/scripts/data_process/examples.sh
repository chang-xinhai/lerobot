#!/bin/bash
# Example commands for processing zarr data

# Set the input data path
INPUT_ZARR="./data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr"
INPUT_ZARR="./data/automoma_manip_summit_franka-task_1object_1scene_20pose-1000.zarr"


# Script 1: Extract first N scenes
# Each scene has 1000 trajectories
# Output format: automoma_manip_summit_franka-task_1object_{n}scene_20pose-{n}000.zarr

echo "Example 1: Extract first 10 scenes (10,000 trajectories)"
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 10

echo "Example 2: Extract first 5 scenes (5,000 trajectories)"
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 5

echo "Example 3: Extract first 15 scenes (15,000 trajectories)"
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 15

# Script 2: Downsample trajectories
# Two modes: uniform or random
# Output format: automoma_manip_summit_franka-task_1object_30scene_20pose-{N}.zarr

echo "Example 4: Uniform sampling - 15,000 trajectories (divisible)"
python scripts/data_process/downsample_trajectories.py \
    --input_zarr $INPUT_ZARR \
    --num_trajectories 300 \
    --mode uniform

echo "Example 5: Uniform sampling - 10,000 trajectories"
python scripts/data_process/downsample_trajectories.py \
    --input_zarr $INPUT_ZARR \
    --num_trajectories 10000 \
    --mode uniform

echo "Example 6: Random sampling - 5,000 trajectories with seed"
python scripts/data_process/downsample_trajectories.py \
    --input_zarr $INPUT_ZARR \
    --num_trajectories 5000 \
    --mode random \
    --seed 42

echo "Example 7: Random sampling - 3,000 trajectories"
python scripts/data_process/downsample_trajectories.py \
    --input_zarr $INPUT_ZARR \
    --num_trajectories 3000 \
    --mode random

echo "Example 8: Uniform sampling - 6,000 trajectories (divisible by 30,000)"
python scripts/data_process/downsample_trajectories.py \
    --input_zarr $INPUT_ZARR \
    --num_trajectories 6000 \
    --mode uniform
