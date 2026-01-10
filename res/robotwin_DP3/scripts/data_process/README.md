# Data Processing Scripts

This directory contains scripts for processing zarr datasets.

## Scripts

### 1. extract_scenes.py

Extracts the first N scenes from a zarr dataset. Each scene contains 1000 trajectories.

**Usage:**
```bash
python extract_scenes.py --input_zarr <path_to_zarr> --num_scenes <N> [--output_dir <dir>]
```

**Example:**
```bash
# Extract first 10 scenes (10,000 trajectories)
python extract_scenes.py \
    --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr \
    --num_scenes 10

# Extract first 5 scenes (5,000 trajectories)
python extract_scenes.py \
    --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr \
    --num_scenes 5
```

**Output:**
- File name: `automoma_manip_summit_franka-task_1object_{n}scene_20pose-{n}000.zarr`
- Location: Same directory as input (unless `--output_dir` is specified)

### 2. downsample_trajectories.py

Downsamples trajectories from a zarr dataset using either random or uniform sampling.

**Usage:**
```bash
python downsample_trajectories.py \
    --input_zarr <path_to_zarr> \
    --num_trajectories <N> \
    --mode <random|uniform> \
    [--seed <seed>] \
    [--output_dir <dir>]
```

**Modes:**
- **uniform**: Sample every (total/N) trajectory uniformly
  - Ensures even distribution across the dataset
  - Deterministic (same result each time)
  
- **random**: Randomly sample N trajectories
  - Random selection from the entire dataset
  - Can set seed for reproducibility

**Examples:**
```bash
# Uniform sampling: sample 10,000 trajectories evenly
python downsample_trajectories.py \
    --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr \
    --num_trajectories 10000 \
    --mode uniform

# Random sampling: randomly select 5,000 trajectories with seed
python downsample_trajectories.py \
    --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr \
    --num_trajectories 5000 \
    --mode random \
    --seed 42

# Uniform sampling with divisible number (for cleaner sampling)
python downsample_trajectories.py \
    --input_zarr ../data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr \
    --num_trajectories 15000 \
    --mode uniform
```

**Output:**
- File name: `automoma_manip_summit_franka-task_1object_30scene_20pose-{N}.zarr`
- Location: Same directory as input (unless `--output_dir` is specified)

## Data Structure

The input zarr file has the following structure:
```
data/
  ├── point_cloud  (shape: [timesteps, 4096, 6])
  ├── state        (shape: [timesteps, 11])
  └── action       (shape: [timesteps, 11])
meta/
  ├── episode_ends (shape: [num_episodes])
  └── attrs: robot_name
```

Each episode typically has 31 timesteps.

## Notes

- Both scripts preserve the original data structure and compression settings
- Episode ends are recalculated to maintain consistency
- Robot name metadata is preserved
- Large datasets are processed in batches to avoid memory issues
- For uniform sampling, if total_episodes / num_trajectories is not an integer, the script will still work but will warn you
