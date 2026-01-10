import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py

import sys
sys.path.append("/home/xinhai/automoma/baseline/RoboTwin")
from policy.config import CollectConfig

def load_hdf5(dataset_path, mobile_base_mode="relative"):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()
    
    with h5py.File(dataset_path, "r") as root:
        # Read robot name from file
        robot_name = root["env_info"]["robot_name"][()].decode("utf-8")
        
        # Load joint data
        joint_group = root["/obs/joint"]
        
        # Get robot-specific joint configuration
        robot_config = CollectConfig.JOINT_CONFIG.get(robot_name, {})
        output_joints = robot_config.get("output_joints", {})
        
        
        # Load joint data based on robot configuration
        joint_data = {}
        for joint_name in output_joints.keys():
            if joint_name in joint_group:
                joint_data[joint_name] = joint_group[joint_name][()]
                        
        # Load point cloud data
        pointcloud = root["/obs/point_cloud"][()]
        
        # Load eef data
        eef_data = root["/obs/eef"][()] if "eef" in root["/obs"] else None
        
        # Determine state vector based on output_joints configuration
        timestamps = pointcloud.shape[0]
        
        # Separate joint data for each timestamp
        joint_states = {joint_name: [] for joint_name in output_joints.keys()}
        
        for t in range(timestamps):
            for joint_name, dim in output_joints.items():
                if joint_name in joint_data:
                    if len(joint_data[joint_name].shape) == 1:
                        # For 1D arrays (gripper) - single value per timestamp
                        joint_states[joint_name].append([joint_data[joint_name][t]])
                    else:
                        # For multi-dimensional arrays (arm, mobile_base, etc.)
                        joint_states[joint_name].append(joint_data[joint_name][t][:dim])
        
        # Convert to numpy arrays and ensure proper dimensions
        for joint_name in joint_states:
            joint_states[joint_name] = np.array(joint_states[joint_name])
            # Ensure 2D array (timestamps, dimensions)
            if len(joint_states[joint_name].shape) == 1:
                joint_states[joint_name] = joint_states[joint_name].reshape(-1, 1)
        
        # Construct state arrays by concatenating all joint states
        state_arrays = np.concatenate([joint_states[joint_name] for joint_name in output_joints.keys()], axis=1)
        
        # Calculate actions based on mobile_base_mode
        if mobile_base_mode == "absolute":
            # All actions are absolute: action[t] = state[t+1]
            actions = state_arrays[1:]
        elif mobile_base_mode == "relative":
            # Only mobile_base action is relative, others are absolute
            action_components = {}
            
            # Process each joint type
            for joint_name in output_joints.keys():
                if joint_name in joint_states:
                    if joint_name == "mobile_base":
                        # Relative action for mobile_base: action[t] = state[t+1] - state[t]
                        action_components[joint_name] = joint_states[joint_name][1:] - joint_states[joint_name][:-1]
                    else:
                        # Absolute action for other joints: action[t] = state[t+1]
                        action_components[joint_name] = joint_states[joint_name][1:]
            
            # Concatenate action components
            actions = np.concatenate([action_components[joint_name] for joint_name in output_joints.keys()], axis=1)
        else:
            raise ValueError(f"Invalid mobile_base_mode: {mobile_base_mode}. Choose 'absolute' or 'relative'.")
        
        return pointcloud[:-1], state_arrays[:-1], actions, eef_data, robot_name

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--mobile_base_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="Mobile base action mode: relative (default) or absolute",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of episodes to process in each batch (default: 100)",
    )
    args = parser.parse_args()
    
    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config
    mobile_base_mode = args.mobile_base_mode
    batch_size = args.batch_size
    
    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # Initialize zarr arrays
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # Initialize variables for batch processing
    current_ep = 0
    total_count = 0
    episode_ends_arrays = []
    robot_name = None
    
    # Get dimensions from first valid episode
    first_valid_ep = None
    for ep in range(num):
        load_path = os.path.join(load_dir, f"data/episode{ep:06d}.hdf5")
        try:
            pointcloud_sample, state_sample, action_sample, _, robot_name = load_hdf5(
                load_path, mobile_base_mode
            )
            first_valid_ep = ep
            break
        except Exception as e:
            print(f"Error reading episode {ep} for dimensions: {e}")
            continue
    
    if first_valid_ep is None:
        print("No valid episodes found!")
        return
    
    # Initialize zarr datasets with proper dimensions
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Estimate total size for zarr arrays
    point_cloud_shape = (0,) + pointcloud_sample.shape[1:]
    state_shape = (0,) + state_sample.shape[1:]
    action_shape = (0,) + action_sample.shape[1:]
    
    zarr_point_cloud = zarr_data.create_dataset(
        "point_cloud",
        shape=point_cloud_shape,
        chunks=(100,) + pointcloud_sample.shape[1:],
        dtype="float32", # TODO: from float64 to float32
        compressor=compressor,
    )
    
    zarr_state = zarr_data.create_dataset(
        "state",
        shape=state_shape,
        chunks=(100,) + state_sample.shape[1:],
        dtype="float32",
        compressor=compressor,
    )
    
    zarr_action = zarr_data.create_dataset(
        "action",
        shape=action_shape,
        chunks=(100,) + action_sample.shape[1:],
        dtype="float32",
        compressor=compressor,
    )
    
    # Process in batches
    while current_ep < num:
        batch_end = min(current_ep + batch_size, num)
        print(f"Processing batch: episodes {current_ep + 1}-{batch_end} / {num}")
        
        # Collect data for current batch
        batch_point_clouds = []
        batch_states = []
        batch_actions = []
        
        for ep in range(current_ep, batch_end):
            load_path = os.path.join(load_dir, f"data/episode{ep:06d}.hdf5")
            
            try:
                pointcloud_all, state_all, action_all, eef_all, _ = load_hdf5(
                    load_path, mobile_base_mode
                )
                
                batch_point_clouds.extend(pointcloud_all)
                batch_states.extend(state_all)
                batch_actions.extend(action_all)
                
                total_count += len(pointcloud_all)
                episode_ends_arrays.append(total_count)
                
            except Exception as e:
                print(f"Error processing episode {ep}: {e}")
                continue
        
        # Append batch data to zarr arrays
        if batch_point_clouds:
            batch_point_clouds = np.array(batch_point_clouds)
            batch_states = np.array(batch_states)
            batch_actions = np.array(batch_actions)
            
            # Resize and append to zarr arrays
            old_size = zarr_point_cloud.shape[0]
            new_size = old_size + len(batch_point_clouds)
            
            zarr_point_cloud.resize(new_size, *zarr_point_cloud.shape[1:])
            zarr_state.resize(new_size, *zarr_state.shape[1:])
            zarr_action.resize(new_size, *zarr_action.shape[1:])
            
            zarr_point_cloud[old_size:new_size] = batch_point_clouds
            zarr_state[old_size:new_size] = batch_states
            zarr_action[old_size:new_size] = batch_actions
            
            # Clear batch data to free memory
            del batch_point_clouds, batch_states, batch_actions
        
        current_ep = batch_end
    
    print()
    
    # Save episode ends and metadata
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
        
        # Save robot name as metadata
        if robot_name:
            zarr_meta.attrs["robot_name"] = robot_name
        
        print(f"Successfully processed {len(episode_ends_arrays)} episodes with {total_count} total timesteps")
        
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()