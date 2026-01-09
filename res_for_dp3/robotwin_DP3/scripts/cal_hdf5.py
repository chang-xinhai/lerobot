import h5py
import sys

def print_hdf5_structure(name, obj, depth=0):
    indent = "  " * depth
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}- {name} (Dataset) shape={obj.shape}, dtype={obj.dtype}")
    else:
        print(f"{indent}+ {name} (Group)")
        for key in obj.keys():
            print_hdf5_structure(key, obj[key], depth + 1)

if __name__ == "__main__":
    # path = "/home/xinhai/automoma/baseline/RoboTwin/policy/ACT/processed_data/sim-automoma_manip_summit_franka/task_1object_3scene_20pose-1000/episode_0.hdf5"
    path = "/home/xinhai/automoma/baseline/RoboTwin/data/automoma_manip_summit_franka/task_1object_3scene_20pose/data/episode000002.hdf5"
    with h5py.File(path, 'r') as f:
        print("HDF5 Structure:")
        for key in f.keys():
            print_hdf5_structure(key, f[key])