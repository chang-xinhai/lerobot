import os, zarr, numpy as np
path='/home/xinhai/automoma/baseline/RoboTwin/policy/DP3/data/automoma_manip_summit_franka-task_1object_15scene_20pose-15000.zarr'
print('ZARR PATH:', path)
if not os.path.exists(path):
    print('Path does not exist')
    raise SystemExit(0)
# open group
grp = zarr.open(path, 'r')
print('Groups: ', list(grp.keys()))
meta = grp['meta']
print('Meta keys:', list(meta.keys()))
print('n_steps (episode_ends last):', int(meta['episode_ends'][-1]))
print('\nData arrays:')
total_uncompressed = 0
for k,v in grp['data'].items():
    shape = v.shape
    dtype = v.dtype
    itemsize = np.dtype(dtype).itemsize
    nbytes = int(np.prod(shape) * itemsize)
    total_uncompressed += nbytes
    print(f"- {k}: shape={shape}, dtype={dtype}, itemsize={itemsize}, uncompressed_bytes={nbytes:,}")
print('\nTotal uncompressed bytes:', f"{total_uncompressed:,}")
print('Total uncompressed GiB:', total_uncompressed/1024**3)
# on-disk size
import subprocess
du = subprocess.run(['du','-sh', path], capture_output=True, text=True)
print('du -sh:', du.stdout.strip())