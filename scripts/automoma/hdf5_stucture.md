HDF5 File Structure:
===================
- env_info (group)
  - grasp_pose (dataset)
    shape: (), dtype: object
  - num_timesteps (dataset)
    shape: (), dtype: int64
  - object_id (dataset)
    shape: (), dtype: object
  - robot_name (dataset)
    shape: (), dtype: object
  - scene_id (dataset)
    shape: (), dtype: object
- obs (group)
  - depth (group)
    - ego_topdown (dataset)
      shape: (32, 240, 320), dtype: float32
    - ego_wrist (dataset)
      shape: (32, 240, 320), dtype: float32
    - fix_local (dataset)
      shape: (32, 240, 320), dtype: float32
  - eef (dataset)
    shape: (32, 7), dtype: float32
  - joint (group)
    - arm (dataset)
      shape: (32, 7), dtype: float64
    - gripper (dataset)
      shape: (32,), dtype: float64
    - mobile_base (dataset)
      shape: (32, 3), dtype: float64
  - point_cloud (dataset)
    shape: (32, 4096, 6), dtype: float64
  - rgb (group)
    - ego_topdown (dataset)
      shape: (32, 240, 320, 3), dtype: uint8
    - ego_wrist (dataset)
      shape: (32, 240, 320, 3), dtype: uint8
    - fix_local (dataset)
      shape: (32, 240, 320, 3), dtype: uint8