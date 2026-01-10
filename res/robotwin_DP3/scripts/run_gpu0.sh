bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_interpolated 1000 0 0

# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_50 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_500 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_900 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_800 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_700 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_600 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_200 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_300 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose_400 1000 0 0


# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 6400 \
#     --num_start_ik 5244


# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 200

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 300

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 400

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 600

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 700

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 800

# python scripts/data_process/downsample_trajectories_start_ik.py \
#     --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
#     --mode downsample \
#     --num_trajectories 1000 \
#     --num_start_ik 900