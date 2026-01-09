
# cd policy/DP3

## 11.20
rsync -avzP /home/xinhai/automoma/baseline/RoboTwin/policy/DP3/data/automoma_manip_summit_franka-task_1object_1scene_20pose_interpolated-1000.zarr \
    xinhai@182.18.90.210:/home/xinhai/automoma/RoboTwin/policy/DP3/data




## 11.19
rsync -avzP /home/xinhai/automoma/baseline/RoboTwin/policy/DP3/data/automoma_manip_summit_franka-task_1object_1scene_20pose_5244-6400.zarr \
    xinhai@182.18.90.210:/home/xinhai/automoma/RoboTwin/policy/DP3/data



## 11.18
python scripts/data_process/merge_scenes.py \
    --input_zarr1 ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
    --input_zarr2 ./data/automoma_manip_summit_franka-task_1object_1scene_20pose_new_new-12800.zarr \
    --output_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-25600.zarr \
    --batch_size 500

## 11.17

python scripts/data_process/downsample_trajectories_start_ik.py \
    --input_zarr "./data/automoma_manip_summit_franka-task_1object_1scene_20pose_new-6400.zarr" \
    --mode collect
# 1439 -> 50, 100, 500, 1000
# 5244 -> 6400

python scripts/data_process/downsample_trajectories_start_ik.py \
    --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
    --mode downsample \
    --num_trajectories 1000 \
    --num_start_ik 500

python scripts/data_process/downsample_trajectories_start_ik.py \
    --input_zarr ./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr \
    --mode downsample \
    --num_trajectories 6400 \
    --num_start_ik 5244

## 11.16

python scripts/data_process/merge_scenes.py \
    --input_zarr1 ./data/automoma_manip_summit_franka_fixed_base-task_1object_1scene_20pose-6400.zarr \
    --input_zarr2 ./data/automoma_manip_summit_franka_fixed_base-task_1object_1scene_20pose_new-6400.zarr \
    --output_zarr ./data/automoma_manip_summit_franka_fixed_base-task_1object_1scene_20pose-12800.zarr \
    --batch_size 500

python scripts/data_process/merge_scenes.py \
    --input_zarr1 "./data/automoma_manip_summit_franka-task_1object_1scene_20pose-6400.zarr" \
    --input_zarr2 "./data/automoma_manip_summit_franka-task_1object_1scene_20pose_new-6400.zarr" \
    --output_zarr "./data/automoma_manip_summit_franka-task_1object_1scene_20pose-12800.zarr" \
    --batch_size 500

bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose_new 6400 0 0

bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 12800 0 1

bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 12800 0 0




## Table 1.1: S=1, T, summit franka mobile
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 100 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 200 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 400 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 800 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 1000 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 1600 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 3200 0 0
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 6400 0 0

## Table 1.2: S=1, T, summit franka fixed base
INPUT_ZARR="./data/automoma_manip_summit_franka-task_1object_1scene_20pose-6400.zarr"
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 100 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 200 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 400 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 800 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 1600 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 3200 --mode uniform

bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 100 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 200 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 400 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 800 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 1000 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 1600 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 3200 0 0
bash train.sh automoma_manip_summit_franka_fixed_base task_1object_1scene_20pose 6400 0 0


# PROCESS DATA
INPUT_ZARR="./data/automoma_manip_summit_franka-task_1object_30scene_20pose-30000.zarr"
# python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 1
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 2
# python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 4
# python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 8
# python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --num_scenes 15

# Extract scenes 0 and 1
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --scenes 1

# Extract scenes 0 and 4
python scripts/data_process/extract_scenes.py --input_zarr $INPUT_ZARR --scenes 0 4


# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 300 --mode uniform
python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 750 --mode uniform

# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 1500 --mode uniform
# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 3000 --mode uniform
# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 6000 --mode uniform
# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 12000 --mode uniform
# python scripts/data_process/downsample_trajectories.py --input_zarr $INPUT_ZARR --num_trajectories 24000 --mode uniform


## Table 2: T=1000, S, summit franka mobile
# bash train.sh automoma_manip_summit_franka task_1object_1scene_20pose 1000 0 0  ✅
bash train.sh automoma_manip_summit_franka task_1object_2scene_20pose 2000_scenes_2_3 0 1
# bash train.sh automoma_manip_summit_franka task_1object_4scene_20pose 4000 0 1
# bash train.sh automoma_manip_summit_franka task_1object_8scene_20pose 8000 0 1
# bash train.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 0  ✅
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 30000 0 0  ✅


# Table 3: S=30, T, summit franka mobile
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 300 0 0  ✅
bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 750 0 0  
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 1500 0 1  
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 3000 0 0  ✅
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 6000 0 1 
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 12000 0 1
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 24000 0 1 
# bash train.sh automoma_manip_summit_franka task_1object_30scene_20pose 30000 0 1

