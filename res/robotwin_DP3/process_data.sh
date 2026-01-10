#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
batch_size=${4:-50}  # Default batch size is 50 if not provided

echo "Processing $expert_data_num episodes in batches of $batch_size"
python scripts/process_data.py $task_name $task_config $expert_data_num --batch_size $batch_size