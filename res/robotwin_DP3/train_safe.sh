#!/bin/bash

# Memory-safe training script for DP3
# Usage: ./train_safe.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [batch_size] [data_batch_size]

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
train_batch_size=${6:-32}        # Default training batch size
data_batch_size=${7:-50}         # Default data processing batch size

if [ -z "$task_name" ] || [ -z "$task_config" ] || [ -z "$expert_data_num" ] || [ -z "$seed" ] || [ -z "$gpu_id" ]; then
    echo "Usage: $0 <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [train_batch_size] [data_batch_size]"
    echo "Example: $0 automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 32 50"
    echo ""
    echo "Parameters:"
    echo "  task_name        - Name of the task"
    echo "  task_config      - Task configuration"
    echo "  expert_data_num  - Number of episodes to process"
    echo "  seed             - Random seed"
    echo "  gpu_id           - GPU ID to use"
    echo "  train_batch_size - Training batch size (default: 32, smaller = less GPU memory)"
    echo "  data_batch_size  - Data processing batch size (default: 50, smaller = less CPU memory)"
    exit 1
fi

echo "=============================================="
echo "Memory-Safe Training Pipeline"
echo "=============================================="
echo "Task: $task_name"
echo "Config: $task_config" 
echo "Episodes: $expert_data_num"
echo "Seed: $seed"
echo "GPU: $gpu_id"
echo "Training batch size: $train_batch_size"
echo "Data processing batch size: $data_batch_size"
echo "=============================================="

# Check system resources
echo "System Resources:"
echo "CPU Memory:"
free -h
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -1
echo ""

# Process data if zarr file doesn't exist
if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    echo "Processing data in batches..."
    echo "Batch size: $data_batch_size episodes per batch"
    echo "This will take longer but use less memory."
    echo ""
    
    # Use our memory-safe data processing
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num} ${data_batch_size}
    
    if [ $? -ne 0 ]; then
        echo "❌ Data processing failed!"
        exit 1
    fi
    echo "✅ Data processing completed successfully!"
else
    echo "✅ Data already processed, found: ./data/${task_name}-${task_config}-${expert_data_num}.zarr"
fi

echo ""
echo "Starting training with memory-safe configuration..."
echo "Training batch size: $train_batch_size (smaller than default 128 for memory safety)"
echo ""

# Start training with memory-safe configuration
bash scripts/train_policy_safe.sh robot_dp3_memory_safe ${task_name} ${task_config} ${expert_data_num} train ${seed} ${gpu_id} ${train_batch_size}

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "❌ Training failed with exit code: $exit_code"
    echo ""
    echo "Memory optimization suggestions:"
    echo "1. Reduce training batch size (current: $train_batch_size)"
    echo "2. Reduce data processing batch size (current: $data_batch_size)"
    echo "3. Check GPU memory usage with: nvidia-smi"
    echo "4. Free up system memory and try again"
fi