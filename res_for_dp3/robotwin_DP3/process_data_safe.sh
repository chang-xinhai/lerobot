#!/bin/bash

# Enhanced process_data script with memory management
# Usage: ./process_data_safe.sh <task_name> <task_config> <expert_data_num> [batch_size] [resume_from]

task_name=${1}
task_config=${2}
expert_data_num=${3}
batch_size=${4:-100}  # Default batch size is 100
resume_from=${5:-0}  # Resume from episode number (default 0)

if [ -z "$task_name" ] || [ -z "$task_config" ] || [ -z "$expert_data_num" ]; then
    echo "Usage: $0 <task_name> <task_config> <expert_data_num> [batch_size] [resume_from]"
    echo "Example: $0 automoma_manip_summit_franka task_1object_15scene_20pose 15000 100 0"
    echo ""
    echo "Parameters:"
    echo "  task_name        - Name of the task"
    echo "  task_config      - Task configuration"
    echo "  expert_data_num  - Total number of episodes to process"
    echo "  batch_size       - Episodes per batch (default: 100, smaller = less memory)"
    echo "  resume_from      - Episode number to resume from (default: 0)"
    exit 1
fi

echo "=============================================="
echo "Memory-Safe Data Processing"
echo "=============================================="
echo "Task: $task_name"
echo "Config: $task_config" 
echo "Total episodes: $expert_data_num"
echo "Batch size: $batch_size"
echo "Resume from episode: $resume_from"
echo "=============================================="

# Check available memory
echo "System Memory Status:"
free -h
echo ""

# Estimate memory requirements
estimated_memory_gb=$((batch_size * 60 / 1024))  # Rough estimate: 60MB per episode
echo "Estimated peak memory usage: ~${estimated_memory_gb} GB per batch"
echo ""

# Ask for confirmation if batch size seems large
if [ $batch_size -gt 100 ]; then
    echo "⚠️  WARNING: Batch size is quite large ($batch_size episodes)"
    echo "This might use significant memory. Consider using smaller batch size."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Starting processing..."
echo "You can monitor memory usage in another terminal with:"
echo "  python scripts/monitor_memory.py \$\$"
echo ""

# Run the processing
python scripts/process_data_safe.py $task_name $task_config $expert_data_num --batch_size $batch_size

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Processing completed successfully!"
    echo "Output saved to: ./data/$task_name-$task_config-$expert_data_num.zarr"
else
    echo ""
    echo "✗ Processing failed with exit code: $exit_code"
    echo ""
    echo "Common solutions:"
    echo "1. Reduce batch size (current: $batch_size)"
    echo "2. Free up system memory"
    echo "3. Check for corrupted episode files"
fi