#!/bin/bash

# Fast training script that loads all data into memory
# Usage: ./train_fast.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [batch_size]

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
batch_size=${6:-64}  # Default batch size for fast training

if [ -z "$task_name" ] || [ -z "$task_config" ] || [ -z "$expert_data_num" ] || [ -z "$seed" ] || [ -z "$gpu_id" ]; then
    echo "Usage: $0 <task_name> <task_config> <expert_data_num> <seed> <gpu_id> [batch_size]"
    echo "Example: $0 automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 64"
    echo ""
    echo "This script loads ALL data into memory for maximum speed!"
    echo "Recommended batch sizes: 32 (safe), 64 (balanced), 96 (aggressive)"
    exit 1
fi

# Warn about very large batch sizes
if [ $batch_size -gt 128 ]; then
    echo "⚠️  WARNING: Batch size $batch_size is very large!"
    echo "   This may cause GPU memory issues."
    echo "   Recommended: 32-96 for RTX 4090"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Try a smaller batch size (32-96)."
        exit 1
    fi
fi

echo "=============================================="
echo "🚀 FAST TRAINING MODE - ALL DATA IN MEMORY"
echo "=============================================="
echo "Task: $task_name"
echo "Config: $task_config" 
echo "Episodes: $expert_data_num"
echo "Seed: $seed"
echo "GPU: $gpu_id"
echo "Batch size: $batch_size"
echo "=============================================="

# Check system resources
echo "System Resources Check:"
echo "CPU Memory:"
free -h
echo ""
echo "GPU Memory:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -2
else
    echo "nvidia-smi not found"
fi
echo ""

# Estimate memory requirement for 70GB data
echo "⚠️  MEMORY REQUIREMENT:"
echo "   Data size: ~70GB"
echo "   Plus overhead: ~20GB"
echo "   Total needed: ~90GB"
echo ""

# Check if we have enough memory
available_memory=$(free | awk 'NR==2{printf "%.0f", $7/1024/1024}')
if [ $available_memory -lt 90 ]; then
    echo "❌ WARNING: Available memory ($available_memory GB) might be insufficient!"
    echo "   Recommended: 90GB+ available memory"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Consider using ./train_safe.sh instead."
        exit 1
    fi
else
    echo "✅ Memory check passed: ${available_memory}GB available"
fi

echo ""

# Process data if zarr file doesn't exist
if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    echo "❌ Data not found. Please process data first:"
    echo "   bash process_data.sh ${task_name} ${task_config} ${expert_data_num}"
    exit 1
else
    echo "✅ Data found: ./data/${task_name}-${task_config}-${expert_data_num}.zarr"
fi

echo ""
echo "🚀 Starting FAST training (all data loaded into memory)..."
echo "This will:"
echo "  1. Load ALL 70GB of data into memory (~2-5 minutes)"
echo "  2. Train at maximum speed with batch size $batch_size"
echo "  3. Use optimized data loading and no lazy access"
echo ""

# Run fast training
bash scripts/train_policy_fast.sh robot_dp3_fast ${task_name} ${task_config} ${expert_data_num} train ${seed} ${gpu_id} ${batch_size}

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 FAST training completed successfully!"
else
    echo ""
    echo "❌ FAST training failed with exit code: $exit_code"
    echo ""
    echo "If you encounter memory issues, try:"
    echo "1. ./train_safe.sh (memory-efficient alternative)"
    echo "2. Reduce batch size: $0 ... $(($batch_size / 2))"
    echo "3. Close other applications to free memory"
fi