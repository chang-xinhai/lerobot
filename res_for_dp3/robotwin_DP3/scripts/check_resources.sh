#!/bin/bash

# Quick resource check script
echo "=============================================="
echo "System Resource Check"
echo "=============================================="

echo "CPU Memory:"
free -h
echo ""

echo "Available CPU cores:"
nproc
echo ""

echo "GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
else
    echo "nvidia-smi not found - no GPU detected"
fi
echo ""

echo "Disk Space (current directory):"
df -h .
echo ""

echo "Current directory size:"
du -sh .
echo ""

# Estimate memory requirements
echo "=============================================="
echo "Memory Requirement Estimates"
echo "=============================================="
echo "For 15,000 episodes (~60MB each):"
echo "- Data processing (batch size 30): ~1.8GB peak CPU memory"
echo "- Data processing (batch size 50): ~3.0GB peak CPU memory"
echo "- Training (batch size 32): ~8-12GB GPU memory"
echo "- Training (batch size 64): ~16-20GB GPU memory"
echo ""

# Get current memory usage
cpu_used=$(free | awk 'NR==2{printf "%.1f", $3/1024/1024}')
cpu_total=$(free | awk 'NR==2{printf "%.1f", $2/1024/1024}')
cpu_available=$(free | awk 'NR==2{printf "%.1f", $7/1024/1024}')

echo "Current CPU Memory: ${cpu_used}GB used / ${cpu_total}GB total (${cpu_available}GB available)"

# Recommendations
echo ""
echo "=============================================="
echo "Recommendations for your system"
echo "=============================================="

if (( $(echo "$cpu_available < 10" | bc -l) )); then
    echo "⚠️  CPU Memory: Use smaller batch sizes"
    echo "   - Data processing batch size: 30 or less"
    echo "   - Consider freeing up some memory first"
else
    echo "✅ CPU Memory: Should be sufficient"
    echo "   - Data processing batch size: 50 is safe"
fi

if command -v nvidia-smi &> /dev/null; then
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    gpu_mem_gb=$((gpu_mem / 1024))
    
    if [ "$gpu_mem_gb" -lt 12 ]; then
        echo "⚠️  GPU Memory: Use smaller training batch sizes"
        echo "   - Training batch size: 16 or 32"
    elif [ "$gpu_mem_gb" -lt 24 ]; then
        echo "✅ GPU Memory: Should be sufficient"
        echo "   - Training batch size: 32 is safe, 64 might work"
    else
        echo "✅ GPU Memory: Plenty available"
        echo "   - Training batch size: 64 or higher is safe"
    fi
else
    echo "❓ GPU Memory: Cannot detect, use conservative settings"
fi

echo ""
echo "Suggested command for your system:"
echo "./train_safe.sh automoma_manip_summit_franka task_1object_15scene_20pose 15000 0 1 32 50"