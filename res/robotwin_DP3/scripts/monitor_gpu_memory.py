#!/usr/bin/env python3
"""
GPU and CPU memory monitoring script for training
"""
import subprocess
import time
import psutil
import sys
import os

def get_gpu_memory():
    """Get GPU memory usage"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                used, total, free = map(int, line.split(', '))
                gpu_info.append({
                    'gpu_id': i,
                    'used': used,
                    'total': total,
                    'free': free,
                    'percent': (used / total) * 100
                })
            return gpu_info
        else:
            return None
    except:
        return None

def monitor_resources(target_pid=None):
    """Monitor system resources"""
    print("Starting memory monitoring...")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    while True:
        try:
            # CPU Memory
            cpu_memory = psutil.virtual_memory()
            
            # Process memory if PID provided
            process_memory = None
            if target_pid:
                try:
                    process = psutil.Process(target_pid)
                    process_memory = process.memory_info()
                except psutil.NoSuchProcess:
                    print(f"Process {target_pid} not found")
                    target_pid = None
            
            # GPU Memory
            gpu_info = get_gpu_memory()
            
            # Display information
            print(f"\r{'='*80}")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            
            # CPU Memory
            print(f"CPU Memory: {cpu_memory.used/1024**3:.1f}GB/{cpu_memory.total/1024**3:.1f}GB ({cpu_memory.percent:.1f}%)")
            
            if cpu_memory.percent > 85:
                print("🚨 WARNING: CPU memory critically high!")
            elif cpu_memory.percent > 70:
                print("⚠️  CAUTION: CPU memory getting high")
            
            # Process Memory
            if process_memory:
                process_gb = process_memory.rss / 1024**3
                process_percent = (process_memory.rss / cpu_memory.total) * 100
                print(f"Process {target_pid}: {process_gb:.1f}GB ({process_percent:.1f}%)")
            
            # GPU Memory
            if gpu_info:
                for gpu in gpu_info:
                    used_gb = gpu['used'] / 1024
                    total_gb = gpu['total'] / 1024
                    free_gb = gpu['free'] / 1024
                    print(f"GPU {gpu['gpu_id']}: {used_gb:.1f}GB/{total_gb:.1f}GB ({gpu['percent']:.1f}%) - Free: {free_gb:.1f}GB")
                    
                    if gpu['percent'] > 90:
                        print("🚨 WARNING: GPU memory critically high!")
                    elif gpu['percent'] > 75:
                        print("⚠️  CAUTION: GPU memory getting high")
            else:
                print("GPU: Not available or nvidia-smi not found")
            
            print("=" * 80, end="")
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

def find_training_process():
    """Find training process automatically"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train_dp3.py' in cmdline or 'python' in proc.info['name'] and any('train' in arg for arg in proc.info['cmdline']):
                    return proc.info['pid']
        except:
            continue
    return None

if __name__ == "__main__":
    target_pid = None
    
    if len(sys.argv) > 1:
        try:
            target_pid = int(sys.argv[1])
        except:
            print("Invalid PID provided")
    else:
        # Try to find training process automatically
        target_pid = find_training_process()
        if target_pid:
            print(f"Found training process: PID {target_pid}")
        else:
            print("No training process found, monitoring system resources only")
    
    monitor_resources(target_pid)