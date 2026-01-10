#!/usr/bin/env python3
"""
Memory-efficient processing script with monitoring
"""
import psutil
import time
import os
import sys

def monitor_memory_usage(pid=None):
    """Monitor memory usage of current process or specific PID"""
    if pid is None:
        process = psutil.Process()
    else:
        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            print(f"Process {pid} not found")
            return
    
    while True:
        try:
            # Get memory info
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            print(f"Process Memory: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB "
                  f"({memory_percent:.1f}% of system)")
            print(f"System Memory: {system_memory.used / 1024 / 1024 / 1024:.2f} GB / "
                  f"{system_memory.total / 1024 / 1024 / 1024:.2f} GB "
                  f"({system_memory.percent:.1f}% used)")
            
            # Warning if memory usage is high
            if memory_percent > 50:
                print("⚠️  WARNING: High memory usage!")
            if system_memory.percent > 85:
                print("🚨 CRITICAL: System memory critically low!")
            
            print("-" * 60)
            time.sleep(10)  # Check every 10 seconds
            
        except psutil.NoSuchProcess:
            print("Process ended")
            break
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        monitor_memory_usage(pid)
    else:
        monitor_memory_usage()