import os
import json
import re
import time
import psutil
from datetime import datetime

def get_training_progress():
    """Extract training progress from logs and state files"""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_progress": 40,  # Base progress (2/5 domains complete)
        "education_progress": 0,
        "education_steps": 0,
        "memory_usage_gb": 0,
        "active_processes": 0,
        "is_training_active": False
    }
    
    # Check for Python processes
    python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                       if p.info['name'] == 'python.exe']
    
    progress_data["active_processes"] = len(python_processes)
    progress_data["is_training_active"] = any("train" in " ".join(p.info['cmdline']) 
                                             if p.info['cmdline'] else "" 
                                             for p in python_processes)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    progress_data["memory_usage_gb"] = round(memory.used / (1024**3), 2)
    
    # Read training log for progress
    log_path = os.path.join("logs", "tara_training.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
                
                # Extract progress percentage
                progress_matches = re.findall(r'(\d+)%\|[█▉▊▋▌▍▎▏▐▕▖▗▘▙▚▛▜▝▞▟]+\s+\|?\s+(\d+)\/400', log_content)
                if progress_matches:
                    # Get the last match (most recent)
                    progress_percent, progress_steps = progress_matches[-1]
                    progress_data["education_progress"] = int(progress_percent)
                    progress_data["education_steps"] = int(progress_steps)
                    
                    # Update overall progress (40% base + education contribution)
                    progress_data["overall_progress"] = 40 + int(int(progress_percent) * 0.12)
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    return progress_data

def update_dashboard():
    """Update the training_status.json file for the dashboard"""
    progress_data = get_training_progress()
    
    # Write to JSON file
    with open("training_status.json", "w") as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"Dashboard updated: Education at {progress_data['education_progress']}%, "
          f"Overall at {progress_data['overall_progress']}%")

if __name__ == "__main__":
    update_dashboard() 