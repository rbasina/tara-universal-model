#!/usr/bin/env python3
"""
TARA Universal Model Training Monitor
Monitors the progress of domain-specific training processes
"""

import os
import json
import time
import glob
from datetime import datetime
from pathlib import Path

def check_adapter_status():
    """Check which domain adapters have been created"""
    adapters_dir = Path("models/adapters")
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    
    status = {}
    for domain in domains:
        domain_dir = adapters_dir / domain
        if domain_dir.exists():
            files = list(domain_dir.glob("*"))
            if files:
                # Check for key files that indicate successful training
                has_model = any(f.name in ["pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"] for f in files)
                has_config = any(f.name in ["config.json", "adapter_config.json"] for f in files)
                status[domain] = {
                    "status": "completed" if (has_model and has_config) else "in_progress",
                    "files": len(files),
                    "file_list": [f.name for f in files[:5]]  # Show first 5 files
                }
            else:
                status[domain] = {"status": "empty", "files": 0, "file_list": []}
        else:
            status[domain] = {"status": "not_started", "files": 0, "file_list": []}
    
    return status

def check_training_data():
    """Check recent training data generation"""
    data_dir = Path("data/synthetic")
    if not data_dir.exists():
        return {}
    
    today = datetime.now().strftime("%Y%m%d")
    recent_files = {}
    
    for domain in ["healthcare", "business", "education", "creative", "leadership"]:
        pattern = f"{domain}_train_{today}_*.json"
        files = list(data_dir.glob(pattern))
        if files:
            latest = max(files, key=lambda x: x.stat().st_mtime)
            recent_files[domain] = {
                "file": latest.name,
                "size_mb": round(latest.stat().st_size / (1024*1024), 2),
                "modified": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S")
            }
    
    return recent_files

def check_training_summaries():
    """Check recent training summary files"""
    summaries = glob.glob("training_summary_*.json")
    if not summaries:
        return None
    
    latest = max(summaries, key=lambda x: os.path.getmtime(x))
    
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        return {
            "file": latest,
            "timestamp": data.get("timestamp", "unknown"),
            "successful_domains": data.get("successful_domains", []),
            "failed_domains": data.get("failed_domains", []),
            "total_time_hours": data.get("total_time_hours", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def check_python_processes():
    """Check running Python processes (Windows)"""
    try:
        import subprocess
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Header + data
                processes = []
                for line in lines[1:]:  # Skip header
                    parts = [p.strip('"') for p in line.split('","')]
                    if len(parts) >= 5:
                        processes.append({
                            "name": parts[0],
                            "pid": parts[1],
                            "memory_kb": parts[4].replace(',', '').replace(' K', '')
                        })
                return processes
        return []
    except Exception as e:
        return [{"error": str(e)}]

def main():
    """Main monitoring function"""
    print("TARA Universal Model Training Monitor")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check adapter status
    print("Domain Adapter Status:")
    adapter_status = check_adapter_status()
    for domain, info in adapter_status.items():
        status_symbol = {
            "completed": "[DONE]",
            "in_progress": "[WORK]", 
            "empty": "[WAIT]",
            "not_started": "[FAIL]"
        }.get(info["status"], "[????]")
        
        print(f"  {status_symbol} {domain.capitalize()}: {info['status']} ({info['files']} files)")
        if info["file_list"]:
            print(f"    Files: {', '.join(info['file_list'])}")
    print()
    
    # Check training data
    print("Recent Training Data:")
    training_data = check_training_data()
    if training_data:
        for domain, info in training_data.items():
            print(f"  [DATA] {domain.capitalize()}: {info['file']} ({info['size_mb']}MB, {info['modified']})")
    else:
        print("  No recent training data found for today")
    print()
    
    # Check training summaries
    print("Latest Training Summary:")
    summary = check_training_summaries()
    if summary:
        if "error" in summary:
            print(f"  [ERROR] Error reading summary: {summary['error']}")
        else:
            print(f"  [SUMM] File: {summary['file']}")
            print(f"  [TIME] Time: {summary['timestamp']}")
            print(f"  [PASS] Successful: {len(summary['successful_domains'])} domains")
            print(f"  [FAIL] Failed: {len(summary['failed_domains'])} domains")
            if summary['failed_domains']:
                print(f"    Failed domains: {', '.join(summary['failed_domains'])}")
    else:
        print("  No training summaries found")
    print()
    
    # Check Python processes
    print("Python Processes:")
    processes = check_python_processes()
    if processes:
        for proc in processes:
            if "error" in proc:
                print(f"  [ERROR] Error: {proc['error']}")
            else:
                memory_mb = round(int(proc['memory_kb']) / 1024, 1)
                print(f"  [PROC] PID {proc['pid']}: {memory_mb}MB")
    else:
        print("  No Python processes found")
    print()
    
    # Summary
    completed = sum(1 for info in adapter_status.values() if info["status"] == "completed")
    in_progress = sum(1 for info in adapter_status.values() if info["status"] in ["in_progress", "empty"])
    
    print("Training Progress Summary:")
    print(f"  [DONE] Completed: {completed}/5 domains")
    print(f"  [WORK] In Progress: {in_progress}/5 domains")
    print(f"  [PROG] Overall Progress: {(completed/5)*100:.1f}%")
    
    if completed == 5:
        print("\n[SUCCESS] All domains completed! TARA Universal Model is ready!")
    elif in_progress > 0:
        print(f"\n[ACTIVE] Training in progress... {5-completed} domains remaining")
    else:
        print("\n[STOPPED] No active training detected")

if __name__ == "__main__":
    main() 