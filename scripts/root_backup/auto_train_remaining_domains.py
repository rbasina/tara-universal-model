#!/usr/bin/env python3
"""
Automated Training Script for Remaining TARA Domains
Ramesh Basina - PhD Research Implementation
"""

import subprocess
import time
import os
import psutil
import json
from datetime import datetime

def check_healthcare_completion():
    """Check if healthcare training is complete"""
    try:
        # Check if healthcare adapter exists and is complete
        healthcare_path = "models/adapters/healthcare"
        if os.path.exists(healthcare_path):
            # Check for training completion markers
            adapter_path = os.path.join(healthcare_path, "adapter")
            if os.path.exists(adapter_path):
                print("✅ Healthcare training completed!")
                return True
        return False
    except Exception as e:
        print(f"Error checking healthcare status: {e}")
        return False

def check_training_process():
    """Check if any training process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'train_domain.py' in ' '.join(proc.info['cmdline']):
                return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False, None

def train_domain(domain_name):
    """Train a specific domain"""
    print(f"\n🚀 Starting {domain_name.upper()} domain training...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    cmd = f"python scripts/train_domain.py --domain {domain_name}"
    
    try:
        # Start training process
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor training progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ {domain_name.upper()} training completed successfully!")
            return True
        else:
            print(f"❌ {domain_name.upper()} training failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error training {domain_name}: {e}")
        return False

def log_progress(domain, status, details=""):
    """Log training progress"""
    progress_file = "training_progress.json"
    
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {"domains": {}, "started": datetime.now().isoformat()}
        
        progress["domains"][domain] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not log progress: {e}")

def main():
    """Main automated training orchestrator"""
    print("🎯 TARA Universal Model - Automated Training Pipeline")
    print("=" * 60)
    print("📋 Training Queue: Business → Creative → Leadership")
    print("⏳ Waiting for Healthcare completion...")
    
    # Wait for healthcare to complete
    while True:
        is_training, pid = check_training_process()
        
        if not is_training:
            if check_healthcare_completion():
                print("✅ Healthcare training detected as complete!")
                break
            else:
                print("⏳ Healthcare training not yet complete, waiting...")
                time.sleep(30)
        else:
            print(f"🔄 Healthcare training still running (PID: {pid})")
            time.sleep(60)
    
    # Training sequence
    domains_to_train = ["business", "creative", "leadership"]
    
    print(f"\n🚀 Starting automated training sequence...")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, domain in enumerate(domains_to_train, 1):
        print(f"\n{'='*60}")
        print(f"📊 Training Domain {i}/3: {domain.upper()}")
        print(f"{'='*60}")
        
        log_progress(domain, "starting")
        
        success = train_domain(domain)
        
        if success:
            log_progress(domain, "completed", "Training successful")
            print(f"✅ {domain.upper()} domain completed successfully!")
            
            # Brief pause between trainings
            if i < len(domains_to_train):
                print("⏸️  Brief pause before next domain...")
                time.sleep(10)
        else:
            log_progress(domain, "failed", "Training encountered errors")
            print(f"❌ {domain.upper()} domain failed!")
            
            # Ask if we should continue
            print("❓ Continue with next domain? (Press Ctrl+C to stop)")
            time.sleep(5)
    
    print(f"\n🎉 AUTOMATED TRAINING COMPLETE!")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("📊 Final Status:")
    
    # Check final status
    completed_domains = []
    for domain in ["healthcare", "education"] + domains_to_train:
        domain_path = f"models/adapters/{domain}"
        if os.path.exists(domain_path):
            completed_domains.append(domain)
            print(f"✅ {domain.upper()}: COMPLETE")
        else:
            print(f"❌ {domain.upper()}: FAILED")
    
    print(f"\n🏆 Phase 1 Arc Reactor Foundation: {len(completed_domains)}/5 domains complete")
    
    if len(completed_domains) == 5:
        print("🎯 PHASE 1 COMPLETE - Ready for Phase 2 Perplexity Intelligence!")
    else:
        print(f"⚠️  {5 - len(completed_domains)} domains need retry")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
        print("📊 Current progress saved in training_progress.json")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("📊 Check logs for details") 