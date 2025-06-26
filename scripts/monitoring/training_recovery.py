#!/usr/bin/env python
# TARA Universal Model - Training Recovery System
# Monitors training progress and automatically resumes interrupted training

import os
import sys
import time
import json
import glob
import subprocess
import argparse
from datetime import datetime, timedelta
import logging
import socket
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/training_recovery.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
CHECKPOINT_DIR = "models/adapters"
RESULTS_DIR = "training_results"
MONITOR_PORT = 8001
CHECK_INTERVAL = 300  # 5 minutes
MAX_RETRIES = 3
TRAINING_SCRIPT = "scripts/training/parameterized_train_domains.py"

def setup_argparse():
    """Configure command line arguments"""
    parser = argparse.ArgumentParser(description="TARA Universal Model Training Recovery System")
    parser.add_argument("--domains", type=str, required=True, help="Comma-separated list of domains to train")
    parser.add_argument("--model", type=str, required=True, help="Model to use for training")
    parser.add_argument("--check_interval", type=int, default=CHECK_INTERVAL, help="Interval between checks in seconds")
    parser.add_argument("--max_runtime", type=int, default=6, help="Maximum runtime in hours before saving state")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume training if interrupted")
    return parser.parse_args()

def is_training_running():
    """Check if training process is running by checking the monitor port"""
    try:
        response = requests.get(f"http://localhost:{MONITOR_PORT}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_latest_checkpoint(domain):
    """Find the latest checkpoint for a specific domain"""
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, domain, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest

def get_training_progress():
    """Get current training progress from the monitoring endpoint"""
    try:
        response = requests.get(f"http://localhost:{MONITOR_PORT}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def start_training(domains, model, resume_from=None):
    """Start or resume training process"""
    cmd = [sys.executable, TRAINING_SCRIPT, "--domains", domains, "--model", model]
    
    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])
    
    logging.info(f"Starting training with command: {' '.join(cmd)}")
    
    # Start process in background
    subprocess.Popen(cmd)
    
    # Wait for training to start
    for _ in range(10):
        time.sleep(5)
        if is_training_running():
            logging.info("Training started successfully")
            return True
    
    logging.error("Failed to start training")
    return False

def start_monitor_server():
    """Start the monitoring server if not already running"""
    if not is_training_running():
        monitor_script = "scripts/monitoring/simple_web_monitor.py"
        cmd = [sys.executable, monitor_script]
        subprocess.Popen(cmd)
        time.sleep(5)
        return is_training_running()
    return True

def save_recovery_state(domains, model, start_time):
    """Save recovery state to a file"""
    state = {
        "domains": domains,
        "model": model,
        "start_time": start_time.isoformat(),
        "last_check": datetime.now().isoformat(),
        "checkpoints": {domain: get_latest_checkpoint(domain) for domain in domains.split(",")}
    }
    
    with open("training_recovery_state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    logging.info(f"Saved recovery state: {state}")
    return state

def load_recovery_state():
    """Load recovery state from file"""
    if not os.path.exists("training_recovery_state.json"):
        return None
    
    with open("training_recovery_state.json", "r") as f:
        return json.load(f)

def create_recovery_script():
    """Create a batch script that can be used to resume training after reboot"""
    script_content = """@echo off
echo TARA Universal Model - Training Recovery
echo Starting recovery process...
python scripts/monitoring/training_recovery.py --auto_resume
"""
    
    with open("resume_training.bat", "w") as f:
        f.write(script_content)
    
    logging.info("Created recovery script: resume_training.bat")

def main():
    """Main function"""
    args = setup_argparse()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    if args.auto_resume:
        # Auto-resume mode
        state = load_recovery_state()
        if not state:
            logging.error("No recovery state found")
            return
        
        logging.info(f"Resuming training from state: {state}")
        domains = state["domains"]
        model = state["model"]
        
        # Find latest checkpoints
        domain_list = domains.split(",")
        checkpoints = {domain: get_latest_checkpoint(domain) for domain in domain_list}
        
        # Start monitor server
        if not start_monitor_server():
            logging.error("Failed to start monitor server")
            return
        
        # Resume training for each domain with checkpoint
        for domain in domain_list:
            if checkpoints[domain]:
                start_training(domain, model, checkpoints[domain])
            else:
                start_training(domain, model)
            time.sleep(5)
    else:
        # Normal monitoring mode
        domains = args.domains
        model = args.model
        check_interval = args.check_interval
        max_runtime = args.max_runtime
        
        logging.info(f"Starting training recovery system for domains: {domains}")
        logging.info(f"Using model: {model}")
        logging.info(f"Check interval: {check_interval} seconds")
        logging.info(f"Maximum runtime: {max_runtime} hours")
        
        # Start monitor server if not running
        if not start_monitor_server():
            logging.error("Failed to start monitor server")
            return
        
        # Start training if not running
        if not is_training_running():
            if not start_training(domains, model):
                logging.error("Failed to start training")
                return
        
        # Create recovery script
        create_recovery_script()
        
        # Monitor training
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=max_runtime)
        
        try:
            while datetime.now() < end_time:
                if not is_training_running():
                    logging.warning("Training process not detected")
                    state = save_recovery_state(domains, model, start_time)
                    
                    for retry in range(MAX_RETRIES):
                        logging.info(f"Attempting to resume training (attempt {retry+1}/{MAX_RETRIES})")
                        if start_training(domains, model):
                            break
                        time.sleep(30)
                    else:
                        logging.error("Failed to resume training after multiple attempts")
                        break
                
                # Get current progress
                progress = get_training_progress()
                if progress:
                    logging.info(f"Current training progress: {progress}")
                
                # Save current state
                save_recovery_state(domains, model, start_time)
                
                # Sleep until next check
                time.sleep(check_interval)
            
            # We've reached the maximum runtime
            logging.info(f"Reached maximum runtime of {max_runtime} hours")
            save_recovery_state(domains, model, start_time)
            logging.info("Training state saved. To resume, run: python scripts/monitoring/training_recovery.py --auto_resume")
            
        except KeyboardInterrupt:
            logging.info("Training recovery system interrupted by user")
            save_recovery_state(domains, model, start_time)
            logging.info("Training state saved. To resume, run: python scripts/monitoring/training_recovery.py --auto_resume")

if __name__ == "__main__":
    main() 