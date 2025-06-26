#!/usr/bin/env python3
"""
TARA Universal Model - Domain Training Restart
Resilient script to restart domain training after Cursor AI restarts

Features:
- Automatically detects interrupted training
- Resumes from latest checkpoints
- Memory optimization
- Handles Cursor AI restarts gracefully
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import psutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/restart_domains.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_training_state():
    """Check the current state of domain training"""
    state_file = "training_state/overall_training_state.json"
    
    if not os.path.exists(state_file):
        logger.info("No previous training state found")
        return None
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            
        return state
    except Exception as e:
        logger.error(f"Failed to read training state: {e}")
        return None

def check_python_process():
    """Check if a Python process is already running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('parameterized_train_domains.py' in cmd for cmd in cmdline):
                    return proc.info['pid']
        return None
    except Exception as e:
        logger.error(f"Failed to check Python processes: {e}")
        return None

def restart_training(domains=None, model=None, resume=True):
    """Restart domain training with the specified parameters"""
    # Build command
    cmd = ["python", "scripts/training/parameterized_train_domains.py"]
    
    if domains:
        cmd.extend(["--domains", domains])
    
    if model:
        cmd.extend(["--model", model])
    
    if resume:
        cmd.append("--resume_from_checkpoint")
        cmd.append("auto")
    
    cmd.append("--memory-efficient")
    
    # Log command
    logger.info(f"Restarting training with command: {' '.join(cmd)}")
    
    # Execute command
    try:
        process = subprocess.Popen(cmd)
        logger.info(f"Started training process with PID: {process.pid}")
        return process.pid
    except Exception as e:
        logger.error(f"Failed to restart training: {e}")
        return None

def main():
    """Main function to restart domain training"""
    parser = argparse.ArgumentParser(description="Restart domain training with resilience to Cursor AI restarts")
    parser.add_argument("--domains", type=str, help="Comma-separated list of domains to train")
    parser.add_argument("--model", type=str, help="Base model to use for training")
    parser.add_argument("--force", action="store_true", help="Force restart even if a training process is already running")
    
    args = parser.parse_args()
    
    # Check if a training process is already running
    existing_pid = check_python_process()
    if existing_pid and not args.force:
        logger.info(f"A training process is already running with PID: {existing_pid}")
        logger.info("Use --force to restart anyway")
        return
    
    # Check training state
    state = check_training_state()
    
    # Determine domains and model
    domains = args.domains
    model = args.model
    
    if state and state.get("status") == "in_progress":
        # Get pending domains from state
        pending_domains = state.get("pending_domains", [])
        if pending_domains:
            if not domains:
                domains = ",".join(pending_domains)
                logger.info(f"Using pending domains from previous run: {domains}")
            
            # Get model from state if not specified
            if not model and state.get("base_model"):
                model = state.get("base_model")
                logger.info(f"Using model from previous run: {model}")
    
    # Restart training
    if domains or model:
        restart_training(domains, model)
    else:
        logger.info("No domains or model specified, and no previous state found")
        logger.info("Please specify domains and model with --domains and --model")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("training_state", exist_ok=True)
    
    main() 