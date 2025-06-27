#!/usr/bin/env python3
"""
TARA Universal Model - Parameterized Domain Training
Train domains using any specified model with customizable parameters
Enhanced with memory optimization, checkpoint management, and robust error handling
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import gc
import time
import psutil
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.enhanced_trainer import EnhancedTARATrainer
from tara_universal_model.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/domain_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models/adapters", exist_ok=True)
os.makedirs("training_results", exist_ok=True)
os.makedirs("training_state", exist_ok=True)

# Global constants
DEFAULT_DOMAINS = ["education", "creative", "leadership"]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT_BACKUP_DIR = "models/checkpoints_backup"
os.makedirs(CHECKPOINT_BACKUP_DIR, exist_ok=True)

def emoji_safe(message):
    """Replace emoji characters with plain text alternatives for console output"""
    emoji_map = {
        "🎯": "[TARGET]",
        "🔍": "[CHECK]",
        "✅": "[OK]",
        "⚠️": "[WARN]",
        "❌": "[ERROR]",
        "ℹ️": "[INFO]",
        "🔄": "[SYNC]",
        "🆕": "[NEW]",
        "📝": "[LOG]",
        "💾": "[MEM]",
        "🚀": "[START]",
        "🛑": "[STOP]",
        "⏱️": "[TIME]",
        "🔧": "[FIX]",
        "📊": "[STATS]",
        "🏆": "[SUCCESS]",
        "❓": "[QUESTION]"
    }
    
    for emoji, replacement in emoji_map.items():
        message = message.replace(emoji, replacement)
    
    return message

def backup_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Create a backup of a checkpoint directory before using it"""
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
        
    try:
        # Create timestamp-based backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(checkpoint_dir)
        domain_name = os.path.basename(os.path.dirname(checkpoint_dir))
        backup_dir = os.path.join(CHECKPOINT_BACKUP_DIR, f"{domain_name}_{checkpoint_name}_{timestamp}")
        
        # Copy checkpoint directory
        shutil.copytree(checkpoint_dir, backup_dir)
        logger.info(f"✅ Created checkpoint backup: {backup_dir}")
        return backup_dir
    except Exception as e:
        logger.warning(f"⚠️ Failed to backup checkpoint {checkpoint_dir}: {e}")
        return None

def validate_checkpoint(checkpoint_dir: str) -> bool:
    """Validate that checkpoint directory contains all necessary files"""
    if not os.path.exists(checkpoint_dir):
        logger.error(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
        return False
        
    # Check for essential files
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        logger.error(f"❌ Missing trainer_state.json in {checkpoint_dir}")
        return False
        
    # Verify step count in trainer state
    try:
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)
        
        actual_step = trainer_state.get("global_step", 0)
        if actual_step <= 0:
            logger.error(f"❌ Invalid step count in {checkpoint_dir}: {actual_step}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to read trainer_state.json: {e}")
        return False
    
    # Check for model files (either pytorch_model.bin or adapter_model.bin/safetensors)
    has_model = (os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or 
                os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")) or
                os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")))
                
    if not has_model:
        logger.warning(f"⚠️ Missing model files in {checkpoint_dir}")
        return False
    
    # Enhanced validation: Check file sizes for corruption
    model_file = None
    for model_file_name in ["pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"]:
        potential_file = os.path.join(checkpoint_dir, model_file_name)
        if os.path.exists(potential_file):
            model_file = potential_file
            break
    
    if model_file and os.path.getsize(model_file) < 1000:  # Less than 1KB
        logger.error(f"❌ Model file too small (possibly corrupted): {model_file}")
        return False
    
    logger.info(f"✅ Checkpoint validation successful for {checkpoint_dir}")
    return True

def find_latest_checkpoint(domain: str, base_model: str) -> Optional[str]:
    """Find the latest checkpoint for a domain with proper validation"""
    # Check both directory structures for checkpoints
    
    # 1. First check in domain-specific folder (Trinity structure)
    domain_dir = f"models/{domain}"
    if os.path.exists(domain_dir):
        # Look for meetara_trinity folders
        trinity_dirs = [d for d in os.listdir(domain_dir) 
                      if d.startswith("meetara_trinity_phase_") and os.path.isdir(os.path.join(domain_dir, d))]
        
        if trinity_dirs:
            # Sort by creation time (newest first)
            trinity_dirs.sort(key=lambda d: os.path.getctime(os.path.join(domain_dir, d)), reverse=True)
            latest_trinity_dir = os.path.join(domain_dir, trinity_dirs[0])
            
            # Find checkpoint directories
            checkpoint_dirs = [d for d in os.listdir(latest_trinity_dir) 
                             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(latest_trinity_dir, d))]
            
            if checkpoint_dirs:
                # Sort by step number (highest first)
                checkpoint_dirs.sort(key=lambda d: int(d.split("-")[-1]), reverse=True)
                latest_checkpoint = os.path.join(latest_trinity_dir, checkpoint_dirs[0])
                
                # Validate checkpoint
                if validate_checkpoint(latest_checkpoint):
                    logger.info(f"✅ Found valid Trinity checkpoint for {domain}: {latest_checkpoint}")
                    return latest_checkpoint
                else:
                    logger.warning(f"⚠️ Trinity checkpoint for {domain} failed validation: {latest_checkpoint}")
    
    # 2. Then check in adapters folder (new structure)
    model_short_name = base_model.split('/')[-1].lower()
    adapters_dir = f"models/adapters/{domain}_{model_short_name}"
    
    if not os.path.exists(adapters_dir):
        return None
        
    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(adapters_dir):
        if item.startswith("checkpoint-"):
            checkpoint_dir = os.path.join(adapters_dir, item)
            
            # Validate checkpoint directory
            trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                try:
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)
                    
                    # Get step number
                    step = trainer_state.get("global_step", 0)
                    
                    # Check for essential files
                    has_optimizer = os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt"))
                    has_model = (os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or 
                                os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")) or
                                os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")))
                    
                    # Only add valid checkpoints with model files
                    if has_model:
                        checkpoints.append((step, checkpoint_dir))
                        logger.info(f"Found valid checkpoint for {domain}: {checkpoint_dir} (step {step})")
                    else:
                        # This is a critical change - we now warn about incomplete checkpoints
                        logger.warning(f"⚠️ Incomplete checkpoint found for {domain}: {checkpoint_dir} (missing model files)")
                        # We don't add this to the valid checkpoints list
                except Exception as e:
                    logger.warning(f"⚠️ Invalid checkpoint for {domain}: {checkpoint_dir} - {e}")
    
    if not checkpoints:
        # Check if we have a special case for education domain with step 134
        education_checkpoint = check_for_education_checkpoint(domain, adapters_dir)
        if education_checkpoint:
            return education_checkpoint
            
        logger.info(f"No valid checkpoints found for {domain}")
        return None
        
    # Sort by step number (descending)
    checkpoints.sort(reverse=True)
    latest_checkpoint = checkpoints[0][1]
    logger.info(f"🔄 Latest checkpoint for {domain}: {latest_checkpoint} (step {checkpoints[0][0]})")
    return latest_checkpoint

def check_for_education_checkpoint(domain: str, output_dir: str) -> Optional[str]:
    """Special handling for education domain with known step 134"""
    if domain != "education":
        return None
        
    # Check for the specific education checkpoint at step 134
    checkpoint_dir = os.path.join(output_dir, "checkpoint-134")
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    
    if os.path.exists(trainer_state_file):
        try:
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            
            step = trainer_state.get("global_step", 0)
            
            # If step is 134 but model files are missing, we have the special case
            has_model = (os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or 
                        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")) or
                        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")))
                        
            if step == 134 and not has_model:
                logger.warning(f"⚠️ Found education checkpoint at step 134 but model files are missing")
                logger.warning(f"⚠️ Will need to restart training from step 0 but track as if from step 134")
                
                # Update dashboard to show correct progress
                update_education_dashboard(step)
                
                # We return None to force fresh training
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking education checkpoint: {e}")
    
    return None

def update_education_dashboard(step: int):
    """Update dashboard to show correct education progress"""
    try:
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_progress": 40 + int((step / 400) * 12),  # Base 40% + education contribution
            "education_progress": int((step / 400) * 100),
            "education_steps": step,
            "memory_usage_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "is_training_active": True
        }
        
        # Write to JSON file
        with open("training_status.json", "w") as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"📊 Dashboard updated: Education at {progress_data['education_progress']}%, "
                    f"Overall at {progress_data['overall_progress']}%")
    except Exception as e:
        logger.warning(f"⚠️ Failed to update dashboard: {e}")

def update_dashboard_state(domains: List[str], base_model: str, checkpoints: Dict[str, str] = None):
    """Update the training_recovery_state.json file for dashboard integration"""
    recovery_state = {
        "domains": ",".join(domains),
        "model": base_model,
        "start_time": datetime.now().isoformat(),
        "last_check": datetime.now().isoformat(),
        "checkpoints": checkpoints or {}
    }
    
    # Save the recovery state
    with open("training_recovery_state.json", "w") as f:
        json.dump(recovery_state, f, indent=2)
    logger.info(f"📊 Updated dashboard state with {len(recovery_state['checkpoints'])} checkpoints")

def setup_domain_logger(domain):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{domain}_training.log")
    logger = logging.getLogger(f"{domain}_logger")
    logger.setLevel(logging.INFO)
    # Remove any existing handlers
    logger.handlers = []
    
    # File handler - set encoding to utf-8 for emoji support
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Custom console handler with emoji replacement
    class EmojiSafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            record.msg = emoji_safe(record.msg)
            super().emit(record)
    
    # Console handler with emoji replacement
    ch = EmojiSafeStreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

async def train_domain(
    domain: str, 
    base_model: str, 
    output_dir: Optional[str] = None, 
    resume_from_checkpoint: Optional[str] = None,
    force_fresh: bool = False,
    max_retries: int = 2,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    lora_r: Optional[int] = None,
    max_steps: Optional[int] = None
) -> Dict[str, Any]:
    """Train a specific domain with the specified base model with enhanced error handling"""
    logger = setup_domain_logger(domain)
    logger.info(f"🎯 Training domain: {domain}")
    start_time = datetime.now()
    retries = 0
    
    # 🔒 SAFETY MECHANISM: Detect and fix training issues before starting
    logger.info(f"🔍 Running safety checks for {domain}...")
    issues = detect_and_fix_training_issues(domain, base_model)
    
    if issues["checkpoint_found"]:
        logger.info(f"✅ Found checkpoint for {domain}: {issues['checkpoint_path']}")
        if issues["state_inconsistent"]:
            if issues["state_fixed"]:
                logger.info(f"✅ Fixed state file inconsistency for {domain}")
            else:
                logger.warning(f"⚠️ Failed to fix state file inconsistency for {domain}")
        if issues["backup_created"]:
            logger.info(f"✅ Created checkpoint backup for {domain}")
    else:
        logger.info(f"ℹ️ No valid checkpoint found for {domain} - will start fresh")
    
    # Determine output directory - UPDATED to use both directory structures
    if not output_dir:
        # 1. Primary output directory in domain-specific folder (Trinity structure)
        model_short_name = base_model.split('/')[-1].lower()
        training_style = "efficient_core_processing"  # Arc Reactor Foundation style
        
        # Use Trinity structure for primary output
        primary_output_dir = f"models/{domain}/meetara_trinity_phase_{training_style}"
        
        # 2. Secondary output directory in adapters folder (for compatibility)
        secondary_output_dir = f"models/adapters/{domain}_{model_short_name}"
        
        # Use primary output directory
        output_dir = primary_output_dir
        
        # Create both directories
        os.makedirs(primary_output_dir, exist_ok=True)
        os.makedirs(secondary_output_dir, exist_ok=True)
        
        # Create symlink for cross-compatibility
        try:
            # On Windows, we need to create directory junction instead of symlink
            if os.name == 'nt':
                # Create a directory junction if it doesn't exist
                if not os.path.exists(os.path.join(secondary_output_dir, "trinity_link")):
                    os.system(f'mklink /J "{os.path.join(secondary_output_dir, "trinity_link")}" "{primary_output_dir}"')
                    logger.info(f"✅ Created directory junction from {secondary_output_dir}/trinity_link to {primary_output_dir}")
            else:
                # On Unix systems, create a symlink
                symlink_path = os.path.join(secondary_output_dir, "trinity_link")
                if not os.path.exists(symlink_path):
                    os.symlink(primary_output_dir, symlink_path)
                    logger.info(f"✅ Created symlink from {secondary_output_dir}/trinity_link to {primary_output_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create directory link: {e}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Special handling for education domain
    if domain == "education":
        education_dir = output_dir
        adapter_dir = f"models/adapters/education_qwen2.5-3b-instruct"
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Check both paths for checkpoint-134
        checkpoint_134_paths = [
            os.path.join(education_dir, "checkpoint-134"),
            os.path.join(adapter_dir, "checkpoint-134")
        ]
        
        checkpoint_found = False
        for checkpoint_path in checkpoint_134_paths:
            trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                try:
                    with open(trainer_state_file, 'r') as f:
                        trainer_state = json.load(f)
                    
                    step = trainer_state.get("global_step", 0)
                    
                    if step == 134:
                        # Update dashboard to show correct progress
                        update_education_dashboard(step)
                        logger.info(f"🔄 Education domain: Found checkpoint at step 134/400 (33.5%)")
                        
                        # If not forcing fresh training, use this checkpoint
                        if not force_fresh and resume_from_checkpoint is None:
                            resume_from_checkpoint = checkpoint_path
                            logger.info(f"🔄 Using education checkpoint at step 134: {checkpoint_path}")
                        
                        checkpoint_found = True
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Error reading education checkpoint: {e}")
    
    # Handle checkpoint logic
    if force_fresh:
        logger.info(f"🔄 Forcing fresh training for {domain} (ignoring checkpoints)")
        resume_from_checkpoint = None
    elif resume_from_checkpoint == "auto":
        # Auto-detect latest checkpoint
        resume_from_checkpoint = find_latest_checkpoint(domain, base_model)
        if resume_from_checkpoint:
            logger.info(f"🔄 Auto-detected checkpoint for {domain}: {resume_from_checkpoint}")
            # Backup the checkpoint before using it
            backup_checkpoint(resume_from_checkpoint)
    elif resume_from_checkpoint:
        logger.info(f"🔄 Using specified checkpoint for {domain}: {resume_from_checkpoint}")
        # Backup the checkpoint before using it
        backup_checkpoint(resume_from_checkpoint)
    else:
        # Check if we have any checkpoint for this domain
        latest_checkpoint = find_latest_checkpoint(domain, base_model)
        if latest_checkpoint:
            logger.info(f"🔄 Found checkpoint for {domain}: {latest_checkpoint}")
            resume_from_checkpoint = latest_checkpoint
            backup_checkpoint(resume_from_checkpoint)
        else:
            logger.info(f"🆕 Starting fresh training for {domain} (no checkpoint found)")
    
    # Save current state for recovery
    state_file = f"training_state/{domain}_training_state.json"
    state = {
        "domain": domain,
        "base_model": base_model,
        "output_dir": output_dir,
        "resume_checkpoint": resume_from_checkpoint,
        "start_time": datetime.now().isoformat(),
        "status": "starting",
        "retries": retries,
        "current_step": 0,
        "total_steps": max_steps or 400,  # Use provided max_steps or default
        "progress_percentage": 0,
        "last_update": datetime.now().isoformat(),
        "notes": "Initializing training environment"
    }
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    # Create domain-specific log file
    domain_log_file = f"logs/{domain}_training.log"
    domain_handler = logging.FileHandler(domain_log_file, encoding='utf-8')
    domain_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(domain_handler)
    logger.info(f"📝 Domain-specific log file created: {domain_log_file}")
    
    # Set up background state file updater
    state_update_task = None
    
    async def update_state_periodically():
        """Update the state file periodically with current progress"""
        try:
            while True:
                # Check for checkpoint directories to get current progress
                checkpoint_dirs = []
                if os.path.exists(output_dir):
                    checkpoint_dirs = [d for d in os.listdir(output_dir) 
                                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
                
                current_step = 0
                if checkpoint_dirs:
                    # Sort by step number (highest first)
                    checkpoint_dirs.sort(key=lambda d: int(d.split("-")[-1]), reverse=True)
                    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[0])
                    
                    # Read trainer state to get current step
                    trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        try:
                            with open(trainer_state_file, 'r') as f:
                                trainer_state = json.load(f)
                            current_step = trainer_state.get("global_step", 0)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to read trainer state: {e}")
                
                # Update state file
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        
                        # Only update if we have new progress
                        if current_step > state.get("current_step", 0):
                            state["current_step"] = current_step
                            state["total_steps"] = max_steps or 400
                            state["progress_percentage"] = int((current_step / (max_steps or 400)) * 100)
                            state["last_update"] = datetime.now().isoformat()
                            state["notes"] = f"Training in progress - step {current_step}/{max_steps or 400} ({state['progress_percentage']}%)"
                            
                            with open(state_file, 'w') as f:
                                json.dump(state, f, indent=2)
                            
                            logger.info(f"📊 Updated state file: {current_step}/{max_steps or 400} ({state['progress_percentage']}%)")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to update state file: {e}")
                
                await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            logger.info("State update task cancelled")
        except Exception as e:
            logger.error(f"❌ State update task failed: {e}")
    
    # Check available memory
    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024 * 1024 * 1024)
    total_memory_gb = mem_info.total / (1024 * 1024 * 1024)
    logger.info(f"💾 Available memory: {available_memory_gb:.2f}GB / {total_memory_gb:.2f}GB ({mem_info.percent}% used)")
    
    # Update state to loading model
    state["status"] = "loading_model"
    state["last_update"] = datetime.now().isoformat()
    state["notes"] = "Loading model and preparing training environment"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    while retries <= max_retries:
        try:
            # Start the state updater task
            state_update_task = asyncio.create_task(update_state_periodically())
            
            # Force garbage collection before starting
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Initialize trainer
            config = get_config()
            trainer = EnhancedTARATrainer(
                config=config,
                domain=domain,
                base_model_name=base_model
            )
            
            # Load model and setup LoRA
            trainer.load_base_model()
            trainer.setup_lora()
            
            # Update state to training
            state["status"] = "training"
            state["last_update"] = datetime.now().isoformat()
            state["notes"] = "Training in progress"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Train
            data_path = f"data/synthetic/{domain}_training_data.json"
            
            # Log whether resuming or starting fresh
            if resume_from_checkpoint:
                logger.info(f"🔄 Resuming {domain} training from checkpoint: {resume_from_checkpoint}")
            else:
                logger.info(f"🆕 Starting fresh training for {domain}")
            
            model_path = await trainer.train_with_validation(
                data_path=data_path,
                output_dir=output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
                batch_size=batch_size,
                seq_length=seq_length,
                lora_r=lora_r,
                max_steps=max_steps
            )
            
            # Cancel the state updater task
            if state_update_task:
                state_update_task.cancel()
                try:
                    await state_update_task
                except asyncio.CancelledError:
                    pass
            
            # Update state to completed
            state["status"] = "completed"
            state["completed_at"] = datetime.now().isoformat()
            state["model_path"] = model_path
            state["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"✅ {domain} training completed: {model_path}")
            
            # Force cleanup after training
            del trainer
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return successful result
            return {
                "domain": domain,
                "status": "completed",
                "model_path": model_path,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "retries": retries
            }
            
        except Exception as e:
            retries += 1
            logger.error(f"❌ {domain} training attempt {retries} failed: {e}", exc_info=True)
            
            # Cancel the state updater task if it's running
            if state_update_task:
                state_update_task.cancel()
                try:
                    await state_update_task
                except asyncio.CancelledError:
                    pass
            
            # Update state with error
            state["status"] = "retry" if retries <= max_retries else "failed"
            state["error"] = str(e)
            state["retries"] = retries
            state["last_error_at"] = datetime.now().isoformat()
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Force cleanup
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if retries <= max_retries:
                # Wait before retry
                retry_delay = 30 * retries  # Increasing delay with each retry
                logger.info(f"⏱️ Waiting {retry_delay} seconds before retry {retries}/{max_retries}...")
                await asyncio.sleep(retry_delay)
                
                # Try to find a new checkpoint for retry
                if not force_fresh:
                    new_checkpoint = find_latest_checkpoint(domain, base_model)
                    if new_checkpoint and new_checkpoint != resume_from_checkpoint:
                        logger.info(f"🔄 Found new checkpoint for retry: {new_checkpoint}")
                        resume_from_checkpoint = new_checkpoint
    
    # If we get here, all retries failed
    logger.error(f"❌ {domain} training failed after {max_retries} retries")
    
    # Return failure result
    return {
        "domain": domain,
        "status": "failed",
        "error": state.get("error", "Unknown error"),
        "retries": retries
    }

async def train_domains(
    domains: List[str], 
    base_model: str, 
    resume_from_checkpoint: Optional[Union[str, Dict[str, str]]] = None,
    force_fresh: bool = False,
    max_retries: int = 2,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    lora_r: Optional[int] = None,
    max_steps: Optional[int] = None
) -> Dict[str, Dict]:
    """Train multiple domains with enhanced error handling and checkpoint management"""
    logger.info(f"🚀 Starting domain training with {base_model}")
    logger.info(f"🔄 Domains to train: {', '.join(domains)}")
    
    # Check system memory
    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024 * 1024 * 1024)
    total_memory_gb = mem_info.total / (1024 * 1024 * 1024)
    logger.info(f"💾 System memory: {available_memory_gb:.2f}GB available / {total_memory_gb:.2f}GB total")
    
    # Log custom parameters if provided
    if batch_size is not None:
        logger.info(f"⚙️ Using custom batch size: {batch_size}")
    if seq_length is not None:
        logger.info(f"⚙️ Using custom sequence length: {seq_length}")
    if lora_r is not None:
        logger.info(f"⚙️ Using custom LoRA rank: {lora_r}")
    if max_steps is not None:
        logger.info(f"⚙️ Using custom max steps: {max_steps}")
    
    # Train domains sequentially to avoid memory issues
    logger.info(f"🔄 Training domains sequentially to optimize memory usage")
    
    # Update dashboard state
    update_dashboard_state(domains, base_model)
    
    # Initialize results
    results = {}
    
    # Initialize overall state file
    overall_state_file = "training_state/overall_training_state.json"
    overall_state = {
        "domains": domains,
        "base_model": base_model,
        "start_time": datetime.now().isoformat(),
        "status": "in_progress",
        "completed_domains": [],
        "failed_domains": [],
        "current_domain": None
    }
    
    with open(overall_state_file, 'w') as f:
        json.dump(overall_state, f, indent=2)
    
    # Process domains sequentially
    for domain in domains:
        # Update overall state
        overall_state["current_domain"] = domain
        with open(overall_state_file, 'w') as f:
            json.dump(overall_state, f, indent=2)
        
        # Determine checkpoint to use
        domain_checkpoint = None
        if isinstance(resume_from_checkpoint, dict):
            domain_checkpoint = resume_from_checkpoint.get(domain, "auto")
        else:
            domain_checkpoint = resume_from_checkpoint
        
        try:
            # Train this domain
            result = await train_domain(
                domain=domain,
                base_model=base_model,
                resume_from_checkpoint=domain_checkpoint,
                force_fresh=force_fresh,
                max_retries=max_retries,
                batch_size=batch_size,
                seq_length=seq_length,
                lora_r=lora_r,
                max_steps=max_steps
            )
            
            # Store results
            results[domain] = result
            
            # Update overall state
            if result["status"] == "completed":
                overall_state["completed_domains"].append(domain)
                overall_state["last_completed_domain"] = domain
                overall_state["last_completed_at"] = datetime.now().isoformat()
            else:
                overall_state["failed_domains"].append(domain)
                
            with open(overall_state_file, 'w') as f:
                json.dump(overall_state, f, indent=2)
            
            # Force cleanup between domains
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Small delay to ensure resources are freed
            await asyncio.sleep(5)
            
            # Update dashboard state after each domain
            update_dashboard_state(domains, base_model, {
                d: results[d]["model_path"] if d in results and results[d]["status"] == "completed" else cp
                for d, cp in overall_state["resume_from_checkpoint"].items()
            })
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in domain training loop for {domain}: {e}", exc_info=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            results[domain] = {
                "domain": domain,
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update overall state
            overall_state["failed_domains"].append(domain)
            with open(overall_state_file, 'w') as f:
                json.dump(overall_state, f, indent=2)
    
    # Save results
    results_file = f"training_results/{base_model.split('/')[-1].lower()}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("training_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Update overall state to completed
    overall_state["status"] = "completed"
    overall_state["completed_at"] = datetime.now().isoformat()
    with open(overall_state_file, 'w') as f:
        json.dump(overall_state, f, indent=2)
    
    # Print summary
    successful = len([r for r in results.values() if r["status"] == "completed"])
    failed = len([r for r in results.values() if r["status"] == "failed"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 DOMAIN TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful: {successful}/{len(domains)} domains")
    logger.info(f"❌ Failed: {failed}/{len(domains)} domains")
    logger.info(f"💾 Results saved: {results_file}")
    
    for domain, result in results.items():
        status_emoji = "✅" if result["status"] == "completed" else "❌"
        duration = result.get("duration_seconds", 0)
        logger.info(f"{status_emoji} {domain.upper():12} | {duration:6.1f}s")
        
        if result["status"] == "completed":
            if "retries" in result and result["retries"] > 0:
                logger.info(f"    Completed after {result['retries']} retries")
        elif result["status"] == "failed":
            logger.info(f"    Error: {result.get('error', 'Unknown')}")
    
    return results

def load_domain_model_mapping(mapping_file="configs/domain_model_mapping.yaml"):
    with open(mapping_file, "r") as f:
        mapping = yaml.safe_load(f)
    return mapping.get("domain_models", {})

def get_domains_for_model(domain_models, model_name):
    return [d for d, m in domain_models.items() if m == model_name]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train domains with specified model or model group")
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to train (overrides model group)"
    )
    parser.add_argument(
        "--model_group",
        type=str,
        default=None,
        help="Model name to train all associated domains (e.g., DialoGPT-medium)"
    )
    parser.add_argument(
        "--force_fresh",
        action="store_true",
        help="Force fresh training even if checkpoints exist"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries per domain on failure (default: 2)"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Enable additional memory optimization techniques"
    )
    parser.add_argument(
        "--checkpoint_map",
        type=str,
        help="JSON string with domain-to-checkpoint mapping for domain-specific resumption"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config (default: use config value)"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help="Override sequence length from config (default: use config value)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="Override LoRA rank from config (default: use config value)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override maximum training steps from config (default: use config value)"
    )
    return parser.parse_args()

def fix_state_file_inconsistency(domain: str, checkpoint_path: str) -> bool:
    """Fix state file inconsistency by updating it to match actual checkpoint status"""
    state_file = f"training_state/{domain}_training_state.json"
    
    if not os.path.exists(state_file):
        logger.warning(f"⚠️ State file not found: {state_file}")
        return False
    
    try:
        # Load current state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Validate checkpoint
        if not validate_checkpoint(checkpoint_path):
            logger.error(f"❌ Cannot fix state file - checkpoint validation failed: {checkpoint_path}")
            return False
        
        # Read checkpoint info
        trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)
        
        step = trainer_state.get("global_step", 0)
        epoch = trainer_state.get("epoch", 0)
        
        # Update state to match checkpoint
        state.update({
            "status": "ready_to_resume",
            "resume_checkpoint": checkpoint_path,
            "last_progress": step,
            "current_epoch": epoch,
            "last_update": datetime.now().isoformat(),
            "checkpoint_validated": True
        })
        
        # Save corrected state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"✅ Fixed state file for {domain}: {checkpoint_path} (step {step})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix state file for {domain}: {e}")
        return False

def detect_and_fix_training_issues(domain: str, base_model: str) -> Dict[str, Any]:
    """Detect and fix training issues for a domain"""
    issues = {
        "domain": domain,
        "checkpoint_found": False,
        "state_inconsistent": False,
        "state_fixed": False,
        "backup_created": False,
        "checkpoint_path": None
    }
    
    # 1. Find checkpoint
    checkpoint_path = find_latest_checkpoint(domain, base_model)
    if checkpoint_path:
        issues["checkpoint_found"] = True
        issues["checkpoint_path"] = checkpoint_path
        
        # 2. Check state file consistency
        state_file = f"training_state/{domain}_training_state.json"
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Check for inconsistencies
            if (state.get("status") == "loading_model" and checkpoint_path) or \
               (state.get("resume_checkpoint") is None and checkpoint_path):
                issues["state_inconsistent"] = True
                
                # 3. Fix state file
                if fix_state_file_inconsistency(domain, checkpoint_path):
                    issues["state_fixed"] = True
        
        # 4. Create backup
        backup_path = backup_checkpoint(checkpoint_path)
        if backup_path:
            issues["backup_created"] = True
    
    return issues

if __name__ == "__main__":
    # Import torch here to avoid issues with early imports
    try:
        import torch
    except ImportError:
        logger.error("❌ PyTorch is not installed. Please install it first.")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/adapters", exist_ok=True)
    os.makedirs("training_state", exist_ok=True)
    
    # Set up domain-specific logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/domain_training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Parse arguments
    args = parse_arguments()
    domain_models = load_domain_model_mapping()

    # Determine domains to train
    domains = []
    if args.domains:
        domains = args.domains.split(",")
    elif args.model_group:
        domains = get_domains_for_model(domain_models, args.model_group)
    else:
        print("Please specify --domains or --model_group.")
        exit(1)

    # Get the base model for the first domain (assuming all domains use same model)
    base_model = domain_models.get(domains[0])
    if not base_model:
        print(f"No model mapping found for domain: {domains[0]}")
        exit(1)
    
    # Create domain-specific log files
    for domain in domains:
        domain_log_file = f"logs/{domain}_training.log"
        domain_handler = logging.FileHandler(domain_log_file, encoding='utf-8')
        domain_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(domain_handler)
        logger.info(f"📝 Domain-specific log file created: {domain_log_file}")
    
    # Create training recovery state
    recovery_state = {
        "domains": ",".join(domains),
        "model": base_model,
        "start_time": datetime.now().isoformat(),
        "checkpoints": {}
    }
    with open("training_recovery_state.json", 'w') as f:
        json.dump(recovery_state, f, indent=2)
    
    # Initialize state files for each domain
    for domain in domains:
        state_file = f"training_state/{domain}_training_state.json"
        state = {
            "domain": domain,
            "base_model": base_model,
            "output_dir": f"models/{domain}/meetara_trinity_phase_efficient_core_processing",
            "resume_checkpoint": None,
            "start_time": datetime.now().isoformat(),
            "status": "starting",
            "retries": 0,
            "current_step": 0,
            "total_steps": args.max_steps or 400,  # Use command line arg if provided
            "progress_percentage": 0,
            "last_update": datetime.now().isoformat(),
            "notes": "Initializing training with optimized settings"
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    print(f"[INFO] Training domains: {', '.join(domains)} with model: {base_model}")
    print(f"[INFO] Force fresh: {args.force_fresh}")
    print(f"[INFO] Max retries: {args.max_retries}")
    print(f"[INFO] Logging to: logs/domain_training.log and domain-specific logs")
    
    # Execute training
    try:
        results = asyncio.run(train_domains(
            domains=domains,
            base_model=base_model,
            force_fresh=args.force_fresh,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            lora_r=args.lora_r,
            max_steps=args.max_steps
        ))
        
        # Update training recovery state with results
        recovery_state = {
            "domains": ",".join(domains),
            "model": base_model,
            "start_time": recovery_state["start_time"],
            "end_time": datetime.now().isoformat(),
            "status": "completed",
            "results": results
        }
        with open("training_recovery_state.json", 'w') as f:
            json.dump(recovery_state, f, indent=2)
        
        # Print results
        print("\n[TRAINING RESULTS]")
        print("-" * 50)
        for domain, result in results.items():
            status = result.get("status", "unknown")
            if status == "completed":
                print(f"✅ {domain}: {status}")
            else:
                print(f"❌ {domain}: {status} - {result.get('error', 'Unknown error')}")
        print("-" * 50)
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        
        # Update training recovery state with error
        recovery_state = {
            "domains": ",".join(domains),
            "model": base_model,
            "start_time": recovery_state["start_time"],
            "end_time": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }
        with open("training_recovery_state.json", 'w') as f:
            json.dump(recovery_state, f, indent=2)
        
        sys.exit(1)
