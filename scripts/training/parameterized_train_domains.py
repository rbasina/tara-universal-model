#!/usr/bin/env python3
"""
TARA Universal Model - Parameterized Domain Training
Train domains using any specified model with customizable parameters
Enhanced with memory optimization and restart resilience
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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
        logging.FileHandler('logs/domain_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create state directory for tracking training progress
os.makedirs("training_state", exist_ok=True)

async def train_domain(domain: str, base_model: str, output_dir: Optional[str] = None, resume_from_checkpoint: Optional[str] = None) -> str:
    """Train a specific domain with the specified base model with memory optimization"""
    logger.info(f"🎯 Training {domain} with {base_model}")
    
    # For clean training, don't use resume_from_checkpoint
    if resume_from_checkpoint:
        logger.info(f"⚠️ Ignoring checkpoint to avoid optimizer issues - starting fresh training")
        resume_from_checkpoint = None
    
    # Save current state for recovery
    state_file = f"training_state/{domain}_training_state.json"
    state = {
        "domain": domain,
        "base_model": base_model,
        "output_dir": output_dir,
        "resume_checkpoint": resume_from_checkpoint,
        "start_time": datetime.now().isoformat(),
        "status": "starting"
    }
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    try:
        # Force garbage collection before starting
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get available memory
        mem_info = psutil.virtual_memory()
        available_memory_gb = mem_info.available / (1024 * 1024 * 1024)
        total_memory_gb = mem_info.total / (1024 * 1024 * 1024)
        logger.info(f"💾 Available memory: {available_memory_gb:.2f}GB / {total_memory_gb:.2f}GB ({mem_info.percent}% used)")
        
        # Initialize trainer
        config = get_config()
        trainer = EnhancedTARATrainer(
            config=config,
            domain=domain,
            base_model_name=base_model
        )
        
        # Update state
        state["status"] = "loading_model"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Load model and setup LoRA
        trainer.load_base_model()
        trainer.setup_lora()
        
        # Update state
        state["status"] = "training"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Train
        data_path = f"data/synthetic/{domain}_training_data.json"
        
        if not output_dir:
            # Generate output directory based on model name
            model_short_name = base_model.split('/')[-1].lower()
            output_dir = f"models/adapters/{domain}_{model_short_name}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Update state with output dir
        state["output_dir"] = output_dir
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        model_path = await trainer.train_with_validation(
            data_path=data_path,
            output_dir=output_dir,
            resume_from_checkpoint=None  # Always start fresh to avoid optimizer issues
        )
        
        # Update state to completed
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        state["model_path"] = model_path
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"✅ {domain} training completed: {model_path}")
        
        # Force cleanup after training
        del trainer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return model_path
        
    except Exception as e:
        logger.error(f"❌ {domain} training failed: {e}")
        
        # Update state to failed
        state["status"] = "failed"
        state["error"] = str(e)
        state["failed_at"] = datetime.now().isoformat()
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        raise

async def train_domains(domains: List[str], base_model: str, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Dict]:
    """Train multiple domains with the specified base model sequentially to avoid memory issues"""
    logger.info(f"🚀 Starting domain training with {base_model}")
    
    # Get system memory information
    mem_info = psutil.virtual_memory()
    available_memory_gb = mem_info.available / (1024 * 1024 * 1024)
    total_memory_gb = mem_info.total / (1024 * 1024 * 1024)
    
    logger.info(f"💾 System memory: {available_memory_gb:.2f}GB available / {total_memory_gb:.2f}GB total")
    logger.info(f"🔄 Training domains sequentially to optimize memory usage")
    
    results = {}
    
    # Save overall state
    overall_state_file = "training_state/overall_training_state.json"
    overall_state = {
        "domains": domains,
        "base_model": base_model,
        "resume_from_checkpoint": resume_from_checkpoint,
        "start_time": datetime.now().isoformat(),
        "status": "in_progress",
        "completed_domains": [],
        "pending_domains": domains.copy(),
        "failed_domains": []
    }
    with open(overall_state_file, 'w') as f:
        json.dump(overall_state, f, indent=2)
    
    for domain in domains:
        start_time = datetime.now()
        try:
            # Update overall state
            overall_state["current_domain"] = domain
            overall_state["current_domain_start_time"] = datetime.now().isoformat()
            overall_state["pending_domains"].remove(domain)
            with open(overall_state_file, 'w') as f:
                json.dump(overall_state, f, indent=2)
            
            # Train domain - always start fresh to avoid optimizer issues
            model_path = await train_domain(
                domain, 
                base_model, 
                resume_from_checkpoint=None  # Always start fresh
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Store results
            results[domain] = {
                "status": "completed",
                "model_path": model_path,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                "resumed_from": None
            }
            
            # Update overall state
            overall_state["completed_domains"].append(domain)
            overall_state["last_completed_domain"] = domain
            overall_state["last_completed_at"] = datetime.now().isoformat()
            with open(overall_state_file, 'w') as f:
                json.dump(overall_state, f, indent=2)
            
            # Force cleanup between domains
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Small delay to ensure resources are freed
            await asyncio.sleep(5)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results[domain] = {
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
        
        if result["status"] == "completed" and result.get("resumed_from"):
            logger.info(f"    Resumed from: {result['resumed_from']}")
        elif result["status"] == "failed":
            logger.info(f"    Error: {result.get('error', 'Unknown')}")
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train domains with specified model")
    parser.add_argument(
        "--domains", 
        type=str,
        help="Comma-separated list of domains to train (default: creative,education,leadership)"
    )
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to use for training (default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory for trained models"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Resume training from a specific checkpoint or 'auto' to find latest"
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Enable additional memory optimization techniques"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/adapters", exist_ok=True)
    os.makedirs("training_results", exist_ok=True)
    os.makedirs("training_state", exist_ok=True)
    
    # Import torch here to avoid issues with early imports
    import torch
    
    args = parse_arguments()
    
    # Parse domains
    domains = ["creative", "education", "leadership"]  # Default
    if args.domains:
        domains = args.domains.split(",")
    
    # Handle auto checkpoint detection - but we'll ignore it to avoid optimizer issues
    resume_checkpoint = None  # Always start fresh
    
    # Print training configuration
    logger.info(f"🚀 Starting domain training with configuration:")
    logger.info(f"   - Domains: {', '.join(domains)}")
    logger.info(f"   - Base model: {args.model}")
    logger.info(f"   - Fresh training: Yes (ignoring checkpoints to avoid optimizer issues)")
    logger.info(f"   - Memory optimization: {'Enabled' if args.memory_efficient else 'Standard'}")
    
    # Run training
    asyncio.run(train_domains(domains, args.model, resume_from_checkpoint=resume_checkpoint))
