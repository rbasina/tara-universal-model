#!/usr/bin/env python3
"""
TARA Universal Model - Parameterized Domain Training
Train domains using any specified model with customizable parameters
"""

import os
import sys
import json
import asyncio
import argparse
import logging
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

async def train_domain(domain: str, base_model: str, output_dir: Optional[str] = None) -> str:
    """Train a specific domain with the specified base model"""
    logger.info(f"🎯 Training {domain} with {base_model}")
    
    try:
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
        
        # Train
        data_path = f"data/synthetic/{domain}_training_data.json"
        
        if not output_dir:
            # Generate output directory based on model name
            model_short_name = base_model.split('/')[-1].lower()
            output_dir = f"models/adapters/{domain}_{model_short_name}"
        
        model_path = await trainer.train_with_validation(
            data_path=data_path,
            output_dir=output_dir
        )
        
        logger.info(f"✅ {domain} training completed: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ {domain} training failed: {e}")
        raise

async def train_domains(domains: List[str], base_model: str) -> Dict[str, Dict]:
    """Train multiple domains with the specified base model"""
    logger.info(f"🚀 Starting domain training with {base_model}")
    
    results = {}
    
    for domain in domains:
        start_time = datetime.now()
        try:
            model_path = await train_domain(domain, base_model)
            
            duration = (datetime.now() - start_time).total_seconds()
            results[domain] = {
                "status": "completed",
                "model_path": model_path,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results[domain] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
    
    # Save results
    results_file = f"training_results/{base_model.split('/')[-1].lower()}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("training_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
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
        
        if result["status"] == "failed":
            logger.info(f"    Error: {result.get('error', 'Unknown')}")
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train domains with specified model")
    parser.add_argument(
        "--domains", 
        nargs="+", 
        default=["creative", "education", "leadership"],
        help="List of domains to train (default: creative education leadership)"
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
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/adapters", exist_ok=True)
    os.makedirs("training_results", exist_ok=True)
    
    args = parse_arguments()
    asyncio.run(train_domains(args.domains, args.model))
