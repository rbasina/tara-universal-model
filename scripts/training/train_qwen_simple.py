#!/usr/bin/env python3
"""
TARA Universal Model - Simple Qwen2.5 Domain Training
Train creative, education, and leadership domains using Qwen2.5-3B-Instruct (simplified version)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.trainer import TARATrainer
from tara_universal_model.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qwen_simple_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_qwen_domains():
    """Train creative, education, and leadership domains with Qwen2.5"""
    logger.info("🚀 Starting simple Qwen2.5 domain training")
    
    domains = ["creative", "education", "leadership"]
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    results = {}
    
    for domain in domains:
        logger.info(f"🎯 Training {domain} with Qwen2.5")
        
        try:
            start_time = datetime.now()
            
            # Initialize trainer
            config = get_config()
            trainer = TARATrainer(
                config=config,
                domain=domain,
                base_model_name=base_model
            )
            
            # Load model and setup LoRA
            logger.info(f"📥 Loading Qwen2.5 base model for {domain}")
            trainer.load_base_model()
            
            logger.info(f"🔧 Setting up LoRA for {domain}")
            trainer.setup_lora()
            
            # Train
            data_path = f"data/synthetic/{domain}_training_data.json"
            output_dir = f"models/adapters/{domain}_qwen25"
            
            logger.info(f"🏋️ Starting training for {domain}")
            model_path = trainer.train(
                data_path=data_path,
                output_dir=output_dir
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results[domain] = {
                "status": "completed",
                "model_path": model_path,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ {domain} training completed in {duration:.2f}s: {model_path}")
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results[domain] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            }
            logger.error(f"❌ {domain} training failed: {e}")
    
    # Save results
    results_file = f"training_results/qwen_simple_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("training_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = len([r for r in results.values() if r["status"] == "completed"])
    failed = len([r for r in results.values() if r["status"] == "failed"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 QWEN2.5 SIMPLE TRAINING SUMMARY")
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

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/adapters", exist_ok=True)
    
    train_qwen_domains() 