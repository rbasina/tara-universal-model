#!/usr/bin/env python3
"""
TARA Universal Model - Qwen2.5 Domain Training
Train creative, education, and leadership domains using Qwen2.5-3B-Instruct
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.enhanced_trainer import EnhancedTARATrainer
from tara_universal_model.utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qwen_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Train creative, education, and leadership domains with Qwen2.5"""
    logger.info("🚀 Starting Qwen2.5 domain training")
    
    domains = ["creative", "education", "leadership"]
    base_model = "Qwen/Qwen2.5-3B-Instruct"
    
    for domain in domains:
        logger.info(f"🎯 Training {domain} with Qwen2.5")
        
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
            output_dir = f"models/adapters/{domain}_qwen25"
            
            model_path = await trainer.train_with_validation(
                data_path=data_path,
                output_dir=output_dir
            )
            
            logger.info(f"✅ {domain} training completed: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ {domain} training failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 