#!/usr/bin/env python3
"""
Simple Domain Training Test Script
Tests training for all 5 domains with proper models and fixed parameters.
"""

import os
import sys
import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.trainer import TARATrainer
from tara_universal_model.utils.config import get_config
from tara_universal_model.utils.data_generator import DataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/domain_training_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_domain_training():
    """Test domain training setup."""
    logger.info("🧪 Testing domain training setup")
    
    config = get_config("configs/config.yaml")
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    
    for domain in domains:
        logger.info(f"\n🎯 Testing {domain} domain")
        
        try:
            # Initialize trainer
            trainer = TARATrainer(
                config=config,
                domain=domain,
                base_model_name=config.base_model_name
            )
            
            # Test model loading
            trainer.load_base_model()
            logger.info(f"✅ {domain}: Base model loaded")
            
            # Test LoRA setup
            trainer.setup_lora()
            logger.info(f"✅ {domain}: LoRA setup complete")
            
        except Exception as e:
            logger.error(f"❌ {domain}: Failed - {e}")

if __name__ == "__main__":
    asyncio.run(test_domain_training()) 