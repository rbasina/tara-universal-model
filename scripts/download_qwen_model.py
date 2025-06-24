#!/usr/bin/env python3
"""
Download Qwen2.5-3B-Instruct model for TARA training
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_qwen_model():
    """Download Qwen2.5-3B-Instruct model"""
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model save path
    model_path = models_dir / "Qwen_Qwen2.5-3B-Instruct"
    
    if model_path.exists():
        logger.info(f"✅ Model already exists at {model_path}")
        return str(model_path)
    
    logger.info(f"📥 Downloading {model_name}")
    
    try:
        # Download tokenizer
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # Download model
        logger.info("📥 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.save_pretrained(model_path)
        
        logger.info(f"✅ Model downloaded to {model_path}")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return None

if __name__ == "__main__":
    download_qwen_model() 