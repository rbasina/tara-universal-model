#!/usr/bin/env python3
"""
Create Working MeeTARA Universal GGUF - FIX CORRUPTION ISSUE
"""
import os
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple approach: Use healthcare domain only (best trained)
base_model_name = "microsoft/DialoGPT-medium" 
adapter_path = Path("models/adapters/healthcare")
output_path = Path("models/meetara-clean")

def create_clean_gguf():
    logger.info(" Creating CLEAN MeeTARA GGUF (Single Domain)")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Fix tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load single adapter (healthcare - best quality)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    
    # Save clean model
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(" Clean model created - ready for GGUF conversion")
    return True

if __name__ == "__main__":
    create_clean_gguf()
