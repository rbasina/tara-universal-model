#!/usr/bin/env python3
"""
TARA Universal Model - Fresh Training Script
Simplified training script to avoid optimizer mismatch issues
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fresh_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def train_domain(domain, model_name, output_dir=None, epochs=3, batch_size=2):
    """Train a domain from scratch to avoid optimizer issues"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    import json
    
    logger.info(f"Starting fresh training for {domain} with {model_name}")
    
    # Set output directory
    if output_dir is None:
        output_dir = f"fresh_training/models/{domain}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_path = f"data/synthetic/{domain}_training_data.json"
    logger.info(f"Loading data from {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Configure LoRA
    logger.info("Setting up LoRA")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "qwen" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA setup complete")
    
    # Prepare simple dataset
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    eval_data = data[train_size:]
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training")
    try:
        trainer.train()
        logger.info("Training completed successfully")
        
        # Save final model
        trainer.save_model(os.path.join(output_dir, "final"))
        logger.info(f"Model saved to {os.path.join(output_dir, 'final')}")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train domains from scratch")
    parser.add_argument("--domains", type=str, default="education,creative,leadership", 
                        help="Comma-separated list of domains to train")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model to use for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("fresh_training/models", exist_ok=True)
    
    # Train domains
    domains = args.domains.split(",")
    results = {}
    
    for domain in domains:
        logger.info(f"=== Training domain: {domain} ===")
        start_time = datetime.now()
        
        success = train_domain(
            domain=domain,
            model_name=args.model,
            output_dir=f"fresh_training/models/{domain}",
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        results[domain] = {
            "success": success,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Domain {domain} training {'succeeded' if success else 'failed'} in {duration:.2f}s")
    
    # Print summary
    logger.info("\n=== Training Summary ===")
    successful = sum(1 for r in results.values() if r["success"])
    logger.info(f"Successfully trained {successful}/{len(domains)} domains")
    
    for domain, result in results.items():
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {domain}: {result['duration']:.2f}s")
    
    return results

if __name__ == "__main__":
    main() 