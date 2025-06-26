#!/usr/bin/env python3
"""
Enhanced Domain Training with Resource Management
Implements efficient training for domain-specific models with:
- Automatic resource scaling
- Robust checkpoint handling
- Cursor AI restart resilience
- Memory optimization
"""

import os
import sys
import json
import logging
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    Enhanced trainer with resource optimization and robust checkpoint handling
    """
    
    def __init__(
        self,
        domain: str,
        base_model: str,
        output_dir: str,
        training_data_path: str,
        config_path: str = "configs/config.yaml",
        batch_size: int = None,
        num_epochs: int = None,
        learning_rate: float = None,
        checkpoint_steps: int = None
    ):
        self.domain = domain
        self.base_model = base_model
        self.output_dir = output_dir
        self.training_data_path = training_data_path
        self.config_path = config_path
        
        # Load config but allow parameter override
        self.config = self._load_config()
        
        # Override config with explicit parameters if provided
        self.batch_size = batch_size or self.config.get("batch_size", 2)
        self.num_epochs = num_epochs or self.config.get("num_epochs", 3)
        self.learning_rate = learning_rate or self.config.get("learning_rate", 2e-4)
        self.checkpoint_steps = checkpoint_steps or self.config.get("checkpoint_steps", 100)
        
        # Initialize state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self.training_state = {
            "domain": domain,
            "base_model": base_model,
            "output_dir": output_dir,
            "last_checkpoint": None,
            "last_step": 0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("training_state", exist_ok=True)
        
        # Save initial state
        self._save_state()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            logger.warning("Using default configuration")
            return {}
    
    def _save_state(self):
        """Save current training state for recovery"""
        state_file = f"training_state/{self.domain}_training_state.json"
        self.training_state["last_update"] = datetime.now().isoformat()
        
        with open(state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2)
        
        logger.debug(f"Training state saved to {state_file}")
    
    def _optimize_batch_size(self) -> int:
        """Dynamically optimize batch size based on available memory"""
        try:
            # Get available memory in GB
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            
            # Calculate optimal batch size based on available memory
            # This is a simple heuristic - adjust based on your models
            if available_memory > 12:  # More than 12GB available
                optimal_batch = 4
            elif available_memory > 8:  # More than 8GB available
                optimal_batch = 2
            else:  # Less memory available
                optimal_batch = 1
            
            logger.info(f"Memory-optimized batch size: {optimal_batch} (Available memory: {available_memory:.2f}GB)")
            return optimal_batch
            
        except Exception as e:
            logger.warning(f"Failed to optimize batch size: {e}")
            logger.warning(f"Using default batch size: {self.batch_size}")
            return self.batch_size
    
    def _load_training_data(self) -> Dict:
        """Load and prepare training data"""
        try:
            with open(self.training_data_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} training samples from {self.training_data_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def _prepare_dataset(self, data: List[Dict]) -> Dict:
        """Prepare dataset from raw data"""
        # This is a placeholder - implement your actual dataset preparation
        # based on your specific data format and requirements
        
        # For demonstration purposes:
        train_size = int(0.9 * len(data))
        train_data = data[:train_size]
        eval_data = data[train_size:]
        
        logger.info(f"Prepared dataset with {len(train_data)} training samples and {len(eval_data)} evaluation samples")
        
        return {
            "train": train_data,
            "eval": eval_data
        }
    
    def load_model(self):
        """Load base model and tokenizer with memory optimization"""
        try:
            # Clear memory before loading model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info(f"Loading model: {self.base_model}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            # Configure model loading for memory efficiency
            model_kwargs = {
                "device_map": "auto" if torch.cuda.is_available() else None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            # Add low_cpu_mem_usage if available memory is limited
            if psutil.virtual_memory().available < 8 * 1024 * 1024 * 1024:  # Less than 8GB
                model_kwargs["low_cpu_mem_usage"] = True
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                **model_kwargs
            )
            
            logger.info(f"Model loaded successfully: {self.model.__class__.__name__}")
            
            # Update state
            self.training_state["model_loaded"] = True
            self._save_state()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.training_state["status"] = "failed"
            self.training_state["error"] = str(e)
            self._save_state()
            raise
    
    def setup_training(self):
        """Set up training configuration and prepare trainer"""
        try:
            # Optimize batch size based on available resources
            optimized_batch_size = self._optimize_batch_size()
            
            # Load and prepare dataset
            raw_data = self._load_training_data()
            self.dataset = self._prepare_dataset(raw_data)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=optimized_batch_size,
                per_device_eval_batch_size=optimized_batch_size,
                gradient_accumulation_steps=4 if optimized_batch_size < 2 else 2,  # Compensate for small batch size
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=10,
                save_steps=self.checkpoint_steps,
                eval_steps=self.checkpoint_steps,
                evaluation_strategy="steps",
                save_total_limit=3,  # Keep only the last 3 checkpoints
                load_best_model_at_end=False,  # Avoid conflicts with checkpoint saving
                report_to=None,  # Disable wandb/tensorboard to save memory
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
                gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
                dataloader_num_workers=0,  # Avoid multiprocessing issues with Cursor AI
                save_safetensors=True,  # Use safetensors format for checkpoints
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["eval"],
                tokenizer=self.tokenizer,
            )
            
            logger.info("Training setup complete")
            
            # Update state
            self.training_state["training_setup"] = True
            self._save_state()
            
        except Exception as e:
            logger.error(f"Failed to setup training: {e}")
            self.training_state["status"] = "failed"
            self.training_state["error"] = str(e)
            self._save_state()
            raise
    
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """Run training with robust checkpoint handling"""
        try:
            # Update state
            self.training_state["status"] = "training"
            self._save_state()
            
            # Determine checkpoint to resume from
            checkpoint_to_use = None
            
            if resume_from_checkpoint == True:
                # Find latest checkpoint
                checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                    checkpoint_to_use = os.path.join(self.output_dir, latest_checkpoint)
                    logger.info(f"Resuming from latest checkpoint: {checkpoint_to_use}")
                else:
                    logger.info("No checkpoints found, starting training from scratch")
            elif isinstance(resume_from_checkpoint, str):
                # Use specified checkpoint
                checkpoint_to_use = resume_from_checkpoint
                logger.info(f"Resuming from specified checkpoint: {checkpoint_to_use}")
            
            # Start training
            logger.info(f"Starting training for {self.domain} with {self.base_model}")
            start_time = time.time()
            
            # Run training
            training_result = self.trainer.train(resume_from_checkpoint=checkpoint_to_use)
            
            # Calculate training duration
            duration = time.time() - start_time
            
            # Update state
            self.training_state["status"] = "completed"
            self.training_state["duration"] = duration
            self.training_state["completed_at"] = datetime.now().isoformat()
            self._save_state()
            
            # Save final model
            self.trainer.save_model(os.path.join(self.output_dir, "final"))
            
            logger.info(f"Training completed in {duration:.2f} seconds")
            
            # Clean up to free memory
            self._cleanup()
            
            return {
                "status": "completed",
                "domain": self.domain,
                "base_model": self.base_model,
                "output_dir": self.output_dir,
                "duration": duration,
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.training_state["status"] = "failed"
            self.training_state["error"] = str(e)
            self._save_state()
            raise
    
    def _cleanup(self):
        """Clean up resources after training"""
        try:
            # Delete model and trainer to free memory
            del self.trainer
            del self.model
            del self.tokenizer
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("Cleanup completed, memory freed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

def train_domain(
    domain: str,
    base_model: str,
    data_path: str,
    output_dir: str = None,
    resume_checkpoint: Union[str, bool] = None,
    config_path: str = "configs/config.yaml"
) -> Dict:
    """
    Train a domain with enhanced resource management
    
    Args:
        domain: Domain name (e.g., "healthcare")
        base_model: Base model name or path
        data_path: Path to training data
        output_dir: Output directory (default: models/adapters/{domain})
        resume_checkpoint: Checkpoint to resume from, or True to find latest
        config_path: Path to configuration file
        
    Returns:
        Training result dictionary
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"models/adapters/{domain}"
    
    # Create trainer
    trainer = EnhancedTrainer(
        domain=domain,
        base_model=base_model,
        output_dir=output_dir,
        training_data_path=data_path,
        config_path=config_path
    )
    
    try:
        # Load model
        trainer.load_model()
        
        # Setup training
        trainer.setup_training()
        
        # Train
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        return result
    
    except Exception as e:
        logger.error(f"Domain training failed for {domain}: {e}")
        return {
            "status": "failed",
            "domain": domain,
            "error": str(e)
        }

async def train_domains_parallel(
    domains: List[str],
    base_model: str,
    resume_from_checkpoint: bool = False
) -> Dict[str, Dict]:
    """
    Train multiple domains in sequence with optimized resource usage
    
    Args:
        domains: List of domain names to train
        base_model: Base model name or path
        resume_from_checkpoint: Whether to resume from latest checkpoints
        
    Returns:
        Dictionary of training results per domain
    """
    import asyncio
    
    logger.info(f"Starting sequential training for domains: {', '.join(domains)}")
    results = {}
    
    for domain in domains:
        logger.info(f"Starting training for domain: {domain}")
        start_time = time.time()
        
        try:
            # Generate data path
            data_path = f"data/synthetic/{domain}_training_data.json"
            
            # Generate output directory
            model_short_name = base_model.split('/')[-1].lower()
            output_dir = f"models/adapters/{domain}_{model_short_name}"
            
            # Check for domain-specific checkpoint
            domain_checkpoint = None
            if resume_from_checkpoint:
                if os.path.exists(output_dir):
                    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                        domain_checkpoint = os.path.join(output_dir, latest_checkpoint)
                        logger.info(f"Found checkpoint for {domain}: {domain_checkpoint}")
            
            # Train domain
            result = train_domain(
                domain=domain,
                base_model=base_model,
                data_path=data_path,
                output_dir=output_dir,
                resume_checkpoint=domain_checkpoint
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Store result
            results[domain] = {
                "status": result.get("status", "unknown"),
                "duration": duration,
                "output_dir": output_dir,
                "completed_at": datetime.now().isoformat()
            }
            
            # Force cleanup between domains
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Small delay to ensure resources are freed
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Failed to train domain {domain}: {e}")
            results[domain] = {
                "status": "failed",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    # Save overall results
    results_file = f"training_results/enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("training_results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            "domains": results,
            "base_model": base_model,
            "completed_at": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Training results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    import asyncio
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced domain training with resource optimization")
    parser.add_argument("--domains", type=str, help="Comma-separated list of domains to train")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model name or path")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Parse domains
    domains = ["creative", "education", "leadership"]
    if args.domains:
        domains = args.domains.split(",")
    
    # Run training
    asyncio.run(train_domains_parallel(
        domains=domains,
        base_model=args.model,
        resume_from_checkpoint=args.resume
    )) 