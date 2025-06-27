"""
Enhanced TARA Trainer with Production Validation
Ensures models work in backend during training - prevents training success ≠ production reliability gap.
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer
from .trainer import TARATrainer
from .production_validator import ProductionValidator
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class EnhancedTARATrainer(TARATrainer):
    """
    Enhanced TARA Trainer with integrated production validation.
    Tests models during training to ensure backend compatibility.
    """
    
    def __init__(self, config, domain: str, base_model_name: str = None):
        super().__init__(config, domain, base_model_name)
        self.validator = ProductionValidator()
        self.training_history = []
        self.validation_checkpoints = []
        
    async def train_with_validation(self, data_path: str, output_dir: str, 
                                  resume_from_checkpoint: str = None) -> str:
        """
        Enhanced training with production validation at checkpoints.
        
        Args:
            data_path: Path to training data
            output_dir: Directory to save the model
            resume_from_checkpoint: Path to checkpoint to resume from, or None
            
        Returns:
            Path to the trained model
        """
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info(f"🚀 Enhanced training starting for {self.domain}")
            
        logger.info(f"📊 Production validation enabled")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start backend validation server check
        await self._ensure_backend_running()
        
        # Load training data
        dataset = self.load_training_data(data_path)
        logger.info(f"📚 Loaded {len(dataset)} training samples")
        
        # Split data for validation during training
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Enhanced training arguments with validation checkpoints
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training_config.num_epochs,
            per_device_train_batch_size=self.config.training_config.batch_size,
            per_device_eval_batch_size=self.config.training_config.batch_size,
            gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
            learning_rate=self.config.training_config.learning_rate,
            weight_decay=self.config.training_config.weight_decay,
            warmup_ratio=self.config.training_config.warmup_ratio,
            logging_steps=self.config.training_config.logging_steps,
            save_steps=self.config.training_config.eval_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
            eval_steps=self.config.training_config.eval_steps,
            save_total_limit=self.config.training_config.save_total_limit,
            load_best_model_at_end=True,  # Re-enabled now that save_steps equals eval_steps
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.config.training_config.fp16,
            gradient_checkpointing=self.config.training_config.use_gradient_checkpointing,
            dataloader_num_workers=self.config.training_config.dataloader_num_workers
        )
        
        # Custom trainer with production validation callbacks
        trainer = ProductionValidatedTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            validator=self.validator,
            domain=self.domain,
            enhanced_trainer=self
        )
        
        # Start training with validation
        if resume_from_checkpoint:
            logger.info(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            logger.info("🎯 Starting enhanced training with production checkpoints...")
            
        start_time = time.time()
        
        # Pass the resume_from_checkpoint parameter to the train method
        training_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_duration = time.time() - start_time
        logger.info(f"✅ Training completed in {training_duration:.2f} seconds")
        
        # Final production validation
        logger.info("🔍 Final production validation...")
        final_validation = await self.validator.validate_model_production_ready(
            self.domain, output_dir
        )
        
        # Save enhanced training results
        enhanced_results = {
            "domain": self.domain,
            "training_duration": training_duration,
            "training_result": {
                "train_runtime": training_result.metrics.get("train_runtime", 0),
                "train_loss": training_result.metrics.get("train_loss", 0),
                "eval_loss": training_result.metrics.get("eval_loss", 0)
            },
            "final_validation": final_validation,
            "production_ready": final_validation.get("production_ready", False),
            "validation_history": self.validation_checkpoints,
            "timestamp": datetime.now().isoformat(),
            "resumed_from_checkpoint": resume_from_checkpoint is not None
        }
        
        # Save results
        results_path = os.path.join(output_dir, "enhanced_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        # Log final status
        if enhanced_results["production_ready"]:
            logger.info(f"🎉 {self.domain} model is PRODUCTION READY!")
            logger.info(f"📊 Validation Score: {final_validation.get('overall_score', 0):.2f}")
        else:
            logger.warning(f"⚠️ {self.domain} model needs improvement for production")
            logger.warning(f"📊 Validation Score: {final_validation.get('overall_score', 0):.2f}")
        
        return output_dir
    
    async def _ensure_backend_running(self):
        """Backend voice server no longer needed - TARA uses embedded GGUF in MeeTARA."""
        logger.info("✅ Backend integration: Using embedded GGUF in MeeTARA (no port 5000 needed)")
        return

class ProductionValidatedTrainer(Trainer):
    """
    Custom Trainer that validates models against backend during training.
    """
    
    def __init__(self, validator, domain, enhanced_trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validator = validator
        self.domain = domain
        self.enhanced_trainer = enhanced_trainer
        self.last_validation_step = 0
        self.last_state_update_step = 0
        
    def on_save(self, args, state, control, **kwargs):
        """Perform production validation when model is saved."""
        if state.global_step - self.last_validation_step >= 200:  # Validate every 200 steps
            logger.info(f"🔍 Production validation at step {state.global_step}")
            
            # Run async validation in sync context
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                validation_result = loop.run_until_complete(
                    self.validator.validate_model_production_ready(
                        self.domain, args.output_dir
                    )
                )
                
                # Store validation result
                checkpoint_result = {
                    "step": state.global_step,
                    "validation": validation_result,
                    "timestamp": datetime.now().isoformat()
                }
                self.enhanced_trainer.validation_checkpoints.append(checkpoint_result)
                
                # Log validation result
                production_ready = validation_result.get("production_ready", False)
                score = validation_result.get("overall_score", 0)
                if production_ready:
                    logger.info(f"✅ Step {state.global_step}: Model is production ready (Score: {score:.2f})")
                else:
                    logger.warning(f"⚠️ Step {state.global_step}: Model needs improvement (Score: {score:.2f})")
                
                self.last_validation_step = state.global_step
                
                # Update state file with validation result
                self._update_state_file(state, validation_result)
            except Exception as e:
                logger.error(f"❌ Production validation failed: {e}")
        
        # Always update state file on save
        self._update_state_file(state)
        
        return super().on_save(args, state, control, **kwargs)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update state file periodically during training."""
        # Update state file every 20 steps
        if state.global_step - self.last_state_update_step >= 20:
            self._update_state_file(state)
            self.last_state_update_step = state.global_step
        
        return super().on_step_end(args, state, control, **kwargs)
    
    def _update_state_file(self, state, validation_result=None):
        """Update the state file with current training progress."""
        try:
            # Get the state file path
            state_file = f"training_state/{self.domain}_training_state.json"
            if not os.path.exists(state_file):
                logger.warning(f"⚠️ State file not found: {state_file}")
                return
            
            # Read current state
            with open(state_file, 'r') as f:
                current_state = json.load(f)
            
            # Update with current progress
            current_state.update({
                "status": "training",
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "progress_percentage": round(state.global_step / state.max_steps * 100, 2) if state.max_steps > 0 else 0,
                "current_epoch": state.epoch,
                "last_update": datetime.now().isoformat(),
                "training_loss": state.log_history[-1].get("loss") if state.log_history else None,
                "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
            })
            
            # Add validation result if available
            if validation_result:
                current_state["last_validation"] = {
                    "timestamp": datetime.now().isoformat(),
                    "step": state.global_step,
                    "production_ready": validation_result.get("production_ready", False),
                    "score": validation_result.get("overall_score", 0)
                }
            
            # Save updated state
            with open(state_file, 'w') as f:
                json.dump(current_state, f, indent=2)
            
            # Also update the log file
            try:
                log_file = f"logs/{self.domain}_training.log"
                with open(log_file, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - Step {state.global_step}/{state.max_steps} "
                           f"({round(state.global_step / state.max_steps * 100, 2)}%) - "
                           f"Loss: {state.log_history[-1].get('loss') if state.log_history else 'N/A'}\n")
            except Exception as log_error:
                logger.warning(f"⚠️ Failed to update log file: {log_error}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to update state file: {e}")

class TrainingOrchestrator:
    """
    Orchestrates training across all domains with enhanced validation.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = get_config(config_path)
        self.training_progress = {}
        self.domains = ["healthcare", "business", "education", "creative", "leadership"]
        
    async def train_all_domains_enhanced(self, resume_from_checkpoint: bool = False):
        """
        Train all domains with enhanced validation.
        
        Args:
            resume_from_checkpoint: Whether to resume from latest checkpoints
        """
        logger.info("🚀 Starting enhanced training for all domains")
        logger.info("📊 Production validation enabled for all models")
        
        start_time = time.time()
        
        for domain in self.domains:
            logger.info(f"\n🎯 Training {domain} domain with production validation")
            
            domain_start = time.time()
            
            try:
                # Initialize enhanced trainer
                trainer = EnhancedTARATrainer(
                    config=self.config,
                    domain=domain,
                    base_model_name=self.config.base_model_name
                )
                
                # Load base model and setup LoRA
                trainer.load_base_model()
                trainer.setup_lora()
                
                # Check for checkpoint if resuming
                checkpoint_path = None
                if resume_from_checkpoint:
                    checkpoint_dir = f"models/{domain}/enhanced_training"
                    if os.path.exists(checkpoint_dir):
                        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if checkpoints:
                            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                            logger.info(f"Found checkpoint for {domain}: {checkpoint_path}")
                
                # Train with validation
                data_path = f"data/synthetic/{domain}_training_data.json"
                output_dir = f"models/{domain}/enhanced_training"
                
                model_path = await trainer.train_with_validation(
                    data_path=data_path,
                    output_dir=output_dir,
                    resume_from_checkpoint=checkpoint_path
                )
                
                domain_duration = time.time() - domain_start
                
                # Record progress
                self.training_progress[domain] = {
                    "status": "completed",
                    "duration": domain_duration,
                    "model_path": model_path,
                    "resumed_from": checkpoint_path
                }
                
                logger.info(f"✅ {domain} training completed in {domain_duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"❌ {domain} training failed: {e}")
                self.training_progress[domain] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        total_duration = time.time() - start_time
        logger.info(f"\n🎉 All domain training completed in {total_duration:.2f} seconds")
        
        # Save overall results
        results_path = "training_results/enhanced_training_summary.json"
        os.makedirs("training_results", exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "domains": self.training_progress,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return self.training_progress 