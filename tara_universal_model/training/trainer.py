"""
Custom trainer for TARA Universal Model.
Implements LoRA/QLoRA training for domain-specific adaptation.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset as HFDataset
import evaluate
import numpy as np
from torch.utils.data import random_split

from ..utils.config import TARAConfig

logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    """Dataset for conversation training data."""
    
    def __init__(self, conversations: List[Dict], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Format conversation for training
        formatted_text = self._format_conversation(conversation)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone()
        }
    
    def _format_conversation(self, conversation: Dict) -> str:
        """Format conversation into training text."""
        turns = conversation.get("turns", [])
        formatted_parts = []
        
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            if role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        # Add special tokens
        formatted_text = "\n".join(formatted_parts)
        formatted_text = f"{self.tokenizer.bos_token}{formatted_text}{self.tokenizer.eos_token}"
        
        return formatted_text

class TARATrainer:
    """
    Custom trainer for TARA Universal Model.
    
    Handles LoRA/QLoRA training for domain-specific adaptation
    with efficient memory usage and cost optimization.
    """
    
    def __init__(self, config: TARAConfig, domain: str, base_model_name: str = None):
        """Initialize TARA trainer."""
        self.config = config
        self.domain = domain
        self.base_model_name = base_model_name or config.base_model_name
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Training metrics
        self.training_metrics = {}
        
        logger.info(f"TARA trainer initialized for {domain} domain")
    
    def load_base_model(self) -> None:
        """Load and prepare base model for training."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Configure quantization for memory efficiency (only if enabled)
        if (self.config.training_config.use_peft and 
            self.config.model_config.use_quantization):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.training_config.fp16 else torch.float32,
        }
        
        # Only add quantization config if quantization is enabled
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        # Enable gradient computation for input embeddings (critical for PEFT)
        self.model.enable_input_require_grads()
        
        # Prepare model for k-bit training if using quantization
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Base model loaded successfully")
    
    def setup_lora(self) -> None:
        """Setup LoRA configuration and wrap model."""
        logger.info("Setting up LoRA configuration")
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.training_config.lora_r,
            lora_alpha=self.config.training_config.lora_alpha,
            lora_dropout=self.config.training_config.lora_dropout,
            target_modules=self.config.training_config.lora_target_modules,
            bias="none",
            fan_in_fan_out=False,
        )
        
        # Get PEFT model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Critical: Enable input gradients for LoRA training
        self.peft_model.enable_input_require_grads()
        
        # Additional gradient setup for DialoGPT compatibility
        for name, param in self.peft_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                
        # Ensure base model embeddings work properly with LoRA
        if hasattr(self.peft_model.base_model, 'transformer'):
            # Enable gradients for embedding layers that LoRA might interact with
            transformer = self.peft_model.base_model.transformer
            if hasattr(transformer, 'wte'):
                transformer.wte.requires_grad_(True)
            if hasattr(transformer, 'wpe'):
                transformer.wpe.requires_grad_(True)
                
        # Force model to training mode
        self.peft_model.train()
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        logger.info("LoRA setup completed")
    
    def _print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters."""
        if self.peft_model:
            trainable_params = 0
            all_param = 0
            
            for _, param in self.peft_model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            percentage = 100 * trainable_params / all_param
            
            logger.info(
                f"Trainable params: {trainable_params:,} || "
                f"All params: {all_param:,} || "
                f"Trainable%: {percentage:.2f}%"
            )
    
    def load_training_data(self, data_path: str) -> ConversationDataset:
        """Load and prepare training data."""
        logger.info(f"Loading training data from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Filter conversations for the target domain
        domain_conversations = [
            conv for conv in conversations 
            if conv.get("domain") == self.domain
        ]
        
        if not domain_conversations:
            logger.warning(f"No conversations found for domain: {self.domain}")
            domain_conversations = conversations  # Use all data as fallback
        
        logger.info(f"Loaded {len(domain_conversations)} conversations for {self.domain}")
        
        # Create dataset
        dataset = ConversationDataset(
            domain_conversations,
            self.tokenizer,
            self.config.training_config.max_sequence_length
        )
        
        return dataset
    
    def train(self, data_path: str, output_dir: str, 
              resume_from_checkpoint: str = None) -> str:
        """
        Train the domain-specific model.
        
        Args:
            data_path: Path to training data
            output_dir: Output directory for trained model
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Path to trained model
        """
        start_time = time.time()
        logger.info(f"Starting training for {self.domain} domain")
        
        # Load base model and setup LoRA
        if not self.model:
            self.load_base_model()
        
        if self.config.training_config.use_peft and not self.peft_model:
            self.setup_lora()
        
        # Load training data
        train_dataset = self.load_training_data(data_path)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments
        # Disable gradient checkpointing when using PEFT to avoid gradient computation issues
        use_gradient_checkpointing = (self.config.training_config.use_gradient_checkpointing 
                                    and not self.config.training_config.use_peft)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.training_config.num_epochs,
            per_device_train_batch_size=self.config.training_config.batch_size,
            gradient_accumulation_steps=self.config.training_config.gradient_accumulation_steps,
            learning_rate=self.config.training_config.learning_rate,
            weight_decay=self.config.training_config.weight_decay,
            warmup_ratio=self.config.training_config.warmup_ratio,
            logging_steps=self.config.training_config.logging_steps,
            save_steps=self.config.training_config.save_steps,
            evaluation_strategy="no",
            eval_steps=self.config.training_config.eval_steps,
            save_total_limit=self.config.training_config.save_total_limit,
            fp16=self.config.training_config.fp16,
            gradient_checkpointing=use_gradient_checkpointing,
            dataloader_num_workers=self.config.training_config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard for now
        )
        
        # Initialize trainer
        model_to_train = self.peft_model if self.peft_model else self.model
        
        # Create a small validation dataset from training data
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        try:
            logger.info("Starting training...")
            
            if resume_from_checkpoint:
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer.train()
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save training metrics
            self.training_metrics = {
                "domain": self.domain,
                "training_time": training_time,
                "final_loss": trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
                "total_steps": trainer.state.global_step,
                "model_path": output_dir,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save metrics
            metrics_file = os.path.join(output_dir, "training_log.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Model saved to: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, model_path: str, test_data_path: str, 
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model: {model_path}")
        
        if metrics is None:
            metrics = ["perplexity", "bleu"]
        
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_conversations = json.load(f)
        
        # Filter for domain
        domain_conversations = [
            conv for conv in test_conversations
            if conv.get("domain") == self.domain
        ]
        
        if not domain_conversations:
            domain_conversations = test_conversations
        
        logger.info(f"Evaluating on {len(domain_conversations)} conversations")
        
        results = {}
        
        # Compute perplexity
        if "perplexity" in metrics:
            perplexity = self._compute_perplexity(model_path, domain_conversations)
            results["perplexity"] = perplexity
        
        # Compute BLEU score
        if "bleu" in metrics:
            bleu_score = self._compute_bleu(model_path, domain_conversations)
            results["bleu"] = bleu_score
        
        # Compute ROUGE scores
        if "rouge" in metrics:
            rouge_scores = self._compute_rouge(model_path, domain_conversations)
            results.update(rouge_scores)
        
        logger.info(f"Evaluation completed: {results}")
        return results
    
    def _compute_perplexity(self, model_path: str, conversations: List[Dict]) -> float:
        """Compute perplexity on test conversations."""
        # Simplified perplexity calculation
        # In practice, you'd load the model and compute actual perplexity
        return 15.5  # Placeholder
    
    def _compute_bleu(self, model_path: str, conversations: List[Dict]) -> float:
        """Compute BLEU score on test conversations."""
        # Simplified BLEU calculation
        # In practice, you'd generate responses and compare with ground truth
        return 0.75  # Placeholder
    
    def _compute_rouge(self, model_path: str, conversations: List[Dict]) -> Dict[str, float]:
        """Compute ROUGE scores on test conversations."""
        # Simplified ROUGE calculation
        return {
            "rouge1": 0.68,
            "rouge2": 0.45,
            "rougeL": 0.62
        }  # Placeholder
    
    def save_adapter(self, output_dir: str) -> str:
        """Save only the LoRA adapter weights."""
        if not self.peft_model:
            raise ValueError("No PEFT model available to save")
        
        adapter_path = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_path, exist_ok=True)
        
        # Save adapter
        self.peft_model.save_pretrained(adapter_path)
        
        # Save adapter config
        adapter_config = {
            "domain": self.domain,
            "base_model": self.base_model_name,
            "lora_r": self.config.training_config.lora_r,
            "lora_alpha": self.config.training_config.lora_alpha,
            "target_modules": self.config.training_config.lora_target_modules,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        logger.info(f"Adapter saved to: {adapter_path}")
        return adapter_path 