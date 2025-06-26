#!/usr/bin/env python3
"""
Domain-specific training script for TARA Universal Model.
Handles data preparation, tokenization, and LoRA adapter training.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.trainer import TARATrainer
from tara_universal_model.utils.config import get_config
from tara_universal_model.utils.data_generator import DataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(domain: str, data_path: str = None, 
                         generate_synthetic: bool = False, 
                         num_samples: int = 5000) -> str:
    """Prepare training data for the domain."""
    
    if generate_synthetic:
        logger.info(f"Generating synthetic data for {domain} domain")
        
        config = get_config()
        data_generator = DataGenerator(config.data_config)
        
        # Generate synthetic data
        output_path = data_generator.generate_domain_data(
            domain=domain,
            num_samples=num_samples,
            quality_threshold=0.8
        )
        
        logger.info(f"Generated synthetic data: {output_path}")
        return output_path
    
    elif data_path and os.path.exists(data_path):
        logger.info(f"Using existing data: {data_path}")
        return data_path
    
    else:
        # Look for existing data in data directory
        data_dir = Path("data")
        possible_paths = [
            data_dir / "synthetic" / f"{domain}_conversations.json",
            data_dir / "processed" / f"{domain}_train.json",
            data_dir / "raw" / f"{domain}.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found existing data: {path}")
                return str(path)
        
        # No data found, generate synthetic
        logger.info(f"No existing data found, generating synthetic data for {domain}")
        return prepare_training_data(domain, None, True, num_samples)

def estimate_training_cost(domain: str, num_samples: int, 
                          model_size: str = "medium") -> None:
    """Estimate training cost for the domain."""
    
    # Training time estimates (hours)
    time_estimates = {
        "small": max(1, num_samples // 3000),
        "medium": max(1.5, num_samples // 2000), 
        "large": max(2, num_samples // 1500)
    }
    
    # Cost per hour for different providers
    provider_costs = {
        "RunPod RTX 3090": 0.44,
        "Vast.ai RTX 3090": 0.35,
        "Google Colab Pro+": 0.00,
        "Local GPU": 0.00
    }
    
    estimated_hours = time_estimates.get(model_size, 2)
    
    print(f"\n>> Training Cost Estimation for {domain.title()}")
    print(f"{'='*50}")
    print(f"Domain: {domain}")
    print(f"Training Samples: {num_samples:,}")
    print(f"Model Size: {model_size}")
    print(f"Estimated Training Time: {estimated_hours} hours")
    print(f"\n>> Cost by Provider:")
    
    for provider, cost_per_hour in provider_costs.items():
        total_cost = cost_per_hour * estimated_hours
        print(f"  {provider}: ${total_cost:.2f}")
    
    print(f"{'='*50}")

def main():
    """Main entry point for domain training."""
    parser = argparse.ArgumentParser(
        description="Train domain-specific TARA Universal Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train healthcare domain with synthetic data
  python scripts/train_domain.py --domain healthcare --generate-data --samples 5000
  
  # Train business domain with existing data
  python scripts/train_domain.py --domain business --data-path data/business_conversations.json
  
  # Train with custom parameters
  python scripts/train_domain.py --domain education --generate-data --epochs 5 --batch-size 8
  
  # Estimate costs only
  python scripts/train_domain.py --domain healthcare --estimate-cost --samples 5000
        """
    )
    
    parser.add_argument('--domain', required=True,
                       choices=['healthcare', 'business', 'education', 'creative', 'leadership'],
                       help='Domain to train')
    parser.add_argument('--data-path', type=str,
                       help='Path to training data (JSON format)')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic training data')
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples (for synthetic data)')
    parser.add_argument('--base-model', type=str,
                       help='Base model to use for training')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file path')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate')
    parser.add_argument('--lora-r', type=int,
                       help='LoRA rank parameter')
    parser.add_argument('--lora-alpha', type=int,
                       help='LoRA alpha parameter')
    
    # Utility options
    parser.add_argument('--estimate-cost', action='store_true',
                       help='Estimate training cost and exit')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without training')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = get_config(args.config)
        
        # Update config with CLI arguments
        if args.epochs:
            config.training_config.num_epochs = args.epochs
        if args.batch_size:
            config.training_config.batch_size = args.batch_size
        if args.learning_rate:
            config.training_config.learning_rate = args.learning_rate
        if args.lora_r:
            config.training_config.lora_r = args.lora_r
        if args.lora_alpha:
            config.training_config.lora_alpha = args.lora_alpha
        
        # Cost estimation only
        if args.estimate_cost:
            estimate_training_cost(args.domain, args.samples)
            return
        
        # Prepare training data
        logger.info(f"Preparing training data for {args.domain} domain")
        data_path = prepare_training_data(
            domain=args.domain,
            data_path=args.data_path,
            generate_synthetic=args.generate_data,
            num_samples=args.samples
        )
        
        # Setup output directory
        output_dir = args.output_dir or f"models/adapters/{args.domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Show training configuration
        print(f"\n>> Training Configuration")
        print(f"{'='*40}")
        print(f"Domain: {args.domain}")
        print(f"Base Model: {args.base_model or config.base_model_name}")
        print(f"Training Data: {data_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Epochs: {config.training_config.num_epochs}")
        print(f"Batch Size: {config.training_config.batch_size}")
        print(f"Learning Rate: {config.training_config.learning_rate}")
        print(f"LoRA Rank: {config.training_config.lora_r}")
        print(f"LoRA Alpha: {config.training_config.lora_alpha}")
        print(f"{'='*40}\n")
        
        if args.dry_run:
            logger.info("Dry run completed. No actual training performed.")
            return
        
        # Initialize trainer
        logger.info(f"Initializing trainer for {args.domain} domain")
        trainer = TARATrainer(
            config=config,
            domain=args.domain,
            base_model_name=args.base_model or config.base_model_name
        )
        
        # Start training
        logger.info("Starting training...")
        trained_model_path = trainer.train(
            data_path=data_path,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume
        )
        
        # Save adapter separately for easy loading
        adapter_path = trainer.save_adapter(output_dir)
        
        # Training completed
        print(f"\n>> Training Completed Successfully!")
        print(f"{'='*40}")
        print(f"Domain: {args.domain}")
        print(f"Model saved to: {trained_model_path}")
        print(f"Adapter saved to: {adapter_path}")
        
        # Show next steps
        print(f"\n>> Next Steps:")
        print(f"1. Test the model:")
        print(f"   python -m tara_universal_model.training.cli evaluate --domain {args.domain} --model-path {trained_model_path}")
        print(f"2. Serve the model:")
        print(f"   python scripts/serve_model.py --preload-domains {args.domain}")
        print(f"3. Integrate with TARA AI Companion")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 