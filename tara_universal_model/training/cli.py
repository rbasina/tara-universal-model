"""
Command-line interface for TARA Universal Model training.
Provides easy-to-use CLI for domain-specific model training.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import torch
from datetime import datetime

from ..utils.config import get_config, TARAConfig
from ..utils.data_generator import DataGenerator
from .trainer import TARATrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingCLI:
    """Command-line interface for TARA model training."""
    
    def __init__(self):
        self.config = get_config()
        self.data_generator = DataGenerator(self.config.data_config)
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI."""
        parser = argparse.ArgumentParser(
            description="TARA Universal Model Training CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train healthcare domain with synthetic data
  python -m tara_universal_model.training.cli train --domain healthcare --generate-data
  
  # Train business domain with existing data
  python -m tara_universal_model.training.cli train --domain business --data-path data/business_conversations.json
  
  # Generate synthetic data only
  python -m tara_universal_model.training.cli generate-data --domain education --samples 1000
  
  # Evaluate trained model
  python -m tara_universal_model.training.cli evaluate --domain healthcare --model-path models/healthcare_adapter
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train domain-specific model')
        train_parser.add_argument('--domain', required=True, 
                                choices=['healthcare', 'business', 'education', 'creative', 'leadership'],
                                help='Domain to train')
        train_parser.add_argument('--base-model', default=None,
                                help='Base model name (default from config)')
        train_parser.add_argument('--data-path', default=None,
                                help='Path to training data (JSON format)')
        train_parser.add_argument('--generate-data', action='store_true',
                                help='Generate synthetic training data')
        train_parser.add_argument('--samples', type=int, default=5000,
                                help='Number of synthetic samples to generate')
        train_parser.add_argument('--output-dir', default=None,
                                help='Output directory for trained model')
        train_parser.add_argument('--epochs', type=int, default=None,
                                help='Number of training epochs')
        train_parser.add_argument('--batch-size', type=int, default=None,
                                help='Training batch size')
        train_parser.add_argument('--learning-rate', type=float, default=None,
                                help='Learning rate')
        train_parser.add_argument('--lora-r', type=int, default=None,
                                help='LoRA rank')
        train_parser.add_argument('--lora-alpha', type=int, default=None,
                                help='LoRA alpha')
        train_parser.add_argument('--resume', default=None,
                                help='Resume training from checkpoint')
        train_parser.add_argument('--dry-run', action='store_true',
                                help='Show training config without actually training')
        
        # Generate data command
        gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic training data')
        gen_parser.add_argument('--domain', required=True,
                              choices=['healthcare', 'business', 'education', 'creative', 'leadership'],
                              help='Domain for data generation')
        gen_parser.add_argument('--samples', type=int, default=5000,
                              help='Number of samples to generate')
        gen_parser.add_argument('--output-path', default=None,
                              help='Output file path')
        gen_parser.add_argument('--quality-threshold', type=float, default=0.8,
                              help='Quality threshold for generated data')
        gen_parser.add_argument('--templates', default=None,
                              help='Path to custom conversation templates')
        
        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
        eval_parser.add_argument('--domain', required=True,
                               choices=['healthcare', 'business', 'education', 'creative', 'leadership'],
                               help='Domain to evaluate')
        eval_parser.add_argument('--model-path', required=True,
                               help='Path to trained model/adapter')
        eval_parser.add_argument('--test-data', default=None,
                               help='Path to test data')
        eval_parser.add_argument('--metrics', nargs='+', 
                               default=['perplexity', 'bleu', 'rouge'],
                               help='Evaluation metrics')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List available models and data')
        list_parser.add_argument('--type', choices=['models', 'data', 'configs'],
                               default='models', help='What to list')
        
        # Cost estimation command
        cost_parser = subparsers.add_parser('estimate-cost', help='Estimate training costs')
        cost_parser.add_argument('--domain', required=True,
                               help='Domain for cost estimation')
        cost_parser.add_argument('--samples', type=int, default=5000,
                               help='Number of training samples')
        cost_parser.add_argument('--cloud-provider', default='runpod',
                               choices=['runpod', 'vast', 'colab', 'local'],
                               help='Cloud provider for cost estimation')
        
        return parser
    
    def train_domain(self, args) -> None:
        """Train domain-specific model."""
        logger.info(f"Starting training for {args.domain} domain")
        
        # Update config with CLI arguments
        if args.epochs:
            self.config.training_config.num_epochs = args.epochs
        if args.batch_size:
            self.config.training_config.batch_size = args.batch_size
        if args.learning_rate:
            self.config.training_config.learning_rate = args.learning_rate
        if args.lora_r:
            self.config.training_config.lora_r = args.lora_r
        if args.lora_alpha:
            self.config.training_config.lora_alpha = args.lora_alpha
        
        # Generate synthetic data if requested
        if args.generate_data:
            logger.info(f"Generating {args.samples} synthetic samples for {args.domain}")
            data_path = self.data_generator.generate_domain_data(
                domain=args.domain,
                num_samples=args.samples,
                quality_threshold=0.8
            )
            logger.info(f"Synthetic data generated: {data_path}")
        else:
            data_path = args.data_path
        
        if not data_path:
            logger.error("No training data specified. Use --data-path or --generate-data")
            return
        
        # Setup output directory
        output_dir = args.output_dir or f"models/adapters/{args.domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Show training configuration
        self._show_training_config(args.domain, data_path, output_dir)
        
        if args.dry_run:
            logger.info("Dry run completed. No actual training performed.")
            return
        
        # Initialize trainer
        trainer = TARATrainer(
            config=self.config,
            domain=args.domain,
            base_model_name=args.base_model or self.config.base_model_name
        )
        
        try:
            # Train the model
            trainer.train(
                data_path=data_path,
                output_dir=output_dir,
                resume_from_checkpoint=args.resume
            )
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            
            # Show post-training summary
            self._show_training_summary(output_dir)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def generate_data(self, args) -> None:
        """Generate synthetic training data."""
        logger.info(f"Generating synthetic data for {args.domain} domain")
        
        output_path = args.output_path or f"data/synthetic/{args.domain}_conversations.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load custom templates if provided
        templates = None
        if args.templates:
            with open(args.templates, 'r', encoding='utf-8') as f:
                templates = json.load(f)
        
        # Generate data
        data_path = self.data_generator.generate_domain_data(
            domain=args.domain,
            num_samples=args.samples,
            output_path=output_path,
            quality_threshold=args.quality_threshold,
            templates=templates
        )
        
        logger.info(f"Generated {args.samples} samples for {args.domain}")
        logger.info(f"Data saved to: {data_path}")
        
        # Show data statistics
        self._show_data_statistics(data_path)
    
    def evaluate_model(self, args) -> None:
        """Evaluate trained model."""
        logger.info(f"Evaluating {args.domain} model")
        
        if not os.path.exists(args.model_path):
            logger.error(f"Model path not found: {args.model_path}")
            return
        
        # Initialize trainer for evaluation
        trainer = TARATrainer(
            config=self.config,
            domain=args.domain,
            base_model_name=self.config.base_model_name
        )
        
        # Load test data
        test_data_path = args.test_data or f"data/processed/{args.domain}_test.json"
        if not os.path.exists(test_data_path):
            logger.warning(f"Test data not found at {test_data_path}")
            logger.info("Generating test data...")
            test_data_path = self.data_generator.generate_domain_data(
                domain=args.domain,
                num_samples=500,
                output_path=f"data/synthetic/{args.domain}_test.json",
                split_type="test"
            )
        
        # Evaluate model
        results = trainer.evaluate(
            model_path=args.model_path,
            test_data_path=test_data_path,
            metrics=args.metrics
        )
        
        # Show evaluation results
        self._show_evaluation_results(results, args.domain)
    
    def list_items(self, args) -> None:
        """List available models, data, or configs."""
        if args.type == 'models':
            self._list_models()
        elif args.type == 'data':
            self._list_data()
        elif args.type == 'configs':
            self._list_configs()
    
    def estimate_cost(self, args) -> None:
        """Estimate training costs."""
        logger.info(f"Estimating training costs for {args.domain} domain")
        
        # Cost estimation based on domain and samples
        cost_estimates = {
            'runpod': {
                'gpu_hour_cost': 0.44,  # RTX 3090
                'estimated_hours': max(1, args.samples // 2000),  # Rough estimate
            },
            'vast': {
                'gpu_hour_cost': 0.35,
                'estimated_hours': max(1, args.samples // 2000),
            },
            'colab': {
                'gpu_hour_cost': 0.00,  # Free tier
                'estimated_hours': max(2, args.samples // 1000),  # Slower
            },
            'local': {
                'gpu_hour_cost': 0.00,
                'estimated_hours': max(3, args.samples // 800),  # CPU fallback
            }
        }
        
        provider_cost = cost_estimates[args.cloud_provider]
        total_cost = provider_cost['gpu_hour_cost'] * provider_cost['estimated_hours']
        
        print(f"\n📊 Training Cost Estimation for {args.domain.title()} Domain")
        print(f"{'='*60}")
        print(f"Training Samples: {args.samples:,}")
        print(f"Cloud Provider: {args.cloud_provider.title()}")
        print(f"Estimated Training Time: {provider_cost['estimated_hours']} hours")
        print(f"Cost per GPU Hour: ${provider_cost['gpu_hour_cost']:.2f}")
        print(f"Total Estimated Cost: ${total_cost:.2f}")
        print(f"{'='*60}")
        
        if args.cloud_provider == 'local':
            print("💡 Note: Local training is free but may be slower")
        elif args.cloud_provider == 'colab':
            print("💡 Note: Colab Pro+ recommended for stable training")
        
        print(f"\n🎯 Target Budget: $750-$3,000 total for all domains")
        print(f"Estimated cost for all 5 domains: ${total_cost * 5:.2f}")
    
    def _show_training_config(self, domain: str, data_path: str, output_dir: str) -> None:
        """Display training configuration."""
        print(f"\n🚀 Training Configuration for {domain.title()} Domain")
        print(f"{'='*60}")
        print(f"Base Model: {self.config.base_model_name}")
        print(f"Training Data: {data_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Epochs: {self.config.training_config.num_epochs}")
        print(f"Batch Size: {self.config.training_config.batch_size}")
        print(f"Learning Rate: {self.config.training_config.learning_rate}")
        print(f"LoRA Rank: {self.config.training_config.lora_r}")
        print(f"LoRA Alpha: {self.config.training_config.lora_alpha}")
        print(f"Use Quantization: {self.config.use_quantization}")
        print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*60}\n")
    
    def _show_training_summary(self, output_dir: str) -> None:
        """Display training summary."""
        print(f"\n✅ Training Summary")
        print(f"{'='*40}")
        print(f"Model saved to: {output_dir}")
        
        # Check if training logs exist
        log_file = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
            print(f"Final Loss: {logs.get('final_loss', 'N/A')}")
            print(f"Training Time: {logs.get('training_time', 'N/A')}")
        
        print(f"{'='*40}")
    
    def _show_data_statistics(self, data_path: str) -> None:
        """Display data statistics."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n📊 Data Statistics")
            print(f"{'='*30}")
            print(f"Total Samples: {len(data)}")
            print(f"Average Length: {sum(len(item.get('conversation', '')) for item in data) // len(data)} chars")
            print(f"Data Size: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")
            print(f"{'='*30}")
        except Exception as e:
            logger.warning(f"Could not show data statistics: {e}")
    
    def _show_evaluation_results(self, results: Dict, domain: str) -> None:
        """Display evaluation results."""
        print(f"\n📈 Evaluation Results for {domain.title()} Domain")
        print(f"{'='*50}")
        for metric, value in results.items():
            print(f"{metric.title()}: {value:.4f}")
        print(f"{'='*50}")
    
    def _list_models(self) -> None:
        """List available models."""
        models_dir = "models"
        adapters_dir = "models/adapters"
        
        print(f"\n🤖 Available Models")
        print(f"{'='*40}")
        
        # List base models
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if os.path.isdir(os.path.join(models_dir, item)) and item != "adapters":
                    print(f"Base Model: {item}")
        
        # List domain adapters
        if os.path.exists(adapters_dir):
            print(f"\nDomain Adapters:")
            for item in os.listdir(adapters_dir):
                if os.path.isdir(os.path.join(adapters_dir, item)):
                    print(f"  - {item}")
        
        print(f"{'='*40}")
    
    def _list_data(self) -> None:
        """List available data."""
        data_dirs = ["data/raw", "data/processed", "data/synthetic"]
        
        print(f"\n💾 Available Data")
        print(f"{'='*40}")
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                print(f"\n{data_dir.split('/')[-1].title()} Data:")
                for file in os.listdir(data_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(data_dir, file)
                        size = os.path.getsize(file_path) / 1024 / 1024
                        print(f"  - {file} ({size:.2f} MB)")
        
        print(f"{'='*40}")
    
    def _list_configs(self) -> None:
        """List available configurations."""
        configs_dir = "configs"
        
        print(f"\n⚙️  Available Configurations")
        print(f"{'='*40}")
        
        if os.path.exists(configs_dir):
            for root, dirs, files in os.walk(configs_dir):
                for file in files:
                    if file.endswith('.yaml') or file.endswith('.yml'):
                        rel_path = os.path.relpath(os.path.join(root, file), configs_dir)
                        print(f"  - {rel_path}")
        
        print(f"{'='*40}")

def main():
    """Main CLI entry point."""
    cli = TrainingCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            cli.train_domain(args)
        elif args.command == 'generate-data':
            cli.generate_data(args)
        elif args.command == 'evaluate':
            cli.evaluate_model(args)
        elif args.command == 'list':
            cli.list_items(args)
        elif args.command == 'estimate-cost':
            cli.estimate_cost(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 