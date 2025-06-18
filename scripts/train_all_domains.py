#!/usr/bin/env python3
"""
Master training script for all TARA Universal Model domains.
Trains all 5 domain models (Healthcare, Business, Education, Creative, Leadership) 
with comprehensive logging, cost tracking, and progress monitoring.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.utils.config import get_config

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"train_all_domains_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DomainTrainingManager:
    """Manages training for all domain models."""
    
    DOMAINS = ['healthcare', 'business', 'education', 'creative', 'leadership']
    
    def __init__(self):
        self.config = get_config()
        self.training_results = {}
        self.total_cost = 0.0
        self.start_time = None
    
    def estimate_total_cost(self, samples_per_domain: int = 5000) -> Dict:
        """Estimate total training cost for all domains."""
        
        # Training time estimates per domain (hours)
        time_per_domain = max(1.5, samples_per_domain // 2000)
        total_hours = time_per_domain * len(self.DOMAINS)
        
        # Cost estimates by provider
        provider_costs = {
            "RunPod RTX 3090": 0.44,
            "Vast.ai RTX 3090": 0.35,
            "Google Colab Pro+": 0.00,
            "Local GPU": 0.00
        }
        
        cost_estimates = {}
        for provider, cost_per_hour in provider_costs.items():
            cost_estimates[provider] = cost_per_hour * total_hours
        
        return {
            "domains": len(self.DOMAINS),
            "samples_per_domain": samples_per_domain,
            "total_samples": samples_per_domain * len(self.DOMAINS),
            "hours_per_domain": time_per_domain,
            "total_hours": total_hours,
            "cost_estimates": cost_estimates
        }
    
    def print_cost_estimate(self, samples_per_domain: int = 5000):
        """Print comprehensive cost estimation."""
        estimates = self.estimate_total_cost(samples_per_domain)
        
        print("\n" + "="*70)
        print("🚀 TARA UNIVERSAL MODEL - FULL TRAINING COST ESTIMATION")
        print("="*70)
        print(f"📊 Training Overview:")
        print(f"   • Domains: {estimates['domains']} (Healthcare, Business, Education, Creative, Leadership)")
        print(f"   • Samples per domain: {estimates['samples_per_domain']:,}")
        print(f"   • Total training samples: {estimates['total_samples']:,}")
        print(f"   • Training time per domain: {estimates['hours_per_domain']:.1f} hours")
        print(f"   • Total training time: {estimates['total_hours']:.1f} hours")
        
        print(f"\n💰 Cost Estimates by Provider:")
        for provider, cost in estimates['cost_estimates'].items():
            emoji = "🆓" if cost == 0 else "💸"
            print(f"   {emoji} {provider}: ${cost:.2f}")
        
        # Savings calculation
        max_cost = max(estimates['cost_estimates'].values())
        free_options = [p for p, c in estimates['cost_estimates'].items() if c == 0]
        
        if free_options and max_cost > 0:
            print(f"\n🎉 Savings with free options ({', '.join(free_options)}): ${max_cost:.2f}")
        
        print(f"\n📈 Budget Impact:")
        print(f"   • Original Budget: $3,000")
        print(f"   • Estimated Cost (RunPod): ${estimates['cost_estimates']['RunPod RTX 3090']:.2f}")
        savings_pct = (1 - estimates['cost_estimates']['RunPod RTX 3090'] / 3000) * 100
        print(f"   • Budget Savings: {savings_pct:.1f}% (${3000 - estimates['cost_estimates']['RunPod RTX 3090']:.2f})")
        
        print("="*70)
    
    def train_domain(self, domain: str, samples: int = 5000, 
                    base_model: str = None) -> Dict:
        """Train a single domain model."""
        
        logger.info(f"🎯 Starting training for {domain.upper()} domain")
        
        # Prepare command
        cmd = [
            sys.executable, "scripts/train_domain.py",
            "--domain", domain,
            "--generate-data",
            "--samples", str(samples),
            "--verbose"
        ]
        
        if base_model:
            cmd.extend(["--base-model", base_model])
        
        # Track timing
        domain_start = time.time()
        
        try:
            # Run training
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            domain_end = time.time()
            training_time = domain_end - domain_start
            
            logger.info(f"✅ {domain.upper()} training completed in {training_time/60:.1f} minutes")
            
            return {
                "domain": domain,
                "status": "success",
                "training_time": training_time,
                "samples": samples,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            domain_end = time.time()
            training_time = domain_end - domain_start
            
            logger.error(f"❌ {domain.upper()} training failed after {training_time/60:.1f} minutes")
            logger.error(f"Error: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            
            return {
                "domain": domain,
                "status": "failed",
                "training_time": training_time,
                "samples": samples,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def train_all_domains(self, samples_per_domain: int = 5000, 
                         base_model: str = None) -> Dict:
        """Train all domain models sequentially."""
        
        logger.info("🚀 Starting TARA Universal Model full training pipeline")
        self.start_time = time.time()
        
        # Print cost estimation
        self.print_cost_estimate(samples_per_domain)
        
        # Confirm with user
        print(f"\n⚡ Ready to train {len(self.DOMAINS)} domain models")
        print(f"This will take approximately {self.estimate_total_cost(samples_per_domain)['total_hours']:.1f} hours")
        
        # Train each domain
        for i, domain in enumerate(self.DOMAINS, 1):
            print(f"\n{'='*50}")
            print(f"🎯 TRAINING DOMAIN {i}/{len(self.DOMAINS)}: {domain.upper()}")
            print(f"{'='*50}")
            
            result = self.train_domain(domain, samples_per_domain, base_model)
            self.training_results[domain] = result
            
            # Progress update
            elapsed = time.time() - self.start_time
            progress = i / len(self.DOMAINS) * 100
            
            print(f"\n📊 Progress: {progress:.1f}% ({i}/{len(self.DOMAINS)} domains)")
            print(f"⏱️  Elapsed time: {elapsed/60:.1f} minutes")
            
            if i < len(self.DOMAINS):
                remaining_domains = len(self.DOMAINS) - i
                avg_time_per_domain = elapsed / i
                eta = remaining_domains * avg_time_per_domain
                print(f"🔮 ETA: ~{eta/60:.1f} minutes remaining")
        
        # Final summary
        self.print_training_summary()
        
        return self.training_results
    
    def print_training_summary(self):
        """Print comprehensive training summary."""
        
        total_time = time.time() - self.start_time
        successful = [d for d, r in self.training_results.items() if r['status'] == 'success']
        failed = [d for d, r in self.training_results.items() if r['status'] == 'failed']
        
        print(f"\n" + "="*70)
        print("🎉 TARA UNIVERSAL MODEL TRAINING COMPLETE!")
        print("="*70)
        
        print(f"⏱️  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        print(f"✅ Successful: {len(successful)}/{len(self.DOMAINS)} domains")
        
        if successful:
            print(f"   🎯 Success: {', '.join([d.title() for d in successful])}")
        
        if failed:
            print(f"❌ Failed: {len(failed)}/{len(self.DOMAINS)} domains")
            print(f"   💥 Failed: {', '.join([d.title() for d in failed])}")
        
        # Domain-specific results
        print(f"\n📊 Domain Training Results:")
        for domain, result in self.training_results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status_emoji} {domain.title()}: {result['training_time']/60:.1f}min ({result['samples']:,} samples)")
        
        # Model locations
        print(f"\n📁 Trained Models Location:")
        print(f"   • Models directory: ./models/")
        print(f"   • Healthcare: ./models/healthcare/")
        print(f"   • Business: ./models/business/")
        print(f"   • Education: ./models/education/")
        print(f"   • Creative: ./models/creative/")
        print(f"   • Leadership: ./models/leadership/")
        
        print(f"\n🔧 Next Steps:")
        print(f"   1. Test models: python scripts/serve_model.py")
        print(f"   2. Evaluate performance: python scripts/evaluate_models.py")
        print(f"   3. Deploy to production: python scripts/deploy.py")
        
        print("="*70)
        
        # Save summary to file
        summary_file = f"training_summary_{timestamp}.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "total_time_minutes": total_time/60,
                "total_time_hours": total_time/3600,
                "domains_trained": len(self.DOMAINS),
                "successful_domains": successful,
                "failed_domains": failed,
                "results": self.training_results
            }, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_file}")

def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train all TARA Universal Model domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all domains with default settings
  python scripts/train_all_domains.py
  
  # Train with more samples per domain
  python scripts/train_all_domains.py --samples 10000
  
  # Use specific base model
  python scripts/train_all_domains.py --base-model microsoft/phi-3-mini-4k-instruct
  
  # Just show cost estimate
  python scripts/train_all_domains.py --estimate-only
        """
    )
    
    parser.add_argument('--samples', type=int, default=5000,
                       help='Number of training samples per domain (default: 5000)')
    parser.add_argument('--base-model', type=str,
                       help='Base model to use for all domains')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only show cost estimation, do not train')
    
    args = parser.parse_args()
    
    try:
        manager = DomainTrainingManager()
        
        if args.estimate_only:
            manager.print_cost_estimate(args.samples)
        else:
            manager.train_all_domains(args.samples, args.base_model)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 