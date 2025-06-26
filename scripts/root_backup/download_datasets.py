#!/usr/bin/env python3
"""
Download script for training datasets for TARA Universal Model.
Downloads domain-specific datasets for fine-tuning.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Available datasets for TARA Universal Model training
AVAILABLE_DATASETS = {
    # Healthcare Domain
    "healthcare": {
        "medical_qa": {
            "name": "medmcqa",
            "description": "Medical Multiple Choice QA dataset",
            "size": "~200MB",
            "samples": "~194k",
            "license": "Apache 2.0"
        },
        "medical_dialogue": {
            "name": "medical_dialog",
            "description": "Medical dialogue dataset",
            "size": "~50MB", 
            "samples": "~1.1M",
            "license": "MIT"
        }
    },
    
    # Business Domain
    "business": {
        "business_qa": {
            "name": "squad",
            "description": "Stanford Question Answering Dataset (general business applicable)",
            "size": "~30MB",
            "samples": "~100k",
            "license": "CC BY-SA 4.0"
        },
        "financial_news": {
            "name": "financial_phrasebank",
            "description": "Financial sentiment analysis dataset",
            "size": "~5MB",
            "samples": "~4.8k",
            "license": "CC BY-SA 3.0"
        }
    },
    
    # Education Domain
    "education": {
        "educational_qa": {
            "name": "sciq",
            "description": "Science Questions dataset for education",
            "size": "~10MB",
            "samples": "~13.7k",
            "license": "CC BY-NC 3.0"
        },
        "math_problems": {
            "name": "gsm8k",
            "description": "Grade School Math Word Problems",
            "size": "~5MB",
            "samples": "~8.5k",
            "license": "MIT"
        }
    },
    
    # Creative Domain
    "creative": {
        "creative_writing": {
            "name": "writingprompts",
            "description": "Creative writing prompts and stories",
            "size": "~500MB",
            "samples": "~300k",
            "license": "CC BY 4.0"
        },
        "story_generation": {
            "name": "rochester_nlp/story_generation",
            "description": "Story generation dataset",
            "size": "~100MB",
            "samples": "~50k",
            "license": "MIT"
        }
    },
    
    # Leadership Domain
    "leadership": {
        "leadership_qa": {
            "name": "squad",  # Using general QA as base
            "description": "General QA dataset (adaptable for leadership)",
            "size": "~30MB",
            "samples": "~100k",
            "license": "CC BY-SA 4.0"
        },
        "business_ethics": {
            "name": "ethics_commonsense_short",
            "description": "Ethics and moral reasoning dataset",
            "size": "~2MB",
            "samples": "~13k",
            "license": "CC BY 4.0"
        }
    },
    
    # General/Universal datasets
    "universal": {
        "general_chat": {
            "name": "daily_dialog",
            "description": "Daily conversation dataset",
            "size": "~20MB",
            "samples": "~13k",
            "license": "CC BY-NC-SA 4.0"
        },
        "instruction_following": {
            "name": "tatsu-lab/alpaca",
            "description": "Instruction-following dataset",
            "size": "~25MB",
            "samples": "~52k",
            "license": "CC BY-NC 4.0"
        }
    }
}

class DatasetDownloader:
    """Download and setup training datasets for TARA."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.synthetic_dir = self.data_dir / "synthetic"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.synthetic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def list_available_datasets(self) -> None:
        """List all available datasets by domain."""
        print("\n📊 Available Training Datasets for TARA Universal Model")
        print("=" * 70)
        
        for domain, datasets in AVAILABLE_DATASETS.items():
            print(f"\n🎯 {domain.upper()} DOMAIN")
            print("-" * 50)
            
            for dataset_key, info in datasets.items():
                print(f"\n📦 {dataset_key}")
                print(f"   Dataset: {info['name']}")
                print(f"   Description: {info['description']}")
                print(f"   Size: {info['size']}")
                print(f"   Samples: {info['samples']}")
                print(f"   License: {info['license']}")
        
        print("=" * 70)
    
    def download_dataset(self, domain: str, dataset_key: str) -> str:
        """Download a specific dataset."""
        if domain not in AVAILABLE_DATASETS:
            available_domains = ", ".join(AVAILABLE_DATASETS.keys())
            raise ValueError(f"Domain {domain} not available. Choose from: {available_domains}")
        
        if dataset_key not in AVAILABLE_DATASETS[domain]:
            available_datasets = ", ".join(AVAILABLE_DATASETS[domain].keys())
            raise ValueError(f"Dataset {dataset_key} not available for {domain}. Choose from: {available_datasets}")
        
        dataset_info = AVAILABLE_DATASETS[domain][dataset_key]
        dataset_name = dataset_info['name']
        
        # Create domain-specific directory
        domain_dir = self.raw_dir / domain
        domain_dir.mkdir(exist_ok=True)
        
        dataset_path = domain_dir / dataset_key
        
        print(f"\n📥 Downloading {dataset_key} for {domain} domain")
        print(f"Dataset: {dataset_name}")
        print(f"Size: {dataset_info['size']} | Samples: {dataset_info['samples']}")
        print("-" * 50)
        
        # Check if dataset already exists
        if dataset_path.exists() and len(list(dataset_path.iterdir())) > 0:
            print(f"✅ Dataset already exists at {dataset_path}")
            return str(dataset_path)
        
        try:
            print("Downloading dataset...")
            start_time = time.time()
            
            # Download using Hugging Face datasets
            dataset = load_dataset(dataset_name, cache_dir=str(dataset_path))
            
            download_time = time.time() - start_time
            print(f"✅ Download completed in {download_time:.1f} seconds")
            
            # Save dataset info
            self._save_dataset_info(dataset_path, domain, dataset_key, dataset_info, dataset)
            
            return str(dataset_path)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            # Clean up partial download
            if dataset_path.exists():
                import shutil
                shutil.rmtree(dataset_path)
            raise
    
    def _save_dataset_info(self, dataset_path: Path, domain: str, dataset_key: str, 
                          dataset_info: Dict, dataset) -> None:
        """Save dataset information and statistics."""
        info_file = dataset_path / "dataset_info.json"
        
        # Get dataset statistics
        stats = {}
        try:
            for split_name, split_data in dataset.items():
                stats[split_name] = len(split_data)
        except:
            stats = {"total": "unknown"}
        
        info_data = {
            "domain": domain,
            "dataset_key": dataset_key,
            "dataset_name": dataset_info['name'],
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "local_path": str(dataset_path),
            "statistics": stats,
            **dataset_info
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
    
    def download_domain_datasets(self, domain: str) -> List[str]:
        """Download all datasets for a specific domain."""
        if domain not in AVAILABLE_DATASETS:
            available_domains = ", ".join(AVAILABLE_DATASETS.keys())
            raise ValueError(f"Domain {domain} not available. Choose from: {available_domains}")
        
        print(f"\n🚀 Downloading all datasets for {domain} domain")
        print("=" * 50)
        
        downloaded_paths = []
        datasets = AVAILABLE_DATASETS[domain]
        
        for dataset_key in datasets.keys():
            try:
                path = self.download_dataset(domain, dataset_key)
                downloaded_paths.append(path)
                print(f"✅ {dataset_key} ready!")
            except Exception as e:
                print(f"❌ Failed to download {dataset_key}: {e}")
        
        print(f"\n🎉 Downloaded {len(downloaded_paths)}/{len(datasets)} datasets for {domain}")
        return downloaded_paths
    
    def download_all_datasets(self) -> Dict[str, List[str]]:
        """Download all available datasets."""
        print("\n🚀 Downloading ALL datasets for TARA Universal Model")
        print("=" * 60)
        
        all_downloads = {}
        
        for domain in AVAILABLE_DATASETS.keys():
            try:
                paths = self.download_domain_datasets(domain)
                all_downloads[domain] = paths
            except Exception as e:
                print(f"❌ Failed to download datasets for {domain}: {e}")
                all_downloads[domain] = []
        
        # Summary
        total_downloaded = sum(len(paths) for paths in all_downloads.values())
        total_available = sum(len(datasets) for datasets in AVAILABLE_DATASETS.values())
        
        print(f"\n📊 Download Summary:")
        print(f"Total downloaded: {total_downloaded}/{total_available} datasets")
        
        for domain, paths in all_downloads.items():
            domain_total = len(AVAILABLE_DATASETS[domain])
            print(f"   {domain}: {len(paths)}/{domain_total}")
        
        print("=" * 60)
        return all_downloads
    
    def list_downloaded_datasets(self) -> None:
        """List all downloaded datasets."""
        print("\n💾 Downloaded Datasets")
        print("=" * 50)
        
        if not self.raw_dir.exists():
            print("No datasets directory found.")
            return
        
        found_datasets = False
        for domain_dir in self.raw_dir.iterdir():
            if domain_dir.is_dir():
                print(f"\n🎯 {domain_dir.name.upper()} DOMAIN")
                print("-" * 30)
                
                for dataset_dir in domain_dir.iterdir():
                    if dataset_dir.is_dir():
                        info_file = dataset_dir / "dataset_info.json"
                        if info_file.exists():
                            with open(info_file) as f:
                                info = json.load(f)
                            
                            print(f"   📦 {info['dataset_key']}")
                            print(f"      Dataset: {info['dataset_name']}")
                            print(f"      Downloaded: {info['downloaded_at']}")
                            print(f"      Statistics: {info.get('statistics', 'N/A')}")
                            found_datasets = True
        
        if not found_datasets:
            print("No datasets downloaded yet.")
        
        print("=" * 50)
    
    def estimate_storage_requirements(self) -> None:
        """Estimate storage requirements for all datasets."""
        print("\n💾 Storage Requirements Estimation")
        print("=" * 60)
        
        total_size_mb = 0
        
        for domain, datasets in AVAILABLE_DATASETS.items():
            domain_size = 0
            print(f"\n🎯 {domain.upper()} DOMAIN")
            print("-" * 30)
            
            for dataset_key, info in datasets.items():
                # Extract size in MB (rough estimation)
                size_str = info['size'].replace('~', '').replace('MB', '').replace('GB', '')
                try:
                    if 'GB' in info['size']:
                        size_mb = float(size_str) * 1024
                    else:
                        size_mb = float(size_str)
                    domain_size += size_mb
                    print(f"   📦 {dataset_key}: {info['size']}")
                except:
                    print(f"   📦 {dataset_key}: {info['size']} (size unknown)")
            
            print(f"   Total: ~{domain_size:.0f}MB")
            total_size_mb += domain_size
        
        print(f"\n📊 TOTAL ESTIMATED STORAGE: ~{total_size_mb:.0f}MB ({total_size_mb/1024:.1f}GB)")
        print(f"💡 Recommended free space: {total_size_mb*1.5/1024:.1f}GB (with 50% buffer)")
        print("=" * 60)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download training datasets for TARA Universal Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python scripts/download_datasets.py --list
  
  # Download specific dataset
  python scripts/download_datasets.py --domain healthcare --dataset medical_qa
  
  # Download all datasets for a domain
  python scripts/download_datasets.py --domain business
  
  # Download all datasets
  python scripts/download_datasets.py --all
  
  # Check storage requirements
  python scripts/download_datasets.py --estimate-storage
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--domain', type=str,
                       help='Domain to download datasets for')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to download (requires --domain)')
    parser.add_argument('--all', action='store_true',
                       help='Download all available datasets')
    parser.add_argument('--data-dir', default='data',
                       help='Directory to store datasets')
    parser.add_argument('--list-downloaded', action='store_true',
                       help='List downloaded datasets')
    parser.add_argument('--estimate-storage', action='store_true',
                       help='Estimate storage requirements')
    
    args = parser.parse_args()
    
    if not any([args.list, args.domain, args.all, args.list_downloaded, args.estimate_storage]):
        parser.print_help()
        return
    
    downloader = DatasetDownloader(args.data_dir)
    
    try:
        if args.list:
            downloader.list_available_datasets()
        
        if args.list_downloaded:
            downloader.list_downloaded_datasets()
        
        if args.estimate_storage:
            downloader.estimate_storage_requirements()
        
        if args.all:
            downloader.download_all_datasets()
        
        if args.domain:
            if args.dataset:
                # Download specific dataset
                downloader.download_dataset(args.domain, args.dataset)
            else:
                # Download all datasets for domain
                downloader.download_domain_datasets(args.domain)
        
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 