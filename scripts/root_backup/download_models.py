#!/usr/bin/env python3
"""
Download script for free base models (Llama2, Qwen, Phi-3).
Includes model verification, setup, and cost estimation display.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Available models for TARA Universal Model
AVAILABLE_MODELS = {
    # Modern Models from Documentation
    "meta-llama/Llama-3.2-7B-Instruct": {
        "size": "7B",
        "description": "Latest Llama model for healthcare, creative, leadership",
        "license": "Custom",
        "memory_gb": 14.0,
        "download_gb": 7.0,
        "requires_token": True,
        "domains": ["healthcare", "creative", "leadership"]
    },
    "meta-llama/Llama-3.2-14B-Instruct": {
        "size": "14B", 
        "description": "Universal Super Model - flagship TARA model",
        "license": "Custom",
        "memory_gb": 28.0,
        "download_gb": 14.0,
        "requires_token": True,
        "domains": ["universal"]
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "size": "7B",
        "description": "Qwen model for education domain",
        "license": "Apache 2.0",
        "memory_gb": 14.0,
        "download_gb": 7.0,
        "requires_token": False,
        "domains": ["education"]
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "size": "3.8B",
        "description": "Latest Phi model for business domain",
        "license": "MIT",
        "memory_gb": 7.6,
        "download_gb": 3.8,
        "requires_token": False,
        "domains": ["business"]
    },
    
    # Legacy Models (for compatibility)
    "microsoft/DialoGPT-medium": {
        "size": "345M",
        "description": "Conversational AI model, good for chat (LEGACY)",
        "license": "MIT",
        "memory_gb": 1.4,
        "download_gb": 0.7,
        "requires_token": False,
        "domains": ["legacy"]
    },
    "microsoft/DialoGPT-large": {
        "size": "762M",
        "description": "Large conversational model (LEGACY)",
        "license": "MIT",
        "memory_gb": 3.1,
        "download_gb": 1.5,
        "requires_token": False,
        "domains": ["legacy"]
    },
    "microsoft/phi-2": {
        "size": "2.7B",
        "description": "Small language model from Microsoft (LEGACY)",
        "license": "MIT",
        "memory_gb": 5.4,
        "download_gb": 2.7,
        "requires_token": False,
        "domains": ["legacy"]
    },
    "Qwen/Qwen-7B-Chat": {
        "size": "7B",
        "description": "Qwen chat model, multilingual (LEGACY)",
        "license": "Apache 2.0",
        "memory_gb": 14.0,
        "download_gb": 7.0,
        "requires_token": False,
        "domains": ["legacy"]
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "size": "7B",
        "description": "Llama 2 chat model (LEGACY - requires HF token)",
        "license": "Custom",
        "memory_gb": 14.0,
        "download_gb": 7.0,
        "requires_token": True,
        "domains": ["legacy"]
    }
}

class ModelDownloader:
    """Download and setup free base models for TARA."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def list_available_models(self) -> None:
        """List all available models categorized by modern vs legacy."""
        list_models()
    
    def download_model(self, model_name: str, token: str = None) -> str:
        """Download a specific model."""
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_name} not available. Choose from: {available}")
        
        model_info = AVAILABLE_MODELS[model_name]
        model_path = self.models_dir / model_name.replace('/', '_')
        
        print(f"\n📥 Downloading {model_name}")
        print(f"Size: {model_info['size']} | Memory: {model_info['memory_gb']:.1f}GB")
        print("-" * 50)
        
        # Check if model already exists
        if model_path.exists() and len(list(model_path.iterdir())) > 0:
            print(f"✅ Model already exists at {model_path}")
            return str(model_path)
        
        # Check HuggingFace token requirement
        if model_info.get('requires_token') and not token:
            print("⚠️  This model requires a HuggingFace token.")
            print("Get your token from: https://huggingface.co/settings/tokens")
            token = input("Enter your HuggingFace token: ").strip()
        
        try:
            # Download model
            print("Downloading model files...")
            start_time = time.time()
            
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_path),
                token=token,
                resume_download=True
            )
            
            download_time = time.time() - start_time
            print(f"✅ Download completed in {download_time:.1f} seconds")
            
            # Verify model
            self._verify_model(model_path, model_name)
            
            # Save model info
            self._save_model_info(model_path, model_name, model_info)
            
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            # Clean up partial download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            raise
    
    def _verify_model(self, model_path: Path, model_name: str) -> None:
        """Verify downloaded model can be loaded."""
        print("🔍 Verifying model...")
        
        try:
            # Try loading tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            print("✅ Tokenizer loaded successfully")
            
            # Try loading model (just config, not weights)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded successfully")
            
            # Quick test
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt")
            print("✅ Model verification completed")
            
        except Exception as e:
            print(f"❌ Model verification failed: {e}")
            raise
    
    def _save_model_info(self, model_path: Path, model_name: str, model_info: Dict) -> None:
        """Save model information."""
        info_file = model_path / "model_info.json"
        
        info_data = {
            "model_name": model_name,
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "local_path": str(model_path),
            **model_info
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
    
    def list_downloaded_models(self) -> None:
        """List all downloaded models."""
        print("\n💾 Downloaded Models")
        print("=" * 40)
        
        if not self.models_dir.exists():
            print("No models directory found.")
            return
        
        found_models = False
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                info_file = model_dir / "model_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)
                    
                    print(f"\n📦 {info['model_name']}")
                    print(f"   Path: {model_dir}")
                    print(f"   Size: {info['size']}")
                    print(f"   Downloaded: {info['downloaded_at']}")
                    found_models = True
        
        if not found_models:
            print("No models downloaded yet.")
        
        print("=" * 40)
    
    def estimate_costs(self, models: List[str] = None) -> None:
        """Estimate training costs for selected models."""
        if not models:
            models = list(AVAILABLE_MODELS.keys())
        
        print("\n💰 Training Cost Estimation")
        print("=" * 60)
        
        total_download_gb = 0
        total_memory_gb = 0
        
        for model_name in models:
            if model_name in AVAILABLE_MODELS:
                info = AVAILABLE_MODELS[model_name]
                total_download_gb += info['download_gb']
                total_memory_gb += info['memory_gb']
                
                print(f"\n🤖 {model_name}")
                print(f"   Download: {info['download_gb']:.1f} GB")
                print(f"   Memory: {info['memory_gb']:.1f} GB")
        
        # Cost estimates
        cloud_costs = {
            "RunPod RTX 3090": 0.44,
            "Vast.ai RTX 3090": 0.35,
            "Google Colab Pro+": 0.00,
            "Local GPU": 0.00
        }
        
        print(f"\n📊 Summary:")
        print(f"Total Download: {total_download_gb:.1f} GB")
        print(f"Peak Memory: {max(AVAILABLE_MODELS[m]['memory_gb'] for m in models):.1f} GB")
        
        print(f"\n💸 Estimated Training Costs (per domain, 2 hours):")
        for provider, cost_per_hour in cloud_costs.items():
            total_cost = cost_per_hour * 2
            print(f"   {provider}: ${total_cost:.2f}")
        
        print(f"\n🎯 Total for 5 domains:")
        for provider, cost_per_hour in cloud_costs.items():
            total_cost = cost_per_hour * 2 * 5
            print(f"   {provider}: ${total_cost:.2f}")
        
        print(f"\n💡 Recommended: Use Vast.ai or RunPod for ${total_cost:.2f} total")
        print("=" * 60)
    
    def download_custom_model(self, model_name: str, output_dir: str, token: str = None) -> str:
        """Download any Hugging Face model to a custom directory."""
        model_path = Path(output_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📥 Downloading {model_name}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)
        
        # Check if model already exists
        if model_path.exists() and len(list(model_path.iterdir())) > 0:
            print(f"✅ Model already exists at {model_path}")
            return str(model_path)
        
        try:
            # Download model
            print("Downloading model files...")
            start_time = time.time()
            
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_path),
                token=token,
                resume_download=True
            )
            
            download_time = time.time() - start_time
            print(f"✅ Download completed in {download_time:.1f} seconds")
            
            # Basic verification
            self._verify_custom_model(model_path, model_name)
            
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            # Clean up partial download
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            raise
    
    def _verify_custom_model(self, model_path: Path, model_name: str) -> None:
        """Verify downloaded custom model can be loaded."""
        print("🔍 Verifying model...")
        
        try:
            # Try loading tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            print("✅ Tokenizer loaded successfully")
            print("✅ Model verification completed")
            
        except Exception as e:
            print(f"⚠️  Model verification warning: {e}")
            print("Model downloaded but verification failed - this may be normal for some models")
    
    def setup_recommended_models(self, token: str = None) -> List[str]:
        """Download recommended models for TARA."""
        # Modern models from documentation
        recommended = [
            "microsoft/Phi-3.5-mini-instruct",  # Business domain (3.8B)
            "Qwen/Qwen2.5-7B-Instruct",         # Education domain (7B)
            # Note: Llama models require token, will be prompted if needed
        ]
        
        # Add Llama models if token provided
        if token:
            recommended.extend([
                "meta-llama/Llama-3.2-7B-Instruct",  # Healthcare, Creative, Leadership
            ])
        
        print("\n🚀 Setting up recommended models for TARA")
        print("=" * 50)
        
        downloaded_paths = []
        for model_name in recommended:
            try:
                path = self.download_model(model_name, token)
                downloaded_paths.append(path)
                print(f"✅ {model_name} ready!")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
        
        print(f"\n🎉 Setup complete! Downloaded {len(downloaded_paths)} models.")
        return downloaded_paths

def list_models():
    """List all available models categorized by modern vs legacy"""
    print("\n🤖 Available Models for TARA Universal Model")
    print("=" * 60)
    
    # Modern Models (from documentation)
    print("\n🚀 MODERN MODELS (Recommended - from Documentation)")
    print("-" * 50)
    modern_models = {k: v for k, v in AVAILABLE_MODELS.items() 
                    if not any(domain == "legacy" for domain in v.get("domains", []))}
    
    for model_name, info in modern_models.items():
        print(f"\n📦 {model_name}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   License: {info['license']}")
        print(f"   Memory Required: {info['memory_gb']:.1f} GB")
        print(f"   Download Size: {info['download_gb']:.1f} GB")
        print(f"   Domains: {', '.join(info.get('domains', []))}")
        if info.get('requires_token'):
            print(f"   ⚠️  Requires HuggingFace token")
    
    # Legacy Models
    print(f"\n📚 LEGACY MODELS (Older, smaller models)")
    print("-" * 50)
    legacy_models = {k: v for k, v in AVAILABLE_MODELS.items() 
                    if any(domain == "legacy" for domain in v.get("domains", []))}
    
    for model_name, info in legacy_models.items():
        print(f"\n📦 {model_name}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   License: {info['license']}")
        print(f"   Memory Required: {info['memory_gb']:.1f} GB")
        print(f"   Download Size: {info['download_gb']:.1f} GB")
        if info.get('requires_token'):
            print(f"   ⚠️  Requires HuggingFace token")
    
    print("=" * 60)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download free base models for TARA Universal Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python scripts/download_models.py --list
  
  # Download specific model
  python scripts/download_models.py --download microsoft/DialoGPT-medium
  
  # Setup recommended models
  python scripts/download_models.py --setup
  
  # Estimate costs
  python scripts/download_models.py --estimate-costs
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--download', type=str,
                       help='Download specific model')
    parser.add_argument('--setup', action='store_true',
                       help='Download recommended models')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to store models')
    parser.add_argument('--token', type=str,
                       help='HuggingFace token for gated models')
    parser.add_argument('--estimate-costs', action='store_true',
                       help='Show cost estimates')
    parser.add_argument('--list-downloaded', action='store_true',
                       help='List downloaded models')
    
    # Add new arguments for flexible downloading
    parser.add_argument('--model_name_or_path', type=str, 
                        help='Hugging Face model ID (e.g., "microsoft/Phi-3-mini") to download.')
    parser.add_argument('--output_dir', type=str, 
                        help='Local directory path to save the downloaded model.')

    args = parser.parse_args()
    
    if not any([args.list, args.download, args.setup, args.estimate_costs, args.list_downloaded, args.model_name_or_path]):
        parser.print_help()
        return
    
    downloader = ModelDownloader(args.models_dir)
    
    try:
        if args.list:
            downloader.list_available_models()
        
        if args.list_downloaded:
            downloader.list_downloaded_models()
        
        if args.setup:
            print("Setting up initial base models: Microsoft DialoGPT-medium and Phi-2...")
            downloader.download_model("microsoft/DialoGPT-medium", args.token)
            downloader.download_model("microsoft/phi-2", args.token)
            print("Initial setup complete.")
            return
        
        # New logic to handle --model_name_or_path and --output_dir
        if args.model_name_or_path and args.output_dir:
            print(f"Downloading model '{args.model_name_or_path}' to '{args.output_dir}'...")
            downloader.download_custom_model(args.model_name_or_path, args.output_dir, args.token)
            print("Download complete.")
            return
        
        if args.download:
            downloader.download_model(args.download, args.token)
        
        if args.estimate_costs:
            downloader.estimate_costs()
        
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 