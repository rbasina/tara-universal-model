#!/usr/bin/env python3
"""
Simple Offline Model Testing Script
Tests completed TARA models directly without requiring backend API
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def check_completed_models() -> Dict[str, Any]:
    """Check which models have actually completed training"""
    print("🔍 CHECKING COMPLETED MODELS")
    print("=" * 50)
    
    results = {
        "completed_models": [],
        "model_details": {},
        "training_data_available": {},
        "status": "checking"
    }
    
    domains = ["healthcare", "education", "business", "creative", "leadership"]
    
    for domain in domains:
        print(f"\n📂 Checking {domain.upper()} domain...")
        
        # Check for adapter files
        adapter_path = f"models/adapters/{domain}"
        adapter_exists = os.path.exists(adapter_path) and len(os.listdir(adapter_path)) > 0
        
        # Check for training data
        data_files = list(Path("data/synthetic").glob(f"{domain}_train_*.json"))
        data_available = len(data_files) > 0
        
        # Check model directory
        model_path = f"models/{domain}"
        model_exists = os.path.exists(model_path)
        
        if adapter_exists:
            adapter_files = os.listdir(adapter_path)
            print(f"  ✅ Adapter files found: {len(adapter_files)} files")
            results["completed_models"].append(domain)
            results["model_details"][domain] = {
                "adapter_path": adapter_path,
                "adapter_files": adapter_files,
                "model_path_exists": model_exists,
                "training_data_files": len(data_files)
            }
        else:
            print(f"  ❌ No adapter files found")
            
        results["training_data_available"][domain] = data_available
        if data_available:
            print(f"  📊 Training data: {len(data_files)} files")
        else:
            print(f"  📊 Training data: Not found")
    
    print(f"\n🎯 SUMMARY: {len(results['completed_models'])}/5 domains completed")
    for domain in results["completed_models"]:
        print(f"  ✅ {domain.upper()}")
    
    return results

def test_model_loading(domain: str, adapter_path: str) -> Dict[str, Any]:
    """Test if a model can be loaded successfully"""
    print(f"\n🧪 Testing {domain.upper()} model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # Load base model
        base_model_name = "microsoft/DialoGPT-medium"
        print(f"  📥 Loading base model: {base_model_name}")
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Load adapter
        print(f"  🔧 Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        load_time = time.time() - start_time
        
        # Test basic inference
        print(f"  💬 Testing basic inference...")
        test_input = f"Hello, I need help with {domain}"
        inputs = tokenizer.encode(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=inputs.shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(test_input):].strip()
        
        print(f"  ✅ Model loaded successfully in {load_time:.2f}s")
        print(f"  💭 Test response: {response[:100]}...")
        
        return {
            "success": True,
            "load_time": load_time,
            "test_response": response,
            "model_size": "345M parameters",
            "adapter_loaded": True
        }
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "load_time": None
        }

def run_offline_tests():
    """Run comprehensive offline testing"""
    print("🎯 TARA UNIVERSAL MODEL - OFFLINE TESTING")
    print("=" * 60)
    print("Testing completed models without backend dependency")
    print()
    
    # Check completed models
    model_status = check_completed_models()
    
    if not model_status["completed_models"]:
        print("\n❌ No completed models found!")
        print("   Models need to finish training before testing.")
        return
    
    # Test each completed model
    print(f"\n🧪 TESTING {len(model_status['completed_models'])} COMPLETED MODELS")
    print("=" * 60)
    
    test_results = {}
    
    for domain in model_status["completed_models"]:
        adapter_path = model_status["model_details"][domain]["adapter_path"]
        test_results[domain] = test_model_loading(domain, adapter_path)
    
    # Generate summary
    print("\n📊 TESTING SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for result in test_results.values() if result["success"])
    total_tests = len(test_results)
    
    print(f"🎯 Overall Success Rate: {successful_tests}/{total_tests} models working")
    print()
    
    for domain, result in test_results.items():
        if result["success"]:
            print(f"✅ {domain.upper():<12} | Load Time: {result['load_time']:.2f}s | Response Generated")
        else:
            print(f"❌ {domain.upper():<12} | Error: {result.get('error', 'Unknown error')}")
    
    # Save results
    with open("offline_test_results.json", "w") as f:
        json.dump({
            "model_status": model_status,
            "test_results": test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_models": total_tests,
                "successful_models": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: offline_test_results.json")
    
    return test_results

if __name__ == "__main__":
    # Import torch here to avoid loading if just checking files
    try:
        import torch
        run_offline_tests()
    except ImportError:
        print("❌ PyTorch not available. Running file-based checks only...")
        check_completed_models() 