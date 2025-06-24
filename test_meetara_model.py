#!/usr/bin/env python3
"""
Test MeeTARA Universal Model Integration
Quick test to verify our 681MB unified model works
"""

import sys
import os
from pathlib import Path

def test_model_loading():
    """Test if our unified model loads correctly"""
    print("🧪 Testing MeeTARA Universal Model Integration...")
    print("=" * 60)
    
    # Check if model file exists
    model_path = Path("../../models/meetara-universal-v1.0.gguf")
    print(f"📁 Model path: {model_path}")
    print(f"📊 Model exists: {model_path.exists()}")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"📦 Model size: {size_mb:.1f}MB")
        
        try:
            print("\n🔄 Loading model...")
            from llama_cpp import Llama
            
            model = Llama(
                model_path=str(model_path),
                n_ctx=512,
                n_threads=2,
                verbose=False
            )
            
            print("✅ Model loaded successfully!")
            print(f"🧠 Vocabulary size: {model.n_vocab()} tokens")
            
            # Test a simple prompt
            print("\n🎯 Testing model response...")
            response = model("Hello, I am MeeTARA. How can I help you?", max_tokens=50, stop=["\n"])
            print(f"🤖 Response: {response['choices'][0]['text']}")
            
            print("\n🎉 SUCCESS: MeeTARA Universal Model is working perfectly!")
            print("✅ Ready for integration with Core Reactor!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ Model file not found!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🚀 MeeTARA Universal Model: INTEGRATION READY!")
    else:
        print("\n💥 Integration test failed - check setup") 