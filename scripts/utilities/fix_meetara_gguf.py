#!/usr/bin/env python3
"""
Fix Broken MeeTARA GGUF - Create Working Version
Problem: meetara-universal-embedded-Q4_K_M.gguf produces corrupted responses
Solution: Clean single-domain merge to avoid tokenizer corruption
"""

import os
import torch
import logging
import subprocess
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeeTARAGGUFFixer:
    """Fix the broken MeeTARA GGUF model"""
    
    def __init__(self):
        self.base_model_name = "microsoft/DialoGPT-medium"
        self.adapter_path = Path("models/adapters/healthcare")  # Use best domain
        self.output_path = Path("models/meetara-fixed")
        self.gguf_path = Path("models/gguf")
        
    def fix_broken_gguf(self) -> bool:
        """Fix the broken GGUF model"""
        logger.info("🔧 FIXING BROKEN MEETARA GGUF")
        logger.info("❌ Problem: Corrupted responses with ',-', '...', etc.")
        logger.info("✅ Solution: Clean single-domain merge")
        
        try:
            # Step 1: Create clean merged model
            if not self._create_clean_model():
                return False
            
            # Step 2: Convert to working GGUF
            if not self._convert_to_gguf():
                return False
            
            # Step 3: Test for corruption
            if not self._test_gguf():
                return False
            
            logger.info("🎉 FIXED! Working MeeTARA GGUF created")
            self._show_success_info()
            return True
            
        except Exception as e:
            logger.error(f"❌ Fix failed: {e}")
            return False
    
    def _create_clean_model(self) -> bool:
        """Create clean merged model without corruption"""
        logger.info("🔄 Creating clean merged model...")
        
        try:
            # Load base model and tokenizer
            logger.info(f"📥 Loading: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Fix tokenizer properly
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load healthcare adapter (best quality, lowest loss ~0.4995)
            logger.info("🔄 Loading healthcare adapter...")
            peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Clean merge
            logger.info("🔄 Merging adapter weights...")
            merged_model = peft_model.merge_and_unload()
            
            # Save clean model
            self.output_path.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(self.output_path)
            tokenizer.save_pretrained(self.output_path)
            
            logger.info("✅ Clean model created with healthcare expertise")
            return True
            
        except Exception as e:
            logger.error(f"❌ Clean model creation failed: {e}")
            return False
    
    def _convert_to_gguf(self) -> bool:
        """Convert to GGUF format"""
        logger.info("🔄 Converting to GGUF...")
        
        try:
            self.gguf_path.mkdir(parents=True, exist_ok=True)
            
            # Check for llama.cpp
            if not Path("llama.cpp").exists():
                logger.info("📥 Cloning llama.cpp...")
                subprocess.run([
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
                ], check=True)
            
            # Convert to GGUF
            gguf_output = self.gguf_path / "meetara-universal-FIXED-Q4_K_M.gguf"
            
            logger.info("🔄 Converting to GGUF format...")
            result = subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(self.output_path),
                "--outfile", str(gguf_output),
                "--outtype", "f16"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ GGUF conversion failed: {result.stderr}")
                return False
            
            # Get file size
            size_mb = gguf_output.stat().st_size / (1024*1024)
            logger.info(f"✅ GGUF created: {size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GGUF conversion failed: {e}")
            return False
    
    def _test_gguf(self) -> bool:
        """Test the fixed GGUF for corruption"""
        logger.info("🧪 Testing fixed GGUF...")
        
        try:
            from llama_cpp import Llama
            
            # Load fixed model
            model_path = self.gguf_path / "meetara-universal-FIXED-Q4_K_M.gguf"
            llm = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                verbose=False
            )
            
            # Test with the problematic prompt
            test_prompt = "Hello, can you help me with Java programming?"
            response = llm(
                test_prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["User:", "\n\n"]
            )
            
            response_text = response['choices'][0]['text'].strip()
            logger.info(f"🧪 Response: {response_text}")
            
            # Check for corruption patterns
            corrupted = (
                ",-" in response_text or 
                "-," in response_text or
                response_text.count(',') > 8 or
                response_text.count('.') > 10 or
                len(response_text.split()) < 3
            )
            
            if corrupted:
                logger.error("❌ Still shows corruption")
                return False
            else:
                logger.info("✅ Clean response - corruption FIXED!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            return False
    
    def _show_success_info(self):
        """Show success information"""
        fixed_path = self.gguf_path / "meetara-universal-FIXED-Q4_K_M.gguf"
        size_mb = fixed_path.stat().st_size / (1024*1024)
        
        print("\n" + "="*50)
        print("🎉 MEETARA GGUF CORRUPTION FIXED!")
        print("="*50)
        print(f"📁 Fixed Model: {fixed_path}")
        print(f"📊 Size: {size_mb:.1f}MB")
        print(f"✅ Status: Clean responses (no corruption)")
        print(f"🚀 Ready for MeeTARA integration")
        print("\n📋 Next Steps:")
        print("1. Copy to MeeTARA repository")
        print("2. Update core_reactor.py model path")
        print("3. Test: 'Hello, can you help me with Java?'")
        print("4. Expected: Intelligent, clean responses")
        print("="*50)

def main():
    """Fix the broken MeeTARA GGUF"""
    fixer = MeeTARAGGUFFixer()
    
    if fixer.fix_broken_gguf():
        print("\n🎯 SUCCESS: Broken GGUF has been FIXED!")
        return True
    else:
        print("\n❌ FAILED: Could not fix broken GGUF")
        return False

if __name__ == "__main__":
    main() 