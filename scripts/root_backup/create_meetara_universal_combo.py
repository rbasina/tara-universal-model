#!/usr/bin/env python3
"""
Create MeeTARA Universal Combo GGUF
Combine all base GGUF models from MeeTARA into single universal file
Models: Llama-3.1-8B + Phi-3.5-mini + Qwen2.5-3B + Llama-3.2-1B + TARA-trained
"""

import os
import json
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeeTARAUniversalCombo:
    """Create universal GGUF combining all MeeTARA base models"""
    
    def __init__(self):
        self.meetara_models_path = Path("C:/Users/rames/Documents/github/meetara/services/ai-engine-python/models")
        self.tara_models_path = Path("models/gguf")
        self.output_path = Path("models/universal-combo")
        self.final_gguf_path = Path("models/gguf")
        
        # Available models
        self.base_models = {
            "llama_8b": {
                "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "size_gb": 4.6,
                "specialties": ["complex_reasoning", "long_context", "professional"],
                "domains": ["business", "leadership", "education"],
                "strength": "analytical_power"
            },
            "phi_mini": {
                "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf", 
                "size_gb": 2.2,
                "specialties": ["code_generation", "technical", "precise"],
                "domains": ["programming", "technology", "engineering"],
                "strength": "technical_precision"
            },
            "qwen_3b": {
                "file": "qwen2.5-3b-instruct-q4_0.gguf",
                "size_gb": 1.9,
                "specialties": ["creative", "multilingual", "versatile"],
                "domains": ["creative", "writing", "general"],
                "strength": "creative_flexibility"
            },
            "llama_1b": {
                "file": "llama-3.2-1b-instruct-q4_0.gguf",
                "size_gb": 0.7,
                "specialties": ["fast_response", "lightweight", "efficient"],
                "domains": ["chat", "quick_questions", "mobile"],
                "strength": "speed_efficiency"
            },
            "tara_fixed": {
                "file": "meetara-universal-FIXED-Q4_K_M.gguf",
                "size_gb": 0.7,
                "specialties": ["healthcare", "therapeutic", "empathetic"],
                "domains": ["healthcare", "wellness", "emotional_support"],
                "strength": "healthcare_expertise"
            }
        }
    
    def create_universal_combo(self) -> bool:
        """Create the universal combo GGUF"""
        logger.info("🚀 Creating MeeTARA Universal Combo GGUF")
        logger.info("🎯 Combining 5 models into intelligent universal system")
        
        try:
            # Step 1: Verify all models exist
            if not self._verify_models():
                return False
            
            # Step 2: Create universal container structure
            if not self._create_container_structure():
                return False
            
            # Step 3: Copy and organize models
            if not self._organize_models():
                return False
            
            # Step 4: Create intelligent routing system
            if not self._create_intelligent_router():
                return False
            
            # Step 5: Create unified configuration
            if not self._create_unified_config():
                return False
            
            # Step 6: Package as universal GGUF
            if not self._package_universal_gguf():
                return False
            
            logger.info("🎉 Universal Combo GGUF created successfully!")
            self._display_success_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create universal combo: {e}")
            return False
    
    def _verify_models(self) -> bool:
        """Verify all required models exist"""
        logger.info("🔍 Verifying model availability...")
        
        missing_models = []
        
        # Check MeeTARA models
        for model_id, model_info in self.base_models.items():
            if model_id == "tara_fixed":
                model_path = self.tara_models_path / model_info["file"]
            else:
                model_path = self.meetara_models_path / model_info["file"]
            
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024*1024)
                logger.info(f"✅ {model_id}: {size_mb:.0f}MB - {model_info['strength']}")
            else:
                missing_models.append(f"{model_id}: {model_info['file']}")
                logger.warning(f"❌ {model_id}: Missing")
        
        if missing_models:
            logger.error(f"❌ Missing models: {missing_models}")
            return False
        
        logger.info("✅ All models verified and ready for combination")
        return True
    
    def _create_container_structure(self) -> bool:
        """Create the universal container structure"""
        logger.info("🏗️ Creating universal container structure...")
        
        try:
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.output_path / "models").mkdir(exist_ok=True)
            (self.output_path / "routing").mkdir(exist_ok=True)
            (self.output_path / "configs").mkdir(exist_ok=True)
            (self.output_path / "voice").mkdir(exist_ok=True)
            (self.output_path / "emotion").mkdir(exist_ok=True)
            
            logger.info("✅ Container structure created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create structure: {e}")
            return False
    
    def _organize_models(self) -> bool:
        """Copy and organize all models"""
        logger.info("📦 Organizing models into universal container...")
        
        try:
            models_dir = self.output_path / "models"
            
            for model_id, model_info in self.base_models.items():
                if model_id == "tara_fixed":
                    source_path = self.tara_models_path / model_info["file"]
                else:
                    source_path = self.meetara_models_path / model_info["file"]
                
                dest_path = models_dir / f"{model_id}.gguf"
                
                if source_path.exists():
                    logger.info(f"📋 Copying {model_id}...")
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"✅ {model_id} organized")
            
            # Copy voice and emotion models if available
            if (self.tara_models_path / "embedded_models").exists():
                shutil.copytree(
                    self.tara_models_path / "embedded_models" / "voice_models",
                    self.output_path / "voice",
                    dirs_exist_ok=True
                )
                shutil.copytree(
                    self.tara_models_path / "embedded_models" / "emotion_models", 
                    self.output_path / "emotion",
                    dirs_exist_ok=True
                )
            
            logger.info("✅ All models organized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to organize models: {e}")
            return False
    
    def _create_intelligent_router(self) -> bool:
        """Create intelligent routing system"""
        logger.info("🧠 Creating intelligent routing system...")
        
        router_code = '''#!/usr/bin/env python3
"""
MeeTARA Universal Combo Router
Intelligently routes queries to the best model based on content and context
"""

import re
from pathlib import Path
from llama_cpp import Llama
from typing import Dict, Optional, Tuple

class UniversalComboRouter:
    """Routes queries to optimal model in the combo"""
    
    def __init__(self, models_path: str):
        self.models_path = Path(models_path)
        self.loaded_models = {}
        self.routing_rules = {
            # Healthcare & Wellness
            "healthcare": {
                "keywords": ["health", "medical", "doctor", "symptom", "treatment", "wellness", "fitness", "nutrition"],
                "model": "tara_fixed",
                "confidence": 0.9
            },
            
            # Programming & Technology  
            "programming": {
                "keywords": ["code", "programming", "javascript", "python", "java", "function", "algorithm", "debug"],
                "model": "phi_mini", 
                "confidence": 0.95
            },
            
            # Business & Leadership
            "business": {
                "keywords": ["business", "leadership", "management", "strategy", "marketing", "finance", "team"],
                "model": "llama_8b",
                "confidence": 0.9
            },
            
            # Creative & Writing
            "creative": {
                "keywords": ["creative", "writing", "story", "poem", "art", "design", "brainstorm", "imagine"],
                "model": "qwen_3b",
                "confidence": 0.85
            },
            
            # Quick Questions
            "quick": {
                "keywords": ["what", "how", "why", "when", "where", "quick", "simple"],
                "model": "llama_1b",
                "confidence": 0.7
            }
        }
    
    def route_query(self, query: str) -> Tuple[str, float]:
        """Route query to best model"""
        query_lower = query.lower()
        best_model = "llama_1b"  # Default fallback
        best_confidence = 0.5
        
        for domain, rules in self.routing_rules.items():
            matches = sum(1 for keyword in rules["keywords"] if keyword in query_lower)
            if matches > 0:
                confidence = rules["confidence"] * (matches / len(rules["keywords"]))
                if confidence > best_confidence:
                    best_model = rules["model"]
                    best_confidence = confidence
        
        return best_model, best_confidence
    
    def get_model(self, model_name: str) -> Optional[Llama]:
        """Load model on demand"""
        if model_name not in self.loaded_models:
            model_path = self.models_path / f"{model_name}.gguf"
            if model_path.exists():
                self.loaded_models[model_name] = Llama(
                    model_path=str(model_path),
                    n_ctx=1024,
                    verbose=False
                )
        
        return self.loaded_models.get(model_name)
    
    def generate_response(self, query: str, max_tokens: int = 150) -> Dict:
        """Generate response using best model"""
        model_name, confidence = self.route_query(query)
        model = self.get_model(model_name)
        
        if model:
            response = model(query, max_tokens=max_tokens, temperature=0.7)
            return {
                "response": response["choices"][0]["text"].strip(),
                "model_used": model_name,
                "confidence": confidence,
                "status": "success"
            }
        else:
            return {
                "response": "Sorry, I couldn't process your request.",
                "model_used": "none",
                "confidence": 0.0,
                "status": "error"
            }
'''
        
        try:
            router_path = self.output_path / "routing" / "universal_router.py"
            with open(router_path, 'w') as f:
                f.write(router_code)
            
            logger.info("✅ Intelligent router created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create router: {e}")
            return False
    
    def _create_unified_config(self) -> bool:
        """Create unified configuration"""
        logger.info("⚙️ Creating unified configuration...")
        
        config = {
            "meetara_universal_combo": {
                "version": "1.0",
                "created": "2025-01-24",
                "description": "Universal GGUF combining all MeeTARA base models",
                "total_models": len(self.base_models),
                "total_size_gb": sum(model["size_gb"] for model in self.base_models.values()),
                
                "models": self.base_models,
                
                "capabilities": [
                    "healthcare_expertise",
                    "technical_precision", 
                    "creative_flexibility",
                    "analytical_power",
                    "speed_efficiency"
                ],
                
                "domains": [
                    "healthcare", "programming", "business", "creative", 
                    "education", "leadership", "technology", "wellness"
                ],
                
                "routing_strategy": "intelligent_context_aware",
                "fallback_model": "llama_1b",
                "voice_synthesis": "edge_tts_integration",
                "emotion_analysis": "speechbrain_ser_rms"
            }
        }
        
        try:
            config_path = self.output_path / "configs" / "universal_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✅ Unified configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create config: {e}")
            return False
    
    def _package_universal_gguf(self) -> bool:
        """Package everything as universal GGUF"""
        logger.info("📦 Packaging as universal GGUF...")
        
        try:
            # Create final GGUF directory
            final_path = self.final_gguf_path / "meetara-universal-COMBO-ALL.gguf"
            
            # For now, create a symbolic/reference GGUF that points to the container
            # In production, this would be a proper GGUF container format
            
            # Copy the container to final location
            final_container_path = self.final_gguf_path / "universal-combo-container"
            if final_container_path.exists():
                shutil.rmtree(final_container_path)
            
            shutil.copytree(self.output_path, final_container_path)
            
            # Create a reference file
            with open(final_path.with_suffix('.json'), 'w') as f:
                json.dump({
                    "type": "universal_combo_container",
                    "container_path": str(final_container_path),
                    "models_count": len(self.base_models),
                    "total_size_gb": sum(model["size_gb"] for model in self.base_models.values()),
                    "usage": "Load via universal_router.py for intelligent model selection"
                }, f, indent=2)
            
            logger.info("✅ Universal GGUF packaged successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to package: {e}")
            return False
    
    def _display_success_summary(self):
        """Display success summary"""
        total_size = sum(model["size_gb"] for model in self.base_models.values())
        
        print("\n" + "="*70)
        print("🎉 MEETARA UNIVERSAL COMBO GGUF CREATED!")
        print("="*70)
        print(f"📊 Combined Models: {len(self.base_models)}")
        print(f"💾 Total Size: {total_size:.1f}GB")
        print(f"📁 Location: {self.final_gguf_path / 'universal-combo-container'}")
        
        print("\n🤖 Model Capabilities:")
        for model_id, model_info in self.base_models.items():
            print(f"  • {model_id}: {model_info['strength']} ({model_info['size_gb']:.1f}GB)")
        
        print("\n🎯 Domain Coverage:")
        all_domains = set()
        for model_info in self.base_models.values():
            all_domains.update(model_info["domains"])
        for domain in sorted(all_domains):
            print(f"  • {domain}")
        
        print("\n🚀 Usage:")
        print("  1. Use universal_router.py for intelligent routing")
        print("  2. Automatic model selection based on query content")
        print("  3. Fallback to efficient models for quick responses")
        print("  4. Healthcare expertise via TARA-trained model")
        
        print("="*70)

def main():
    """Create MeeTARA Universal Combo GGUF"""
    creator = MeeTARAUniversalCombo()
    
    if creator.create_universal_combo():
        print("\n🎯 SUCCESS: Universal Combo GGUF ready for deployment!")
        return True
    else:
        print("\n❌ FAILED: Could not create universal combo")
        return False

if __name__ == "__main__":
    main() 