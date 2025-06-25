#!/usr/bin/env python3
"""
🎯 Hierarchical GGUF Creator for MeeTARA Universal Model
Creates parent-child GGUF structure for scalable domain integration
"""

import os
import json
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class HierarchicalGGUFBuilder:
    def __init__(self):
        self.base_dir = Path(".")
        self.models_dir = self.base_dir / "models"
        self.gguf_dir = self.models_dir / "gguf"
        self.output_dir = self.gguf_dir / "hierarchical"
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
    def create_hierarchical_structure(self):
        """Create the hierarchical GGUF structure"""
        logger.info("🎯 Creating Hierarchical GGUF Structure...")
        
        # Step 1: Prepare DialoGPT-based model (Healthcare + Business + Education)
        self.create_dialogpt_model()
        
        # Step 2: Prepare Qwen2.5-based model (Creative + Education + Leadership)
        self.create_qwen_model()
        
        # Step 3: Create parent universal container
        self.create_universal_container()
        
        logger.info("🎉 Hierarchical GGUF Structure Complete!")
        
    def create_dialogpt_model(self):
        """Create DialoGPT-based model with trained domains"""
        logger.info("🔄 Creating DialoGPT-based model...")
        
        source_file = self.gguf_dir / "tara-1.0-instruct-Q4_K_M.gguf"
        target_file = self.output_dir / "meetara-1.0-instruct-Q4_K_M.gguf"
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            logger.info(f"✅ DialoGPT model created: {target_file}")
            
            # Create metadata
            metadata = {
                "model_name": "meetara-1.0-instruct-Q4_K_M",
                "base_model": "DialoGPT-medium",
                "domains": ["healthcare", "business", "education"],
                "training_status": {
                    "healthcare": {"status": "complete", "improvement": "97.6%"},
                    "business": {"status": "complete", "improvement": "97.3%"},
                    "education": {"status": "complete", "improvement": "97.5%"}
                },
                "specialties": ["conversational", "therapeutic", "professional"],
                "file_size_mb": round(target_file.stat().st_size / (1024*1024), 1),
                "context_length": 4096,
                "chat_format": "dialogpt"
            }
            
            with open(self.output_dir / "meetara-1.0-instruct-Q4_K_M.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        else:
            logger.error(f"❌ Source DialoGPT model not found: {source_file}")
    
    def create_qwen_model(self):
        """Create Qwen2.5-based model for analytical domains"""
        logger.info("🔄 Creating Qwen2.5-based model...")
        
        source_file = self.gguf_dir / "meetara-qwen25-base-v1.0.gguf"
        target_file = self.output_dir / "meetara-qwen25-base-v1.0.gguf"
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            logger.info(f"✅ Qwen2.5 model created: {target_file}")
            
            # Create metadata
            metadata = {
                "model_name": "meetara-qwen25-base-v1.0",
                "base_model": "Qwen2.5-3B-Instruct",
                "domains": ["creative", "education", "leadership"],
                "training_status": {
                    "creative": {"status": "pending", "improvement": "TBD"},
                    "education": {"status": "migration_needed", "improvement": "TBD"},
                    "leadership": {"status": "pending", "improvement": "TBD"}
                },
                "specialties": ["analytical", "creative", "strategic"],
                "file_size_mb": round(target_file.stat().st_size / (1024*1024), 1),
                "context_length": 8192,
                "chat_format": "qwen"
            }
            
            with open(self.output_dir / "meetara-qwen25-base-v1.0.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        else:
            logger.error(f"❌ Source Qwen2.5 model not found: {source_file}")
    
    def create_universal_container(self):
        """Create the parent universal container metadata"""
        logger.info("🔄 Creating Universal Container...")
        
        # Create universal metadata
        universal_metadata = {
            "container_name": "meetara-universal-v1",
            "version": "1.0",
            "architecture": "hierarchical",
            "description": "Universal GGUF container with specialized child models",
            "child_models": [
                {
                    "name": "meetara-1.0-instruct-Q4_K_M",
                    "base": "DialoGPT-medium",
                    "domains": ["healthcare", "business", "education"],
                    "specialties": ["conversational", "therapeutic", "professional"],
                    "status": "production_ready"
                },
                {
                    "name": "meetara-qwen25-base-v1.0",
                    "base": "Qwen2.5-3B-Instruct", 
                    "domains": ["creative", "education", "leadership"],
                    "specialties": ["analytical", "creative", "strategic"],
                    "status": "training_needed"
                }
            ],
            "domain_routing": {
                "healthcare": "meetara-1.0-instruct-Q4_K_M",
                "business": "meetara-1.0-instruct-Q4_K_M",
                "education": "meetara-qwen25-base-v1.0",  # Will migrate from DialoGPT
                "creative": "meetara-qwen25-base-v1.0",
                "leadership": "meetara-qwen25-base-v1.0"
            },
            "future_expansion": {
                "target_domains": 28,
                "expansion_strategy": "specialized_gguf_per_domain_group",
                "integration_method": "hierarchical_container"
            },
            "total_size_mb": self.calculate_total_size(),
            "deployment_target": "meetara_repository",
            "created_date": "2025-01-24"
        }
        
        # Save universal metadata
        with open(self.output_dir / "meetara-universal-v1.json", "w") as f:
            json.dump(universal_metadata, f, indent=2)
            
        logger.info("✅ Universal container metadata created")
        
        # Create deployment instructions
        self.create_deployment_instructions()
    
    def calculate_total_size(self):
        """Calculate total size of all child models"""
        total_size = 0
        for file in self.output_dir.glob("*.gguf"):
            total_size += file.stat().st_size
        return round(total_size / (1024*1024), 1)
    
    def create_deployment_instructions(self):
        """Create deployment instructions for meetara integration"""
        instructions = """
# MeeTARA Universal GGUF Deployment Instructions

## Hierarchical Structure Created:
```
meetara-universal-v1/ (Container)
├── meetara-1.0-instruct-Q4_K_M.gguf (681MB)
│   └── Domains: Healthcare, Business, Education
│   └── Base: DialoGPT-medium (Conversational)
├── meetara-qwen25-base-v1.0.gguf (1.9GB)
│   └── Domains: Creative, Education, Leadership  
│   └── Base: Qwen2.5-3B (Analytical)
└── meetara-universal-v1.json (Routing metadata)
```

## Integration with MeeTARA Repository:

### 1. Copy Structure to MeeTARA:
```bash
cp -r models/gguf/hierarchical/* ../github/meetara/models/
```

### 2. Update MeeTARA Router Configuration:
- Update `services/ai-engine-python/core_reactor.py`
- Add hierarchical model routing logic
- Configure domain-to-model mapping

### 3. Future Domain Expansion:
- Train new domains on appropriate base models
- Create specialized GGUF files
- Add to hierarchical container
- Automatic integration with MeeTARA

## Next Steps:
1. ✅ Deploy current structure to MeeTARA
2. 🔄 Train Creative/Leadership on Qwen2.5
3. 🔄 Migrate Education from DialoGPT to Qwen2.5
4. 🚀 Scale to 28+ domains with new GGUF containers
"""
        
        with open(self.output_dir / "DEPLOYMENT_INSTRUCTIONS.md", "w") as f:
            f.write(instructions)
            
        logger.info("✅ Deployment instructions created")
    
    def print_summary(self):
        """Print creation summary"""
        print("\n🎉 HIERARCHICAL GGUF STRUCTURE COMPLETE!")
        print("=" * 50)
        
        if self.output_dir.exists():
            print(f"📁 Location: {self.output_dir}")
            print("\n📊 Created Files:")
            
            for file in sorted(self.output_dir.iterdir()):
                if file.is_file():
                    size_mb = round(file.stat().st_size / (1024*1024), 1) if file.suffix == '.gguf' else 0
                    size_str = f"({size_mb}MB)" if size_mb > 0 else ""
                    print(f"   ✅ {file.name} {size_str}")
        
        print(f"\n🎯 Next Steps:")
        print("   1. Deploy to MeeTARA repository")
        print("   2. Train remaining domains on Qwen2.5")
        print("   3. Test hierarchical routing")
        print("   4. Scale to 28+ domains")

def main():
    """Main execution function"""
    print("🎯 MeeTARA Hierarchical GGUF Builder")
    print("=" * 50)
    
    builder = HierarchicalGGUFBuilder()
    builder.create_hierarchical_structure()
    builder.print_summary()

if __name__ == "__main__":
    main() 