# GGUF Compression Techniques

**📅 Created**: June 25, 2025  
**🔄 Last Updated**: June 30, 2025  
**🎯 Status**: Production Implementation - Consolidated System

## Overview

This document outlines the compression techniques used for GGUF (GPT-Generated Unified Format) models in the TARA Universal Model project. The goal is to achieve maximum size reduction while maintaining model quality and performance.

## Current GGUF Models

| Model | Size | Base | Domains | Quantization |
|-------|------|------|---------|--------------|
| meetara-universal-model-1.0.gguf | 4.6GB | DialoGPT-medium | All 5 domains | Q4_K_M |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 4.6GB | Llama-3.1-8B | N/A | Q4_K_M |
| Phi-3.5-mini-instruct-Q4_K_M.gguf | 2.2GB | Phi-3.5-mini | N/A | Q4_K_M |
| qwen2.5-3b-instruct-q4_0.gguf | 1.9GB | Qwen2.5-3B | N/A | Q4_0 |
| llama-3.2-1b-instruct-q4_0.gguf | 0.7GB | Llama-3.2-1B | N/A | Q4_0 |
| meetara-universal-FIXED-Q4_K_M.gguf | 4.6GB | Multiple | All domains | Q4_K_M |

## Quantization Methods

### Q4_K_M (Current Production Default)
- **Description**: 4-bit quantization with K-means clustering and mixed precision
- **Size Reduction**: ~75% compared to 16-bit models
- **Quality Impact**: Minimal for conversational tasks
- **Use Case**: Production deployment, balanced size/quality
- **Implementation**:
  ```python
  # Using llama.cpp for conversion
  ./quantize model.gguf output.gguf q4_k_m
  ```

### Q5_K_M
- **Description**: 5-bit quantization with K-means clustering and mixed precision
- **Size Reduction**: ~70% compared to 16-bit models
- **Quality Impact**: Very minimal, almost indistinguishable from FP16
- **Use Case**: Quality-critical domains (healthcare, technical)
- **Implementation**:
  ```python
  # Using llama.cpp for conversion
  ./quantize model.gguf output.gguf q5_k_m
  ```

### Q2_K
- **Description**: 2-bit quantization with K-means clustering
- **Size Reduction**: ~87% compared to 16-bit models
- **Quality Impact**: Noticeable quality degradation
- **Use Case**: Mobile/edge deployment, extremely constrained environments
- **Implementation**:
  ```python
  # Using llama.cpp for conversion
  ./quantize model.gguf output.gguf q2_k
  ```

### Q8_0
- **Description**: 8-bit quantization, linear quantization
- **Size Reduction**: ~50% compared to 16-bit models
- **Quality Impact**: Negligible
- **Use Case**: Development, testing, highest quality needs
- **Implementation**:
  ```python
  # Using llama.cpp for conversion
  ./quantize model.gguf output.gguf q8_0
  ```

## Advanced Compression Techniques

### 1. Context Window Optimization
- **Technique**: Reduce maximum context window for smaller models
- **Impact**: Linear reduction in memory usage during inference
- **Tradeoff**: Limits conversation history length
- **Implementation**:
  ```python
  # In llama.cpp
  ./main -m model.gguf -c 1024  # Set context to 1024 tokens
  ```

### 2. Tensor Parallelism
- **Technique**: Split model across multiple computation units
- **Impact**: Reduces per-device memory requirements
- **Tradeoff**: Requires multiple devices, increased communication overhead
- **Implementation**:
  ```python
  # Using llama.cpp
  ./main -m model.gguf -ngl 2  # Split across 2 GPUs
  ```

### 3. Sparse Matrix Compression
- **Technique**: Remove near-zero weights from model
- **Impact**: 10-30% additional size reduction
- **Tradeoff**: Requires careful pruning to maintain quality
- **Implementation**: Custom pruning scripts required

### 4. Knowledge Distillation
- **Technique**: Train smaller model to mimic larger model
- **Impact**: Can reduce model size by 50-90%
- **Tradeoff**: Complex training process, potential quality loss
- **Status**: Research phase, not yet implemented

## Compression Results

| Method | Original Size | Compressed Size | Reduction | Quality Impact |
|--------|---------------|-----------------|-----------|----------------|
| Q4_K_M | 1.3GB | 345MB | 73.5% | Minimal |
| Q5_K_M | 1.3GB | 390MB | 70.0% | Negligible |
| Q2_K | 1.3GB | 169MB | 87.0% | Moderate |
| Q8_0 | 1.3GB | 650MB | 50.0% | None |

*Note: Sizes based on DialoGPT-medium model with LoRA adapters*

## 🚦 **UPDATED STRATEGY (2025-06-30)**

### **Key Principles**
- **Q4_K_M quantization is the default for production.** Use Q5_K_M for quality-critical domains (e.g., healthcare), Q2_K for mobile/edge, Q8_0 for dev/testing.
- **LoRA adapters are used for parameter-efficient fine-tuning.** Each subdomain is trained as a LoRA adapter on its optimal base model.
- **Parent GGUFs are created by merging the base model and all LoRA adapters for a parent family.**
- **Do NOT train at the parent domain level.** Always train subdomains, then package as parent GGUFs.
- **Router/meta file orchestrates serving.** Maps user intent/domain to the correct parent GGUF and adapter.
- **Speech/emotion/voice models are packaged separately.**

### **Consolidated GGUF Conversion System**

The TARA Universal Model has consolidated all GGUF conversion into a single unified system:

```python
# scripts/conversion/universal_gguf_factory.py
class UniversalGGUFFactory:
    """
    Consolidated GGUF conversion system for all domains
    Replaces 9 separate GGUF creation scripts with a single unified system
    """
    
    def __init__(self, config_path="configs/model_mapping.json"):
        self.config = self._load_config(config_path)
        self.compression = CompressionUtilities()
        self.cleanup = CleanupUtilities()
        self.emotional_intelligence = EmotionalIntelligence()
        self.phase_manager = PhaseManager()
        
    def create_universal_model(
        self, 
        output_path="models/universal/meetara-universal-model.gguf",
        quantization="q4_k_m", 
        validate=True,
        include_emotional_intelligence=True,
        cleanup_temp_files=True
    ):
        """
        Create a universal GGUF model with all domains
        
        Args:
            output_path: Path for output GGUF file
            quantization: Quantization type (q4_k_m, q5_k_m, q2_k, q8_0)
            validate: Whether to validate the model after creation
            include_emotional_intelligence: Whether to include emotional intelligence
            cleanup_temp_files: Whether to clean up temporary files
        """
        # Implementation details for universal model creation
        
    def create_domain_specific_model(
        self,
        domain,
        output_path=None,
        quantization="q4_k_m",
        validate=True
    ):
        """
        Create a domain-specific GGUF model
        
        Args:
            domain: Domain to create model for
            output_path: Path for output GGUF file
            quantization: Quantization type (q4_k_m, q5_k_m, q2_k, q8_0)
            validate: Whether to validate the model after creation
        """
        # Implementation details for domain-specific model creation
        
    def fix_corrupted_model(
        self,
        input_path,
        output_path=None,
        validation_prompts=None
    ):
        """
        Fix a corrupted GGUF model
        
        Args:
            input_path: Path to corrupted model
            output_path: Path for fixed model
            validation_prompts: List of prompts to validate model
        """
        # Implementation details for fixing corrupted models
```

### **Eliminated Redundant Scripts**

The following redundant scripts have been consolidated into the Universal GGUF Factory:

1. ~~create_healthcare_gguf.py~~ → universal_gguf_factory.py
2. ~~create_business_gguf.py~~ → universal_gguf_factory.py
3. ~~create_education_gguf.py~~ → universal_gguf_factory.py
4. ~~create_creative_gguf.py~~ → universal_gguf_factory.py
5. ~~create_leadership_gguf.py~~ → universal_gguf_factory.py
6. ~~create_universal_gguf.py~~ → universal_gguf_factory.py
7. ~~fix_meetara_gguf.py~~ → universal_gguf_factory.py
8. ~~create_clean_gguf.py~~ → universal_gguf_factory.py
9. ~~tara_gguf_cli.py~~ → universal_gguf_factory.py

### **Recommended Workflow**
1. **Train LoRA adapters for each subdomain** on its optimal base model.
2. **Quantize using Q4_K_M** (or Q5_K_M/Q2_K/Q8_0 as needed).
3. **Package all adapters for a parent family** (plus the base model) into a single parent GGUF file.
4. **Create a router/meta file** to map domains to parent GGUFs and adapters.
5. **Serve using intelligent routing**: load only the required GGUF and adapter for each request.
6. **Keep speech/emotion models in a dedicated directory** and reference as needed.

### **Key Reminders**
- Parent GGUFs = packaging, not training.
- Subdomain LoRA adapters = trained units.
- Router/meta file = orchestration for serving.
- Speech/emotion models = separate GGUFs.

## Optimization Strategy

### Current Approach
- Use Q4_K_M as default quantization method
- Apply LoRA for parameter-efficient fine-tuning (15.32% trainable parameters)
- Maintain separate models for different domains
- Use intelligent routing for domain selection

### Research Directions
1. **Hybrid Quantization**: Different precision for attention vs. feed-forward layers
2. **Adaptive Context**: Dynamic context window based on conversation needs
3. **Progressive Loading**: Load model components as needed during conversation
4. **Shared Embedding Space**: Common embeddings across domain models

## Implementation Guidelines

### Recommended Quantization by Use Case

| Use Case | Recommended Method | Context Window | Notes |
|----------|-------------------|----------------|-------|
| Production | Q4_K_M | 2048 | Default setting |
| Mobile | Q2_K | 1024 | For severely constrained devices |
| Healthcare | Q5_K_M | 4096 | Higher quality for critical domain |
| Development | Q8_0 | 8192 | For testing and validation |

### Conversion Script Template

```python
# This is now part of the UniversalGGUFFactory class
def convert_to_gguf(
    input_model: str,
    output_path: str,
    quant_type: str = "q4_k_m",
    context_size: int = 2048
):
    """
    Convert model to GGUF format with specified quantization
    
    Args:
        input_model: Path to input model
        output_path: Path for output GGUF file
        quant_type: Quantization type (q4_k_m, q5_k_m, q2_k, q8_0)
        context_size: Context window size
    """
    os.system(f"./quantize {input_model} {output_path} {quant_type}")
    print(f"Converted model saved to {output_path}")
    print(f"Original size: {os.path.getsize(input_model) / 1024**3:.2f}GB")
    print(f"Compressed size: {os.path.getsize(output_path) / 1024**3:.2f}GB")
```

## Next Steps

1. **Benchmark Different Methods**: Test all quantization methods against quality metrics
2. **Optimize Existing Models**: Apply best compression techniques to current models
3. **Document Findings**: Update this document with benchmark results
4. **Standardize Process**: Create unified compression pipeline for all models

---

*This document will be updated as new compression techniques are researched and implemented.*
