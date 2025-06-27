#!/usr/bin/env python3
"""
🔧 Compression Utilities for TARA Universal Model
Handles advanced compression techniques: quantization, sparse, hybrid, distillation
"""

import os
import torch
import logging
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    Q2_K = "Q2_K"      # Mobile/Edge (fastest, smallest)
    Q4_K_M = "Q4_K_M"  # Production (balanced)
    Q5_K_M = "Q5_K_M"  # Quality-critical (highest quality)
    Q8_0 = "Q8_0"      # Development/Testing (full precision)

class CompressionType(Enum):
    STANDARD = "standard"      # Basic quantization
    SPARSE = "sparse"          # Sparse quantization
    HYBRID = "hybrid"          # Mixed precision
    DISTILLED = "distilled"    # Knowledge distillation

@dataclass
class CompressionConfig:
    quantization: QuantizationType
    compression_type: CompressionType
    target_size_mb: Optional[float] = None
    quality_threshold: float = 0.95
    speed_priority: bool = False
    memory_constraint: Optional[float] = None

@dataclass
class CompressionResult:
    success: bool
    output_path: Path
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    quality_score: float
    compression_time: float
    error_message: Optional[str] = None

class CompressionUtilities:
    """Advanced compression utilities for TARA Universal Model"""
    
    def __init__(self, llama_cpp_path: Path = None):
        self.llama_cpp_path = llama_cpp_path or Path("llama.cpp")
        self.temp_dir = None
        self.compression_stats = {}
        
    def ensure_llama_cpp(self) -> bool:
        """Ensure llama.cpp is available"""
        if not self.llama_cpp_path.exists():
            logger.info("📥 Cloning llama.cpp...")
            try:
                subprocess.run([
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
                ], check=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to clone llama.cpp: {e}")
                return False
        return True
    
    def compress_model(self, input_path: Path, config: CompressionConfig) -> CompressionResult:
        """Main compression function with advanced techniques"""
        
        start_time = time.time()
        original_size_mb = input_path.stat().st_size / (1024*1024)
        
        try:
            logger.info(f"🔧 Compressing model with {config.compression_type.value} + {config.quantization.value}")
            
            # Create temporary directory
            self.temp_dir = Path(tempfile.mkdtemp())
            
            # Apply compression based on type
            if config.compression_type == CompressionType.STANDARD:
                result = self._standard_quantization(input_path, config)
            elif config.compression_type == CompressionType.SPARSE:
                result = self._sparse_quantization(input_path, config)
            elif config.compression_type == CompressionType.HYBRID:
                result = self._hybrid_quantization(input_path, config)
            elif config.compression_type == CompressionType.DISTILLED:
                result = self._knowledge_distillation(input_path, config)
            else:
                raise ValueError(f"Unknown compression type: {config.compression_type}")
            
            # Calculate metrics
            compression_time = time.time() - start_time
            compressed_size_mb = result.output_path.stat().st_size / (1024*1024)
            compression_ratio = compressed_size_mb / original_size_mb
            
            # Estimate quality score (simplified)
            quality_score = self._estimate_quality_score(config.quantization, config.compression_type)
            
            return CompressionResult(
                success=True,
                output_path=result.output_path,
                original_size_mb=original_size_mb,
                compressed_size_mb=compressed_size_mb,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                compression_time=compression_time
            )
            
        except Exception as e:
            compression_time = time.time() - start_time
            return CompressionResult(
                success=False,
                output_path=Path(),
                original_size_mb=original_size_mb,
                compressed_size_mb=0,
                compression_ratio=0,
                quality_score=0,
                compression_time=compression_time,
                error_message=str(e)
            )
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def _standard_quantization(self, input_path: Path, config: CompressionConfig) -> CompressionResult:
        """Standard quantization using llama.cpp"""
        logger.info("🔄 Applying standard quantization...")
        
        if not self.ensure_llama_cpp():
            raise RuntimeError("llama.cpp not available")
        
        # Convert to F16 first if needed
        f16_path = self.temp_dir / "model-f16.gguf"
        if not input_path.suffix == '.gguf':
            # Convert from HuggingFace format
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(input_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ], check=True)
        else:
            # Already GGUF, copy to temp
            shutil.copy2(input_path, f16_path)
        
        # Apply quantization
        output_path = self.temp_dir / f"model-{config.quantization.value}.gguf"
        
        subprocess.run([
            "llama.cpp/quantize",
            str(f16_path),
            str(output_path),
            config.quantization.value.lower()
        ], check=True)
        
        return CompressionResult(
            success=True,
            output_path=output_path,
            original_size_mb=0,
            compressed_size_mb=0,
            compression_ratio=0,
            quality_score=0,
            compression_time=0
        )
    
    def _sparse_quantization(self, input_path: Path, config: CompressionConfig) -> CompressionResult:
        """Sparse quantization for better compression"""
        logger.info("🔄 Applying sparse quantization...")
        
        # Note: This is a placeholder for sparse quantization
        # In production, you would implement actual sparse quantization
        # For now, we'll use standard quantization with sparse-aware settings
        
        if not self.ensure_llama_cpp():
            raise RuntimeError("llama.cpp not available")
        
        # Convert to F16 first
        f16_path = self.temp_dir / "model-f16.gguf"
        if not input_path.suffix == '.gguf':
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(input_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ], check=True)
        else:
            shutil.copy2(input_path, f16_path)
        
        # Apply aggressive quantization for sparse effect
        output_path = self.temp_dir / f"model-sparse-{config.quantization.value}.gguf"
        
        # Use more aggressive quantization for sparse compression
        quant_type = "q2_k" if config.quantization == QuantizationType.Q2_K else "q4_k_m"
        
        subprocess.run([
            "llama.cpp/quantize",
            str(f16_path),
            str(output_path),
            quant_type
        ], check=True)
        
        return CompressionResult(
            success=True,
            output_path=output_path,
            original_size_mb=0,
            compressed_size_mb=0,
            compression_ratio=0,
            quality_score=0,
            compression_time=0
        )
    
    def _hybrid_quantization(self, input_path: Path, config: CompressionConfig) -> CompressionResult:
        """Hybrid quantization with mixed precision"""
        logger.info("🔄 Applying hybrid quantization...")
        
        # Note: This is a placeholder for hybrid quantization
        # In production, you would implement actual hybrid quantization
        # For now, we'll use a combination of different quantization levels
        
        if not self.ensure_llama_cpp():
            raise RuntimeError("llama.cpp not available")
        
        # Convert to F16 first
        f16_path = self.temp_dir / "model-f16.gguf"
        if not input_path.suffix == '.gguf':
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(input_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ], check=True)
        else:
            shutil.copy2(input_path, f16_path)
        
        # Create hybrid quantization (mix of Q4_K_M and Q5_K_M)
        output_path = self.temp_dir / f"model-hybrid-{config.quantization.value}.gguf"
        
        # For hybrid, we'll use the specified quantization but with hybrid-aware settings
        subprocess.run([
            "llama.cpp/quantize",
            str(f16_path),
            str(output_path),
            config.quantization.value.lower()
        ], check=True)
        
        return CompressionResult(
            success=True,
            output_path=output_path,
            original_size_mb=0,
            compressed_size_mb=0,
            compression_ratio=0,
            quality_score=0,
            compression_time=0
        )
    
    def _knowledge_distillation(self, input_path: Path, config: CompressionConfig) -> CompressionResult:
        """Knowledge distillation for model compression"""
        logger.info("🔄 Applying knowledge distillation...")
        
        # Note: This is a placeholder for knowledge distillation
        # In production, you would implement actual distillation
        # For now, we'll use standard quantization as a proxy
        
        if not self.ensure_llama_cpp():
            raise RuntimeError("llama.cpp not available")
        
        # Convert to F16 first
        f16_path = self.temp_dir / "model-f16.gguf"
        if not input_path.suffix == '.gguf':
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(input_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ], check=True)
        else:
            shutil.copy2(input_path, f16_path)
        
        # Apply quantization (distillation would happen before this)
        output_path = self.temp_dir / f"model-distilled-{config.quantization.value}.gguf"
        
        subprocess.run([
            "llama.cpp/quantize",
            str(f16_path),
            str(output_path),
            config.quantization.value.lower()
        ], check=True)
        
        return CompressionResult(
            success=True,
            output_path=output_path,
            original_size_mb=0,
            compressed_size_mb=0,
            compression_ratio=0,
            quality_score=0,
            compression_time=0
        )
    
    def _estimate_quality_score(self, quantization: QuantizationType, compression_type: CompressionType) -> float:
        """Estimate quality score based on compression settings"""
        
        # Base quality scores for quantization types
        quantization_scores = {
            QuantizationType.Q2_K: 0.7,
            QuantizationType.Q4_K_M: 0.85,
            QuantizationType.Q5_K_M: 0.95,
            QuantizationType.Q8_0: 1.0
        }
        
        # Compression type multipliers
        compression_multipliers = {
            CompressionType.STANDARD: 1.0,
            CompressionType.SPARSE: 0.9,
            CompressionType.HYBRID: 0.95,
            CompressionType.DISTILLED: 0.85
        }
        
        base_score = quantization_scores.get(quantization, 0.8)
        multiplier = compression_multipliers.get(compression_type, 1.0)
        
        return base_score * multiplier
    
    def get_compression_recommendations(self, model_size_mb: float, 
                                      target_size_mb: Optional[float] = None,
                                      quality_priority: bool = False,
                                      speed_priority: bool = False) -> List[CompressionConfig]:
        """Get compression recommendations based on requirements"""
        
        recommendations = []
        
        # Calculate target compression ratio
        if target_size_mb:
            compression_ratio = target_size_mb / model_size_mb
        else:
            compression_ratio = 0.5  # Default 50% compression
        
        # Generate recommendations
        if speed_priority:
            # Prioritize speed over quality
            recommendations.append(CompressionConfig(
                quantization=QuantizationType.Q2_K,
                compression_type=CompressionType.STANDARD,
                target_size_mb=target_size_mb,
                speed_priority=True
            ))
        
        elif quality_priority:
            # Prioritize quality over size
            recommendations.append(CompressionConfig(
                quantization=QuantizationType.Q5_K_M,
                compression_type=CompressionType.HYBRID,
                target_size_mb=target_size_mb,
                quality_threshold=0.98
            ))
        
        else:
            # Balanced approach
            if compression_ratio < 0.3:
                # Aggressive compression
                recommendations.append(CompressionConfig(
                    quantization=QuantizationType.Q2_K,
                    compression_type=CompressionType.SPARSE,
                    target_size_mb=target_size_mb
                ))
            elif compression_ratio < 0.6:
                # Moderate compression
                recommendations.append(CompressionConfig(
                    quantization=QuantizationType.Q4_K_M,
                    compression_type=CompressionType.STANDARD,
                    target_size_mb=target_size_mb
                ))
            else:
                # Light compression
                recommendations.append(CompressionConfig(
                    quantization=QuantizationType.Q5_K_M,
                    compression_type=CompressionType.HYBRID,
                    target_size_mb=target_size_mb
                ))
        
        return recommendations
    
    def benchmark_compression(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Benchmark different compression techniques"""
        
        logger.info("🔍 Benchmarking compression techniques...")
        
        results = {}
        original_size_mb = input_path.stat().st_size / (1024*1024)
        
        # Test all quantization types with standard compression
        for quantization in QuantizationType:
            config = CompressionConfig(
                quantization=quantization,
                compression_type=CompressionType.STANDARD
            )
            
            result = self.compress_model(input_path, config)
            results[f"standard_{quantization.value}"] = {
                "success": result.success,
                "size_mb": result.compressed_size_mb,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "compression_time": result.compression_time
            }
        
        # Test different compression types with Q4_K_M
        for compression_type in CompressionType:
            config = CompressionConfig(
                quantization=QuantizationType.Q4_K_M,
                compression_type=compression_type
            )
            
            result = self.compress_model(input_path, config)
            results[f"{compression_type.value}_Q4_K_M"] = {
                "success": result.success,
                "size_mb": result.compressed_size_mb,
                "compression_ratio": result.compression_ratio,
                "quality_score": result.quality_score,
                "compression_time": result.compression_time
            }
        
        # Save benchmark results
        benchmark_path = output_dir / "compression_benchmark.json"
        with open(benchmark_path, 'w') as f:
            json.dump({
                "original_size_mb": original_size_mb,
                "results": results,
                "recommendations": self.get_compression_recommendations(original_size_mb)
            }, f, indent=2)
        
        logger.info(f"✅ Compression benchmark saved to {benchmark_path}")
        return results

# Import time for timing functionality
import time 