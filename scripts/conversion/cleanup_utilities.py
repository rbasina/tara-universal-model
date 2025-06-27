#!/usr/bin/env python3
"""
🧹 Cleanup Utilities for TARA Universal Model
Handles garbage data removal, model validation, and preparation for GGUF conversion
"""

import os
import torch
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import pickle

logger = logging.getLogger(__name__)

@dataclass
class ModelValidationResult:
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    model_size_mb: float
    tokenizer_vocab_size: int
    model_config: Dict[str, Any]
    validation_score: float  # 0-1

@dataclass
class CleanupResult:
    success: bool
    cleaned_path: Path
    original_size_mb: float
    cleaned_size_mb: float
    removed_files: List[str]
    validation_result: ModelValidationResult
    error_message: Optional[str] = None

class ModelCleanupUtilities:
    """Utilities for cleaning and validating models before GGUF conversion"""
    
    def __init__(self):
        self.required_files = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        self.optional_files = [
            'special_tokens_map.json',
            'added_tokens.json',
            'merges.txt',
            'vocab.json',
            'generation_config.json'
        ]
        
        self.garbage_patterns = [
            '*.tmp', '*.temp', '*.bak', '*.backup',
            '*.log', '*.cache', '*.lock',
            'checkpoint-*', 'runs/', 'logs/',
            'wandb/', '.git/', '__pycache__/',
            '*.pyc', '*.pyo', '*.pyd'
        ]
    
    def clean_model_directory(self, model_path: Path, output_path: Path = None) -> CleanupResult:
        """Clean model directory by removing garbage and validating structure"""
        
        try:
            logger.info(f"🧹 Cleaning model directory: {model_path}")
            
            # Create output directory
            if output_path is None:
                output_path = model_path.parent / f"{model_path.name}_cleaned"
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Track original size
            original_size_mb = self._calculate_directory_size(model_path)
            
            # Step 1: Remove garbage files
            removed_files = self._remove_garbage_files(model_path, output_path)
            
            # Step 2: Validate model structure
            validation_result = self._validate_model_structure(output_path)
            
            # Step 3: Fix common issues
            if validation_result.is_valid:
                self._fix_common_issues(output_path)
            
            # Calculate cleaned size
            cleaned_size_mb = self._calculate_directory_size(output_path)
            
            return CleanupResult(
                success=validation_result.is_valid,
                cleaned_path=output_path,
                original_size_mb=original_size_mb,
                cleaned_size_mb=cleaned_size_mb,
                removed_files=removed_files,
                validation_result=validation_result
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to clean model directory: {e}")
            return CleanupResult(
                success=False,
                cleaned_path=Path(),
                original_size_mb=0,
                cleaned_size_mb=0,
                removed_files=[],
                validation_result=ModelValidationResult(
                    is_valid=False,
                    issues=[str(e)],
                    warnings=[],
                    model_size_mb=0,
                    tokenizer_vocab_size=0,
                    model_config={},
                    validation_score=0.0
                ),
                error_message=str(e)
            )
    
    def _remove_garbage_files(self, source_path: Path, target_path: Path) -> List[str]:
        """Remove garbage files and copy valid files"""
        
        removed_files = []
        
        # Copy only valid files
        for item in source_path.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(source_path)
                target_item = target_path / relative_path
                
                # Check if it's a garbage file
                is_garbage = self._is_garbage_file(item)
                
                if is_garbage:
                    removed_files.append(str(relative_path))
                    logger.debug(f"🗑️ Removed garbage file: {relative_path}")
                else:
                    # Copy valid file
                    target_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_item)
                    logger.debug(f"✅ Copied valid file: {relative_path}")
        
        logger.info(f"🧹 Removed {len(removed_files)} garbage files")
        return removed_files
    
    def _is_garbage_file(self, file_path: Path) -> bool:
        """Check if file is garbage"""
        
        file_name = file_path.name
        file_suffix = file_path.suffix.lower()
        
        # Check garbage patterns
        for pattern in self.garbage_patterns:
            if pattern.startswith('*.'):
                if file_suffix == pattern[1:]:
                    return True
            elif pattern.endswith('/'):
                if file_path.is_dir() and pattern[:-1] in file_name:
                    return True
            else:
                if pattern in file_name:
                    return True
        
        # Check for temporary files
        if file_name.startswith('.') or file_name.endswith('~'):
            return True
        
        # Check for very large files that might be corrupted
        if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024 * 1024:  # 10GB
            return True
        
        return False
    
    def _validate_model_structure(self, model_path: Path) -> ModelValidationResult:
        """Validate model structure and files"""
        
        issues = []
        warnings = []
        validation_score = 1.0
        
        # Check required files
        missing_required = []
        for file_name in self.required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_required.append(file_name)
                validation_score -= 0.2
        
        if missing_required:
            issues.append(f"Missing required files: {missing_required}")
        
        # Check optional files
        missing_optional = []
        for file_name in self.optional_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_optional.append(file_name)
                validation_score -= 0.05
        
        if missing_optional:
            warnings.append(f"Missing optional files: {missing_optional}")
        
        # Validate adapter files
        adapter_config_path = model_path / 'adapter_config.json'
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                # Check adapter configuration
                if 'base_model_name_or_path' not in adapter_config:
                    issues.append("Missing base_model_name_or_path in adapter config")
                    validation_score -= 0.1
                
                if 'target_modules' not in adapter_config:
                    issues.append("Missing target_modules in adapter config")
                    validation_score -= 0.1
                    
            except Exception as e:
                issues.append(f"Invalid adapter config: {e}")
                validation_score -= 0.3
        
        # Validate model config
        config_path = model_path / 'config.json'
        model_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Check essential config fields
                required_config_fields = ['vocab_size', 'hidden_size', 'num_attention_heads']
                for field in required_config_fields:
                    if field not in model_config:
                        issues.append(f"Missing {field} in model config")
                        validation_score -= 0.1
                        
            except Exception as e:
                issues.append(f"Invalid model config: {e}")
                validation_score -= 0.3
        
        # Validate tokenizer
        tokenizer_config_path = model_path / 'tokenizer_config.json'
        tokenizer_vocab_size = 0
        if tokenizer_config_path.exists():
            try:
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                
                # Check tokenizer configuration
                if 'model_max_length' not in tokenizer_config:
                    warnings.append("Missing model_max_length in tokenizer config")
                    validation_score -= 0.05
                    
            except Exception as e:
                issues.append(f"Invalid tokenizer config: {e}")
                validation_score -= 0.2
        
        # Calculate model size
        model_size_mb = self._calculate_directory_size(model_path)
        
        # Check for reasonable model size
        if model_size_mb < 10:  # Less than 10MB
            warnings.append(f"Model seems very small: {model_size_mb:.1f}MB")
            validation_score -= 0.1
        elif model_size_mb > 10000:  # More than 10GB
            warnings.append(f"Model seems very large: {model_size_mb:.1f}MB")
            validation_score -= 0.1
        
        # Ensure validation score is between 0 and 1
        validation_score = max(0.0, min(1.0, validation_score))
        
        is_valid = len(issues) == 0 and validation_score > 0.5
        
        return ModelValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            model_size_mb=model_size_mb,
            tokenizer_vocab_size=tokenizer_vocab_size,
            model_config=model_config,
            validation_score=validation_score
        )
    
    def _fix_common_issues(self, model_path: Path):
        """Fix common model issues"""
        
        logger.info("🔧 Fixing common model issues...")
        
        # Fix missing pad token in tokenizer config
        tokenizer_config_path = model_path / 'tokenizer_config.json'
        if tokenizer_config_path.exists():
            try:
                with open(tokenizer_config_path, 'r') as f:
                    tokenizer_config = json.load(f)
                
                # Add pad token if missing
                if 'pad_token' not in tokenizer_config:
                    tokenizer_config['pad_token'] = tokenizer_config.get('eos_token', '[PAD]')
                    logger.info("✅ Added missing pad_token to tokenizer config")
                
                # Save updated config
                with open(tokenizer_config_path, 'w') as f:
                    json.dump(tokenizer_config, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not fix tokenizer config: {e}")
        
        # Fix model config issues
        config_path = model_path / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Add missing fields with defaults
                if 'pad_token_id' not in model_config:
                    model_config['pad_token_id'] = model_config.get('eos_token_id', 0)
                    logger.info("✅ Added missing pad_token_id to model config")
                
                if 'bos_token_id' not in model_config:
                    model_config['bos_token_id'] = model_config.get('eos_token_id', 1)
                    logger.info("✅ Added missing bos_token_id to model config")
                
                # Save updated config
                with open(config_path, 'w') as f:
                    json.dump(model_config, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not fix model config: {e}")
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def validate_adapter_compatibility(self, adapter_path: Path, base_model_name: str) -> bool:
        """Validate adapter compatibility with base model"""
        
        try:
            adapter_config_path = adapter_path / 'adapter_config.json'
            if not adapter_config_path.exists():
                logger.error("❌ Adapter config not found")
                return False
            
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            # Check base model compatibility
            configured_base = adapter_config.get('base_model_name_or_path', '')
            if configured_base != base_model_name:
                logger.warning(f"⚠️ Base model mismatch: {configured_base} vs {base_model_name}")
                return False
            
            # Check target modules
            target_modules = adapter_config.get('target_modules', [])
            if not target_modules:
                logger.warning("⚠️ No target modules specified")
                return False
            
            logger.info(f"✅ Adapter compatible with {base_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to validate adapter compatibility: {e}")
            return False
    
    def create_clean_model_manifest(self, model_path: Path) -> Dict[str, Any]:
        """Create a manifest of the cleaned model"""
        
        manifest = {
            "model_path": str(model_path),
            "cleaned_date": datetime.now().isoformat(),
            "files": {},
            "validation": {},
            "checksums": {}
        }
        
        # Scan all files
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(model_path))
                
                # File info
                stat = file_path.stat()
                manifest["files"][relative_path] = {
                    "size_bytes": stat.st_size,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                # Calculate checksum
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        checksum = hashlib.md5(content).hexdigest()
                        manifest["checksums"][relative_path] = checksum
                except Exception as e:
                    logger.warning(f"⚠️ Could not calculate checksum for {relative_path}: {e}")
        
        # Add validation info
        validation_result = self._validate_model_structure(model_path)
        manifest["validation"] = {
            "is_valid": validation_result.is_valid,
            "validation_score": validation_result.validation_score,
            "issues": validation_result.issues,
            "warnings": validation_result.warnings,
            "model_size_mb": validation_result.model_size_mb
        }
        
        return manifest
    
    def save_cleanup_report(self, cleanup_result: CleanupResult, output_path: Path):
        """Save detailed cleanup report"""
        
        report = {
            "cleanup_summary": {
                "success": cleanup_result.success,
                "original_size_mb": cleanup_result.original_size_mb,
                "cleaned_size_mb": cleanup_result.cleaned_size_mb,
                "size_reduction_mb": cleanup_result.original_size_mb - cleanup_result.cleaned_size_mb,
                "size_reduction_percent": ((cleanup_result.original_size_mb - cleanup_result.cleaned_size_mb) / cleanup_result.original_size_mb) * 100 if cleanup_result.original_size_mb > 0 else 0,
                "removed_files_count": len(cleanup_result.removed_files)
            },
            "validation_result": {
                "is_valid": cleanup_result.validation_result.is_valid,
                "validation_score": cleanup_result.validation_result.validation_score,
                "issues": cleanup_result.validation_result.issues,
                "warnings": cleanup_result.validation_result.warnings,
                "model_size_mb": cleanup_result.validation_result.model_size_mb,
                "tokenizer_vocab_size": cleanup_result.validation_result.tokenizer_vocab_size
            },
            "removed_files": cleanup_result.removed_files,
            "model_config": cleanup_result.validation_result.model_config,
            "error_message": cleanup_result.error_message
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Cleanup report saved to {output_path}")

# Import datetime for timestamp functionality
from datetime import datetime 