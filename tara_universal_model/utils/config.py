"""
Configuration management for TARA Universal Model.
Handles settings for all components including training, serving, and domain experts.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class EmotionConfig:
    """Configuration for emotion detection."""
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    threshold: float = 0.3
    professional_context: bool = True
    voice_detection_enabled: bool = False
    cache_embeddings: bool = True

@dataclass
class DomainConfig:
    """Configuration for domain routing."""
    domains_config_path: str = "configs/domains"
    default_domain: str = "universal"
    confidence_threshold: float = 0.6
    safety_enabled: bool = True
    supported_domains: List[str] = field(default_factory=lambda: [
        "healthcare", "business", "education", "creative", "leadership", "universal"
    ])

@dataclass
class ModelConfig:
    """Configuration for model serving."""
    base_model_name: str = "models/microsoft_Phi-3.5-mini-instruct"
    max_input_length: int = 2048
    max_response_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_quantization: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # LoRA Configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj"
    ])
    
    # Training Parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Data Parameters
    max_sequence_length: int = 512
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Optimization
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Advanced
    use_peft: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

@dataclass
class ServingConfig:
    """Configuration for model serving."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    enable_cors: bool = True
    enable_docs: bool = True
    log_level: str = "INFO"
    
    # Model Loading
    model_cache_dir: str = "models/cache"
    adapters_path: str = "models/adapters"
    preload_domains: List[str] = field(default_factory=lambda: [
        "healthcare", "business", "education"
    ])
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour

@dataclass
class DataConfig:
    """Configuration for data processing."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    synthetic_data_path: str = "data/synthetic"
    
    # Synthetic Data Generation
    samples_per_domain: int = 5000
    quality_threshold: float = 0.8
    diversity_threshold: float = 0.7
    
    # Data Processing
    tokenizer_name: str = "models/microsoft_Phi-3.5-mini-instruct"
    add_special_tokens: bool = True
    padding: str = "max_length"
    truncation: bool = True

@dataclass
class SecurityConfig:
    """Security and privacy configuration."""
    enable_encryption: bool = True
    encryption_key_path: str = "configs/security/encryption.key"
    
    # Privacy Settings
    local_processing_only: bool = True
    log_conversations: bool = False
    anonymous_logging: bool = True
    
    # HIPAA Compliance (for healthcare domain)
    hipaa_compliant: bool = True
    audit_logging: bool = True
    data_retention_days: int = 30

@dataclass
class TARAConfig:
    """Main TARA Universal Model configuration."""
    # Component Configurations
    emotion_config: EmotionConfig = field(default_factory=EmotionConfig)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    serving_config: ServingConfig = field(default_factory=ServingConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global Settings
    project_name: str = "TARA Universal Model"
    version: str = "1.0.0"
    debug: bool = False
    
    # Paths
    base_path: str = "."
    config_path: str = "configs"
    models_path: str = "models"
    data_path: str = "data"
    logs_path: str = "logs"
    
    # Model Properties (derived from model_config)
    @property
    def base_model_name(self) -> str:
        return self.model_config.base_model_name
    
    @property
    def max_input_length(self) -> int:
        return self.model_config.max_input_length
    
    @property
    def max_response_length(self) -> int:
        return self.model_config.max_response_length
    
    @property
    def temperature(self) -> float:
        return self.model_config.temperature
    
    @property
    def top_p(self) -> float:
        return self.model_config.top_p
    
    @property
    def use_quantization(self) -> bool:
        return self.model_config.use_quantization
    
    @property
    def supported_domains(self) -> List[str]:
        return self.domain_config.supported_domains
    
    @property
    def adapters_path(self) -> str:
        return self.serving_config.adapters_path
    
    @property
    def context_window(self) -> int:
        return 10  # Default conversation context window

class ConfigManager:
    """Configuration manager for TARA Universal Model."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = TARAConfig()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                self._update_config_from_dict(config_data)
                print(f"Configuration loaded from {self.config_path}")
            else:
                print(f"Config file not found at {self.config_path}, using defaults")
                self.save_config()  # Save default config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config_dict = self._config_to_dict()
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _update_config_from_dict(self, config_data: Dict) -> None:
        """Update configuration from dictionary."""
        # Update emotion config
        if 'emotion' in config_data:
            emotion_data = config_data['emotion']
            for key, value in emotion_data.items():
                if hasattr(self.config.emotion_config, key):
                    setattr(self.config.emotion_config, key, value)
        
        # Update domain config
        if 'domain' in config_data:
            domain_data = config_data['domain']
            for key, value in domain_data.items():
                if hasattr(self.config.domain_config, key):
                    setattr(self.config.domain_config, key, value)
        
        # Update model config
        if 'model' in config_data:
            model_data = config_data['model']
            for key, value in model_data.items():
                if hasattr(self.config.model_config, key):
                    setattr(self.config.model_config, key, value)
        
        # Update training config
        if 'training' in config_data:
            training_data = config_data['training']
            for key, value in training_data.items():
                if hasattr(self.config.training_config, key):
                    setattr(self.config.training_config, key, value)
        
        # Update serving config
        if 'serving' in config_data:
            serving_data = config_data['serving']
            for key, value in serving_data.items():
                if hasattr(self.config.serving_config, key):
                    setattr(self.config.serving_config, key, value)
        
        # Update data config
        if 'data' in config_data:
            data_config_data = config_data['data']
            for key, value in data_config_data.items():
                if hasattr(self.config.data_config, key):
                    setattr(self.config.data_config, key, value)
        
        # Update security config
        if 'security' in config_data:
            security_data = config_data['security']
            for key, value in security_data.items():
                if hasattr(self.config.security_config, key):
                    setattr(self.config.security_config, key, value)
        
        # Update global settings
        global_fields = ['project_name', 'version', 'debug', 'base_path', 
                        'config_path', 'models_path', 'data_path', 'logs_path']
        for field in global_fields:
            if field in config_data:
                setattr(self.config, field, config_data[field])
    
    def _config_to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'project_name': self.config.project_name,
            'version': self.config.version,
            'debug': self.config.debug,
            'base_path': self.config.base_path,
            'config_path': self.config.config_path,
            'models_path': self.config.models_path,
            'data_path': self.config.data_path,
            'logs_path': self.config.logs_path,
            
            'emotion': {
                'model_name': self.config.emotion_config.model_name,
                'threshold': self.config.emotion_config.threshold,
                'professional_context': self.config.emotion_config.professional_context,
                'voice_detection_enabled': self.config.emotion_config.voice_detection_enabled,
                'cache_embeddings': self.config.emotion_config.cache_embeddings
            },
            
            'domain': {
                'domains_config_path': self.config.domain_config.domains_config_path,
                'default_domain': self.config.domain_config.default_domain,
                'confidence_threshold': self.config.domain_config.confidence_threshold,
                'safety_enabled': self.config.domain_config.safety_enabled,
                'supported_domains': self.config.domain_config.supported_domains
            },
            
            'model': {
                'base_model_name': self.config.model_config.base_model_name,
                'max_input_length': self.config.model_config.max_input_length,
                'max_response_length': self.config.model_config.max_response_length,
                'temperature': self.config.model_config.temperature,
                'top_p': self.config.model_config.top_p,
                'top_k': self.config.model_config.top_k,
                'repetition_penalty': self.config.model_config.repetition_penalty,
                'use_quantization': self.config.model_config.use_quantization,
                'device': self.config.model_config.device
            },
            
            'training': {
                'lora_r': self.config.training_config.lora_r,
                'lora_alpha': self.config.training_config.lora_alpha,
                'lora_dropout': self.config.training_config.lora_dropout,
                'lora_target_modules': self.config.training_config.lora_target_modules,
                'batch_size': self.config.training_config.batch_size,
                'gradient_accumulation_steps': self.config.training_config.gradient_accumulation_steps,
                'learning_rate': self.config.training_config.learning_rate,
                'num_epochs': self.config.training_config.num_epochs,
                'max_steps': self.config.training_config.max_steps,
                'warmup_ratio': self.config.training_config.warmup_ratio,
                'weight_decay': self.config.training_config.weight_decay,
                'max_sequence_length': self.config.training_config.max_sequence_length,
                'use_gradient_checkpointing': self.config.training_config.use_gradient_checkpointing,
                'fp16': self.config.training_config.fp16
            },
            
            'serving': {
                'host': self.config.serving_config.host,
                'port': self.config.serving_config.port,
                'workers': self.config.serving_config.workers,
                'max_concurrent_requests': self.config.serving_config.max_concurrent_requests,
                'request_timeout': self.config.serving_config.request_timeout,
                'enable_cors': self.config.serving_config.enable_cors,
                'model_cache_dir': self.config.serving_config.model_cache_dir,
                'adapters_path': self.config.serving_config.adapters_path,
                'preload_domains': self.config.serving_config.preload_domains
            },
            
            'data': {
                'raw_data_path': self.config.data_config.raw_data_path,
                'processed_data_path': self.config.data_config.processed_data_path,
                'synthetic_data_path': self.config.data_config.synthetic_data_path,
                'samples_per_domain': self.config.data_config.samples_per_domain,
                'quality_threshold': self.config.data_config.quality_threshold,
                'tokenizer_name': self.config.data_config.tokenizer_name
            },
            
            'security': {
                'enable_encryption': self.config.security_config.enable_encryption,
                'local_processing_only': self.config.security_config.local_processing_only,
                'log_conversations': self.config.security_config.log_conversations,
                'hipaa_compliant': self.config.security_config.hipaa_compliant,
                'data_retention_days': self.config.security_config.data_retention_days
            }
        }
    
    def get_domain_config(self, domain: str) -> Dict:
        """Get configuration for specific domain."""
        domain_config_file = os.path.join(
            self.config.domain_config.domains_config_path, 
            f"{domain}.yaml"
        )
        
        if os.path.exists(domain_config_file):
            with open(domain_config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        return {}
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}'")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check required paths exist
        required_paths = [
            self.config.data_path,
            self.config.models_path,
            self.config.config_path
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                issues.append(f"Required path does not exist: {path}")
        
        # Validate model settings
        if self.config.model_config.temperature < 0 or self.config.model_config.temperature > 2:
            issues.append("Temperature should be between 0 and 2")
        
        if self.config.model_config.top_p < 0 or self.config.model_config.top_p > 1:
            issues.append("top_p should be between 0 and 1")
        
        # Validate training settings
        if self.config.training_config.learning_rate <= 0:
            issues.append("Learning rate should be positive")
        
        if self.config.training_config.batch_size <= 0:
            issues.append("Batch size should be positive")
        
        return issues

# Global config instance
_config_manager = None

def get_config(config_path: str = "configs/config.yaml") -> TARAConfig:
    """Get global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager.config

def reload_config(config_path: str = "configs/config.yaml") -> TARAConfig:
    """Reload configuration from file."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.config 