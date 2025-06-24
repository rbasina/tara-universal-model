#!/usr/bin/env python3
"""
GGUF Model Integration for TARA Universal Model.
Provides support for quantized GGUF models using llama-cpp-python.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None  # Define Llama as None for type hints when not available
    logging.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

from ..utils.config import TARAConfig

logger = logging.getLogger(__name__)

class GGUFModelManager:
    """Manages GGUF models for TARA Universal Model."""
    
    def __init__(self, config: TARAConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.model_configs = {
            "tara-1.0": {
                "file": "tara-1.0-instruct-Q4_K_M.gguf",
                "context_length": 4096,
                "domains": ["healthcare", "business", "education", "creative", "leadership", "universal"],
                "chat_format": "dialogpt",
                "system_prompt": "You are TARA, an intelligent AI companion with expertise across healthcare, business, education, creative, and leadership domains. You provide empathetic, knowledgeable assistance while maintaining therapeutic relationships.",
                "specialty": "unified_domain_expert"
            },
            "phi-3.5": {
                "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
                "context_length": 4096,
                "domains": ["business", "universal"],
                "chat_format": "phi-3",
                "system_prompt": "You are TARA, a helpful AI assistant."
            },
            "llama-3.1": {
                "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 
                "context_length": 8192,
                "domains": ["healthcare", "education", "leadership"],
                "chat_format": "llama-3",
                "system_prompt": "You are TARA, a knowledgeable and caring AI assistant."
            },
            "llama-3.2": {
                "file": "llama-3.2-1b-instruct-q4_0.gguf",
                "context_length": 2048,
                "domains": ["creative"],
                "chat_format": "llama-3",
                "system_prompt": "You are TARA, a creative and inspiring AI assistant."
            },
            "qwen-2.5": {
                "file": "qwen2.5-3b-instruct-q4_0.gguf",
                "context_length": 4096,
                "domains": ["universal"],
                "chat_format": "chatml",
                "system_prompt": "You are TARA, an intelligent AI assistant."
            }
        }
        
    def get_model_path(self, model_name: str) -> Path:
        """Get the full path to a GGUF model file."""
        model_file = self.model_configs[model_name]["file"]
        return Path("models/gguf") / model_file
        
    def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """Load a GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available")
            return None
            
        if model_name in self.models:
            return self.models[model_name]
            
        model_path = self.get_model_path(model_name)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
            
        try:
            config = self.model_configs[model_name]
            
            # Default parameters optimized for each model
            model_params = {
                "model_path": str(model_path),
                "n_ctx": config["context_length"],
                "n_threads": kwargs.get("n_threads", 8),
                "n_gpu_layers": kwargs.get("n_gpu_layers", 0),  # CPU by default
                "verbose": False,
                "chat_format": config.get("chat_format"),
            }
            
            # Override with user parameters
            model_params.update(kwargs)
            
            logger.info(f"Loading GGUF model: {model_name} from {model_path}")
            model = Llama(**model_params)
            
            self.models[model_name] = model
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def get_model_for_domain(self, domain: str) -> Optional[str]:
        """Get the best model for a specific domain."""
        for model_name, config in self.model_configs.items():
            if domain in config["domains"]:
                return model_name
        
        # Fallback to universal model
        for model_name, config in self.model_configs.items():
            if "universal" in config["domains"]:
                return model_name
                
        return None
    
    def generate_response(
        self, 
        model_name: str, 
        prompt: str, 
        domain: str = "universal",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using a GGUF model."""
        
        model = self.load_model(model_name)
        if not model:
            return {"error": f"Failed to load model: {model_name}"}
        
        try:
            config = self.model_configs[model_name]
            system_prompt = config["system_prompt"]
            
            # Format prompt based on chat format
            if config.get("chat_format") == "phi-3":
                formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            elif config.get("chat_format") == "llama-3":
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif config.get("chat_format") == "chatml":
                formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Generate response
            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
                stop=kwargs.get("stop", ["<|end|>", "<|eot_id|>", "<|im_end|>", "\n\nUser:", "\n\nHuman:"]),
                echo=False
            )
            
            return {
                "response": response["choices"][0]["text"].strip(),
                "model": model_name,
                "domain": domain,
                "tokens_used": response["usage"]["total_tokens"],
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"]
            }
            
        except Exception as e:
            logger.error(f"Error generating response with {model_name}: {e}")
            return {"error": str(e)}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available GGUF models."""
        available = []
        for model_name, config in self.model_configs.items():
            model_path = self.get_model_path(model_name)
            available.append({
                "name": model_name,
                "file": config["file"],
                "available": model_path.exists(),
                "size_mb": round(model_path.stat().st_size / (1024*1024), 1) if model_path.exists() else 0,
                "domains": config["domains"],
                "context_length": config["context_length"]
            })
        return available
    
    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all models from memory."""
        self.models.clear()
        logger.info("Unloaded all models")


class TARAGGUFModel:
    """Main TARA GGUF Model interface."""
    
    def __init__(self, config: TARAConfig):
        self.config = config
        self.manager = GGUFModelManager(config)
        
    def chat(
        self, 
        message: str, 
        domain: str = "universal",
        model_preference: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with TARA using GGUF models."""
        
        # Determine which model to use
        if model_preference and model_preference in self.manager.model_configs:
            model_name = model_preference
        else:
            model_name = self.manager.get_model_for_domain(domain)
            
        if not model_name:
            return {"error": "No suitable model found for domain"}
        
        return self.manager.generate_response(
            model_name=model_name,
            prompt=message,
            domain=domain,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "available_models": self.manager.get_available_models(),
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "total_models": len(self.manager.model_configs)
        } 