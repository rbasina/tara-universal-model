"""
Integration adapter for TARA Universal Model.
Provides backward compatibility with tara-ai-companion and API interface for existing codebase.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
from pathlib import Path

from ..serving.model import TARAUniversalModel, ChatResponse, ChatMessage
from ..utils.config import get_config, TARAConfig

logger = logging.getLogger(__name__)

@dataclass
class AdapterConfig:
    """Configuration for the integration adapter."""
    model_path: Optional[str] = None
    config_path: str = "configs/config.yaml"
    default_domain: str = "universal"
    enable_emotion_detection: bool = True
    enable_domain_routing: bool = True

class TARAAdapter:
    """
    Integration adapter for TARA Universal Model.
    
    Provides a clean interface for integrating with tara-ai-companion
    while maintaining backward compatibility.
    """
    
    def __init__(self, config: Union[AdapterConfig, Dict] = None):
        """Initialize TARA adapter."""
        if isinstance(config, dict):
            self.adapter_config = AdapterConfig(**config)
        elif config is None:
            self.adapter_config = AdapterConfig()
        else:
            self.adapter_config = config
        
        # Load TARA configuration
        self.tara_config = get_config(self.adapter_config.config_path)
        
        # Initialize TARA model
        self.tara_model = None
        self.is_initialized = False
        
        logger.info("TARA adapter initialized")
    
    async def initialize(self) -> None:
        """Initialize the TARA model asynchronously."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing TARA Universal Model...")
            
            # Initialize TARA model
            self.tara_model = TARAUniversalModel(self.tara_config)
            
            # Load base model
            base_model_name = self.tara_config.base_model_name
            self.tara_model.load_base_model(base_model_name)
            
            # Load domain adapters if available
            if self.adapter_config.model_path:
                domain = self.adapter_config.default_domain
                self.tara_model.load_domain_adapter(domain, self.adapter_config.model_path)
            
            self.is_initialized = True
            logger.info("TARA model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TARA model: {e}")
            raise
    
    def initialize_sync(self) -> None:
        """Initialize the TARA model synchronously."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing TARA Universal Model...")
            
            # Initialize TARA model
            self.tara_model = TARAUniversalModel(self.tara_config)
            
            # Load base model
            base_model_name = self.tara_config.base_model_name
            self.tara_model.load_base_model(base_model_name)
            
            # Load domain adapters if available
            if self.adapter_config.model_path:
                domain = self.adapter_config.default_domain
                self.tara_model.load_domain_adapter(domain, self.adapter_config.model_path)
            
            self.is_initialized = True
            logger.info("TARA model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TARA model: {e}")
            raise
    
    def process_message(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a user message and return response.
        
        This is the main interface method for tara-ai-companion integration.
        
        Args:
            user_message: The user's input message
            context: Optional conversation context
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            self.initialize_sync()
        
        try:
            # Process with TARA model
            response = self.tara_model.process_user_input(user_message)
            
            # Convert to compatible format
            return {
                "response": response.message,
                "emotion_detected": response.emotion_detected,
                "domain_used": response.domain_used,
                "confidence": response.confidence_score,
                "processing_time": response.processing_time,
                "safe": response.safety_check_passed,
                "metadata": {
                    "model_version": "1.0.0",
                    "timestamp": response.processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "emotion_detected": {"primary_emotion": "neutral"},
                "domain_used": "universal",
                "confidence": 0.0,
                "processing_time": 0.0,
                "safe": True,
                "error": str(e)
            }
    
    async def process_message_async(self, user_message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Async version of process_message.
        
        Args:
            user_message: The user's input message
            context: Optional conversation context
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Run processing in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_message, user_message, context)
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return self.tara_config.supported_domains
    
    def switch_domain(self, domain: str) -> bool:
        """
        Switch to a specific domain.
        
        Args:
            domain: Target domain name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            return False
        
        if domain not in self.tara_config.supported_domains:
            logger.warning(f"Unsupported domain: {domain}")
            return False
        
        try:
            # Try to load domain adapter if available
            adapter_path = f"{self.tara_config.adapters_path}/{domain}"
            if Path(adapter_path).exists():
                self.tara_model.load_domain_adapter(domain, adapter_path)
            
            # Switch domain context
            self.tara_model.current_domain = domain
            logger.info(f"Switched to {domain} domain")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to {domain} domain: {e}")
            return False
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not self.is_initialized or not self.tara_model:
            return {"error": "Model not initialized"}
        
        return self.tara_model.get_conversation_summary()
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.is_initialized and self.tara_model:
            self.tara_model.clear_conversation()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "initialized": self.is_initialized,
            "current_domain": self.tara_model.current_domain if self.tara_model else None,
            "available_domains": self.get_available_domains(),
            "model_path": self.adapter_config.model_path,
            "config_path": self.adapter_config.config_path
        }
    
    def configure_emotions(self, enable: bool = True, threshold: float = 0.3) -> None:
        """Configure emotion detection settings."""
        self.adapter_config.enable_emotion_detection = enable
        if self.tara_model:
            self.tara_model.emotion_detector.config.threshold = threshold
    
    def configure_domains(self, enable: bool = True, default_domain: str = "universal") -> None:
        """Configure domain routing settings."""
        self.adapter_config.enable_domain_routing = enable
        self.adapter_config.default_domain = default_domain
    
    # Legacy compatibility methods for existing tara-ai-companion code
    
    def chat(self, message: str, user_id: Optional[str] = None) -> str:
        """
        Legacy chat method for backward compatibility.
        
        Args:
            message: User message
            user_id: Optional user identifier
            
        Returns:
            Response message string
        """
        result = self.process_message(message)
        return result.get("response", "I'm sorry, I couldn't process your request.")
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Legacy emotion analysis method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotion analysis results
        """
        if not self.is_initialized:
            self.initialize_sync()
        
        if self.tara_model and self.tara_model.emotion_detector:
            return self.tara_model.emotion_detector.detect_emotions(text)
        
        return {"primary_emotion": "neutral", "confidence": 0.5}
    
    def get_domain_suggestion(self, text: str) -> str:
        """
        Legacy domain suggestion method.
        
        Args:
            text: Text to analyze for domain
            
        Returns:
            Suggested domain name
        """
        if not self.is_initialized:
            self.initialize_sync()
        
        if self.tara_model and self.tara_model.domain_router:
            return self.tara_model.domain_router.route_domain(text)
        
        return self.adapter_config.default_domain
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "model_loaded": self.tara_model is not None,
            "version": "1.0.0",
            "timestamp": None  # Add timestamp if needed
        }

# Factory function for easy instantiation
def create_tara_adapter(model_path: Optional[str] = None, 
                       config_path: str = "configs/config.yaml",
                       domain: str = "universal") -> TARAAdapter:
    """
    Factory function to create a TARA adapter.
    
    Args:
        model_path: Path to trained model/adapter
        config_path: Path to configuration file
        domain: Default domain to use
        
    Returns:
        Configured TARAAdapter instance
    """
    adapter_config = AdapterConfig(
        model_path=model_path,
        config_path=config_path,
        default_domain=domain
    )
    
    return TARAAdapter(adapter_config)

# Context manager for automatic initialization and cleanup
class TARAAdapterContext:
    """Context manager for TARA adapter."""
    
    def __init__(self, adapter: TARAAdapter):
        self.adapter = adapter
    
    def __enter__(self) -> TARAAdapter:
        self.adapter.initialize_sync()
        return self.adapter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self.adapter.tara_model:
            self.adapter.clear_conversation()
        return False

# Async context manager
class AsyncTARAAdapterContext:
    """Async context manager for TARA adapter."""
    
    def __init__(self, adapter: TARAAdapter):
        self.adapter = adapter
    
    async def __aenter__(self) -> TARAAdapter:
        await self.adapter.initialize()
        return self.adapter
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self.adapter.tara_model:
            self.adapter.clear_conversation()
        return False 