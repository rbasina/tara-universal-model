"""
Model Integration Layer - TARA Universal Model
Loads trained domain models and serves intelligent responses to MeeTARA frontend
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from ..core.perplexity_intelligence import PerplexityIntelligence

class ModelIntegrationService:
    """
    Service that loads trained domain models and provides intelligent routing
    """
    
    def __init__(self, models_path: str = "models"):
        self.models_path = models_path
        self.base_model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.domain_models = {}
        self.perplexity_intelligence = PerplexityIntelligence()
        self.logger = logging.getLogger(__name__)
        
        # Track which models are available
        self.domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
        
    def initialize_service(self) -> Dict[str, Any]:
        """Initialize the model integration service"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check and load available models
            available_models = self.check_available_models()
            loaded_models = self.load_available_models(available_models)
            
            return {
                'status': 'success',
                'available_models': available_models,
                'loaded_models': loaded_models,
                'ready_for_requests': len(loaded_models) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return {'status': 'error', 'error': str(e), 'ready_for_requests': False}
    
    def check_available_models(self) -> List[str]:
        """Check which domain models have completed training"""
        available = []
        
        for domain in self.domains:
            model_path = os.path.join(self.models_path, domain)
            if os.path.exists(model_path):
                required_files = ['adapter_config.json', 'adapter_model.safetensors']
                if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                    available.append(domain)
                    self.logger.info(f"✅ {domain} model ready")
                else:
                    self.logger.info(f"⏳ {domain} model incomplete")
            else:
                self.logger.info(f"⏳ {domain} model not found")
        
        return available
    
    def load_available_models(self, available_models: List[str]) -> List[str]:
        """Load all available trained models"""
        loaded = []
        
        for domain in available_models:
            try:
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
                
                # Load LoRA adapter
                model_path = os.path.join(self.models_path, domain)
                domain_model = PeftModel.from_pretrained(base_model, model_path)
                
                self.domain_models[domain] = domain_model
                loaded.append(domain)
                self.logger.info(f"✅ {domain} model loaded")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {domain}: {e}")
        
        return loaded
    
    def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Main API: Process user message with intelligent routing
        This is what MeeTARA calls
        """
        try:
            # Intelligent domain detection
            routing = self.perplexity_intelligence.process_message(message, user_id)
            
            # Handle crisis
            if routing.get('crisis_detected'):
                return {
                    'success': True,
                    'response': routing.get('crisis_response', ''),
                    'domain': 'healthcare',
                    'crisis_detected': True,
                    'crisis_type': routing.get('crisis_type')
                }
            
            # Generate response
            domain = routing['domain']
            if domain in self.domain_models:
                response = self.generate_response(message, domain)
            else:
                response = self.get_fallback_response(domain)
            
            return {
                'success': True,
                'response': response,
                'domain': domain,
                'confidence': routing.get('confidence', 0.5),
                'crisis_detected': False
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                'success': False,
                'response': "I'm experiencing technical difficulties. Please try again.",
                'error': str(e)
            }
    
    def generate_response(self, message: str, domain: str) -> str:
        """Generate response using trained domain model"""
        try:
            model = self.domain_models[domain]
            
            # Prepare input
            input_text = f"User: {message}\nAssistant:"
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(input_text):].strip()
            
            return response if response else self.get_default_response(domain)
            
        except Exception as e:
            self.logger.error(f"Error generating {domain} response: {e}")
            return self.get_default_response(domain)
    
    def get_fallback_response(self, domain: str) -> str:
        """Fallback when model not available"""
        fallbacks = {
            'healthcare': "I understand you're reaching out about health matters. While I'm still learning, I'm here to support you.",
            'business': "I can help with business matters. While developing my expertise, I'm ready to assist.",
            'education': "I'm here to help with learning. Let me know what you'd like to explore.",
            'creative': "I love supporting creativity! I'm excited to help with your projects.",
            'leadership': "Leadership is important. I'm here to help develop your skills."
        }
        return fallbacks.get(domain, "I'm here to help! What can I assist you with?")
    
    def get_default_response(self, domain: str) -> str:
        """Default response when generation fails"""
        defaults = {
            'healthcare': "I'm here to support your wellbeing. How can I help?",
            'business': "I'm ready to assist with your business needs.",
            'education': "I'm here to support your learning journey.",
            'creative': "I'm excited to collaborate on your creative projects.",
            'leadership': "I'm here to help develop your leadership skills."
        }
        return defaults.get(domain, "I'm here to help. How can I assist you?")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'models_loaded': len(self.domain_models),
            'available_domains': list(self.domain_models.keys()),
            'ready_for_production': len(self.domain_models) >= 3
        }


# API wrapper for MeeTARA integration
class TaraApiService:
    """API service for MeeTARA frontend integration"""
    
    def __init__(self):
        self.model_service = ModelIntegrationService()
        self.initialized = False
        
    def start(self) -> Dict[str, Any]:
        """Start TARA service"""
        result = self.model_service.initialize_service()
        self.initialized = result.get('ready_for_requests', False)
        return result
    
    def chat(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Main chat endpoint for MeeTARA"""
        if not self.initialized:
            return {
                'success': False,
                'response': "TARA is initializing. Please wait...",
                'status': 'initializing'
            }
        
        return self.model_service.process_message(message, user_id)
    
    def status(self) -> Dict[str, Any]:
        """Status endpoint"""
        return self.model_service.get_status()


# Example usage for testing
if __name__ == "__main__":
    # Test the integration service
    service = TaraApiService()
    
    print("🚀 Starting TARA Model Integration Service...")
    start_result = service.start()
    print(f"Initialization: {start_result}")
    
    if start_result.get('ready_for_requests'):
        # Test chat
        test_messages = [
            "I have a headache and feel stressed",
            "How can I improve my business strategy?",
            "I want to learn about leadership"
        ]
        
        for message in test_messages:
            print(f"\nTesting: {message}")
            result = service.chat(message, "test_user")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Domain: {result.get('domain', 'unknown')}")
    else:
        print("Service not ready - models still training") 