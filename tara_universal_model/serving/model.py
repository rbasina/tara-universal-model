"""
Main TARA Universal Model serving module.
Integrates emotion detection and domain routing for professional AI assistance.
"""

import logging
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel, LoraConfig

from ..emotional_intelligence.detector import EmotionDetector
from ..domain_experts.router import DomainRouter
from ..utils.config import TARAConfig

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    emotion_context: Optional[Dict[str, Any]] = None
    domain_context: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class ChatResponse:
    """Chat response with metadata."""
    message: str
    confidence_score: float
    emotion_detected: Dict[str, Any]
    domain_used: str
    safety_check_passed: bool
    processing_time: float

class TARAUniversalModel:
    """
    Main TARA Universal Model class.
    
    Provides privacy-first conversational AI with emotional intelligence
    and professional domain expertise.
    """
    
    def __init__(self, config: TARAConfig):
        """Initialize TARA Universal Model."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.emotion_detector = EmotionDetector(config.emotion_config)
        self.domain_router = DomainRouter(config.domain_config)
        
        # Model components
        self.tokenizer = None
        self.base_model = None
        self.domain_adapters = {}
        
        # Chat context
        self.conversation_history = []
        self.current_domain = "universal"
        self.user_emotion_profile = {}
        
        logger.info("TARA Universal Model initialized successfully")
    
    def load_base_model(self, model_name: str = None) -> None:
        """Load the base model and tokenizer."""
        model_name = model_name or self.config.base_model_name
        
        logger.info(f"Loading base model: {model_name}")
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if self.config.use_quantization else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        logger.info(f"Base model loaded successfully on {self.device}")
    
    def load_domain_adapter(self, domain: str, adapter_path: str = None) -> None:
        """Load domain-specific LoRA adapter."""
        if not adapter_path:
            adapter_path = f"{self.config.adapters_path}/{domain}"
        
        try:
            logger.info(f"Loading {domain} domain adapter from {adapter_path}")
            
            # Load LoRA adapter
            adapter_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=domain
            )
            
            self.domain_adapters[domain] = adapter_model
            logger.info(f"Domain adapter '{domain}' loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load {domain} adapter: {e}")
            logger.info(f"Using base model for {domain} domain")
    
    def process_user_input(self, user_input: str) -> ChatResponse:
        """
        Process user input with emotion detection and domain routing.
        """
        import time
        start_time = time.time()
        
        try:
            # Detect emotions in user input
            emotion_context = self.emotion_detector.detect_emotions(user_input)
            
            # Route to appropriate domain
            suggested_domain = self.domain_router.route_domain(
                user_input, 
                current_domain=self.current_domain,
                emotion_context=emotion_context
            )
            
            # Switch domain if needed
            if suggested_domain != self.current_domain:
                self.current_domain = suggested_domain
            
            # Generate response
            response_text = self._generate_response(user_input, emotion_context)
            
            # Safety check
            safety_passed = self._safety_check(response_text, self.current_domain)
            
            return ChatResponse(
                message=response_text,
                confidence_score=0.95,
                emotion_detected=emotion_context,
                domain_used=self.current_domain,
                safety_check_passed=safety_passed,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return ChatResponse(
                message="I apologize, but I encountered an error processing your request.",
                confidence_score=0.0,
                emotion_detected={},
                domain_used=self.current_domain,
                safety_check_passed=True,
                processing_time=time.time() - start_time
            )
    
    def _generate_response(self, user_input: str, emotion_context: Dict) -> str:
        """Generate response using the appropriate model."""
        # Build conversation context
        context = self._build_conversation_context(user_input, emotion_context)
        
        # Tokenize input
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length
        ).to(self.device)
        
        # Get current model (base or adapter)
        current_model = self.domain_adapters.get(self.current_domain, self.base_model)
        
        # Generate response
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            outputs = current_model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def _build_conversation_context(self, user_input: str, emotion_context: Dict) -> str:
        """Build conversation context with system prompt."""
        # Get domain-specific system prompt
        system_prompt = self.domain_router.get_system_prompt(
            self.current_domain,
            emotion_context
        )
        
        # Build conversation history
        context_parts = [system_prompt]
        
        # Add recent conversation history
        recent_history = self.conversation_history[-self.config.context_window:]
        for message in recent_history:
            if message.role == "user":
                context_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                context_parts.append(f"Assistant: {message.content}")
        
        # Add current user input
        context_parts.append(f"User: {user_input}")
        context_parts.append("Assistant:")
        
        return "\n\n".join(context_parts)
    
    def _safety_check(self, response: str, domain: str) -> bool:
        """Perform safety check on generated response."""
        # Basic safety checks
        safety_violations = [
            "harmful", "dangerous", "illegal", "unethical",
            "personal information", "confidential"
        ]
        
        response_lower = response.lower()
        for violation in safety_violations:
            if violation in response_lower:
                logger.warning(f"Safety violation detected: {violation}")
                return False
        
        # Domain-specific safety checks
        if domain == "healthcare":
            healthcare_violations = ["diagnosis", "treatment", "medication dosage"]
            for violation in healthcare_violations:
                if violation in response_lower:
                    logger.warning(f"Healthcare safety violation: {violation}")
                    return False
        
        return True
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        self.current_domain = "universal"
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        emotion_counts = {}
        domain_usage = {}
        
        for message in self.conversation_history:
            if message.emotion_context:
                emotion = message.emotion_context.get('primary_emotion', 'neutral')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if message.domain_context:
                domain_usage[message.domain_context] = domain_usage.get(message.domain_context, 0) + 1
        
        return {
            "message_count": len(self.conversation_history),
            "emotion_distribution": emotion_counts,
            "domain_usage": domain_usage,
            "current_domain": self.current_domain
        }
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation history to file."""
        import json
        from datetime import datetime
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "emotion_context": msg.emotion_context,
                    "domain_context": msg.domain_context
                }
                for msg in self.conversation_history
            ],
            "summary": self.get_conversation_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved to {filepath}") 