"""
Universal Router - Phase 2 Perplexity Intelligence
TARA Universal Model - Domain Detection and Intelligent Routing
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class UniversalRouter:
    """
    Universal Router for TARA Universal Model Phase 2
    Implements Perplexity Intelligence with context-aware domain detection
    """
    
    def __init__(self, model_base_path: str = "models"):
        self.model_base_path = model_base_path
        self.domain_models = {}
        self.tokenizer = None
        self.context_manager = ContextManager()
        self.crisis_detector = CrisisDetector()
        
        # Domain detection keywords and patterns
        self.domain_patterns = {
            'healthcare': [
                r'\b(health|medical|doctor|hospital|medicine|symptoms|diagnosis|treatment|therapy|wellness|pain|illness|disease|prescription|medication)\b',
                r'\b(feeling sick|not well|headache|fever|anxiety|depression|stress|mental health)\b'
            ],
            'business': [
                r'\b(business|company|management|strategy|marketing|sales|finance|revenue|profit|investment|startup|entrepreneur)\b',
                r'\b(meeting|presentation|project|team|leadership|decision|budget|client|customer)\b'
            ],
            'education': [
                r'\b(learn|study|education|school|university|student|teacher|course|lesson|homework|exam|research|knowledge)\b',
                r'\b(explain|understand|help me learn|teach me|tutorial|how to)\b'
            ],
            'creative': [
                r'\b(creative|art|design|music|writing|story|poem|draw|paint|compose|imagine|inspiration|innovative)\b',
                r'\b(brainstorm|idea|create|make|build|craft|artistic|aesthetic)\b'
            ],
            'leadership': [
                r'\b(leadership|manage|team|lead|motivate|inspire|guide|mentor|coach|delegate|organize)\b',
                r'\b(conflict|communication|decision making|problem solving|vision|goals)\b'
            ]
        }
        
        self.logger = logging.getLogger(__name__)
        
    def load_domain_models(self):
        """Load all trained domain models"""
        try:
            # Load base tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load each domain model
            for domain in ['healthcare', 'business', 'education', 'creative', 'leadership']:
                try:
                    model_path = f"{self.model_base_path}/{domain}"
                    base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                    domain_model = PeftModel.from_pretrained(base_model, model_path)
                    self.domain_models[domain] = domain_model
                    self.logger.info(f"Loaded {domain} model successfully")
                except Exception as e:
                    self.logger.error(f"Failed to load {domain} model: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            
    def detect_domain(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Detect the most appropriate domain for the message
        Uses pattern matching and context awareness
        """
        message_lower = message.lower()
        domain_scores = {}
        
        # Score based on keyword patterns
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower, re.IGNORECASE))
                score += matches
            domain_scores[domain] = score
            
        # Consider conversation context
        if context and 'current_domain' in context:
            current_domain = context['current_domain']
            if current_domain in domain_scores:
                domain_scores[current_domain] += 1  # Slight preference for continuity
                
        # Crisis detection override
        if self.crisis_detector.detect_crisis(message):
            return 'healthcare'  # Route crisis to healthcare domain
            
        # Return domain with highest score, default to healthcare
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'healthcare'  # Default domain
            
        return max(domain_scores, key=domain_scores.get)
    
    def route_conversation(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Route conversation to appropriate domain and generate response
        """
        try:
            # Detect domain
            detected_domain = self.detect_domain(message, context)
            
            # Update context
            updated_context = self.context_manager.update_context(
                message, detected_domain, context
            )
            
            # Check for crisis
            crisis_info = self.crisis_detector.analyze_message(message)
            
            # Generate response
            if detected_domain in self.domain_models:
                response = self._generate_response(
                    message, detected_domain, updated_context
                )
            else:
                response = self._fallback_response(message, detected_domain)
                
            return {
                'response': response,
                'domain': detected_domain,
                'context': updated_context,
                'crisis_detected': crisis_info['is_crisis'],
                'crisis_level': crisis_info.get('level', 'none'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in route_conversation: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again.",
                'domain': 'healthcare',
                'context': context or {},
                'crisis_detected': False,
                'crisis_level': 'none',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_response(self, message: str, domain: str, context: Dict) -> str:
        """Generate response using the appropriate domain model"""
        try:
            model = self.domain_models[domain]
            
            # Prepare input with context
            conversation_history = context.get('conversation_history', [])
            input_text = self._prepare_input(message, conversation_history)
            
            # Tokenize input
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(input_text):].strip()
            
            return response if response else f"I understand you're asking about {domain}. How can I help you further?"
            
        except Exception as e:
            self.logger.error(f"Error generating response for {domain}: {e}")
            return f"I'm here to help with {domain} matters. Could you please rephrase your question?"
    
    def _prepare_input(self, message: str, history: List[Dict]) -> str:
        """Prepare input text with conversation history"""
        context_messages = []
        
        # Add recent conversation history (last 3 exchanges)
        for exchange in history[-3:]:
            if 'user' in exchange:
                context_messages.append(f"User: {exchange['user']}")
            if 'assistant' in exchange:
                context_messages.append(f"Assistant: {exchange['assistant']}")
                
        context_messages.append(f"User: {message}")
        context_messages.append("Assistant:")
        
        return "\n".join(context_messages)
    
    def _fallback_response(self, message: str, domain: str) -> str:
        """Fallback response when domain model is not available"""
        return f"I understand you're asking about {domain}. While I'm still learning in this area, I'm here to help. Could you provide more details about what you need?"


class ContextManager:
    """
    Manages conversation context and memory across sessions
    """
    
    def __init__(self):
        self.conversation_memory = []
        self.user_profile = {}
        self.session_context = {}
        
    def update_context(self, message: str, domain: str, context: Optional[Dict] = None) -> Dict:
        """Update conversation context with new message"""
        current_context = context.copy() if context else {}
        
        # Update current domain
        current_context['current_domain'] = domain
        current_context['last_message'] = message
        current_context['timestamp'] = datetime.now().isoformat()
        
        # Update conversation history
        if 'conversation_history' not in current_context:
            current_context['conversation_history'] = []
            
        current_context['conversation_history'].append({
            'user': message,
            'domain': domain,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges to manage memory
        if len(current_context['conversation_history']) > 10:
            current_context['conversation_history'] = current_context['conversation_history'][-10:]
            
        return current_context
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences and patterns"""
        return self.user_profile.get(user_id, {})
    
    def update_user_profile(self, user_id: str, preferences: Dict):
        """Update user profile with new preferences"""
        if user_id not in self.user_profile:
            self.user_profile[user_id] = {}
        self.user_profile[user_id].update(preferences)


class CrisisDetector:
    """
    Detects crisis situations and emergency scenarios
    """
    
    def __init__(self):
        self.crisis_patterns = {
            'mental_health_emergency': [
                r'\b(suicide|kill myself|end it all|want to die|hurt myself|self harm)\b',
                r'\b(can\'t go on|no point|hopeless|worthless|better off dead)\b'
            ],
            'medical_emergency': [
                r'\b(chest pain|heart attack|can\'t breathe|stroke|emergency|911|ambulance)\b',
                r'\b(severe pain|bleeding|unconscious|overdose|poisoning)\b'
            ],
            'abuse_situation': [
                r'\b(being hurt|abuse|violence|scared|help me|threatening)\b',
                r'\b(unsafe|danger|hitting|attacking)\b'
            ]
        }
        
        self.emergency_responses = {
            'mental_health_emergency': {
                'message': "I'm very concerned about you. Please reach out to a mental health professional immediately. In the US, you can call 988 (Suicide & Crisis Lifeline) or text 'HELLO' to 741741.",
                'level': 'critical'
            },
            'medical_emergency': {
                'message': "This sounds like a medical emergency. Please call 911 (or your local emergency number) immediately or go to the nearest emergency room.",
                'level': 'critical'
            },
            'abuse_situation': {
                'message': "Your safety is important. If you're in immediate danger, please call 911. For domestic violence support, you can call 1-800-799-7233 (National Domestic Violence Hotline).",
                'level': 'critical'
            }
        }
    
    def detect_crisis(self, message: str) -> bool:
        """Quick crisis detection for routing decisions"""
        message_lower = message.lower()
        
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return True
        return False
    
    def analyze_message(self, message: str) -> Dict:
        """Detailed crisis analysis with response recommendations"""
        message_lower = message.lower()
        
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return {
                        'is_crisis': True,
                        'type': crisis_type,
                        'level': self.emergency_responses[crisis_type]['level'],
                        'recommended_response': self.emergency_responses[crisis_type]['message']
                    }
        
        return {
            'is_crisis': False,
            'type': 'none',
            'level': 'none'
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize router
    router = UniversalRouter()
    
    # Test domain detection
    test_messages = [
        "I have a headache and feel nauseous",
        "How can I improve my business strategy?",
        "Can you help me learn calculus?",
        "I want to write a creative story",
        "How do I motivate my team better?"
    ]
    
    print("Testing domain detection:")
    for message in test_messages:
        domain = router.detect_domain(message)
        print(f"'{message}' -> {domain}") 