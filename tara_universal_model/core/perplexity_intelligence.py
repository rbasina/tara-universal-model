"""
Phase 2: Perplexity Intelligence Enhancement
TARA Universal Model - Context-Aware Routing and Crisis Detection
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

class PerplexityIntelligence:
    """
    Phase 2 enhancement for TARA Universal Model
    Adds context-aware reasoning, crisis detection, and intelligent routing
    """
    
    def __init__(self):
        self.conversation_contexts = {}  # user_id -> context
        self.logger = logging.getLogger(__name__)
        
        # Crisis detection patterns
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
        
        self.crisis_responses = {
            'mental_health_emergency': {
                'message': "I'm very concerned about you. Please reach out immediately: 988 (Suicide & Crisis Lifeline) or text 'HELLO' to 741741.",
                'urgency': 'critical',
                'domain': 'healthcare'
            },
            'medical_emergency': {
                'message': "This sounds like a medical emergency. Please call 911 immediately or go to the nearest emergency room.",
                'urgency': 'critical', 
                'domain': 'healthcare'
            },
            'abuse_situation': {
                'message': "Your safety is important. If in immediate danger, call 911. For support: 1-800-799-7233 (National Domestic Violence Hotline).",
                'urgency': 'critical',
                'domain': 'healthcare'
            }
        }
    
    def process_message(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Main processing method with Perplexity Intelligence
        """
        try:
            # Get or create user context
            if user_id not in self.conversation_contexts:
                self.conversation_contexts[user_id] = {
                    'conversation_history': [],
                    'current_domain': None,
                    'domain_switches': 0,
                    'crisis_history': []
                }
            
            user_context = self.conversation_contexts[user_id]
            
            # Crisis detection (highest priority)
            crisis_info = self.detect_crisis(message)
            if crisis_info['is_crisis']:
                self.logger.warning(f"Crisis detected for user {user_id}: {crisis_info['type']}")
                
                return {
                    'domain': crisis_info['domain'],
                    'confidence': 1.0,
                    'crisis_detected': True,
                    'crisis_type': crisis_info['type'],
                    'crisis_response': crisis_info['message'],
                    'urgency': crisis_info['urgency'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Context-aware domain routing
            routing_result = self.intelligent_routing(message, user_context)
            
            # Update conversation context
            self.update_context(user_id, message, routing_result)
            
            return routing_result
            
        except Exception as e:
            self.logger.error(f"Error in process_message: {e}")
            return {
                'domain': 'healthcare',  # Safe fallback
                'confidence': 0.5,
                'crisis_detected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def detect_crisis(self, message: str) -> Dict[str, Any]:
        """Enhanced crisis detection with specific response protocols"""
        message_lower = message.lower()
        
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    crisis_response = self.crisis_responses[crisis_type]
                    return {
                        'is_crisis': True,
                        'type': crisis_type,
                        'message': crisis_response['message'],
                        'urgency': crisis_response['urgency'],
                        'domain': crisis_response['domain']
                    }
        
        return {'is_crisis': False}
    
    def intelligent_routing(self, message: str, user_context: Dict) -> Dict[str, Any]:
        """Context-aware intelligent routing with conversation memory"""
        
        # Simple domain detection patterns
        domain_patterns = {
            'healthcare': [
                r'\b(health|medical|doctor|hospital|medicine|symptoms|diagnosis|treatment|therapy|wellness|pain|illness|disease)\b'
            ],
            'business': [
                r'\b(business|company|management|strategy|marketing|sales|finance|revenue|profit|investment|startup)\b'
            ],
            'education': [
                r'\b(learn|study|education|school|university|student|teacher|course|lesson|homework|exam|research)\b'
            ],
            'creative': [
                r'\b(creative|art|design|music|writing|story|poem|draw|paint|compose|imagine|inspiration)\b'
            ],
            'leadership': [
                r'\b(leadership|manage|team|lead|motivate|inspire|guide|mentor|coach|delegate|organize)\b'
            ]
        }
        
        message_lower = message.lower()
        domain_scores = {}
        
        # Score based on keyword patterns
        for domain, patterns in domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower, re.IGNORECASE))
                score += matches
            domain_scores[domain] = score
        
        # Apply context weighting
        current_domain = user_context.get('current_domain')
        if current_domain and current_domain in domain_scores:
            domain_scores[current_domain] += 0.5  # Continuity bonus
        
        # Select best domain
        if not domain_scores or max(domain_scores.values()) == 0:
            suggested_domain = 'healthcare'  # Safe default
            confidence = 0.5
        else:
            suggested_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(0.9, 0.5 + (domain_scores[suggested_domain] * 0.1))
        
        # Determine if domain switch is beneficial
        should_switch = self.should_switch_domain(
            current_domain, suggested_domain, confidence, user_context
        )
        
        final_domain = suggested_domain if should_switch else (current_domain or suggested_domain)
        
        return {
            'domain': final_domain,
            'confidence': confidence,
            'crisis_detected': False,
            'domain_switched': should_switch,
            'suggested_domain': suggested_domain,
            'reasoning': f"Selected {final_domain} domain with {confidence:.1%} confidence",
            'timestamp': datetime.now().isoformat()
        }
    
    def should_switch_domain(self, current_domain: str, suggested_domain: str, 
                           confidence: float, user_context: Dict) -> bool:
        """Intelligent domain switching decision"""
        
        if not current_domain:
            return True
        
        if current_domain == suggested_domain:
            return False
        
        # High confidence always switches
        if confidence > 0.8:
            return True
        
        # Low confidence never switches
        if confidence < 0.6:
            return False
        
        # Consider conversation length
        conversation_length = len(user_context.get('conversation_history', []))
        
        if conversation_length < 3:
            return confidence > 0.65
        else:
            return confidence > 0.75
    
    def update_context(self, user_id: str, message: str, routing_result: Dict):
        """Update user conversation context"""
        
        user_context = self.conversation_contexts[user_id]
        
        # Track domain switches
        current_domain = user_context.get('current_domain')
        new_domain = routing_result['domain']
        
        if current_domain and current_domain != new_domain:
            user_context['domain_switches'] += 1
        
        user_context['current_domain'] = new_domain
        
        # Add to conversation history
        user_context['conversation_history'].append({
            'message': message,
            'domain': new_domain,
            'confidence': routing_result['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 turns for memory efficiency
        if len(user_context['conversation_history']) > 20:
            user_context['conversation_history'] = user_context['conversation_history'][-20:]


# Example usage
if __name__ == "__main__":
    pi = PerplexityIntelligence()
    
    test_messages = [
        "I have a headache and feel nauseous",
        "How can I improve my business strategy?",
        "I want to hurt myself",  # Crisis test
        "Can you help me learn calculus?",
        "I want to write a creative story"
    ]
    
    print("Testing Perplexity Intelligence:")
    for message in test_messages:
        result = pi.process_message(message, "test_user")
        print(f"'{message}' -> {result['domain']} (confidence: {result.get('confidence', 0):.2f})")
        if result.get('crisis_detected'):
            print(f"  CRISIS DETECTED: {result['crisis_type']}") 