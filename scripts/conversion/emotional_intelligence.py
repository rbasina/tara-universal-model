#!/usr/bin/env python3
"""
💙 Emotional Intelligence Engine
Handles emotional intelligence and response modulation for TARA Universal Model
"""

import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class EmotionalResponse:
    tone: str
    modifiers: List[str]
    empathy_level: str
    response_style: str
    emotional_intensity: float

class EmotionalIntelligenceEngine:
    """Handles emotional intelligence and response modulation"""
    
    def __init__(self):
        self.emotion_models = {}
        self.response_templates = {}
        self.emotional_context_history = []
        self._load_emotional_models()
    
    def _load_emotional_models(self):
        """Load emotional intelligence models and templates"""
        
        # Emotional response templates with detailed configurations
        self.response_templates = {
            'joy': {
                'tone': 'enthusiastic',
                'modifiers': ['wonderful', 'amazing', 'fantastic', 'excellent'],
                'empathy_level': 'high',
                'response_style': 'celebratory',
                'emotional_intensity': 0.8
            },
            'sadness': {
                'tone': 'gentle',
                'modifiers': ['I understand', 'I\'m here for you', 'it\'s okay', 'you\'re not alone'],
                'empathy_level': 'very_high',
                'response_style': 'supportive',
                'emotional_intensity': 0.9
            },
            'anger': {
                'tone': 'calm',
                'modifiers': ['I hear you', 'let\'s work through this', 'I understand your frustration'],
                'empathy_level': 'high',
                'response_style': 'de-escalating',
                'emotional_intensity': 0.7
            },
            'fear': {
                'tone': 'reassuring',
                'modifiers': ['you\'re safe', 'we\'ll figure this out', 'I\'m here to help'],
                'empathy_level': 'very_high',
                'response_style': 'protective',
                'emotional_intensity': 0.9
            },
            'surprise': {
                'tone': 'excited',
                'modifiers': ['wow', 'that\'s incredible', 'amazing discovery'],
                'empathy_level': 'medium',
                'response_style': 'enthusiastic',
                'emotional_intensity': 0.6
            },
            'disgust': {
                'tone': 'neutral',
                'modifiers': ['I understand your concern', 'let\'s address this properly'],
                'empathy_level': 'medium',
                'response_style': 'objective',
                'emotional_intensity': 0.5
            },
            'trust': {
                'tone': 'confident',
                'modifiers': ['I appreciate your trust', 'we\'ll work together on this'],
                'empathy_level': 'high',
                'response_style': 'collaborative',
                'emotional_intensity': 0.7
            },
            'anticipation': {
                'tone': 'encouraging',
                'modifiers': ['I\'m excited for you', 'this will be great', 'let\'s make it happen'],
                'empathy_level': 'high',
                'response_style': 'motivational',
                'emotional_intensity': 0.8
            },
            'neutral': {
                'tone': 'professional',
                'modifiers': [],
                'empathy_level': 'medium',
                'response_style': 'informative',
                'emotional_intensity': 0.5
            }
        }
        
        # Domain-specific emotional responses
        self.domain_emotional_responses = {
            'healthcare': {
                'crisis': {
                    'tone': 'urgent_caring',
                    'modifiers': ['I\'m here immediately', 'let\'s address this right now', 'your safety is priority'],
                    'empathy_level': 'very_high',
                    'response_style': 'crisis_intervention',
                    'emotional_intensity': 1.0
                },
                'wellness': {
                    'tone': 'nurturing',
                    'modifiers': ['your health matters', 'let\'s build healthy habits', 'I care about your wellbeing'],
                    'empathy_level': 'very_high',
                    'response_style': 'supportive_care',
                    'emotional_intensity': 0.8
                }
            },
            'business': {
                'stress': {
                    'tone': 'calm_professional',
                    'modifiers': ['let\'s approach this strategically', 'we\'ll find the best solution'],
                    'empathy_level': 'high',
                    'response_style': 'problem_solving',
                    'emotional_intensity': 0.6
                },
                'success': {
                    'tone': 'celebratory_professional',
                    'modifiers': ['excellent work', 'this is a great achievement', 'well done'],
                    'empathy_level': 'high',
                    'response_style': 'recognition',
                    'emotional_intensity': 0.7
                }
            },
            'education': {
                'struggle': {
                    'tone': 'patient_encouraging',
                    'modifiers': ['learning takes time', 'you\'re making progress', 'let\'s break this down'],
                    'empathy_level': 'very_high',
                    'response_style': 'educational_support',
                    'emotional_intensity': 0.8
                },
                'breakthrough': {
                    'tone': 'excited_proud',
                    'modifiers': ['you did it!', 'this is fantastic progress', 'you\'ve got this'],
                    'empathy_level': 'high',
                    'response_style': 'celebration',
                    'emotional_intensity': 0.9
                }
            }
        }
    
    def analyze_emotional_context(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive emotional context analysis"""
        
        query_lower = query.lower()
        
        # Enhanced emotional keywords with intensity levels
        emotion_keywords = {
            'joy': {
                'keywords': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'joy'],
                'intensity_boosters': ['extremely', 'incredibly', 'absolutely', 'so much']
            },
            'sadness': {
                'keywords': ['sad', 'depressed', 'lonely', 'hurt', 'pain', 'crying', 'miss', 'lost'],
                'intensity_boosters': ['very', 'so', 'extremely', 'completely']
            },
            'anger': {
                'keywords': ['angry', 'frustrated', 'mad', 'upset', 'annoyed', 'hate', 'furious'],
                'intensity_boosters': ['very', 'extremely', 'completely', 'absolutely']
            },
            'fear': {
                'keywords': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'panic'],
                'intensity_boosters': ['very', 'extremely', 'completely', 'absolutely']
            },
            'surprise': {
                'keywords': ['wow', 'unexpected', 'shocked', 'surprised', 'incredible', 'unbelievable'],
                'intensity_boosters': ['completely', 'totally', 'absolutely']
            },
            'disgust': {
                'keywords': ['disgusting', 'gross', 'terrible', 'awful', 'horrible', 'nasty'],
                'intensity_boosters': ['absolutely', 'completely', 'totally']
            },
            'trust': {
                'keywords': ['trust', 'believe', 'confident', 'sure', 'reliable', 'dependable'],
                'intensity_boosters': ['completely', 'absolutely', 'fully']
            },
            'anticipation': {
                'keywords': ['hope', 'expect', 'look forward', 'plan', 'excited for', 'can\'t wait'],
                'intensity_boosters': ['really', 'so much', 'absolutely']
            }
        }
        
        # Calculate emotion scores with intensity
        emotion_scores = {}
        for emotion, config in emotion_keywords.items():
            base_score = sum(1 for keyword in config['keywords'] if keyword in query_lower)
            
            # Check for intensity boosters
            intensity_multiplier = 1.0
            for booster in config['intensity_boosters']:
                if booster in query_lower:
                    intensity_multiplier = 2.0
                    break
            
            emotion_scores[emotion] = base_score * intensity_multiplier
        
        # Normalize scores
        total = sum(emotion_scores.values()) or 1
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # Add user context if available
        if user_context:
            if 'emotional_state' in user_context:
                for emotion, intensity in user_context['emotional_state'].items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion] = (emotion_scores[emotion] + intensity) / 2
            
            if 'emotional_history' in user_context:
                # Consider emotional history for context
                history = user_context['emotional_history']
                if len(history) > 0:
                    recent_emotion = history[-1]['dominant_emotion']
                    if recent_emotion in emotion_scores:
                        emotion_scores[recent_emotion] *= 1.2  # Boost recent emotion
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate emotional stability
        sorted_emotions = sorted(emotion_scores.values(), reverse=True)
        emotional_stability = 1.0 - (sorted_emotions[0] - sorted_emotions[-1]) if len(sorted_emotions) > 1 else 1.0
        
        # Store in history
        emotional_context = {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': max(emotion_scores.values()),
            'emotional_stability': emotional_stability,
            'timestamp': time.time()
        }
        
        self.emotional_context_history.append(emotional_context)
        
        # Keep only last 10 entries
        if len(self.emotional_context_history) > 10:
            self.emotional_context_history = self.emotional_context_history[-10:]
        
        return emotional_context
    
    def modulate_response(self, response: str, emotional_context: Dict[str, Any], 
                         domain: str = None, user_context: Dict[str, Any] = None) -> str:
        """Modulate response based on emotional context and domain"""
        
        dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
        intensity = emotional_context.get('emotional_intensity', 0.0)
        
        # Get base template
        template = self.response_templates.get(dominant_emotion, self.response_templates['neutral'])
        
        # Check for domain-specific emotional response
        if domain and domain in self.domain_emotional_responses:
            domain_templates = self.domain_emotional_responses[domain]
            
            # Check for specific emotional scenarios
            if intensity > 0.8 and dominant_emotion in ['sadness', 'fear']:
                if 'crisis' in domain_templates:
                    template = domain_templates['crisis']
            elif intensity > 0.7 and dominant_emotion in ['joy', 'anticipation']:
                if 'success' in domain_templates:
                    template = domain_templates['success']
            elif intensity > 0.6 and dominant_emotion in ['anger', 'frustration']:
                if 'stress' in domain_templates:
                    template = domain_templates['stress']
        
        # Apply emotional modulation based on intensity
        modulated_response = response
        
        if intensity > 0.7:  # High emotional intensity
            if template['empathy_level'] in ['high', 'very_high'] and template['modifiers']:
                # Add emotional modifier at the beginning
                modifier = template['modifiers'][0]
                modulated_response = f"{modifier}, {response}"
            
            # Adjust tone based on emotional intensity
            if template['tone'] == 'gentle' and intensity > 0.9:
                modulated_response = f"I want you to know that {modulated_response.lower()}"
        
        elif intensity > 0.4:  # Medium emotional intensity
            if template['modifiers'] and len(template['modifiers']) > 1:
                # Add a subtle modifier
                modifier = template['modifiers'][1] if len(template['modifiers']) > 1 else template['modifiers'][0]
                if not modulated_response.startswith(modifier):
                    modulated_response = f"{modifier}. {modulated_response}"
        
        # Add emotional punctuation for very high intensity
        if intensity > 0.8:
            if dominant_emotion in ['joy', 'excitement']:
                modulated_response = modulated_response.replace('.', '!')
            elif dominant_emotion in ['sadness', 'fear']:
                modulated_response = modulated_response.replace('.', '...')
        
        return modulated_response
    
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get summary of emotional intelligence capabilities"""
        return {
            "supported_emotions": list(self.response_templates.keys()),
            "domain_specific_responses": list(self.domain_emotional_responses.keys()),
            "empathy_levels": ["low", "medium", "high", "very_high"],
            "response_styles": list(set(template['response_style'] for template in self.response_templates.values())),
            "emotional_context_history_length": len(self.emotional_context_history),
            "features": [
                "real_time_emotional_analysis",
                "context_aware_response_modulation",
                "domain_specific_emotional_responses",
                "emotional_stability_tracking",
                "intensity_based_modulation"
            ]
        }
    
    def save_emotional_config(self, output_path: Path):
        """Save emotional intelligence configuration"""
        config = {
            "response_templates": self.response_templates,
            "domain_emotional_responses": self.domain_emotional_responses,
            "emotional_context_history": self.emotional_context_history[-5:],  # Last 5 entries
            "capabilities": self.get_emotional_summary()
        }
        
        with open(output_path / "emotional_intelligence_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Emotional intelligence config saved to {output_path}")

# Import time for timestamp functionality
import time 