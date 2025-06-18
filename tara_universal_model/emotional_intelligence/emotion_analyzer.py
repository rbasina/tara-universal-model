"""
Simple Emotion Analyzer for TARA
================================

A basic emotion analysis system for processing user emotions
in the context of the reinforcement learning system.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime

class EmotionAnalyzer:
    """Simple emotion analyzer for TARA's reinforcement learning system."""
    
    def __init__(self):
        # Basic emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'excellent'],
            'sadness': ['sad', 'disappointed', 'upset', 'down', 'depressed', 'unhappy', 'terrible'],
            'anger': ['angry', 'mad', 'frustrated', 'annoyed', 'furious', 'irritated', 'hate'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'concerned', 'frightened'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'],
            'neutral': ['okay', 'fine', 'normal', 'average', 'standard', 'regular']
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'high': ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally'],
            'medium': ['quite', 'fairly', 'somewhat', 'rather', 'pretty'],
            'low': ['a bit', 'slightly', 'a little', 'kind of', 'sort of']
        }
    
    def analyze_emotion(self, text: str) -> Dict:
        """
        Analyze emotion in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not text:
            return self._default_emotion_context()
        
        text_lower = text.lower()
        
        # Detect emotions
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score / len(keywords)  # Normalize
        
        # Find primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[primary_emotion] == 0:
            primary_emotion = 'neutral'
        
        # Detect intensity
        intensity = self._detect_intensity(text_lower)
        
        # Check if empathy is required
        requires_empathy = self._requires_empathy(text_lower, primary_emotion)
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'intensity': intensity,
            'confidence': min(emotion_scores[primary_emotion] * 2, 1.0),
            'requires_empathy': requires_empathy,
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_intensity(self, text: str) -> str:
        """Detect emotional intensity in text."""
        for intensity, modifiers in self.intensity_modifiers.items():
            if any(modifier in text for modifier in modifiers):
                return intensity
        return 'medium'
    
    def _requires_empathy(self, text: str, primary_emotion: str) -> bool:
        """Determine if the text requires empathetic response."""
        empathy_triggers = [
            'help', 'problem', 'issue', 'trouble', 'difficult', 'hard',
            'confused', 'lost', 'stuck', 'wrong', 'error', 'mistake'
        ]
        
        # Check for empathy triggers
        has_triggers = any(trigger in text for trigger in empathy_triggers)
        
        # Negative emotions often require empathy
        negative_emotions = ['sadness', 'anger', 'fear']
        is_negative = primary_emotion in negative_emotions
        
        return has_triggers or is_negative
    
    def _default_emotion_context(self) -> Dict:
        """Return default emotion context."""
        return {
            'primary_emotion': 'neutral',
            'emotion_scores': {emotion: 0.0 for emotion in self.emotion_keywords.keys()},
            'intensity': 'medium',
            'confidence': 0.5,
            'requires_empathy': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_conversation_emotion(self, conversation_history: List[Dict]) -> Dict:
        """
        Analyze emotion across a conversation.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Aggregated emotion analysis
        """
        if not conversation_history:
            return self._default_emotion_context()
        
        # Analyze each turn
        turn_emotions = []
        for turn in conversation_history:
            if turn.get('role') == 'user' and turn.get('content'):
                emotion = self.analyze_emotion(turn['content'])
                turn_emotions.append(emotion)
        
        if not turn_emotions:
            return self._default_emotion_context()
        
        # Aggregate emotions (simple average)
        aggregated_scores = {}
        for emotion in self.emotion_keywords.keys():
            scores = [e['emotion_scores'][emotion] for e in turn_emotions]
            aggregated_scores[emotion] = sum(scores) / len(scores)
        
        # Find dominant emotion
        primary_emotion = max(aggregated_scores, key=aggregated_scores.get)
        
        # Check if any turn requires empathy
        requires_empathy = any(e['requires_empathy'] for e in turn_emotions)
        
        # Average intensity
        intensities = [e['intensity'] for e in turn_emotions]
        intensity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for intensity in intensities:
            intensity_counts[intensity] += 1
        
        dominant_intensity = max(intensity_counts, key=intensity_counts.get)
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_scores': aggregated_scores,
            'intensity': dominant_intensity,
            'confidence': aggregated_scores[primary_emotion],
            'requires_empathy': requires_empathy,
            'conversation_length': len(conversation_history),
            'timestamp': datetime.now().isoformat()
        } 