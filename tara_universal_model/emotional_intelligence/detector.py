"""
Emotion detection module for TARA Universal Model.
Provides text-based emotion detection with professional context.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmotionConfig:
    """Configuration for emotion detection."""
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    threshold: float = 0.3
    professional_context: bool = True
    voice_detection_enabled: bool = False

class EmotionDetector:
    """
    Emotion detection system for professional AI interactions.
    
    Detects emotions in text and provides professional context
    for appropriate responses.
    """
    
    def __init__(self, config: EmotionConfig):
        """Initialize emotion detector."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Professional emotion mappings
        self.professional_emotions = {
            "joy": ["enthusiasm", "satisfaction", "optimism", "confidence"],
            "sadness": ["disappointment", "concern", "frustration", "worry"],
            "anger": ["irritation", "impatience", "dissatisfaction", "stress"],
            "fear": ["anxiety", "uncertainty", "hesitation", "caution"],
            "surprise": ["curiosity", "interest", "amazement", "confusion"],
            "disgust": ["disapproval", "skepticism", "rejection", "discomfort"],
            "neutral": ["professional", "focused", "analytical", "composed"]
        }
        
        # Emotion intensity keywords
        self.intensity_keywords = {
            "high": ["extremely", "very", "really", "totally", "completely", "absolutely"],
            "medium": ["quite", "fairly", "somewhat", "rather", "pretty"],
            "low": ["slightly", "a bit", "kind of", "sort of", "maybe"]
        }
        
        # Professional context keywords
        self.professional_contexts = {
            "healthcare": ["patient", "medical", "health", "treatment", "symptoms", "diagnosis"],
            "business": ["strategy", "revenue", "market", "competition", "profit", "growth"],
            "education": ["learning", "student", "teaching", "exam", "study", "knowledge"],
            "leadership": ["team", "management", "decision", "leadership", "responsibility", "vision"],
            "creative": ["creative", "design", "artistic", "innovative", "inspiration", "imagination"]
        }
        
        self.emotion_classifier = None
        self.load_emotion_model()
        
        logger.info("Emotion detector initialized successfully")
    
    def load_emotion_model(self) -> None:
        """Load the emotion classification model."""
        try:
            logger.info(f"Loading emotion model: {self.config.model_name}")
            
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            logger.info("Using rule-based emotion detection as fallback")
            self.emotion_classifier = None
    
    def detect_emotions(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Detect emotions in text with professional context.
        
        Args:
            text: Input text to analyze
            context: Optional professional context
            
        Returns:
            Dictionary with emotion analysis results
        """
        try:
            # Basic emotion detection
            if self.emotion_classifier:
                emotions = self._model_based_detection(text)
            else:
                emotions = self._rule_based_detection(text)
            
            # Enhance with professional context
            professional_analysis = self._analyze_professional_context(text, context)
            
            # Detect emotion intensity
            intensity = self._detect_intensity(text)
            
            # Create emotion trajectory (simplified)
            trajectory = self._analyze_emotion_trajectory(text)
            
            return {
                "primary_emotion": emotions["primary"],
                "emotion_scores": emotions["scores"],
                "professional_context": professional_analysis,
                "intensity": intensity,
                "confidence": emotions["confidence"],
                "trajectory": trajectory,
                "detected_keywords": self._extract_emotion_keywords(text),
                "requires_empathy": self._requires_empathetic_response(emotions["primary"], intensity)
            }
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return self._get_default_emotion_result()
    
    def _model_based_detection(self, text: str) -> Dict[str, Any]:
        """Use transformer model for emotion detection."""
        try:
            results = self.emotion_classifier(text)
            
            # Process results
            emotion_scores = {}
            for result in results[0]:  # results is a list with one element
                emotion_scores[result['label'].lower()] = result['score']
            
            # Get primary emotion
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            return {
                "primary": primary_emotion,
                "scores": emotion_scores,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Model-based detection failed: {e}")
            return self._rule_based_detection(text)
    
    def _rule_based_detection(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based emotion detection."""
        text_lower = text.lower()
        
        # Emotion keywords
        emotion_keywords = {
            "joy": ["happy", "excited", "great", "wonderful", "amazing", "fantastic", "love", "enjoy"],
            "sadness": ["sad", "disappointed", "upset", "down", "terrible", "awful", "bad"],
            "anger": ["angry", "mad", "frustrated", "annoyed", "furious", "irritated"],
            "fear": ["worried", "anxious", "scared", "nervous", "concerned", "afraid"],
            "surprise": ["wow", "amazing", "incredible", "unbelievable", "shocking"],
            "disgust": ["disgusting", "awful", "terrible", "horrible", "gross"],
            "neutral": ["okay", "fine", "normal", "regular", "standard"]
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score / len(keywords)  # Normalize
        
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        # If no clear emotion detected, default to neutral
        if confidence < 0.1:
            primary_emotion = "neutral"
            confidence = 0.8
        
        return {
            "primary": primary_emotion,
            "scores": emotion_scores,
            "confidence": confidence
        }
    
    def _analyze_professional_context(self, text: str, context: str = None) -> Dict[str, Any]:
        """Analyze professional context of the text."""
        text_lower = text.lower()
        
        # Detect professional domain from text
        domain_scores = {}
        for domain, keywords in self.professional_contexts.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        detected_domain = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else "general"
        
        # Professional indicators
        professional_indicators = [
            "please", "thank you", "appreciate", "understand", "consider",
            "professional", "business", "work", "meeting", "project"
        ]
        
        formality_score = sum(1 for indicator in professional_indicators if indicator in text_lower)
        formality_level = "high" if formality_score >= 3 else "medium" if formality_score >= 1 else "low"
        
        return {
            "detected_domain": detected_domain,
            "domain_confidence": domain_scores.get(detected_domain, 0),
            "formality_level": formality_level,
            "requires_professional_tone": formality_level in ["high", "medium"]
        }
    
    def _detect_intensity(self, text: str) -> str:
        """Detect emotion intensity from text."""
        text_lower = text.lower()
        
        # Count intensity indicators
        high_count = sum(1 for keyword in self.intensity_keywords["high"] if keyword in text_lower)
        medium_count = sum(1 for keyword in self.intensity_keywords["medium"] if keyword in text_lower)
        low_count = sum(1 for keyword in self.intensity_keywords["low"] if keyword in text_lower)
        
        # Check for punctuation intensity
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Determine intensity
        if high_count > 0 or exclamation_count >= 2 or caps_ratio > 0.3:
            return "high"
        elif medium_count > 0 or exclamation_count == 1 or caps_ratio > 0.1:
            return "medium"
        else:
            return "low"
    
    def _analyze_emotion_trajectory(self, text: str) -> List[str]:
        """Analyze emotion changes within the text (simplified)."""
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        trajectory = []
        
        for sentence in sentences:
            if sentence.strip():
                emotion_result = self._rule_based_detection(sentence.strip())
                trajectory.append(emotion_result["primary"])
        
        return trajectory
    
    def _extract_emotion_keywords(self, text: str) -> List[str]:
        """Extract emotion-related keywords from text."""
        text_lower = text.lower()
        keywords = []
        
        # Common emotion keywords
        emotion_words = [
            "happy", "sad", "angry", "excited", "worried", "confident",
            "frustrated", "satisfied", "disappointed", "anxious", "calm",
            "stressed", "relaxed", "motivated", "discouraged"
        ]
        
        for word in emotion_words:
            if word in text_lower:
                keywords.append(word)
        
        return keywords
    
    def _requires_empathetic_response(self, emotion: str, intensity: str) -> bool:
        """Determine if the emotion requires an empathetic response."""
        empathy_emotions = ["sadness", "fear", "anger", "disgust"]
        
        if emotion in empathy_emotions:
            return True
        
        if intensity == "high" and emotion != "joy":
            return True
        
        return False
    
    def _get_default_emotion_result(self) -> Dict[str, Any]:
        """Return default emotion result when detection fails."""
        return {
            "primary_emotion": "neutral",
            "emotion_scores": {"neutral": 0.8},
            "professional_context": {
                "detected_domain": "general",
                "domain_confidence": 0,
                "formality_level": "medium",
                "requires_professional_tone": True
            },
            "intensity": "medium",
            "confidence": 0.5,
            "trajectory": ["neutral"],
            "detected_keywords": [],
            "requires_empathy": False
        }
    
    def get_emotion_insights(self, emotion_history: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion patterns over time."""
        if not emotion_history:
            return {"message": "No emotion history available"}
        
        # Extract primary emotions
        emotions = [entry.get("primary_emotion", "neutral") for entry in emotion_history]
        
        # Calculate emotion distribution
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Detect patterns
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        emotion_volatility = len(set(emotions)) / len(emotions)
        
        # Professional recommendations
        recommendations = []
        if dominant_emotion in ["sadness", "fear", "anger"]:
            recommendations.append("Consider offering additional support or resources")
        
        if emotion_volatility > 0.7:
            recommendations.append("User shows high emotional variability - adapt responses accordingly")
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "volatility_score": emotion_volatility,
            "total_interactions": len(emotions),
            "recommendations": recommendations
        } 