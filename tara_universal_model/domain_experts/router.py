"""
Domain expert routing module for TARA Universal Model.
Handles professional domain classification and context switching.
"""

import logging
import yaml
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    """Configuration for domain routing."""
    domains_config_path: str = "configs/domains"
    default_domain: str = "universal"
    confidence_threshold: float = 0.6
    safety_enabled: bool = True

class DomainRouter:
    """
    Professional domain routing system.
    
    Routes conversations to appropriate domain experts and manages
    professional context switching with safety validation.
    """
    
    def __init__(self, config: DomainConfig):
        """Initialize domain router."""
        self.config = config
        self.domain_configs = {}
        self.domain_keywords = {}
        self.system_prompts = {}
        self.safety_rules = {}
        
        # Professional domains
        self.supported_domains = [
            "healthcare", "business", "education", 
            "creative", "leadership", "universal"
        ]
        
        # Load domain configurations
        self.load_domain_configs()
        
        logger.info("Domain router initialized successfully")
    
    def load_domain_configs(self) -> None:
        """Load domain-specific configurations."""
        try:
            # Initialize default configurations if config files don't exist
            self._initialize_default_configs()
            
            # Load from config files if they exist
            if os.path.exists(self.config.domains_config_path):
                for domain in self.supported_domains:
                    config_file = os.path.join(self.config.domains_config_path, f"{domain}.yaml")
                    if os.path.exists(config_file):
                        with open(config_file, 'r', encoding='utf-8') as f:
                            domain_config = yaml.safe_load(f)
                            self.domain_configs[domain] = domain_config
                            self.domain_keywords[domain] = domain_config.get('keywords', [])
                            self.system_prompts[domain] = domain_config.get('system_prompt', '')
                            self.safety_rules[domain] = domain_config.get('safety_rules', [])
            
            logger.info(f"Loaded configurations for {len(self.domain_configs)} domains")
            
        except Exception as e:
            logger.error(f"Error loading domain configs: {e}")
            self._initialize_default_configs()
    
    def _initialize_default_configs(self) -> None:
        """Initialize default domain configurations."""
        # Healthcare domain
        self.domain_keywords["healthcare"] = [
            "health", "medical", "doctor", "patient", "symptoms", "treatment",
            "medicine", "hospital", "diagnosis", "therapy", "wellness", "care",
            "medication", "nursing", "clinical", "healthcare", "medical advice"
        ]
        
        self.system_prompts["healthcare"] = """
        You are a healthcare communication assistant. You provide supportive, empathetic responses 
        while maintaining strict professional boundaries. You do NOT provide medical advice, 
        diagnoses, or treatment recommendations. Always encourage users to consult with qualified 
        healthcare professionals for medical concerns.
        """
        
        self.safety_rules["healthcare"] = [
            "Never provide medical diagnosis",
            "Never recommend specific treatments",
            "Always encourage professional consultation",
            "Maintain HIPAA-compliant language",
            "Avoid definitive medical statements"
        ]
        
        # Business domain
        self.domain_keywords["business"] = [
            "business", "strategy", "revenue", "profit", "market", "competition",
            "sales", "marketing", "finance", "management", "leadership", "team",
            "project", "growth", "investment", "customer", "client", "corporate"
        ]
        
        self.system_prompts["business"] = """
        You are a business strategy assistant. You provide professional insights on business 
        matters, strategic thinking, and professional communication. You maintain a professional 
        tone while being helpful and analytical.
        """
        
        self.safety_rules["business"] = [
            "Avoid specific financial advice",
            "No insider trading information",
            "Maintain professional confidentiality",
            "Provide general strategic guidance only"
        ]
        
        # Education domain
        self.domain_keywords["education"] = [
            "education", "learning", "student", "teacher", "study", "exam",
            "school", "university", "course", "lesson", "homework", "research",
            "academic", "knowledge", "tutorial", "textbook", "curriculum"
        ]
        
        self.system_prompts["education"] = """
        You are an educational support assistant. You help with learning, studying, and 
        educational guidance. You encourage critical thinking and provide explanations 
        that promote understanding rather than just giving answers.
        """
        
        self.safety_rules["education"] = [
            "Encourage original thinking",
            "Provide guidance, not direct answers",
            "Promote academic integrity",
            "Age-appropriate content"
        ]
        
        # Creative domain
        self.domain_keywords["creative"] = [
            "creative", "design", "art", "writing", "music", "artistic",
            "imagination", "inspiration", "brainstorm", "innovative", "craft",
            "visual", "creative writing", "storytelling", "poetry", "painting"
        ]
        
        self.system_prompts["creative"] = """
        You are a creative collaboration assistant. You help spark creativity, provide 
        artistic inspiration, and support creative processes. You encourage original 
        thinking and artistic expression.
        """
        
        self.safety_rules["creative"] = [
            "Respect intellectual property",
            "Encourage originality",
            "Maintain appropriate content",
            "Support creative expression"
        ]
        
        # Leadership domain
        self.domain_keywords["leadership"] = [
            "leadership", "management", "team", "decision", "responsibility",
            "vision", "coaching", "mentor", "executive", "manager", "leader",
            "delegation", "motivation", "performance", "development", "supervision"
        ]
        
        self.system_prompts["leadership"] = """
        You are an executive coaching assistant. You provide insights on leadership, 
        team management, and professional development. You help develop leadership 
        skills and strategic thinking.
        """
        
        self.safety_rules["leadership"] = [
            "Maintain professional boundaries",
            "Encourage ethical leadership",
            "Respect organizational confidentiality",
            "Promote inclusive leadership"
        ]
        
        # Universal domain (default)
        self.domain_keywords["universal"] = [
            "help", "question", "general", "information", "advice", "support"
        ]
        
        self.system_prompts["universal"] = """
        You are TARA, a professional AI assistant with emotional intelligence. You provide 
        helpful, accurate, and empathetic responses while maintaining professional standards. 
        You adapt your communication style to the user's emotional state and professional context.
        """
        
        self.safety_rules["universal"] = [
            "Maintain professional tone",
            "Provide accurate information",
            "Respect user privacy",
            "Encourage professional consultation when appropriate"
        ]
    
    def route_domain(self, text: str, current_domain: str = None, 
                    emotion_context: Dict = None, conversation_context: List[Dict] = None) -> str:
        """
        Route text to appropriate professional domain.
        
        Args:
            text: Input text to analyze
            current_domain: Current domain context
            emotion_context: Emotion analysis results
            
        Returns:
            Suggested domain name
        """
        try:
            # Calculate domain scores
            domain_scores = self._calculate_domain_scores(text)
            
            # Apply emotion context
            if emotion_context:
                domain_scores = self._apply_emotion_weighting(domain_scores, emotion_context)
            
            # Get highest scoring domain
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]
            
            # Check confidence threshold
            if best_score < self.config.confidence_threshold:
                logger.info(f"Low confidence ({best_score:.2f}) for domain routing, using current or default")
                return current_domain or self.config.default_domain
            
            # Domain switching logic
            if current_domain and current_domain != best_domain:
                # Only switch if significantly better
                current_score = domain_scores.get(current_domain, 0)
                if best_score - current_score < 0.2:  # Sticky domain behavior
                    return current_domain
            
            logger.info(f"Routed to domain: {best_domain} (confidence: {best_score:.2f})")
            return best_domain
            
        except Exception as e:
            logger.error(f"Error in domain routing: {e}")
            return current_domain or self.config.default_domain
    
    def _calculate_domain_scores(self, text: str) -> Dict[str, float]:
        """Calculate domain relevance scores for text."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact match gets full score
                if keyword.lower() in text_lower:
                    score += 1
                # Partial match gets partial score
                elif any(word in text_lower for word in keyword.split()):
                    score += 0.5
            
            # Normalize by keyword count
            domain_scores[domain] = score / len(keywords) if keywords else 0
        
        return domain_scores
    
    def _apply_emotion_weighting(self, domain_scores: Dict[str, float], 
                               emotion_context: Dict) -> Dict[str, float]:
        """Apply emotion-based weighting to domain scores."""
        primary_emotion = emotion_context.get("primary_emotion", "neutral")
        intensity = emotion_context.get("intensity", "medium")
        
        # Emotion-domain preferences
        emotion_domain_boost = {
            "sadness": {"healthcare": 0.2, "leadership": 0.1},
            "fear": {"healthcare": 0.2, "education": 0.1},
            "anger": {"business": 0.1, "leadership": 0.2},
            "joy": {"creative": 0.2, "business": 0.1},
            "surprise": {"education": 0.1, "creative": 0.1}
        }
        
        # Apply emotion-based boosts
        if primary_emotion in emotion_domain_boost:
            for domain, boost in emotion_domain_boost[primary_emotion].items():
                if domain in domain_scores:
                    domain_scores[domain] += boost
                    
                    # Increase boost for high intensity emotions
                    if intensity == "high":
                        domain_scores[domain] += boost * 0.5
        
        return domain_scores
    
    def get_system_prompt(self, domain: str, emotion_context: Dict = None) -> str:
        """Get domain-specific system prompt with emotion context."""
        base_prompt = self.system_prompts.get(domain, self.system_prompts["universal"])
        
        if not emotion_context:
            return base_prompt
        
        # Add emotion-aware instructions
        primary_emotion = emotion_context.get("primary_emotion", "neutral")
        intensity = emotion_context.get("intensity", "medium")
        requires_empathy = emotion_context.get("requires_empathy", False)
        
        emotion_instructions = ""
        
        if requires_empathy:
            emotion_instructions += "\nThe user appears to be experiencing some distress. Respond with empathy and understanding."
        
        if intensity == "high":
            emotion_instructions += "\nThe user is expressing strong emotions. Acknowledge their feelings appropriately."
        
        if primary_emotion in ["sadness", "fear", "anger"]:
            emotion_instructions += f"\nThe user seems to be feeling {primary_emotion}. Provide supportive and calming responses."
        
        return base_prompt + emotion_instructions
    
    def validate_domain_safety(self, text: str, domain: str) -> Dict[str, Any]:
        """Validate safety rules for domain-specific content."""
        safety_violations = []
        warnings = []
        
        safety_rules = self.safety_rules.get(domain, [])
        text_lower = text.lower()
        
        # Domain-specific safety checks
        if domain == "healthcare":
            medical_advice_patterns = [
                r'\byou should take\b', r'\btake this medication\b',
                r'\bdiagnosis is\b', r'\byou have\b.*\bdisease\b'
            ]
            
            for pattern in medical_advice_patterns:
                if re.search(pattern, text_lower):
                    safety_violations.append("Potential medical advice detected")
                    break
        
        elif domain == "business":
            financial_advice_patterns = [
                r'\binvest in\b', r'\bbuy.*stock\b', r'\bguaranteed profit\b'
            ]
            
            for pattern in financial_advice_patterns:
                if re.search(pattern, text_lower):
                    warnings.append("Financial advice detected")
                    break
        
        # General safety checks
        inappropriate_content = [
            "illegal", "harmful", "dangerous", "unethical"
        ]
        
        for content in inappropriate_content:
            if content in text_lower:
                safety_violations.append(f"Inappropriate content: {content}")
        
        return {
            "safe": len(safety_violations) == 0,
            "violations": safety_violations,
            "warnings": warnings,
            "domain": domain
        }
    
    def get_cross_domain_insights(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-domain patterns in conversation."""
        if not conversation_history:
            return {"message": "No conversation history available"}
        
        # Extract domains used
        domains_used = []
        for entry in conversation_history:
            domain = entry.get("domain_context", "universal")
            domains_used.append(domain)
        
        # Calculate domain distribution
        domain_counts = {}
        for domain in domains_used:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Detect domain switching patterns
        domain_switches = 0
        for i in range(1, len(domains_used)):
            if domains_used[i] != domains_used[i-1]:
                domain_switches += 1
        
        # Professional insights
        insights = []
        
        if len(set(domains_used)) > 3:
            insights.append("User shows diverse professional interests")
        
        if domain_switches > len(domains_used) * 0.3:
            insights.append("Frequent domain switching detected - user may have complex needs")
        
        primary_domain = max(domain_counts, key=domain_counts.get)
        if domain_counts[primary_domain] > len(domains_used) * 0.7:
            insights.append(f"Strong focus on {primary_domain} domain")
        
        return {
            "domains_used": list(set(domains_used)),
            "domain_distribution": domain_counts,
            "primary_domain": primary_domain,
            "domain_switches": domain_switches,
            "total_messages": len(domains_used),
            "insights": insights
        }
    
    def suggest_domain_expertise(self, user_query: str) -> List[str]:
        """Suggest relevant domain expertise for complex queries."""
        domain_scores = self._calculate_domain_scores(user_query)
        
        # Get top domains above threshold
        relevant_domains = [
            domain for domain, score in domain_scores.items() 
            if score > 0.3
        ]
        
        # Sort by relevance
        relevant_domains.sort(key=lambda d: domain_scores[d], reverse=True)
        
        return relevant_domains[:3]  # Return top 3 relevant domains 