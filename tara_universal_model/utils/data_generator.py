"""
Synthetic data generation for TARA Universal Model training.
Creates domain-specific conversation templates and professional scenarios.
"""

import json
import os
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data generation."""
    output_dir: str = "data/synthetic"
    quality_threshold: float = 0.8
    diversity_threshold: float = 0.7
    samples_per_domain: int = 5000
    synthetic_data_path: str = "data/synthetic"

class DataGenerator:
    """
    Synthetic conversation data generator for TARA training.
    
    Generates professional conversations with emotional context
    for domain-specific training.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize data generator."""
        self.config = config
        self.domain_templates = {}
        self.emotion_patterns = {}
        
        # Initialize conversation templates
        self._initialize_templates()
        
        # Ensure output directory exists
        os.makedirs(self.config.synthetic_data_path, exist_ok=True)
        
        logger.info("Data generator initialized successfully")
    
    def _initialize_templates(self) -> None:
        """Initialize conversation templates for each domain."""
        
        # Healthcare domain templates
        self.domain_templates["healthcare"] = {
            "scenarios": [
                "patient_consultation", "staff_support", "health_education",
                "emergency_situation", "treatment_discussion", "wellness_check"
            ],
            "user_intents": [
                "seeking_medical_information", "expressing_health_concerns",
                "requesting_appointment", "asking_about_symptoms",
                "discussing_treatment_options", "seeking_emotional_support"
            ],
            "conversation_starters": [
                "I've been feeling anxious about my upcoming procedure",
                "My patient seems very worried about their diagnosis",
                "Can you help me understand these symptoms better?",
                "I'm struggling with work-life balance as a healthcare worker",
                "How should I communicate bad news to a patient's family?",
                "I need guidance on managing my stress levels"
            ],
            "response_patterns": [
                "empathetic_validation", "professional_guidance", 
                "information_sharing", "emotional_support", "referral_suggestion"
            ]
        }
        
        # Business domain templates
        self.domain_templates["business"] = {
            "scenarios": [
                "strategic_planning", "team_leadership", "client_management",
                "performance_review", "market_analysis", "crisis_management"
            ],
            "user_intents": [
                "seeking_business_advice", "discussing_strategy",
                "team_management_help", "market_insights",
                "leadership_guidance", "problem_solving"
            ],
            "conversation_starters": [
                "Our quarterly results are below expectations",
                "I'm having trouble motivating my remote team",
                "How should we approach this new market opportunity?",
                "I need to make some difficult budget decisions",
                "My client is unhappy with our service delivery",
                "I'm considering a major strategic pivot"
            ],
            "response_patterns": [
                "analytical_approach", "strategic_thinking",
                "leadership_coaching", "market_insights", "solution_oriented"
            ]
        }
        
        # Education domain templates  
        self.domain_templates["education"] = {
            "scenarios": [
                "student_tutoring", "exam_preparation", "learning_difficulties",
                "curriculum_planning", "academic_guidance", "motivation_support"
            ],
            "user_intents": [
                "understanding_concepts", "exam_preparation",
                "homework_help", "learning_strategies",
                "academic_planning", "motivation_seeking"
            ],
            "conversation_starters": [
                "I'm struggling to understand calculus concepts",
                "My students seem disengaged in remote learning",
                "How can I prepare effectively for my finals?",
                "I need help organizing my study schedule",
                "This research paper seems overwhelming",
                "I'm losing motivation to continue my studies"
            ],
            "response_patterns": [
                "educational_scaffolding", "concept_explanation",
                "study_strategies", "motivational_support", "resource_sharing"
            ]
        }
        
        # Creative domain templates
        self.domain_templates["creative"] = {
            "scenarios": [
                "creative_brainstorming", "writer_block", "artistic_feedback",
                "project_development", "creative_collaboration", "inspiration_seeking"
            ],
            "user_intents": [
                "seeking_inspiration", "creative_feedback",
                "brainstorming_ideas", "overcoming_blocks",
                "artistic_guidance", "project_planning"
            ],
            "conversation_starters": [
                "I'm experiencing writer's block on my novel",
                "I need fresh ideas for my art project",
                "How can I make my presentation more engaging?",
                "I'm struggling with the direction of my creative work",
                "Can you help me brainstorm marketing campaign ideas?",
                "I need feedback on my latest design concept"
            ],
            "response_patterns": [
                "creative_encouragement", "idea_generation",
                "constructive_feedback", "inspiration_sharing", "technique_suggestions"
            ]
        }
        
        # Leadership domain templates
        self.domain_templates["leadership"] = {
            "scenarios": [
                "team_management", "conflict_resolution", "decision_making",
                "performance_coaching", "organizational_change", "vision_setting"
            ],
            "user_intents": [
                "leadership_development", "team_building",
                "conflict_management", "strategic_decisions",
                "performance_improvement", "change_management"
            ],
            "conversation_starters": [
                "Two of my team members are in constant conflict",
                "I need to make a difficult decision about layoffs",
                "How can I better communicate our company vision?",
                "My team is resistant to the new changes",
                "I'm struggling with giving negative feedback",
                "How do I build trust with a new team?"
            ],
            "response_patterns": [
                "leadership_coaching", "conflict_mediation",
                "strategic_guidance", "team_dynamics", "communication_skills"
            ]
        }
        
        # Emotion patterns for realistic conversations
        self.emotion_patterns = {
            "stressed": ["overwhelmed", "anxious", "pressured", "tense"],
            "confident": ["assured", "optimistic", "determined", "focused"],
            "frustrated": ["annoyed", "stuck", "blocked", "irritated"],
            "curious": ["interested", "eager", "inquisitive", "engaged"],
            "concerned": ["worried", "troubled", "uncertain", "cautious"],
            "excited": ["enthusiastic", "energized", "motivated", "passionate"]
        }
    
    def generate_domain_data(self, domain: str, num_samples: int = 5000,
                           output_path: str = None, quality_threshold: float = 0.8,
                           templates: Dict = None, split_type: str = "train") -> str:
        """
        Generate synthetic conversation data for a specific domain.
        
        Args:
            domain: Target domain for data generation
            num_samples: Number of conversation samples to generate
            output_path: Output file path
            quality_threshold: Quality threshold for generated samples
            templates: Custom conversation templates
            split_type: Data split type ('train', 'val', 'test')
            
        Returns:
            Path to generated data file
        """
        if domain not in self.domain_templates:
            raise ValueError(f"Unsupported domain: {domain}")
        
        logger.info(f"Generating {num_samples} samples for {domain} domain")
        
        # Use custom templates if provided
        domain_config = templates or self.domain_templates[domain]
        
        # Generate conversations
        conversations = []
        for i in range(num_samples):
            try:
                conversation = self._generate_conversation(domain, domain_config)
                
                # Quality check
                if self._evaluate_quality(conversation) >= quality_threshold:
                    conversations.append(conversation)
                
                # Progress logging
                if (i + 1) % 500 == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue
        
        logger.info(f"Generated {len(conversations)} high-quality samples")
        
        # Save to file
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{split_type}_{timestamp}.json"
            output_path = os.path.join(self.config.synthetic_data_path, filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to: {output_path}")
        return output_path
    
    def _generate_conversation(self, domain: str, domain_config: Dict) -> Dict[str, Any]:
        """Generate a single conversation for the domain."""
        
        # Select scenario and emotion
        scenario = random.choice(domain_config["scenarios"])
        user_intent = random.choice(domain_config["user_intents"])
        starter = random.choice(domain_config["conversation_starters"])
        response_pattern = random.choice(domain_config["response_patterns"])
        
        # Select emotion context
        primary_emotion = random.choice(list(self.emotion_patterns.keys()))
        emotion_descriptors = self.emotion_patterns[primary_emotion]
        
        # Generate conversation turns
        conversation_turns = []
        
        # Initial user message
        user_message = self._personalize_message(starter, scenario, primary_emotion)
        conversation_turns.append({
            "role": "user",
            "content": user_message,
            "emotion": primary_emotion,
            "intent": user_intent
        })
        
        # Generate assistant response
        assistant_response = self._generate_assistant_response(
            user_message, domain, response_pattern, primary_emotion
        )
        conversation_turns.append({
            "role": "assistant", 
            "content": assistant_response,
            "domain": domain,
            "response_pattern": response_pattern
        })
        
        # Generate follow-up turns (2-4 additional exchanges)
        num_followups = random.randint(1, 3)
        for _ in range(num_followups):
            # User follow-up
            followup_user = self._generate_followup_user(
                conversation_turns, scenario, primary_emotion
            )
            conversation_turns.append({
                "role": "user",
                "content": followup_user,
                "emotion": primary_emotion
            })
            
            # Assistant follow-up
            followup_assistant = self._generate_followup_assistant(
                conversation_turns, domain, response_pattern
            )
            conversation_turns.append({
                "role": "assistant",
                "content": followup_assistant,
                "domain": domain
            })
        
        return {
            "conversation_id": str(uuid.uuid4()),
            "domain": domain,
            "scenario": scenario,
            "primary_emotion": primary_emotion,
            "user_intent": user_intent,
            "turns": conversation_turns,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "quality_score": 0.85,  # Placeholder
                "conversation_length": len(conversation_turns)
            }
        }
    
    def _personalize_message(self, starter: str, scenario: str, emotion: str) -> str:
        """Add personalization and emotion to conversation starter."""
        emotion_modifiers = {
            "stressed": ["I'm really", "I'm quite", "I feel so"],
            "confident": ["I'm", "I feel", "I'm fairly"],
            "frustrated": ["I'm getting", "I'm becoming", "I feel"],
            "curious": ["I'm", "I'm really", "I'm quite"],
            "concerned": ["I'm", "I'm somewhat", "I'm getting"],
            "excited": ["I'm so", "I'm really", "I'm incredibly"]
        }
        
        if emotion in emotion_modifiers:
            modifier = random.choice(emotion_modifiers[emotion])
            if random.random() < 0.3:  # 30% chance to add emotion modifier
                return f"{modifier} {starter.lower()}"
        
        return starter
    
    def _generate_assistant_response(self, user_message: str, domain: str, 
                                   pattern: str, emotion: str) -> str:
        """Generate contextual assistant response."""
        
        # Response templates by domain and pattern
        response_templates = {
            "healthcare": {
                "empathetic_validation": [
                    "I understand this must be a difficult time for you. Let me help you work through this.",
                    "Your concerns are completely valid. It's natural to feel this way.",
                    "I can hear that you're going through a challenging situation."
                ],
                "professional_guidance": [
                    "Based on what you've shared, I'd recommend speaking with a qualified healthcare professional.",
                    "Here are some evidence-based approaches that might help in this situation.",
                    "Let's explore some professional resources that could support you."
                ]
            },
            "business": {
                "analytical_approach": [
                    "Let's break this down into key components and analyze each factor.",
                    "From a strategic perspective, we should consider multiple angles.",
                    "I'd suggest we examine the data and market conditions first."
                ],
                "strategic_thinking": [
                    "This presents both challenges and opportunities. Let's explore both.",
                    "Have you considered the long-term implications of this decision?",
                    "What are your key success metrics for this initiative?"
                ]
            },
            "education": {
                "educational_scaffolding": [
                    "Let's start with the fundamentals and build your understanding step by step.",
                    "I'll help you break this complex topic into manageable pieces.",
                    "What specific aspect would you like to focus on first?"
                ],
                "motivational_support": [
                    "Learning can be challenging, but you're making great progress.",
                    "Every expert was once a beginner. You're on the right path.",
                    "Let's find a study approach that works better for your learning style."
                ]
            },
            "creative": {
                "creative_encouragement": [
                    "Creative blocks are temporary. Let's explore some techniques to spark new ideas.",
                    "Your creative instincts are valuable. Let's find ways to nurture them.",
                    "Every creative journey has ups and downs. You're not alone in this."
                ],
                "idea_generation": [
                    "Let's try some brainstorming techniques to generate fresh perspectives.",
                    "What if we approached this from a completely different angle?",
                    "Have you considered combining elements from different sources of inspiration?"
                ]
            },
            "leadership": {
                "leadership_coaching": [
                    "Great leaders face these challenges regularly. Let's work through this together.",
                    "Your leadership instincts are developing. This is part of the growth process.",
                    "What leadership principles do you think apply to this situation?"
                ],
                "team_dynamics": [
                    "Understanding team dynamics is crucial for effective leadership.",
                    "Each team member brings unique perspectives. How can we leverage that?",
                    "Building trust takes time, but these steps can help accelerate the process."
                ]
            }
        }
        
        # Get appropriate response template
        domain_responses = response_templates.get(domain, {})
        pattern_responses = domain_responses.get(pattern, [
            "I understand your situation. Let me help you think through this.",
            "This is an important topic. Let's explore it together.",
            "I appreciate you sharing this with me. Here's how I can help."
        ])
        
        return random.choice(pattern_responses)
    
    def _generate_followup_user(self, conversation_history: List[Dict], 
                              scenario: str, emotion: str) -> str:
        """Generate realistic user follow-up message."""
        
        followup_templates = [
            "That's helpful, but I'm still concerned about...",
            "Thank you for that insight. Could you also help me with...",
            "I appreciate your guidance. However, I'm also wondering...",
            "That makes sense. What about when...",
            "I understand that approach. But what if...",
            "That's a good point. Can you elaborate on..."
        ]
        
        return random.choice(followup_templates)
    
    def _generate_followup_assistant(self, conversation_history: List[Dict],
                                   domain: str, pattern: str) -> str:
        """Generate assistant follow-up response."""
        
        followup_templates = [
            "Absolutely, let me address that specific concern.",
            "That's an excellent question. Here's another perspective to consider.",
            "I'm glad you brought that up. It's an important aspect to explore.",
            "You're thinking critically about this, which is great.",
            "Let's dive deeper into that particular aspect.",
            "That shows you're really engaging with this topic thoughtfully."
        ]
        
        return random.choice(followup_templates)
    
    def _evaluate_quality(self, conversation: Dict[str, Any]) -> float:
        """Evaluate the quality of a generated conversation."""
        score = 0.8  # Base score
        
        # Check conversation length
        if len(conversation["turns"]) >= 4:
            score += 0.1
        
        # Check emotional consistency
        emotions_present = [turn.get("emotion") for turn in conversation["turns"] if turn.get("emotion")]
        if len(set(emotions_present)) <= 2:  # Consistent emotions
            score += 0.05
        
        # Check domain relevance
        if conversation.get("domain") and conversation.get("scenario"):
            score += 0.05
        
        return min(score, 1.0)
    
    def generate_all_domains(self, samples_per_domain: int = 1000) -> Dict[str, str]:
        """Generate data for all supported domains."""
        results = {}
        
        for domain in self.domain_templates.keys():
            try:
                output_path = self.generate_domain_data(domain, samples_per_domain)
                results[domain] = output_path
                logger.info(f"✅ Generated data for {domain}: {output_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate data for {domain}: {e}")
                results[domain] = None
        
        return results 