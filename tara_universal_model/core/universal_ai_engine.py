"""
TARA Universal Model - Core AI Engine
Robust Backend AI System to Support All Human Needs

This is the heart of TARA - the Universal AI Engine that provides:
- Multi-domain AI expertise (Healthcare, Business, Education, Creative, Leadership, Universal)
- Robust model inference and processing
- Context-aware responses
- HAI-enhanced human support
- Scalable architecture for all human needs
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, Pipeline
)
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class AIRequest:
    """Universal AI request structure"""
    user_input: str
    domain: str
    context: Optional[Dict] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict] = None
    urgency_level: str = "normal"  # normal, high, emergency

@dataclass
class AIResponse:
    """Universal AI response structure"""
    response_text: str
    domain: str
    confidence: float
    processing_time: float
    suggestions: List[str]
    follow_up_questions: List[str]
    resources: List[Dict]
    emotional_tone: str
    hai_context: str

class DomainExpert:
    """Base class for domain-specific AI experts"""
    
    def __init__(self, domain: str, model_path: Optional[str] = None):
        self.domain = domain
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        
        # Domain-specific configurations
        self.domain_config = self._get_domain_config()
        
    def _get_domain_config(self) -> Dict:
        """Get domain-specific configuration"""
        configs = {
            "healthcare": {
                "personality": "Compassionate, professional, evidence-based",
                "response_style": "thorough, empathetic, safety-focused",
                "specialties": ["general_health", "mental_health", "wellness", "prevention"],
                "disclaimers": True,
                "emergency_protocols": True
            },
            "business": {
                "personality": "Strategic, analytical, results-oriented",
                "response_style": "data-driven, actionable, growth-focused",
                "specialties": ["strategy", "operations", "finance", "marketing", "leadership"],
                "disclaimers": False,
                "emergency_protocols": False
            },
            "education": {
                "personality": "Patient, encouraging, knowledgeable",
                "response_style": "clear, structured, engaging",
                "specialties": ["learning", "teaching", "research", "skill_development"],
                "disclaimers": False,
                "emergency_protocols": False
            },
            "creative": {
                "personality": "Inspiring, imaginative, innovative",
                "response_style": "creative, open-ended, possibility-focused",
                "specialties": ["writing", "design", "art", "innovation", "brainstorming"],
                "disclaimers": False,
                "emergency_protocols": False
            },
            "leadership": {
                "personality": "Wise, authoritative, empowering",
                "response_style": "strategic, inspiring, decision-focused",
                "specialties": ["team_management", "decision_making", "communication", "vision"],
                "disclaimers": False,
                "emergency_protocols": False
            },
            "universal": {
                "personality": "Adaptable, helpful, comprehensive",
                "response_style": "balanced, informative, supportive",
                "specialties": ["general_assistance", "problem_solving", "information", "guidance"],
                "disclaimers": False,
                "emergency_protocols": True
            }
        }
        return configs.get(self.domain, configs["universal"])
    
    async def load_model(self):
        """Load domain-specific model"""
        try:
            if self.is_loaded:
                return
            
            logger.info(f"🔄 Loading {self.domain} AI model...")
            
            # Try to load domain-specific model first
            model_paths = [
                f"models/{self.domain}",
                f"models/microsoft_Phi-3.5-mini-instruct",
                f"models/microsoft_DialoGPT-medium",
                "microsoft/DialoGPT-medium"
            ]
            
            for path in model_paths:
                try:
                    if Path(path).exists() or not path.startswith("models/"):
                        logger.info(f"🔧 Attempting to load model from: {path}")
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(path)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                        
                        # Create text generation pipeline
                        self.pipeline = pipeline(
                            "text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            max_length=512,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        self.is_loaded = True
                        logger.info(f"✅ {self.domain} model loaded successfully from {path}")
                        return
                        
                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")
                    continue
            
            # Fallback to rule-based responses
            logger.warning(f"⚠️ No model loaded for {self.domain}, using rule-based responses")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.domain} model: {e}")
            self.is_loaded = True  # Continue with rule-based responses
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate domain-specific AI response"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                await self.load_model()
            
            # Generate response based on available capabilities
            if self.pipeline:
                response_text = await self._generate_model_response(request)
            else:
                response_text = await self._generate_rule_based_response(request)
            
            # Enhance response with domain expertise
            enhanced_response = await self._enhance_response(request, response_text)
            
            processing_time = time.time() - start_time
            
            return AIResponse(
                response_text=enhanced_response["text"],
                domain=self.domain,
                confidence=enhanced_response["confidence"],
                processing_time=processing_time,
                suggestions=enhanced_response["suggestions"],
                follow_up_questions=enhanced_response["follow_up_questions"],
                resources=enhanced_response["resources"],
                emotional_tone=enhanced_response["emotional_tone"],
                hai_context=enhanced_response["hai_context"]
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating {self.domain} response: {e}")
            return self._generate_fallback_response(request, time.time() - start_time)
    
    async def _generate_model_response(self, request: AIRequest) -> str:
        """Generate response using loaded AI model"""
        try:
            # Create domain-specific prompt
            prompt = self._create_domain_prompt(request)
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract and clean response
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return await self._generate_rule_based_response(request)
    
    def _create_domain_prompt(self, request: AIRequest) -> str:
        """Create domain-specific prompt"""
        domain_context = {
            "healthcare": f"As a compassionate healthcare AI assistant, provide helpful, evidence-based guidance. Remember to suggest consulting healthcare professionals for medical decisions.\n\nUser: {request.user_input}\nAssistant:",
            "business": f"As a strategic business AI advisor, provide data-driven insights and actionable recommendations.\n\nUser: {request.user_input}\nAssistant:",
            "education": f"As a patient and knowledgeable education AI tutor, explain concepts clearly and encourage learning.\n\nUser: {request.user_input}\nAssistant:",
            "creative": f"As an inspiring creative AI partner, help explore innovative ideas and possibilities.\n\nUser: {request.user_input}\nAssistant:",
            "leadership": f"As a wise leadership AI mentor, provide strategic guidance for effective leadership.\n\nUser: {request.user_input}\nAssistant:",
            "universal": f"As TARA, a helpful AI assistant, provide comprehensive support for any human need.\n\nUser: {request.user_input}\nAssistant:"
        }
        
        return domain_context.get(self.domain, domain_context["universal"])
    
    async def _generate_rule_based_response(self, request: AIRequest) -> str:
        """Generate rule-based response when no model is available"""
        domain_responses = {
            "healthcare": self._generate_healthcare_response(request),
            "business": self._generate_business_response(request),
            "education": self._generate_education_response(request),
            "creative": self._generate_creative_response(request),
            "leadership": self._generate_leadership_response(request),
            "universal": self._generate_universal_response(request)
        }
        
        return domain_responses.get(self.domain, domain_responses["universal"])
    
    def _generate_healthcare_response(self, request: AIRequest) -> str:
        """Generate healthcare-specific response"""
        keywords = request.user_input.lower()
        
        if any(word in keywords for word in ["pain", "hurt", "ache", "sick", "ill"]):
            return f"I understand you're experiencing discomfort regarding '{request.user_input}'. While I can provide general wellness information, it's important to consult with a healthcare professional for proper medical evaluation and treatment. In the meantime, consider rest, hydration, and monitoring your symptoms. If this is an emergency, please contact emergency services immediately."
        
        elif any(word in keywords for word in ["mental", "stress", "anxiety", "depression", "mood"]):
            return f"Thank you for sharing your mental health concern about '{request.user_input}'. Mental wellness is crucial for overall health. Consider speaking with a mental health professional who can provide personalized support. In the meantime, practices like deep breathing, regular exercise, adequate sleep, and connecting with supportive people can be helpful. If you're having thoughts of self-harm, please reach out to a crisis helpline immediately."
        
        elif any(word in keywords for word in ["diet", "nutrition", "food", "eating"]):
            return f"Regarding your nutrition question about '{request.user_input}', a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins is generally beneficial. However, nutritional needs vary by individual. Consider consulting with a registered dietitian for personalized dietary advice, especially if you have specific health conditions or goals."
        
        else:
            return f"Thank you for your healthcare question about '{request.user_input}'. I'm here to provide general wellness information and support. For specific medical concerns, please consult with qualified healthcare professionals who can provide personalized medical advice. Is there a particular aspect of health and wellness I can help you explore?"
    
    def _generate_business_response(self, request: AIRequest) -> str:
        """Generate business-specific response"""
        keywords = request.user_input.lower()
        
        if any(word in keywords for word in ["strategy", "plan", "planning", "growth"]):
            return f"Regarding your strategic question about '{request.user_input}', successful business strategy typically involves: 1) Clear goal setting and vision alignment, 2) Market analysis and competitive positioning, 3) Resource allocation and capability building, 4) Performance measurement and adaptation. What specific aspect of your strategy would you like to explore further?"
        
        elif any(word in keywords for word in ["marketing", "sales", "customer", "revenue"]):
            return f"For your marketing and sales inquiry about '{request.user_input}', consider focusing on: 1) Understanding your target customer deeply, 2) Creating compelling value propositions, 3) Choosing the right channels for your audience, 4) Measuring and optimizing performance. What's your current biggest challenge in reaching your customers?"
        
        elif any(word in keywords for word in ["team", "employee", "management", "leadership"]):
            return f"Regarding your team management question about '{request.user_input}', effective leadership involves: 1) Clear communication of expectations and goals, 2) Regular feedback and recognition, 3) Professional development opportunities, 4) Creating a positive work environment. What specific team challenge are you facing?"
        
        else:
            return f"Thank you for your business question about '{request.user_input}'. I'm here to help you think through strategic, operational, and growth challenges. Whether it's planning, execution, or optimization, let's explore how to move your business forward. What's your primary business objective right now?"
    
    def _generate_education_response(self, request: AIRequest) -> str:
        """Generate education-specific response"""
        return f"Great question about '{request.user_input}'! Learning is most effective when we break complex topics into manageable parts. Let me help you understand this concept step by step, and then we can explore how it connects to your broader learning goals. What specific aspect would you like to dive deeper into?"
    
    def _generate_creative_response(self, request: AIRequest) -> str:
        """Generate creative-specific response"""
        return f"I love your creative thinking about '{request.user_input}'! Let's explore some innovative approaches and possibilities. Creativity thrives when we combine existing ideas in new ways, challenge assumptions, and remain open to unexpected connections. What creative direction feels most exciting to you right now?"
    
    def _generate_leadership_response(self, request: AIRequest) -> str:
        """Generate leadership-specific response"""
        return f"Your leadership question about '{request.user_input}' touches on important aspects of effective leadership. Great leaders focus on vision, communication, empowerment, and results. They also understand that leadership is about serving others and creating conditions for success. What leadership challenge are you currently navigating?"
    
    def _generate_universal_response(self, request: AIRequest) -> str:
        """Generate universal response"""
        return f"I'm here to help with '{request.user_input}'. As your AI companion, I can assist across many areas - from problem-solving and planning to learning and creative thinking. Let me provide you with comprehensive support tailored to your specific needs. What would be most helpful for you right now?"
    
    async def _enhance_response(self, request: AIRequest, base_response: str) -> Dict:
        """Enhance response with domain-specific features"""
        config = self.domain_config
        
        # Generate suggestions based on domain
        suggestions = await self._generate_suggestions(request, base_response)
        
        # Generate follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(request)
        
        # Generate relevant resources
        resources = await self._generate_resources(request)
        
        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(request, base_response)
        
        # Create HAI context
        hai_context = f"TARA is engaging with you in {self.domain} mode, applying {config['personality']} approach to provide {config['response_style']} assistance."
        
        # Add disclaimers if needed
        if config.get("disclaimers"):
            base_response += "\n\n⚠️ This information is for educational purposes only and should not replace professional medical advice."
        
        return {
            "text": base_response,
            "confidence": 0.85,  # Base confidence, would be calculated from model
            "suggestions": suggestions,
            "follow_up_questions": follow_up_questions,
            "resources": resources,
            "emotional_tone": emotional_tone,
            "hai_context": hai_context
        }
    
    async def _generate_suggestions(self, request: AIRequest, response: str) -> List[str]:
        """Generate helpful suggestions"""
        domain_suggestions = {
            "healthcare": [
                "Consider keeping a symptom diary",
                "Schedule a check-up with your healthcare provider",
                "Explore stress management techniques",
                "Focus on preventive care measures"
            ],
            "business": [
                "Analyze your key performance metrics",
                "Consider market research and customer feedback",
                "Explore strategic partnerships",
                "Review and optimize your processes"
            ],
            "education": [
                "Practice with real-world examples",
                "Create a structured learning plan",
                "Find additional learning resources",
                "Connect with others learning similar topics"
            ],
            "creative": [
                "Brainstorm without judgment",
                "Explore different creative mediums",
                "Seek inspiration from diverse sources",
                "Collaborate with other creative minds"
            ],
            "leadership": [
                "Gather feedback from your team",
                "Develop your emotional intelligence",
                "Practice active listening",
                "Focus on empowering others"
            ],
            "universal": [
                "Break the challenge into smaller steps",
                "Consider multiple perspectives",
                "Seek additional information or expertise",
                "Take action on what you can control"
            ]
        }
        
        return domain_suggestions.get(self.domain, domain_suggestions["universal"])[:3]
    
    async def _generate_follow_up_questions(self, request: AIRequest) -> List[str]:
        """Generate relevant follow-up questions"""
        questions = [
            "What specific aspect would you like to explore further?",
            "How does this relate to your current goals?",
            "What resources or support do you need to move forward?"
        ]
        return questions
    
    async def _generate_resources(self, request: AIRequest) -> List[Dict]:
        """Generate relevant resources"""
        return [
            {
                "type": "guidance",
                "title": f"{self.domain.title()} Best Practices",
                "description": f"Expert guidance for {self.domain} challenges"
            }
        ]
    
    def _determine_emotional_tone(self, request: AIRequest, response: str) -> str:
        """Determine appropriate emotional tone"""
        urgency = request.urgency_level
        
        if urgency == "emergency":
            return "urgent_supportive"
        elif urgency == "high":
            return "focused_helpful"
        else:
            return self.domain_config["personality"].split(",")[0].lower().strip()
    
    def _generate_fallback_response(self, request: AIRequest, processing_time: float) -> AIResponse:
        """Generate fallback response when all else fails"""
        return AIResponse(
            response_text=f"I'm here to help with '{request.user_input}'. While I'm experiencing some technical difficulties, I'm committed to supporting you. Let me know how I can assist you in a different way.",
            domain=self.domain,
            confidence=0.5,
            processing_time=processing_time,
            suggestions=["Try rephrasing your request", "Specify what type of help you need"],
            follow_up_questions=["What's the most important thing I can help you with?"],
            resources=[],
            emotional_tone="supportive",
            hai_context="TARA is working to provide the best possible support despite technical challenges."
        )

class UniversalAIEngine:
    """
    Core Universal AI Engine - Robust Backend for All Human Needs
    
    This is the heart of TARA that orchestrates all domain experts
    and provides comprehensive AI support for every human need.
    """
    
    def __init__(self):
        self.domain_experts: Dict[str, DomainExpert] = {}
        self.is_initialized = False
        self.supported_domains = [
            "healthcare", "business", "education", 
            "creative", "leadership", "universal"
        ]
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.domain_usage_stats = {}
        
        logger.info("🚀 Initializing Universal AI Engine...")
    
    async def initialize(self):
        """Initialize all domain experts"""
        if self.is_initialized:
            return
        
        logger.info("🔧 Loading domain experts...")
        
        # Initialize domain experts
        for domain in self.supported_domains:
            try:
                expert = DomainExpert(domain)
                self.domain_experts[domain] = expert
                self.domain_usage_stats[domain] = 0
                logger.info(f"✅ {domain.title()} expert initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {domain} expert: {e}")
        
        self.is_initialized = True
        logger.info("🎉 Universal AI Engine initialized successfully!")
        logger.info(f"📊 Loaded {len(self.domain_experts)} domain experts")
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """
        Process AI request with full HAI support
        This is the main entry point for all AI interactions
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate and enhance request
            enhanced_request = await self._enhance_request(request)
            
            # Route to appropriate domain expert
            domain_expert = await self._get_domain_expert(enhanced_request.domain)
            
            # Generate AI response
            response = await domain_expert.generate_response(enhanced_request)
            
            # Post-process response
            final_response = await self._post_process_response(response, enhanced_request)
            
            # Update statistics
            self._update_stats(enhanced_request.domain, time.time() - start_time)
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error processing AI request: {e}")
            return self._generate_error_response(request, str(e))
    
    async def _enhance_request(self, request: AIRequest) -> AIRequest:
        """Enhance request with context and validation"""
        # Validate domain
        if request.domain not in self.supported_domains:
            logger.warning(f"Unknown domain '{request.domain}', defaulting to 'universal'")
            request.domain = "universal"
        
        # Add context if missing
        if request.context is None:
            request.context = {}
        
        # Add timestamp
        request.context["timestamp"] = datetime.now().isoformat()
        request.context["request_id"] = f"req_{int(time.time() * 1000)}"
        
        return request
    
    async def _get_domain_expert(self, domain: str) -> DomainExpert:
        """Get domain expert, loading if necessary"""
        if domain not in self.domain_experts:
            # Create expert on-demand
            self.domain_experts[domain] = DomainExpert(domain)
            self.domain_usage_stats[domain] = 0
        
        expert = self.domain_experts[domain]
        
        # Ensure expert is loaded
        if not expert.is_loaded:
            await expert.load_model()
        
        return expert
    
    async def _post_process_response(self, response: AIResponse, request: AIRequest) -> AIResponse:
        """Post-process response with additional enhancements"""
        # Add HAI enhancements
        response.hai_context += f" | Request processed in {response.processing_time:.2f}s"
        
        # Add emergency protocols if needed
        if request.urgency_level == "emergency":
            response.response_text = "🚨 **Emergency Support Mode** 🚨\n\n" + response.response_text
            response.response_text += "\n\n⚠️ If this is a life-threatening emergency, please contact emergency services immediately."
        
        return response
    
    def _update_stats(self, domain: str, processing_time: float):
        """Update usage statistics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        self.domain_usage_stats[domain] += 1
    
    def _generate_error_response(self, request: AIRequest, error_message: str) -> AIResponse:
        """Generate error response with HAI support"""
        return AIResponse(
            response_text=f"I encountered an issue while processing your request about '{request.user_input}'. I'm still here to help you in any way I can. Please try rephrasing your request or let me know how else I can assist you.",
            domain=request.domain,
            confidence=0.0,
            processing_time=0.0,
            suggestions=["Try rephrasing your request", "Specify your needs more clearly"],
            follow_up_questions=["What's the most important thing I can help you with right now?"],
            resources=[],
            emotional_tone="supportive",
            hai_context=f"TARA encountered a technical issue but remains committed to helping you. Error: {error_message}"
        )
    
    async def get_capabilities(self) -> Dict:
        """Get comprehensive engine capabilities"""
        return {
            "supported_domains": self.supported_domains,
            "total_experts": len(self.domain_experts),
            "loaded_experts": len([e for e in self.domain_experts.values() if e.is_loaded]),
            "request_count": self.request_count,
            "average_processing_time": self.total_processing_time / max(self.request_count, 1),
            "domain_usage": self.domain_usage_stats,
            "hai_features": [
                "Multi-domain expertise",
                "Context-aware responses", 
                "Emergency protocols",
                "Personalized assistance",
                "Continuous learning",
                "Privacy protection"
            ],
            "status": "ready" if self.is_initialized else "initializing"
        }
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        health_status = {
            "engine_status": "healthy" if self.is_initialized else "initializing",
            "domain_experts": {},
            "performance": {
                "total_requests": self.request_count,
                "average_response_time": self.total_processing_time / max(self.request_count, 1),
                "uptime": "active"
            },
            "capabilities": "full"
        }
        
        # Check each domain expert
        for domain, expert in self.domain_experts.items():
            health_status["domain_experts"][domain] = {
                "loaded": expert.is_loaded,
                "model_available": expert.model is not None,
                "pipeline_ready": expert.pipeline is not None,
                "usage_count": self.domain_usage_stats.get(domain, 0)
            }
        
        return health_status

# Global Universal AI Engine instance
_universal_engine = None

async def get_universal_engine() -> UniversalAIEngine:
    """Get global Universal AI Engine instance"""
    global _universal_engine
    if _universal_engine is None:
        _universal_engine = UniversalAIEngine()
        await _universal_engine.initialize()
    return _universal_engine 