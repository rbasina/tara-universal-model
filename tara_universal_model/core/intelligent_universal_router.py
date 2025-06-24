"""
Intelligent Universal Router - Revolutionary Multi-Domain Intelligence
TARA Universal Model Integration with me²TARA Trinity Architecture

This router implements the breakthrough approach:
✨ Single universal GGUF model handles ALL domains
🧠 Seamless intelligence detects multiple domains automatically  
🎯 Blends knowledge naturally like human intelligence
🔄 Integrates perfectly with me²TARA hybrid routing (Ports 2025/8765/8766)
🚀 Scales effortlessly as we add more domain training

No rigid routing, no hard-coded rules - just pure intelligence!
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

from .seamless_intelligence import SeamlessIntelligenceEngine
from ..serving.gguf_model import GGUFModelManager, TARAGGUFModel
from ..utils.config import TARAConfig

logger = logging.getLogger(__name__)

class IntelligentUniversalRouter:
    """
    Revolutionary universal router that mirrors human intelligence.
    
    Key breakthrough: Instead of routing TO different models,
    we use ONE universal model with intelligent prompting that
    adapts based on multi-domain analysis.
    
    This is how human intelligence actually works - we don't switch
    brains for different topics, we adapt our thinking and communication
    style based on the context and domains involved.
    """
    
    def __init__(self, config: Optional[TARAConfig] = None):
        self.config = config or TARAConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize the seamless intelligence engine
        self.intelligence_engine = SeamlessIntelligenceEngine()
        
        # Initialize GGUF model manager
        self.gguf_manager = GGUFModelManager(self.config)
        self.tara_model = TARAGGUFModel(self.config)
        
        # Universal model configuration - the breakthrough approach
        self.universal_model_config = {
            "primary_model": "tara-1.0",  # Our trained universal model
            "fallback_models": ["phi-3.5", "llama-3.1"],  # Backup options
            "context_length": 4096,
            "supports_all_domains": True,
            "trained_domains": ["healthcare", "business", "education", "creative", "leadership"]
        }
        
        # Integration with me²TARA system
        self.meetara_integration = {
            "port_coordination": {
                # Port 5000 no longer needed - using embedded GGUF integration
                "meetara_router": 2025,  # me²TARA main routing
                "meetara_backup_1": 8765,  # me²TARA fallback 1
                "meetara_backup_2": 8766   # me²TARA fallback 2
            },
            "cost_optimization": {
                "local_processing": True,  # Keep costs near zero
                "api_fallback": False,     # Stay local-first
                "token_efficiency": True   # Optimize prompt engineering
            }
        }
        
        self.logger.info("🧠 Intelligent Universal Router initialized")
        self.logger.info(f"🎯 Primary model: {self.universal_model_config['primary_model']}")
        self.logger.info(f"🔗 me²TARA integration ready")
    
    async def process_message(self, message: str, user_id: str = "default", 
                            conversation_context: Optional[List[Dict]] = None,
                            meetara_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Revolutionary message processing that mirrors human intelligence.
        
        This is the core breakthrough:
        1. Analyze message for ALL relevant domains simultaneously
        2. Use ONE universal model with intelligent prompting
        3. Blend expertise naturally based on domain analysis
        4. Return response that feels like talking to a human expert
        """
        
        try:
            # 1. Seamless intelligence analysis
            analysis = await self.intelligence_engine.analyze_message(
                message, user_id, conversation_context
            )
            
            # 2. Generate intelligent system prompt based on analysis
            system_prompt = await self._generate_intelligent_system_prompt(analysis)
            
            # 3. Create context-aware user prompt
            enhanced_prompt = await self._enhance_user_prompt(message, analysis, meetara_context)
            
            # 4. Generate response using universal model
            response_data = await self._generate_universal_response(
                system_prompt, enhanced_prompt, analysis
            )
            
            # 5. Post-process for me²TARA integration
            final_response = await self._post_process_for_meetara(response_data, analysis)
            
            return {
                "success": True,
                "response": final_response["text"],
                "analysis": analysis,
                "model_used": response_data.get("model", "tara-1.0"),
                "domains_detected": analysis["domain_analysis"]["all_domain_scores"],
                "primary_domain": analysis["domain_analysis"]["primary_domain"],
                "secondary_domains": analysis["domain_analysis"]["secondary_domains"],
                "integration_type": analysis["response_strategy"]["response_approach"],
                "empathy_score": final_response.get("empathy_score", 0.7),
                "meetara_compatible": True,
                "processing_time_ms": response_data.get("processing_time_ms", 0),
                "tokens_used": response_data.get("tokens_used", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in process_message: {e}")
            return await self._generate_fallback_response(message, str(e))
    
    async def _generate_intelligent_system_prompt(self, analysis: Dict) -> str:
        """
        Generate intelligent system prompt based on seamless intelligence analysis.
        """
        
        domain_analysis = analysis["domain_analysis"]
        response_strategy = analysis["response_strategy"]
        emotional_context = analysis["emotional_context"]
        
        if response_strategy["response_approach"] == "integrated_multi_domain":
            return f"""You are TARA, an intelligent AI companion with integrated expertise across healthcare, business, education, creative, and leadership domains.

Based on the user's query, you've identified multiple relevant domains: {', '.join([domain_analysis['primary_domain']] + domain_analysis['secondary_domains'])}

Your response should:
1. Address the primary domain ({domain_analysis['primary_domain']}) with deep expertise and appropriate tone
2. Seamlessly integrate insights from secondary domains: {', '.join(domain_analysis['secondary_domains'])}
3. Blend the knowledge naturally, like human intelligence works across disciplines
4. Adapt your communication style to match the dominant domain context
5. Provide holistic, integrated guidance that addresses all aspects of the user's needs

Integration strategy: {response_strategy['integration_strategy']}
Communication tone: {emotional_context['communication_tone_needed']}
Empathy level: {emotional_context['empathy_level']}"""
            
        else:
            # Single domain focused response
            primary_domain = domain_analysis["primary_domain"]
            domain_prompts = {
                "healthcare": "You are TARA, an empathetic AI companion specializing in healthcare and wellness. You provide supportive, caring responses while maintaining professional boundaries.",
                "business": "You are TARA, a strategic AI business partner with deep analytical capabilities. You provide professional insights and practical solutions.",
                "education": "You are TARA, an inspiring AI learning companion. You help users understand concepts and achieve learning goals with clear explanations.",
                "creative": "You are TARA, an innovative AI creative partner that sparks imagination. You help explore creative possibilities and artistic expression.",
                "leadership": "You are TARA, a wise AI leadership coach. You provide guidance on leadership challenges and team management."
            }
            
            return domain_prompts.get(primary_domain, domain_prompts["healthcare"])
    
    async def _enhance_user_prompt(self, message: str, analysis: Dict, 
                                 meetara_context: Optional[Dict] = None) -> str:
        """
        Enhance the user prompt with context and intelligence insights.
        """
        
        enhanced_prompt = message
        
        # Add emotional context if high empathy needed
        if analysis["emotional_context"]["empathy_level"] == "high":
            enhanced_prompt = f"[User needs emotional support] {message}"
        
        # Add domain context for multi-domain queries
        if analysis["domain_analysis"]["is_multi_domain"]:
            domains = [analysis["domain_analysis"]["primary_domain"]] + analysis["domain_analysis"]["secondary_domains"]
            enhanced_prompt = f"[Multi-domain query: {', '.join(domains)}] {message}"
        
        return enhanced_prompt
    
    async def _generate_universal_response(self, system_prompt: str, user_prompt: str, 
                                         analysis: Dict) -> Dict[str, Any]:
        """
        Generate response using the universal GGUF model.
        """
        
        start_time = datetime.now()
        
        try:
            # Use the universal TARA model
            response_data = self.gguf_manager.generate_response(
                model_name=self.universal_model_config["primary_model"],
                prompt=user_prompt,
                domain=analysis["domain_analysis"]["primary_domain"],
                max_tokens=self._calculate_optimal_tokens(analysis),
                temperature=self._calculate_optimal_temperature(analysis)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            response_data["processing_time_ms"] = processing_time
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error generating universal response: {e}")
            return {
                "response": "I'm here to help you with that. Let me provide some thoughtful guidance based on your needs.",
                "model": "fallback",
                "tokens_used": 0
            }
    
    async def _post_process_for_meetara(self, response_data: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Post-process response for me²TARA integration.
        """
        
        response_text = response_data.get("response", "I'm here to help you.")
        
        # Calculate empathy score
        empathy_score = self.intelligence_engine._calculate_empathy_score(response_text, analysis)
        
        return {
            "text": response_text,
            "empathy_score": empathy_score,
            "tokens_used": response_data.get("tokens_used", 0),
            "model_used": response_data.get("model", "tara-1.0")
        }
    
    def _calculate_optimal_tokens(self, analysis: Dict) -> int:
        """Calculate optimal token count based on analysis."""
        
        base_tokens = 150
        
        # Add tokens for multi-domain responses
        if analysis["domain_analysis"]["is_multi_domain"]:
            base_tokens += 100
        
        # Add tokens for high empathy needs
        if analysis["emotional_context"]["empathy_level"] == "high":
            base_tokens += 50
        
        return min(base_tokens, 512)  # Cap at reasonable limit
    
    def _calculate_optimal_temperature(self, analysis: Dict) -> float:
        """Calculate optimal temperature based on analysis."""
        
        base_temp = 0.7
        
        # Lower temperature for healthcare (more precise)
        if analysis["domain_analysis"]["primary_domain"] == "healthcare":
            base_temp = 0.6
        
        # Higher temperature for creative (more diverse)
        elif analysis["domain_analysis"]["primary_domain"] == "creative":
            base_temp = 0.8
        
        return min(base_temp, 0.9)
    
    async def _generate_fallback_response(self, message: str, error: str) -> Dict[str, Any]:
        """Generate fallback response when everything fails."""
        
        return {
            "success": False,
            "response": "I'm here to help you, though I'm experiencing some technical difficulties. Could you please try rephrasing your question?",
            "error": error,
            "model_used": "fallback",
            "domains_detected": {"universal": 1.0},
            "primary_domain": "universal",
            "secondary_domains": [],
            "integration_type": "error_fallback",
            "empathy_score": 0.8,
            "meetara_compatible": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring and integration."""
        
        # Check model availability
        available_models = self.gguf_manager.get_available_models()
        primary_model_available = any(
            model["name"] == self.universal_model_config["primary_model"] and model["available"]
            for model in available_models
        )
        
        return {
            "system_name": "TARA Universal Intelligent Router",
            "version": "2.0.0",
            "status": "operational" if primary_model_available else "degraded",
            "primary_model": {
                "name": self.universal_model_config["primary_model"],
                "available": primary_model_available,
                "supports_domains": self.universal_model_config["trained_domains"]
            },
            "meetara_integration": {
                "port": self.meetara_integration["port_coordination"]["tara_universal"],
                "cost_optimization": self.meetara_integration["cost_optimization"]["local_processing"]
            },
            "performance": {
                "local_processing": True,
                "cost_per_interaction": "$0.00",
                "uptime_target": "99.5%"
            },
            "last_updated": datetime.now().isoformat()
        }

# Integration with me²TARA - Export class for easy import
__all__ = ["IntelligentUniversalRouter"]

# Example usage and testing
if __name__ == "__main__":
    async def test_intelligent_router():
        """Test the intelligent universal router."""
        
        print("🧠 Testing Intelligent Universal Router")
        print("=" * 60)
        
        # Initialize router
        router = IntelligentUniversalRouter()
        
        # Test system status
        print("\n📊 System Status:")
        status = await router.get_system_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test sample message processing
        print("\n💬 Sample Message Processing:")
        test_message = "I'm feeling stressed about my work deadlines and it's affecting my health"
        
        try:
            result = await router.process_message(test_message, "test_user")
            print(f"   Query: {test_message}")
            print(f"   Primary Domain: {result['primary_domain']}")
            print(f"   Secondary Domains: {result['secondary_domains']}")
            print(f"   Integration Type: {result['integration_type']}")
            print(f"   Empathy Score: {result['empathy_score']:.2f}")
            print(f"   me²TARA Compatible: {result['meetara_compatible']}")
            print(f"   Response Preview: {result['response'][:100]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n✅ Intelligent Universal Router Test Complete")
    
    # Run the test
    asyncio.run(test_intelligent_router()) 