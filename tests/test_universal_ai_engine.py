"""
Test Suite for TARA Universal AI Engine
Tests all domain experts and core AI functionality
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from tara_universal_model.core import (
    UniversalAIEngine, 
    DomainExpert, 
    AIRequest, 
    AIResponse, 
    get_universal_engine
)

class TestAIRequest:
    """Test AIRequest data structure"""
    
    def test_ai_request_creation(self):
        """Test creating AIRequest with all fields"""
        request = AIRequest(
            user_input="How can I improve my health?",
            domain="healthcare",
            context={"user_id": "test123"},
            user_id="test123",
            session_id="session456",
            preferences={"language": "en"},
            urgency_level="high"
        )
        
        assert request.user_input == "How can I improve my health?"
        assert request.domain == "healthcare"
        assert request.context["user_id"] == "test123"
        assert request.urgency_level == "high"
    
    def test_ai_request_minimal(self):
        """Test creating AIRequest with minimal fields"""
        request = AIRequest(
            user_input="Hello TARA",
            domain="universal"
        )
        
        assert request.user_input == "Hello TARA"
        assert request.domain == "universal"
        assert request.context is None
        assert request.urgency_level == "normal"

class TestDomainExpert:
    """Test individual domain experts"""
    
    @pytest.mark.asyncio
    async def test_healthcare_expert(self):
        """Test healthcare domain expert"""
        expert = DomainExpert("healthcare")
        
        # Test domain configuration
        assert expert.domain == "healthcare"
        assert expert.domain_config["personality"] == "Compassionate, professional, evidence-based"
        assert expert.domain_config["disclaimers"] is True
        assert expert.domain_config["emergency_protocols"] is True
        
        # Test response generation
        request = AIRequest(
            user_input="I have a headache",
            domain="healthcare"
        )
        
        response = await expert.generate_response(request)
        
        assert isinstance(response, AIResponse)
        assert response.domain == "healthcare"
        assert "healthcare" in response.response_text.lower() or "medical" in response.response_text.lower()
        assert response.confidence > 0
        assert response.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_business_expert(self):
        """Test business domain expert"""
        expert = DomainExpert("business")
        
        assert expert.domain == "business"
        assert expert.domain_config["personality"] == "Strategic, analytical, results-oriented"
        assert expert.domain_config["disclaimers"] is False
        
        request = AIRequest(
            user_input="How can I improve my sales?",
            domain="business"
        )
        
        response = await expert.generate_response(request)
        
        assert isinstance(response, AIResponse)
        assert response.domain == "business"
        assert len(response.suggestions) > 0
        assert len(response.follow_up_questions) > 0
    
    @pytest.mark.asyncio
    async def test_education_expert(self):
        """Test education domain expert"""
        expert = DomainExpert("education")
        
        request = AIRequest(
            user_input="Explain quantum physics",
            domain="education"
        )
        
        response = await expert.generate_response(request)
        
        assert response.domain == "education"
        assert "learn" in response.response_text.lower() or "understand" in response.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_creative_expert(self):
        """Test creative domain expert"""
        expert = DomainExpert("creative")
        
        request = AIRequest(
            user_input="Help me write a story",
            domain="creative"
        )
        
        response = await expert.generate_response(request)
        
        assert response.domain == "creative"
        assert "creative" in response.response_text.lower() or "idea" in response.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_leadership_expert(self):
        """Test leadership domain expert"""
        expert = DomainExpert("leadership")
        
        request = AIRequest(
            user_input="How to motivate my team?",
            domain="leadership"
        )
        
        response = await expert.generate_response(request)
        
        assert response.domain == "leadership"
        assert "leadership" in response.response_text.lower() or "team" in response.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_universal_expert(self):
        """Test universal domain expert"""
        expert = DomainExpert("universal")
        
        request = AIRequest(
            user_input="I need help with everything",
            domain="universal"
        )
        
        response = await expert.generate_response(request)
        
        assert response.domain == "universal"
        assert "help" in response.response_text.lower()

class TestUniversalAIEngine:
    """Test the main Universal AI Engine"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        assert engine.is_initialized is True
        assert len(engine.domain_experts) == 6
        assert "healthcare" in engine.domain_experts
        assert "business" in engine.domain_experts
        assert "education" in engine.domain_experts
        assert "creative" in engine.domain_experts
        assert "leadership" in engine.domain_experts
        assert "universal" in engine.domain_experts
    
    @pytest.mark.asyncio
    async def test_process_request_all_domains(self):
        """Test processing requests for all domains"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        test_cases = [
            ("healthcare", "I feel sick"),
            ("business", "Increase revenue"),
            ("education", "Teach me math"),
            ("creative", "Write a poem"),
            ("leadership", "Lead my team"),
            ("universal", "Help me please")
        ]
        
        for domain, user_input in test_cases:
            request = AIRequest(
                user_input=user_input,
                domain=domain
            )
            
            response = await engine.process_request(request)
            
            assert isinstance(response, AIResponse)
            assert response.domain == domain
            assert len(response.response_text) > 0
            assert response.confidence > 0
            assert response.processing_time >= 0
            
            # Check HAI context is present
            assert "TARA" in response.hai_context
    
    @pytest.mark.asyncio
    async def test_invalid_domain_fallback(self):
        """Test fallback to universal domain for invalid domains"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        request = AIRequest(
            user_input="Help me",
            domain="invalid_domain"
        )
        
        response = await engine.process_request(request)
        
        assert response.domain == "universal"  # Should fallback to universal
    
    @pytest.mark.asyncio
    async def test_emergency_request(self):
        """Test emergency request handling"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        request = AIRequest(
            user_input="I need immediate help!",
            domain="healthcare",
            urgency_level="emergency"
        )
        
        response = await engine.process_request(request)
        
        assert "Emergency Support Mode" in response.response_text
        assert "emergency services" in response.response_text.lower()
    
    @pytest.mark.asyncio
    async def test_engine_capabilities(self):
        """Test engine capabilities reporting"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        capabilities = await engine.get_capabilities()
        
        assert "supported_domains" in capabilities
        assert len(capabilities["supported_domains"]) == 6
        assert "hai_features" in capabilities
        assert "Multi-domain expertise" in capabilities["hai_features"]
        assert capabilities["status"] == "ready"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test engine health check"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        health = await engine.health_check()
        
        assert health["engine_status"] == "healthy"
        assert "domain_experts" in health
        assert "performance" in health
        assert health["capabilities"] == "full"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance statistics tracking"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        # Process some requests
        for i in range(3):
            request = AIRequest(
                user_input=f"Test request {i}",
                domain="universal"
            )
            await engine.process_request(request)
        
        assert engine.request_count == 3
        assert engine.total_processing_time > 0
        assert engine.domain_usage_stats["universal"] == 3

class TestGlobalEngineInstance:
    """Test global engine instance management"""
    
    @pytest.mark.asyncio
    async def test_get_universal_engine(self):
        """Test getting global engine instance"""
        engine1 = await get_universal_engine()
        engine2 = await get_universal_engine()
        
        # Should return the same instance
        assert engine1 is engine2
        assert engine1.is_initialized is True

class TestErrorHandling:
    """Test error handling and fallback scenarios"""
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test graceful handling of model loading failures"""
        expert = DomainExpert("healthcare")
        
        # Mock model loading failure
        with patch.object(expert, '_initialize_tts', side_effect=Exception("Model load failed")):
            await expert.load_model()
            
            # Should still be marked as loaded (fallback mode)
            assert expert.is_loaded is True
    
    @pytest.mark.asyncio
    async def test_response_generation_failure(self):
        """Test fallback response when generation fails"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        # Test with empty input
        request = AIRequest(
            user_input="",
            domain="universal"
        )
        
        response = await engine.process_request(request)
        
        # Should still return a valid response
        assert isinstance(response, AIResponse)
        assert len(response.response_text) > 0

class TestPerformanceBenchmarks:
    """Performance benchmarks for the AI engine"""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self):
        """Test that responses are generated within reasonable time"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        request = AIRequest(
            user_input="Quick test",
            domain="universal"
        )
        
        start_time = time.time()
        response = await engine.process_request(request)
        end_time = time.time()
        
        # Should respond within 5 seconds for fallback responses
        assert (end_time - start_time) < 5.0
        assert response.processing_time < 5.0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        engine = UniversalAIEngine()
        await engine.initialize()
        
        # Create multiple concurrent requests
        requests = [
            AIRequest(user_input=f"Concurrent test {i}", domain="universal")
            for i in range(5)
        ]
        
        # Process all requests concurrently
        tasks = [engine.process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(responses) == 5
        for response in responses:
            assert isinstance(response, AIResponse)
            assert len(response.response_text) > 0

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"]) 