"""
Test Suite for TARA Voice Server
Tests all voice server endpoints and TTS functionality
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json

# Import the voice server app
import sys
sys.path.append('..')
from voice_server import app, HAIConfig, InputValidator, RateLimiter

class TestVoiceServerAPI:
    """Test Voice Server API endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_tts_status_endpoint(self):
        """Test TTS status endpoint"""
        response = self.client.get("/tts/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ready"
        assert data["hai_principle"] == "Help Anytime, Everywhere"
        assert "systems" in data
        assert "edge_tts" in data["systems"]
        assert "pyttsx3" in data["systems"]
        assert "ai_engine" in data["systems"]
        assert "domains" in data
        assert len(data["domains"]) == 6
        assert "safety_features" in data
        assert data["version"].startswith("2.0.0")
    
    def test_tts_synthesize_endpoint(self):
        """Test TTS synthesize endpoint"""
        request_data = {
            "text": "Hello, this is a test",
            "domain": "universal"
        }
        
        response = self.client.post("/tts/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["domain"] == "universal"
        assert data["text_response"] == "Hello, this is a test"
        assert "synthesis_method" in data
        assert "hai_message" in data
    
    def test_tts_synthesize_all_domains(self):
        """Test TTS synthesis for all domains"""
        domains = ["universal", "healthcare", "business", "education", "creative", "leadership"]
        
        for domain in domains:
            request_data = {
                "text": f"Testing {domain} domain",
                "domain": domain
            }
            
            response = self.client.post("/tts/synthesize", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["domain"] == domain
            assert data["success"] is True
    
    def test_chat_with_voice_endpoint(self):
        """Test chat with voice endpoint"""
        request_data = {
            "message": "How are you today?",
            "domain": "universal"
        }
        
        response = self.client.post("/chat_with_voice", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "How are you today?"
        assert data["domain"] == "universal"
        assert "response" in data
        assert "audio_synthesis" in data
        assert "hai_context" in data
    
    def test_ai_chat_endpoint(self):
        """Test direct AI chat endpoint"""
        request_data = {
            "message": "What is artificial intelligence?",
            "domain": "education"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["message"] == "What is artificial intelligence?"
        assert data["domain"] == "education"
        assert "response" in data
        assert "hai_context" in data
    
    def test_ai_health_endpoint(self):
        """Test AI health check endpoint"""
        response = self.client.get("/ai/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ai_engine_available" in data
        assert "status" in data
        assert "timestamp" in data

class TestInputValidation:
    """Test input validation and sanitization"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        malicious_input = "<script>alert('xss')</script>Hello"
        sanitized = InputValidator.sanitize_text(malicious_input)
        
        assert "<script>" not in sanitized
        assert "Hello" in sanitized
    
    def test_input_validation_too_long(self):
        """Test input validation for text too long"""
        long_text = "A" * (HAIConfig.MAX_TEXT_LENGTH + 1)
        is_valid, message = InputValidator.validate_text(long_text)
        
        assert is_valid is False
        assert "too long" in message.lower()
    
    def test_input_validation_too_short(self):
        """Test input validation for text too short"""
        is_valid, message = InputValidator.validate_text("")
        
        assert is_valid is False
        assert "too short" in message.lower()
    
    def test_malicious_input_blocked(self):
        """Test that malicious input is blocked"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "eval(alert(1))"
        ]
        
        for malicious_input in malicious_inputs:
            is_valid, message = InputValidator.validate_text(malicious_input)
            assert is_valid is False
            assert "harmful content" in message.lower()
    
    def test_tts_request_validation(self):
        """Test TTS request validation"""
        # Test valid request
        valid_request = {
            "text": "Hello world",
            "domain": "universal"
        }
        response = self.client.post("/tts/synthesize", json=valid_request)
        assert response.status_code == 200
        
        # Test invalid domain
        invalid_domain_request = {
            "text": "Hello world",
            "domain": "invalid_domain"
        }
        response = self.client.post("/tts/synthesize", json=invalid_domain_request)
        assert response.status_code == 422  # Validation error
        
        # Test missing text
        missing_text_request = {
            "domain": "universal"
        }
        response = self.client.post("/tts/synthesize", json=missing_text_request)
        assert response.status_code == 422  # Validation error

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiter_allows_normal_usage(self):
        """Test rate limiter allows normal usage"""
        client_ip = "127.0.0.1"
        
        # Should allow first request
        assert RateLimiter.is_allowed(client_ip) is True
        
        # Should allow subsequent requests within limit
        for _ in range(10):
            assert RateLimiter.is_allowed(client_ip) is True
    
    def test_rate_limiter_blocks_excessive_requests(self):
        """Test rate limiter blocks excessive requests"""
        client_ip = "192.168.1.100"
        
        # Make requests up to the limit
        for _ in range(HAIConfig.RATE_LIMIT_PER_MINUTE):
            RateLimiter.is_allowed(client_ip)
        
        # Next request should be blocked
        assert RateLimiter.is_allowed(client_ip) is False

class TestDomainSpecificBehavior:
    """Test domain-specific behavior and responses"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_healthcare_domain_disclaimers(self):
        """Test healthcare domain includes appropriate disclaimers"""
        request_data = {
            "message": "I have a headache",
            "domain": "healthcare"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        # Healthcare responses should mention professional consultation
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["professional", "doctor", "healthcare", "medical"])
    
    def test_business_domain_strategic_focus(self):
        """Test business domain provides strategic insights"""
        request_data = {
            "message": "How to increase sales?",
            "domain": "business"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["strategy", "business", "growth", "recommend"])
    
    def test_education_domain_learning_focus(self):
        """Test education domain focuses on learning"""
        request_data = {
            "message": "Explain photosynthesis",
            "domain": "education"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["learn", "understand", "explain", "concept"])
    
    def test_creative_domain_inspiration(self):
        """Test creative domain provides inspiration"""
        request_data = {
            "message": "Help me write a story",
            "domain": "creative"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["creative", "idea", "inspire", "imagination"])
    
    def test_leadership_domain_guidance(self):
        """Test leadership domain provides leadership guidance"""
        request_data = {
            "message": "How to motivate my team?",
            "domain": "leadership"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["leadership", "team", "motivate", "guide"])

class TestVoiceConfiguration:
    """Test voice configuration and personality traits"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_domain_voice_mapping(self):
        """Test that each domain has proper voice configuration"""
        from voice_server import DOMAIN_VOICES
        
        required_domains = ["universal", "healthcare", "business", "education", "creative", "leadership"]
        
        for domain in required_domains:
            assert domain in DOMAIN_VOICES
            config = DOMAIN_VOICES[domain]
            assert "voice" in config
            assert "personality" in config
            assert "fallback_voice" in config
            assert len(config["personality"]) > 0
    
    def test_voice_synthesis_with_custom_voice(self):
        """Test voice synthesis with custom voice specification"""
        request_data = {
            "text": "Testing custom voice",
            "domain": "healthcare",
            "voice": "en-US-SaraNeural"
        }
        
        response = self.client.post("/tts/synthesize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

class TestErrorHandling:
    """Test error handling and graceful degradation"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_graceful_degradation_no_audio(self):
        """Test graceful degradation when audio synthesis fails"""
        # This would require mocking TTS failures
        # For now, test that the endpoint always returns something useful
        request_data = {
            "text": "Test graceful degradation",
            "domain": "universal"
        }
        
        response = self.client.post("/tts/synthesize", json=request_data)
        
        # Should always return a response, even if audio fails
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "text_response" in data
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON requests"""
        response = self.client.post(
            "/tts/synthesize",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        # Missing text field
        request_data = {
            "domain": "universal"
        }
        
        response = self.client.post("/tts/synthesize", json=request_data)
        assert response.status_code == 422
    
    def test_server_error_handling(self):
        """Test server error handling"""
        # Test with extremely long text that might cause issues
        request_data = {
            "text": "A" * 10000,  # Very long text
            "domain": "universal"
        }
        
        response = self.client.post("/tts/synthesize", json=request_data)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 422, 500]

class TestFileCleanup:
    """Test automatic file cleanup functionality"""
    
    def test_temp_file_cleanup_scheduling(self):
        """Test that temp files are scheduled for cleanup"""
        from voice_server import FileCleanupManager
        
        test_file = "test_temp_file.mp3"
        FileCleanupManager.schedule_cleanup(test_file)
        
        # Check that file is in cleanup list
        from voice_server import temp_files_created
        assert len(temp_files_created) > 0
        assert any(f['path'] == test_file for f in temp_files_created)

class TestHAIFeatures:
    """Test HAI-specific features"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_hai_context_in_responses(self):
        """Test that HAI context is included in responses"""
        request_data = {
            "message": "Help me understand AI",
            "domain": "education"
        }
        
        response = self.client.post("/ai/chat", json=request_data)
        data = response.json()
        
        assert "hai_context" in data
        assert "TARA" in data["hai_context"]
        assert "education" in data["hai_context"]
    
    def test_safety_features_status(self):
        """Test that safety features are properly reported"""
        response = self.client.get("/tts/status")
        data = response.json()
        
        safety_features = data["safety_features"]
        assert safety_features["rate_limiting"] is True
        assert safety_features["input_validation"] is True
        assert safety_features["auto_cleanup"] is True
        assert safety_features["offline_capable"] is True
        assert safety_features["privacy_protection"] is True

class TestCORSConfiguration:
    """Test CORS configuration for frontend integration"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        # Test preflight request
        response = self.client.options(
            "/tts/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_allowed_origins(self):
        """Test that allowed origins work correctly"""
        response = self.client.get(
            "/tts/status",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"]) 