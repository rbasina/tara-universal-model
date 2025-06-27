#!/usr/bin/env python3
"""
🧪 TDD Tests for TARA Universal GGUF Conversion System
Comprehensive test coverage for all conversion system components
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "conversion"))

from universal_gguf_factory import (
    UniversalGGUFFactory, 
    IntelligentRouter, 
    QuantizationType, 
    CompressionType,
    DomainInfo,
    RoutingDecision
)
from cleanup_utilities import ModelCleanupUtilities, CleanupResult, ModelValidationResult
from phase_manager import PhaseManager, PhaseInfo, DomainStatus
from compression_utilities import CompressionUtilities, CompressionConfig, CompressionResult

class TestIntelligentRouter(unittest.TestCase):
    """Test Intelligent Router functionality"""
    
    def setUp(self):
        self.router = IntelligentRouter()
        
        # Create test domain info
        self.healthcare_domain = DomainInfo(
            name="healthcare",
            base_model="microsoft/DialoGPT-medium",
            adapter_path=Path("test_adapters/healthcare"),
            training_quality=0.97,
            response_speed=0.8,
            emotional_intensity=0.9,
            context_length=4096,
            specialties=["medical", "therapeutic"],
            phase=1
        )
        
        self.business_domain = DomainInfo(
            name="business",
            base_model="microsoft/DialoGPT-medium",
            adapter_path=Path("test_adapters/business"),
            training_quality=0.95,
            response_speed=0.7,
            emotional_intensity=0.5,
            context_length=4096,
            specialties=["strategy", "leadership"],
            phase=1
        )
    
    def test_add_domain(self):
        """Test adding domain to router"""
        self.router.add_domain(self.healthcare_domain)
        self.assertIn("healthcare", self.router.domain_models)
        self.assertEqual(self.router.domain_models["healthcare"], self.healthcare_domain)
    
    def test_route_query_healthcare(self):
        """Test routing healthcare query"""
        self.router.add_domain(self.healthcare_domain)
        self.router.add_domain(self.business_domain)
        
        query = "I have a medical emergency and need help"
        decision = self.router.route_query(query)
        
        self.assertEqual(decision.primary_model, "healthcare")
        self.assertGreater(decision.confidence, 0.5)
        self.assertIn("healthcare", decision.reasoning.lower())
    
    def test_route_query_business(self):
        """Test routing business query"""
        self.router.add_domain(self.healthcare_domain)
        self.router.add_domain(self.business_domain)
        
        query = "I need help with business strategy and market analysis"
        decision = self.router.route_query(query)
        
        self.assertEqual(decision.primary_model, "business")
        self.assertGreater(decision.confidence, 0.5)
        self.assertIn("business", decision.reasoning.lower())
    
    def test_emotional_analysis(self):
        """Test emotional context analysis"""
        query = "I'm feeling very sad and lonely today"
        emotional_context = self.router._analyze_emotional_context(query)
        
        self.assertIn("sadness", emotional_context["emotions"])
        self.assertEqual(emotional_context["dominant_emotion"], "sadness")
        self.assertGreater(emotional_context["emotional_intensity"], 0.5)
    
    def test_content_analysis(self):
        """Test content analysis"""
        query = "I need urgent medical help for chest pain"
        content_analysis = self.router._analyze_query_content(query)
        
        self.assertEqual(content_analysis["primary_domain"], "healthcare")
        self.assertGreater(content_analysis["urgency"], 0.5)
        self.assertIn("healthcare", content_analysis["domain_scores"])
    
    def test_caching(self):
        """Test routing decision caching"""
        self.router.add_domain(self.healthcare_domain)
        
        query = "I have a medical question"
        decision1 = self.router.route_query(query)
        decision2 = self.router.route_query(query)
        
        # Should return same decision from cache
        self.assertEqual(decision1.primary_model, decision2.primary_model)
        self.assertEqual(decision1.confidence, decision2.confidence)

class TestEmotionalIntelligence(unittest.TestCase):
    """Test Emotional Intelligence Engine"""
    
    def setUp(self):
        from emotional_intelligence import EmotionalIntelligenceEngine
        self.ei_engine = EmotionalIntelligenceEngine()
    
    def test_emotional_analysis(self):
        """Test emotional context analysis"""
        query = "I'm so happy and excited about my new job!"
        context = self.ei_engine.analyze_emotional_context(query)
        
        self.assertEqual(context["dominant_emotion"], "joy")
        self.assertGreater(context["emotional_intensity"], 0.5)
        self.assertIn("joy", context["emotions"])
    
    def test_response_modulation(self):
        """Test response modulation"""
        response = "That's great news!"
        emotional_context = {
            "dominant_emotion": "joy",
            "emotional_intensity": 0.8,
            "emotions": {"joy": 0.8, "sadness": 0.1, "anger": 0.1}
        }
        
        modulated = self.ei_engine.modulate_response(response, emotional_context)
        self.assertIsInstance(modulated, str)
        self.assertGreater(len(modulated), len(response))
    
    def test_domain_specific_responses(self):
        """Test domain-specific emotional responses"""
        # Test healthcare crisis response
        crisis_response = self.ei_engine.domain_emotional_responses["healthcare"]["crisis"]
        self.assertEqual(crisis_response["tone"], "urgent_caring")
        self.assertEqual(crisis_response["empathy_level"], "very_high")
        self.assertEqual(crisis_response["emotional_intensity"], 1.0)

class TestCompressionUtilities(unittest.TestCase):
    """Test Compression Utilities"""
    
    def setUp(self):
        self.compression_utils = CompressionUtilities()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_quantization_types(self):
        """Test quantization type enumeration"""
        self.assertEqual(QuantizationType.Q2_K.value, "Q2_K")
        self.assertEqual(QuantizationType.Q4_K_M.value, "Q4_K_M")
        self.assertEqual(QuantizationType.Q5_K_M.value, "Q5_K_M")
        self.assertEqual(QuantizationType.Q8_0.value, "Q8_0")
    
    def test_compression_types(self):
        """Test compression type enumeration"""
        self.assertEqual(CompressionType.STANDARD.value, "standard")
        self.assertEqual(CompressionType.SPARSE.value, "sparse")
        self.assertEqual(CompressionType.HYBRID.value, "hybrid")
        self.assertEqual(CompressionType.DISTILLED.value, "distilled")
    
    def test_compression_config(self):
        """Test compression configuration"""
        config = CompressionConfig(
            quantization=QuantizationType.Q4_K_M,
            compression_type=CompressionType.STANDARD,
            target_size_mb=1000,
            quality_threshold=0.95,
            speed_priority=False
        )
        
        self.assertEqual(config.quantization, QuantizationType.Q4_K_M)
        self.assertEqual(config.compression_type, CompressionType.STANDARD)
        self.assertEqual(config.target_size_mb, 1000)
        self.assertEqual(config.quality_threshold, 0.95)
        self.assertFalse(config.speed_priority)
    
    def test_quality_estimation(self):
        """Test quality score estimation"""
        quality = self.compression_utils._estimate_quality_score(
            QuantizationType.Q4_K_M, 
            CompressionType.STANDARD
        )
        self.assertGreater(quality, 0.8)
        self.assertLess(quality, 1.0)
    
    def test_compression_recommendations(self):
        """Test compression recommendations"""
        recommendations = self.compression_utils.get_compression_recommendations(
            model_size_mb=2000,
            target_size_mb=500,
            quality_priority=False,
            speed_priority=True
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(all(isinstance(r, CompressionConfig) for r in recommendations))

class TestCleanupUtilities(unittest.TestCase):
    """Test Model Cleanup Utilities"""
    
    def setUp(self):
        self.cleanup_utils = ModelCleanupUtilities()
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_model_structure()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_test_model_structure(self):
        """Create test model structure"""
        model_dir = Path(self.temp_dir) / "test_model"
        model_dir.mkdir()
        
        # Create required files
        (model_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (model_dir / "adapter_model.safetensors").write_text("test")
        (model_dir / "config.json").write_text('{"vocab_size": 1000, "hidden_size": 768}')
        (model_dir / "tokenizer.json").write_text("test")
        (model_dir / "tokenizer_config.json").write_text('{"model_max_length": 2048}')
        
        # Create garbage files
        (model_dir / "temp.tmp").write_text("temp")
        (model_dir / "cache.log").write_text("log")
        (model_dir / "checkpoint-1").mkdir()
    
    def test_clean_model_directory(self):
        """Test model directory cleaning"""
        model_path = Path(self.temp_dir) / "test_model"
        output_path = Path(self.temp_dir) / "cleaned_model"
        
        result = self.cleanup_utils.clean_model_directory(model_path, output_path)
        
        self.assertTrue(result.success)
        self.assertEqual(result.cleaned_path, output_path)
        self.assertGreater(result.original_size_mb, 0)
        self.assertGreater(result.cleaned_size_mb, 0)
        self.assertGreater(len(result.removed_files), 0)
    
    def test_validate_model_structure(self):
        """Test model structure validation"""
        model_path = Path(self.temp_dir) / "test_model"
        validation = self.cleanup_utils._validate_model_structure(model_path)
        
        self.assertTrue(validation.is_valid)
        self.assertGreater(validation.validation_score, 0.5)
        self.assertEqual(len(validation.issues), 0)
    
    def test_garbage_detection(self):
        """Test garbage file detection"""
        temp_file = Path(self.temp_dir) / "test.tmp"
        temp_file.write_text("test")
        
        self.assertTrue(self.cleanup_utils._is_garbage_file(temp_file))
        
        valid_file = Path(self.temp_dir) / "valid.json"
        valid_file.write_text("{}")
        
        self.assertFalse(self.cleanup_utils._is_garbage_file(valid_file))
    
    def test_adapter_compatibility(self):
        """Test adapter compatibility validation"""
        adapter_path = Path(self.temp_dir) / "test_model"
        is_compatible = self.cleanup_utils.validate_adapter_compatibility(
            adapter_path, "microsoft/DialoGPT-medium"
        )
        
        self.assertTrue(is_compatible)

class TestPhaseManager(unittest.TestCase):
    """Test Phase Manager"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.phase_manager = PhaseManager(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_phase(self):
        """Test phase creation"""
        success = self.phase_manager.create_phase(
            1, 
            ["healthcare", "business"],
            {"quantization": "Q4_K_M"}
        )
        
        self.assertTrue(success)
        phase_info = self.phase_manager.get_phase_info(1)
        self.assertIsNotNone(phase_info)
        self.assertEqual(phase_info.phase_number, 1)
        self.assertEqual(phase_info.domains, ["healthcare", "business"])
    
    def test_add_domain_to_phase(self):
        """Test adding domain to phase"""
        self.phase_manager.create_phase(1, ["healthcare"])
        
        success = self.phase_manager.add_domain_to_phase(
            1, 
            "business",
            Path("test_adapters/business")
        )
        
        self.assertTrue(success)
        phase_info = self.phase_manager.get_phase_info(1)
        self.assertIn("business", phase_info.domains)
    
    def test_update_domain_status(self):
        """Test domain status update"""
        self.phase_manager.create_phase(1, ["healthcare"])
        
        success = self.phase_manager.update_domain_status(
            "healthcare",
            "complete",
            0.97,
            {"training_loss": 0.03}
        )
        
        self.assertTrue(success)
        domain_status = self.phase_manager.get_domain_status("healthcare")
        self.assertEqual(domain_status.training_status, "complete")
        self.assertEqual(domain_status.training_quality, 0.97)
    
    def test_get_ready_domains(self):
        """Test getting ready domains"""
        self.phase_manager.create_phase(1, ["healthcare", "business"])
        self.phase_manager.update_domain_status("healthcare", "complete", 0.97)
        self.phase_manager.update_domain_status("business", "training", 0.5)
        
        ready_domains = self.phase_manager.get_ready_domains(1)
        self.assertIn("healthcare", ready_domains)
        self.assertNotIn("business", ready_domains)
    
    def test_advance_phase(self):
        """Test phase advancement"""
        initial_phase = self.phase_manager.current_phase
        new_phase = self.phase_manager.advance_phase()
        
        self.assertEqual(new_phase, initial_phase + 1)
        self.assertEqual(self.phase_manager.current_phase, new_phase)

class TestUniversalGGUFFactory(unittest.TestCase):
    """Test Universal GGUF Factory"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.factory = UniversalGGUFFactory(Path(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_add_domain_phase(self):
        """Test adding domain to phase"""
        adapter_path = Path(self.temp_dir) / "test_adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (adapter_path / "adapter_model.safetensors").write_text("test")
        
        success = self.factory.add_domain_phase(
            "healthcare",
            adapter_path,
            training_quality=0.97,
            response_speed=0.8,
            emotional_intensity=0.9
        )
        
        self.assertTrue(success)
        self.assertIn("healthcare", self.factory.router.domain_models)
    
    def test_validate_adapter(self):
        """Test adapter validation"""
        adapter_path = Path(self.temp_dir) / "valid_adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}")
        (adapter_path / "adapter_model.safetensors").write_text("test")
        
        self.assertTrue(self.factory._validate_adapter(adapter_path))
        
        invalid_path = Path(self.temp_dir) / "invalid_adapter"
        invalid_path.mkdir()
        self.assertFalse(self.factory._validate_adapter(invalid_path))
    
    def test_get_phase_summary(self):
        """Test phase summary generation"""
        # Add a domain first
        adapter_path = Path(self.temp_dir) / "test_adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (adapter_path / "adapter_model.safetensors").write_text("test")
        
        self.factory.add_domain_phase("healthcare", adapter_path)
        
        summary = self.factory.get_phase_summary(1)
        self.assertIsInstance(summary, dict)
        self.assertIn("domains", summary)
        self.assertIn("emotional_intelligence", summary)
        self.assertIn("intelligent_routing", summary)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.factory = UniversalGGUFFactory(Path(self.temp_dir))
        self.phase_manager = PhaseManager(Path(self.temp_dir))
        self.cleanup_utils = ModelCleanupUtilities()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from phase creation to GGUF building"""
        # 1. Create phase
        self.phase_manager.create_phase(1, ["healthcare", "business"])
        
        # 2. Create test adapters
        for domain in ["healthcare", "business"]:
            adapter_path = Path(self.temp_dir) / f"test_{domain}"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
            (adapter_path / "adapter_model.safetensors").write_text("test")
            
            # Add to factory
            self.factory.add_domain_phase(domain, adapter_path)
            
            # Update status
            self.phase_manager.update_domain_status(domain, "complete", 0.95)
        
        # 3. Get ready domains
        ready_domains = self.phase_manager.get_ready_domains(1)
        self.assertEqual(len(ready_domains), 2)
        self.assertIn("healthcare", ready_domains)
        self.assertIn("business", ready_domains)
        
        # 4. Test routing
        query = "I need medical help"
        decision = self.factory.router.route_query(query)
        self.assertEqual(decision.primary_model, "healthcare")
        
        # 5. Test phase summary
        summary = self.factory.get_phase_summary(1)
        self.assertIsInstance(summary, dict)
        self.assertIn("domains", summary)

def run_tests():
    """Run all tests with coverage reporting"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestIntelligentRouter,
        TestEmotionalIntelligence,
        TestCompressionUtilities,
        TestCleanupUtilities,
        TestPhaseManager,
        TestUniversalGGUFFactory,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🧪 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 