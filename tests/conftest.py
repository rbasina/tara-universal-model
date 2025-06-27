#!/usr/bin/env python3
"""
🧪 Pytest Configuration & Fixtures for TARA Universal Model
Comprehensive test setup and fixtures for TDD implementation
"""

import pytest
import tempfile
import json
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project paths to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "conversion"))
sys.path.insert(0, str(project_root / "scripts" / "training"))
sys.path.insert(0, str(project_root / "scripts" / "monitoring"))

@pytest.fixture(scope="session")
def project_root_path():
    """Return project root path"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def temp_project_dir():
    """Create temporary project directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_test_dir():
    """Create temporary directory for individual tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def mock_training_state():
    """Mock training state for testing"""
    return {
        "current_domain": "healthcare",
        "current_step": 134,
        "total_steps": 400,
        "start_time": time.time(),
        "last_checkpoint": "checkpoint-134",
        "domains": {
            "healthcare": {
                "status": "training",
                "progress": 33.5,
                "current_step": 134,
                "checkpoint_path": "models/healthcare/checkpoint-134"
            },
            "business": {
                "status": "complete",
                "progress": 100.0,
                "current_step": 400,
                "checkpoint_path": "models/business/checkpoint-400"
            }
        }
    }

@pytest.fixture(scope="function")
def mock_domain_info():
    """Mock domain information for testing"""
    return {
        "healthcare": {
            "name": "healthcare",
            "base_model": "microsoft/DialoGPT-medium",
            "adapter_path": Path("test_adapters/healthcare"),
            "training_quality": 0.97,
            "response_speed": 0.8,
            "emotional_intensity": 0.9,
            "context_length": 4096,
            "specialties": ["medical", "therapeutic"],
            "phase": 1
        },
        "business": {
            "name": "business",
            "base_model": "microsoft/DialoGPT-medium",
            "adapter_path": Path("test_adapters/business"),
            "training_quality": 0.95,
            "response_speed": 0.7,
            "emotional_intensity": 0.5,
            "context_length": 4096,
            "specialties": ["strategy", "leadership"],
            "phase": 1
        }
    }

@pytest.fixture(scope="function")
def mock_compression_config():
    """Mock compression configuration for testing"""
    return {
        "quantization": "Q4_K_M",
        "compression_type": "standard",
        "target_size_mb": 1000,
        "quality_threshold": 0.95,
        "speed_priority": False
    }

@pytest.fixture(scope="function")
def mock_phase_info():
    """Mock phase information for testing"""
    return {
        "phase_number": 1,
        "domains": ["healthcare", "business"],
        "status": "active",
        "start_time": time.time(),
        "config": {
            "quantization": "Q4_K_M",
            "compression_type": "standard"
        }
    }

@pytest.fixture(scope="function")
def mock_connection_state():
    """Mock connection state for testing"""
    return {
        "last_heartbeat": time.time(),
        "connection_active": True,
        "recovery_attempts": 0,
        "training_state": {
            "current_domain": "healthcare",
            "current_step": 134,
            "checkpoint_path": "models/healthcare/checkpoint-134"
        }
    }

@pytest.fixture(scope="function")
def mock_checkpoint_structure(temp_test_dir):
    """Create mock checkpoint structure for testing"""
    checkpoint_dir = temp_test_dir / "models" / "healthcare" / "checkpoint-134"
    checkpoint_dir.mkdir(parents=True)
    
    # Create required checkpoint files
    (checkpoint_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
    (checkpoint_dir / "adapter_model.safetensors").write_text("test")
    (checkpoint_dir / "training_args.bin").write_text("test")
    (checkpoint_dir / "config.json").write_text('{"vocab_size": 1000, "hidden_size": 768}')
    (checkpoint_dir / "tokenizer.json").write_text("test")
    (checkpoint_dir / "tokenizer_config.json").write_text('{"model_max_length": 2048}')
    
    # Create some garbage files
    (checkpoint_dir / "temp.tmp").write_text("temp")
    (checkpoint_dir / "cache.log").write_text("log")
    (checkpoint_dir / "checkpoint-1").mkdir()
    
    return checkpoint_dir

@pytest.fixture(scope="function")
def mock_parallel_training_state():
    """Mock parallel training state for testing"""
    return {
        "parallel_training": True,
        "active_domains": ["education", "creative", "leadership"],
        "domain_states": {
            "education": {
                "status": "training",
                "progress": 25.0,
                "current_step": 100,
                "checkpoint_path": "models/education/checkpoint-100"
            },
            "creative": {
                "status": "training",
                "progress": 15.0,
                "current_step": 60,
                "checkpoint_path": "models/creative/checkpoint-60"
            },
            "leadership": {
                "status": "training",
                "progress": 10.0,
                "current_step": 40,
                "checkpoint_path": "models/leadership/checkpoint-40"
            }
        },
        "resource_usage": {
            "cpu_percent": 85.0,
            "memory_percent": 70.0,
            "gpu_percent": 0.0
        }
    }

@pytest.fixture(scope="function")
def mock_dashboard_data():
    """Mock dashboard data for testing"""
    return {
        "timestamp": time.time(),
        "overall_progress": 45.5,
        "active_domains": ["healthcare", "education", "creative"],
        "domain_progress": {
            "healthcare": {"progress": 33.5, "status": "training"},
            "education": {"progress": 25.0, "status": "training"},
            "creative": {"progress": 15.0, "status": "training"}
        },
        "system_metrics": {
            "cpu_usage": 85.0,
            "memory_usage": 70.0,
            "disk_usage": 45.0
        }
    }

@pytest.fixture(scope="function")
def mock_emotional_context():
    """Mock emotional context for testing"""
    return {
        "dominant_emotion": "joy",
        "emotional_intensity": 0.8,
        "emotions": {
            "joy": 0.8,
            "sadness": 0.1,
            "anger": 0.1
        },
        "context": "positive_experience"
    }

@pytest.fixture(scope="function")
def mock_routing_decision():
    """Mock routing decision for testing"""
    return {
        "primary_model": "healthcare",
        "confidence": 0.85,
        "reasoning": "Query contains medical terminology and healthcare context",
        "fallback_model": "business",
        "emotional_context": {
            "dominant_emotion": "concern",
            "emotional_intensity": 0.7
        }
    }

@pytest.fixture(scope="function")
def mock_cleanup_result():
    """Mock cleanup result for testing"""
    return {
        "success": True,
        "cleaned_path": Path("cleaned_model"),
        "original_size_mb": 150.0,
        "cleaned_size_mb": 120.0,
        "removed_files": ["temp.tmp", "cache.log"],
        "validation_score": 0.95,
        "issues": []
    }

@pytest.fixture(scope="function")
def mock_compression_result():
    """Mock compression result for testing"""
    return {
        "success": True,
        "compressed_path": Path("compressed_model.gguf"),
        "original_size_mb": 2000.0,
        "compressed_size_mb": 500.0,
        "compression_ratio": 4.0,
        "quality_score": 0.92,
        "compression_time": 120.5,
        "quantization_type": "Q4_K_M",
        "compression_type": "standard"
    }

@pytest.fixture(scope="function")
def mock_phase_manager(temp_test_dir):
    """Create mock phase manager for testing"""
    from phase_manager import PhaseManager
    return PhaseManager(temp_test_dir)

@pytest.fixture(scope="function")
def mock_cleanup_utilities():
    """Create mock cleanup utilities for testing"""
    from cleanup_utilities import ModelCleanupUtilities
    return ModelCleanupUtilities()

@pytest.fixture(scope="function")
def mock_compression_utilities():
    """Create mock compression utilities for testing"""
    from compression_utilities import CompressionUtilities
    return CompressionUtilities()

@pytest.fixture(scope="function")
def mock_intelligent_router():
    """Create mock intelligent router for testing"""
    from universal_gguf_factory import IntelligentRouter
    return IntelligentRouter()

@pytest.fixture(scope="function")
def mock_emotional_intelligence():
    """Create mock emotional intelligence engine for testing"""
    from emotional_intelligence import EmotionalIntelligenceEngine
    return EmotionalIntelligenceEngine()

# Test data fixtures
@pytest.fixture(scope="session")
def test_queries():
    """Test queries for routing tests"""
    return {
        "healthcare": [
            "I have a medical emergency and need help",
            "What are the symptoms of diabetes?",
            "I'm feeling chest pain, what should I do?",
            "Can you help me understand my medication?",
            "I need mental health support"
        ],
        "business": [
            "I need help with business strategy",
            "How can I improve my company's performance?",
            "What are the best marketing strategies?",
            "I need financial planning advice",
            "How do I manage my team effectively?"
        ],
        "education": [
            "I need help with my homework",
            "Can you explain quantum physics?",
            "How do I study more effectively?",
            "I need help with math problems",
            "What are good study techniques?"
        ],
        "creative": [
            "I need help with my creative writing",
            "Can you help me brainstorm ideas?",
            "I'm stuck with my art project",
            "How can I improve my creativity?",
            "I need inspiration for my design"
        ],
        "leadership": [
            "How can I be a better leader?",
            "I need help managing conflict in my team",
            "What are effective communication strategies?",
            "How do I motivate my employees?",
            "I need advice on decision making"
        ]
    }

@pytest.fixture(scope="session")
def test_emotional_queries():
    """Test queries with emotional context"""
    return {
        "joy": [
            "I'm so happy and excited about my new job!",
            "I just got engaged and I'm over the moon!",
            "My team won the championship, I'm thrilled!"
        ],
        "sadness": [
            "I'm feeling very sad and lonely today",
            "I lost my job and I don't know what to do",
            "My relationship ended and I'm heartbroken"
        ],
        "anger": [
            "I'm so frustrated with this situation",
            "My colleague keeps undermining my work",
            "I'm angry about the unfair treatment"
        ],
        "fear": [
            "I'm scared about my medical test results",
            "I'm worried about losing my job",
            "I'm afraid of public speaking"
        ],
        "concern": [
            "I'm concerned about my child's health",
            "I'm worried about the company's future",
            "I'm anxious about the upcoming presentation"
        ]
    }

# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_benchmark():
    """Performance benchmark fixture"""
    return {
        "start_time": None,
        "end_time": None,
        "memory_usage": [],
        "cpu_usage": [],
        "operations_count": 0
    }

# Error simulation fixtures
@pytest.fixture(scope="function")
def error_scenarios():
    """Error scenarios for testing"""
    return {
        "connection_loss": {
            "type": "connection_loss",
            "duration": 300,  # 5 minutes
            "recovery_time": 60
        },
        "memory_overflow": {
            "type": "memory_overflow",
            "memory_usage": 95.0,
            "recovery_action": "restart_process"
        },
        "checkpoint_corruption": {
            "type": "checkpoint_corruption",
            "corrupted_files": ["adapter_model.safetensors"],
            "recovery_action": "restore_backup"
        },
        "training_interruption": {
            "type": "training_interruption",
            "interruption_step": 134,
            "recovery_action": "resume_from_checkpoint"
        }
    }

# Configuration fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "test_mode": True,
        "temp_dir": tempfile.gettempdir(),
        "max_test_duration": 300,  # 5 minutes
        "cleanup_after_tests": True,
        "verbose_output": True
    }

# Coverage reporting configuration
def pytest_configure(config):
    """Configure pytest for coverage reporting"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for better organization"""
    for item in items:
        # Mark tests based on their names
        if "test_gguf_conversion_system" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "test_connection_recovery" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "test_training_recovery" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

# Test result reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Custom test result summary"""
    print("\n" + "="*60)
    print("🧪 TARA UNIVERSAL MODEL TEST SUMMARY")
    print("="*60)
    
    # Count test results
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    errors = len(terminalreporter.stats.get('error', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    total = passed + failed + errors + skipped
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if total > 0:
        success_rate = (passed / total) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("="*60) 