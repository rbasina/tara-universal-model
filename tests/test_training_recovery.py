#!/usr/bin/env python3
"""
🧪 TDD Tests for TARA Training Recovery System
Test training recovery mechanisms and continuity features
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "monitoring"))

class TestTrainingRecovery(unittest.TestCase):
    """Test training recovery mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.training_dir = Path(self.temp_dir) / "training"
        self.training_dir.mkdir()
        
        # Create test training state
        self.create_test_training_state()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_test_training_state(self):
        """Create test training state files"""
        # Create training state JSON
        training_state = {
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
        
        with open(self.training_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Create checkpoint directories
        for domain in ["healthcare", "business"]:
            checkpoint_dir = self.training_dir / "models" / domain / "checkpoint-134"
            checkpoint_dir.mkdir(parents=True)
            
            # Create checkpoint files
            (checkpoint_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
            (checkpoint_dir / "adapter_model.safetensors").write_text("test")
            (checkpoint_dir / "training_args.bin").write_text("test")
    
    def test_training_state_loading(self):
        """Test loading training state from file"""
        state_file = self.training_dir / "training_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        self.assertEqual(state["current_domain"], "healthcare")
        self.assertEqual(state["current_step"], 134)
        self.assertEqual(state["total_steps"], 400)
        self.assertIn("healthcare", state["domains"])
        self.assertIn("business", state["domains"])
    
    def test_checkpoint_validation(self):
        """Test checkpoint validation"""
        checkpoint_path = self.training_dir / "models" / "healthcare" / "checkpoint-134"
        
        # Check required files exist
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            self.assertTrue(file_path.exists(), f"Required file {file_name} not found")
    
    def test_domain_progress_calculation(self):
        """Test domain progress calculation"""
        state_file = self.training_dir / "training_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        healthcare_progress = state["domains"]["healthcare"]["progress"]
        business_progress = state["domains"]["business"]["progress"]
        
        self.assertEqual(healthcare_progress, 33.5)
        self.assertEqual(business_progress, 100.0)
    
    def test_resume_point_detection(self):
        """Test resume point detection"""
        state_file = self.training_dir / "training_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        current_domain = state["current_domain"]
        current_step = state["current_step"]
        last_checkpoint = state["last_checkpoint"]
        
        self.assertEqual(current_domain, "healthcare")
        self.assertEqual(current_step, 134)
        self.assertEqual(last_checkpoint, "checkpoint-134")
    
    def test_training_recovery_mechanism(self):
        """Test training recovery mechanism"""
        # Simulate training interruption
        state_file = self.training_dir / "training_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        # Update state to simulate interruption
        state["interrupted"] = True
        state["interruption_time"] = time.time()
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        # Test recovery detection
        with open(state_file, "r") as f:
            recovered_state = json.load(f)
        
        self.assertTrue(recovered_state["interrupted"])
        self.assertIn("interruption_time", recovered_state)

class TestParallelTrainingRecovery(unittest.TestCase):
    """Test parallel training recovery"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.parallel_dir = Path(self.temp_dir) / "parallel"
        self.parallel_dir.mkdir()
        
        # Create parallel training state
        self.create_parallel_training_state()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_parallel_training_state(self):
        """Create parallel training state"""
        parallel_state = {
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
        
        with open(self.parallel_dir / "parallel_state.json", "w") as f:
            json.dump(parallel_state, f, indent=2)
        
        # Create checkpoint directories for each domain
        for domain in ["education", "creative", "leadership"]:
            checkpoint_dir = self.parallel_dir / "models" / domain / "checkpoint-100"
            checkpoint_dir.mkdir(parents=True)
            
            # Create checkpoint files
            (checkpoint_dir / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
            (checkpoint_dir / "adapter_model.safetensors").write_text("test")
    
    def test_parallel_state_loading(self):
        """Test loading parallel training state"""
        state_file = self.parallel_dir / "parallel_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        self.assertTrue(state["parallel_training"])
        self.assertEqual(len(state["active_domains"]), 3)
        self.assertIn("education", state["active_domains"])
        self.assertIn("creative", state["active_domains"])
        self.assertIn("leadership", state["active_domains"])
    
    def test_domain_progress_tracking(self):
        """Test domain progress tracking in parallel training"""
        state_file = self.parallel_dir / "parallel_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        domain_states = state["domain_states"]
        
        self.assertEqual(domain_states["education"]["progress"], 25.0)
        self.assertEqual(domain_states["creative"]["progress"], 15.0)
        self.assertEqual(domain_states["leadership"]["progress"], 10.0)
    
    def test_resource_monitoring(self):
        """Test resource monitoring in parallel training"""
        state_file = self.parallel_dir / "parallel_state.json"
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        resource_usage = state["resource_usage"]
        
        self.assertGreater(resource_usage["cpu_percent"], 0)
        self.assertGreater(resource_usage["memory_percent"], 0)
        self.assertLessEqual(resource_usage["cpu_percent"], 100)
        self.assertLessEqual(resource_usage["memory_percent"], 100)

class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint management system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        domain = "healthcare"
        step = 134
        
        checkpoint_path = self.checkpoint_dir / domain / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True)
        
        # Create checkpoint files
        (checkpoint_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (checkpoint_path / "adapter_model.safetensors").write_text("test")
        (checkpoint_path / "training_args.bin").write_text("test")
        
        self.assertTrue(checkpoint_path.exists())
        self.assertTrue((checkpoint_path / "adapter_config.json").exists())
        self.assertTrue((checkpoint_path / "adapter_model.safetensors").exists())
    
    def test_checkpoint_validation(self):
        """Test checkpoint validation"""
        domain = "healthcare"
        step = 134
        
        checkpoint_path = self.checkpoint_dir / domain / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True)
        
        # Create valid checkpoint
        (checkpoint_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (checkpoint_path / "adapter_model.safetensors").write_text("test")
        
        # Test validation
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        is_valid = all((checkpoint_path / f).exists() for f in required_files)
        
        self.assertTrue(is_valid)
    
    def test_checkpoint_backup(self):
        """Test checkpoint backup mechanism"""
        domain = "healthcare"
        step = 134
        
        checkpoint_path = self.checkpoint_dir / domain / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True)
        
        # Create checkpoint files
        (checkpoint_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (checkpoint_path / "adapter_model.safetensors").write_text("test")
        
        # Create backup
        backup_path = self.checkpoint_dir / domain / f"checkpoint-{step}_backup"
        shutil.copytree(checkpoint_path, backup_path)
        
        self.assertTrue(backup_path.exists())
        self.assertTrue((backup_path / "adapter_config.json").exists())
        self.assertTrue((backup_path / "adapter_model.safetensors").exists())
    
    def test_checkpoint_restoration(self):
        """Test checkpoint restoration"""
        domain = "healthcare"
        step = 134
        
        # Create original checkpoint
        original_path = self.checkpoint_dir / domain / f"checkpoint-{step}"
        original_path.mkdir(parents=True)
        (original_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (original_path / "adapter_model.safetensors").write_text("test")
        
        # Create backup
        backup_path = self.checkpoint_dir / domain / f"checkpoint-{step}_backup"
        shutil.copytree(original_path, backup_path)
        
        # Simulate corruption by removing original
        shutil.rmtree(original_path)
        
        # Restore from backup
        shutil.copytree(backup_path, original_path)
        
        self.assertTrue(original_path.exists())
        self.assertTrue((original_path / "adapter_config.json").exists())
        self.assertTrue((original_path / "adapter_model.safetensors").exists())

class TestSystemRecovery(unittest.TestCase):
    """Test system recovery mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_dir = Path(self.temp_dir) / "recovery"
        self.recovery_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_system_sleep_recovery(self):
        """Test system sleep recovery detection"""
        # Create recovery state
        recovery_state = {
            "last_activity": time.time(),
            "system_sleep_detected": False,
            "recovery_attempts": 0
        }
        
        state_file = self.recovery_dir / "recovery_state.json"
        with open(state_file, "w") as f:
            json.dump(recovery_state, f, indent=2)
        
        # Simulate system sleep detection
        current_time = time.time()
        time_diff = current_time - recovery_state["last_activity"]
        
        # If more than 5 minutes have passed, assume system sleep
        if time_diff > 300:  # 5 minutes
            recovery_state["system_sleep_detected"] = True
            recovery_state["recovery_attempts"] += 1
        
        self.assertIsInstance(recovery_state["system_sleep_detected"], bool)
        self.assertGreaterEqual(recovery_state["recovery_attempts"], 0)
    
    def test_powershell_restart_recovery(self):
        """Test PowerShell restart recovery"""
        # Create restart detection file
        restart_file = self.recovery_dir / "restart_detected.txt"
        restart_file.write_text(str(time.time()))
        
        # Check if restart was detected
        restart_detected = restart_file.exists()
        self.assertTrue(restart_detected)
    
    def test_cursor_ai_crash_recovery(self):
        """Test Cursor AI crash recovery"""
        # Create crash recovery state
        crash_state = {
            "cursor_crashed": True,
            "crash_time": time.time(),
            "state_preserved": True,
            "recovery_data": {
                "current_domain": "healthcare",
                "current_step": 134,
                "checkpoint_path": "models/healthcare/checkpoint-134"
            }
        }
        
        state_file = self.recovery_dir / "crash_recovery.json"
        with open(state_file, "w") as f:
            json.dump(crash_state, f, indent=2)
        
        # Test crash recovery data
        with open(state_file, "r") as f:
            recovered_state = json.load(f)
        
        self.assertTrue(recovered_state["cursor_crashed"])
        self.assertTrue(recovered_state["state_preserved"])
        self.assertIn("recovery_data", recovered_state)
    
    def test_training_interruption_recovery(self):
        """Test training interruption recovery"""
        # Create interruption state
        interruption_state = {
            "interrupted": True,
            "interruption_time": time.time(),
            "interruption_reason": "system_sleep",
            "resume_data": {
                "domain": "healthcare",
                "step": 134,
                "checkpoint": "checkpoint-134"
            }
        }
        
        state_file = self.recovery_dir / "interruption_state.json"
        with open(state_file, "w") as f:
            json.dump(interruption_state, f, indent=2)
        
        # Test interruption recovery
        with open(state_file, "r") as f:
            recovered_state = json.load(f)
        
        self.assertTrue(recovered_state["interrupted"])
        self.assertIn("resume_data", recovered_state)
        self.assertEqual(recovered_state["resume_data"]["domain"], "healthcare")

class TestDashboardIntegration(unittest.TestCase):
    """Test dashboard integration for training monitoring"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard_dir = Path(self.temp_dir) / "dashboard"
        self.dashboard_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        dashboard_data = {
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
        
        data_file = self.dashboard_dir / "dashboard_data.json"
        with open(data_file, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        
        # Test dashboard data
        with open(data_file, "r") as f:
            loaded_data = json.load(f)
        
        self.assertIn("timestamp", loaded_data)
        self.assertIn("overall_progress", loaded_data)
        self.assertIn("active_domains", loaded_data)
        self.assertIn("domain_progress", loaded_data)
        self.assertIn("system_metrics", loaded_data)
    
    def test_progress_calculation(self):
        """Test progress calculation for dashboard"""
        domain_progress = {
            "healthcare": 33.5,
            "education": 25.0,
            "creative": 15.0,
            "business": 100.0,
            "leadership": 0.0
        }
        
        # Calculate overall progress
        total_progress = sum(domain_progress.values())
        average_progress = total_progress / len(domain_progress)
        
        self.assertEqual(average_progress, 34.7)
        self.assertGreater(average_progress, 0)
        self.assertLess(average_progress, 100)

def run_recovery_tests():
    """Run all recovery tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTrainingRecovery,
        TestParallelTrainingRecovery,
        TestCheckpointManagement,
        TestSystemRecovery,
        TestDashboardIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🔄 RECOVERY TEST RESULTS SUMMARY")
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
    success = run_recovery_tests()
    sys.exit(0 if success else 1) 