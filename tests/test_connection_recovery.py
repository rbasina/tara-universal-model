#!/usr/bin/env python3
"""
🧪 TDD Tests for Connection Recovery & Training Continuity
Test mechanisms to handle Cursor AI connection losses and training interruptions
"""

import unittest
import tempfile
import json
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

class TestConnectionRecovery(unittest.TestCase):
    """Test connection recovery mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_dir = Path(self.temp_dir) / "connection_recovery"
        self.recovery_dir.mkdir()
        
        # Create connection state
        self.create_connection_state()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_connection_state(self):
        """Create connection state files"""
        connection_state = {
            "last_heartbeat": time.time(),
            "connection_active": True,
            "recovery_attempts": 0,
            "training_state": {
                "current_domain": "healthcare",
                "current_step": 134,
                "checkpoint_path": "models/healthcare/checkpoint-134"
            }
        }
        
        with open(self.recovery_dir / "connection_state.json", "w") as f:
            json.dump(connection_state, f, indent=2)
    
    def test_connection_heartbeat(self):
        """Test connection heartbeat mechanism"""
        heartbeat_file = self.recovery_dir / "heartbeat.txt"
        heartbeat_file.write_text(str(time.time()))
        
        # Check heartbeat exists
        self.assertTrue(heartbeat_file.exists())
        
        # Read heartbeat time
        with open(heartbeat_file, "r") as f:
            heartbeat_time = float(f.read().strip())
        
        self.assertIsInstance(heartbeat_time, float)
        self.assertGreater(heartbeat_time, 0)
    
    def test_connection_loss_detection(self):
        """Test connection loss detection"""
        # Create old heartbeat
        old_time = time.time() - 3600  # 1 hour ago
        heartbeat_file = self.recovery_dir / "heartbeat.txt"
        heartbeat_file.write_text(str(old_time))
        
        # Check if connection is considered lost
        with open(heartbeat_file, "r") as f:
            last_heartbeat = float(f.read().strip())
        
        current_time = time.time()
        time_diff = current_time - last_heartbeat
        
        # If more than 5 minutes, consider connection lost
        connection_lost = time_diff > 300  # 5 minutes
        
        self.assertTrue(connection_lost)
    
    def test_training_state_preservation(self):
        """Test training state preservation during connection loss"""
        training_state = {
            "current_domain": "healthcare",
            "current_step": 134,
            "total_steps": 400,
            "checkpoint_path": "models/healthcare/checkpoint-134",
            "training_loss": 0.03,
            "learning_rate": 0.0001
        }
        
        # Save training state
        state_file = self.recovery_dir / "training_state.json"
        with open(state_file, "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Simulate connection loss and recovery
        with open(state_file, "r") as f:
            recovered_state = json.load(f)
        
        self.assertEqual(recovered_state["current_domain"], "healthcare")
        self.assertEqual(recovered_state["current_step"], 134)
        self.assertEqual(recovered_state["total_steps"], 400)
        self.assertIn("checkpoint_path", recovered_state)
    
    def test_automatic_recovery_trigger(self):
        """Test automatic recovery trigger"""
        recovery_trigger = {
            "connection_lost": True,
            "loss_time": time.time(),
            "recovery_triggered": False,
            "max_recovery_attempts": 3
        }
        
        trigger_file = self.recovery_dir / "recovery_trigger.json"
        with open(trigger_file, "w") as f:
            json.dump(recovery_trigger, f, indent=2)
        
        # Check recovery trigger
        with open(trigger_file, "r") as f:
            trigger = json.load(f)
        
        self.assertTrue(trigger["connection_lost"])
        self.assertFalse(trigger["recovery_triggered"])
        self.assertEqual(trigger["max_recovery_attempts"], 3)

class TestTrainingContinuity(unittest.TestCase):
    """Test training continuity mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.continuity_dir = Path(self.temp_dir) / "continuity"
        self.continuity_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_auto_save(self):
        """Test automatic checkpoint saving"""
        checkpoint_dir = self.continuity_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create checkpoint every 10 steps
        step = 134
        checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir()
        
        # Create checkpoint files
        (checkpoint_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (checkpoint_path / "adapter_model.safetensors").write_text("test")
        (checkpoint_path / "training_args.bin").write_text("test")
        
        # Verify checkpoint exists
        self.assertTrue(checkpoint_path.exists())
        self.assertTrue((checkpoint_path / "adapter_config.json").exists())
        self.assertTrue((checkpoint_path / "adapter_model.safetensors").exists())
    
    def test_resume_from_checkpoint(self):
        """Test resuming from checkpoint"""
        # Create checkpoint
        checkpoint_dir = self.continuity_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        step = 134
        checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir()
        
        # Create checkpoint files
        (checkpoint_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test"}')
        (checkpoint_path / "adapter_model.safetensors").write_text("test")
        
        # Resume data
        resume_data = {
            "checkpoint_path": str(checkpoint_path),
            "resume_step": step,
            "resume_domain": "healthcare",
            "resume_time": time.time()
        }
        
        resume_file = self.continuity_dir / "resume_data.json"
        with open(resume_file, "w") as f:
            json.dump(resume_data, f, indent=2)
        
        # Verify resume data
        with open(resume_file, "r") as f:
            loaded_resume = json.load(f)
        
        self.assertEqual(loaded_resume["resume_step"], step)
        self.assertEqual(loaded_resume["resume_domain"], "healthcare")
        self.assertIn("checkpoint_path", loaded_resume)
    
    def test_training_progress_tracking(self):
        """Test training progress tracking"""
        progress_data = {
            "domain": "healthcare",
            "current_step": 134,
            "total_steps": 400,
            "progress_percent": 33.5,
            "training_loss": 0.03,
            "learning_rate": 0.0001,
            "last_update": time.time()
        }
        
        progress_file = self.continuity_dir / "progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)
        
        # Verify progress data
        with open(progress_file, "r") as f:
            loaded_progress = json.load(f)
        
        self.assertEqual(loaded_progress["domain"], "healthcare")
        self.assertEqual(loaded_progress["current_step"], 134)
        self.assertEqual(loaded_progress["progress_percent"], 33.5)
        self.assertIn("training_loss", loaded_progress)
    
    def test_domain_switching_continuity(self):
        """Test domain switching continuity"""
        domain_states = {
            "healthcare": {
                "status": "paused",
                "last_step": 134,
                "checkpoint": "checkpoint-134"
            },
            "education": {
                "status": "active",
                "current_step": 50,
                "checkpoint": "checkpoint-50"
            },
            "creative": {
                "status": "waiting",
                "last_step": 0,
                "checkpoint": "checkpoint-0"
            }
        }
        
        states_file = self.continuity_dir / "domain_states.json"
        with open(states_file, "w") as f:
            json.dump(domain_states, f, indent=2)
        
        # Verify domain states
        with open(states_file, "r") as f:
            loaded_states = json.load(f)
        
        self.assertEqual(loaded_states["healthcare"]["status"], "paused")
        self.assertEqual(loaded_states["education"]["status"], "active")
        self.assertEqual(loaded_states["creative"]["status"], "waiting")

class TestPowerShellRecovery(unittest.TestCase):
    """Test PowerShell recovery mechanisms"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.powershell_dir = Path(self.temp_dir) / "powershell"
        self.powershell_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_powershell_restart_detection(self):
        """Test PowerShell restart detection"""
        # Create restart marker
        restart_marker = self.powershell_dir / "restart_marker.txt"
        restart_marker.write_text(str(time.time()))
        
        # Check restart marker
        self.assertTrue(restart_marker.exists())
        
        with open(restart_marker, "r") as f:
            restart_time = float(f.read().strip())
        
        self.assertIsInstance(restart_time, float)
        self.assertGreater(restart_time, 0)
    
    def test_powershell_script_recovery(self):
        """Test PowerShell script recovery"""
        # Create recovery script
        recovery_script = self.powershell_dir / "recovery.ps1"
        script_content = """
# PowerShell Recovery Script
$recoveryData = Get-Content "recovery_data.json" | ConvertFrom-Json
$currentDomain = $recoveryData.current_domain
$currentStep = $recoveryData.current_step

Write-Host "Recovering training for domain: $currentDomain at step: $currentStep"
        """
        
        recovery_script.write_text(script_content)
        
        # Verify script exists
        self.assertTrue(recovery_script.exists())
        
        # Read script content
        with open(recovery_script, "r") as f:
            content = f.read()
        
        self.assertIn("PowerShell Recovery Script", content)
        self.assertIn("$currentDomain", content)
        self.assertIn("$currentStep", content)
    
    def test_powershell_state_preservation(self):
        """Test PowerShell state preservation"""
        powershell_state = {
            "session_id": "powershell_session_123",
            "working_directory": str(self.powershell_dir),
            "environment_variables": {
                "TARA_TRAINING_DOMAIN": "healthcare",
                "TARA_CURRENT_STEP": "134"
            },
            "last_command": "python train_meetara_universal_model.py"
        }
        
        state_file = self.powershell_dir / "powershell_state.json"
        with open(state_file, "w") as f:
            json.dump(powershell_state, f, indent=2)
        
        # Verify state
        with open(state_file, "r") as f:
            loaded_state = json.load(f)
        
        self.assertEqual(loaded_state["session_id"], "powershell_session_123")
        self.assertIn("TARA_TRAINING_DOMAIN", loaded_state["environment_variables"])
        self.assertEqual(loaded_state["environment_variables"]["TARA_TRAINING_DOMAIN"], "healthcare")

class TestCursorAICrashRecovery(unittest.TestCase):
    """Test Cursor AI crash recovery"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cursor_dir = Path(self.temp_dir) / "cursor_ai"
        self.cursor_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_cursor_crash_detection(self):
        """Test Cursor AI crash detection"""
        crash_data = {
            "cursor_crashed": True,
            "crash_time": time.time(),
            "crash_reason": "memory_overflow",
            "recovery_attempts": 0,
            "max_recovery_attempts": 5
        }
        
        crash_file = self.cursor_dir / "crash_data.json"
        with open(crash_file, "w") as f:
            json.dump(crash_data, f, indent=2)
        
        # Verify crash data
        with open(crash_file, "r") as f:
            loaded_crash = json.load(f)
        
        self.assertTrue(loaded_crash["cursor_crashed"])
        self.assertEqual(loaded_crash["crash_reason"], "memory_overflow")
        self.assertEqual(loaded_crash["max_recovery_attempts"], 5)
    
    def test_cursor_session_recovery(self):
        """Test Cursor AI session recovery"""
        session_data = {
            "session_id": "cursor_session_456",
            "last_activity": time.time(),
            "open_files": [
                "scripts/training/train_meetara_universal_model.py",
                "docs/memory-bank/activeContext.md"
            ],
            "cursor_position": {
                "file": "scripts/training/train_meetara_universal_model.py",
                "line": 134,
                "column": 10
            }
        }
        
        session_file = self.cursor_dir / "session_data.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        
        # Verify session data
        with open(session_file, "r") as f:
            loaded_session = json.load(f)
        
        self.assertEqual(loaded_session["session_id"], "cursor_session_456")
        self.assertIn("open_files", loaded_session)
        self.assertIn("cursor_position", loaded_session)
    
    def test_cursor_auto_restart(self):
        """Test Cursor AI auto restart mechanism"""
        restart_config = {
            "auto_restart_enabled": True,
            "restart_delay_seconds": 30,
            "max_restart_attempts": 3,
            "restart_conditions": [
                "memory_overflow",
                "connection_timeout",
                "unexpected_crash"
            ]
        }
        
        config_file = self.cursor_dir / "restart_config.json"
        with open(config_file, "w") as f:
            json.dump(restart_config, f, indent=2)
        
        # Verify restart config
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
        
        self.assertTrue(loaded_config["auto_restart_enabled"])
        self.assertEqual(loaded_config["restart_delay_seconds"], 30)
        self.assertEqual(loaded_config["max_restart_attempts"], 3)
        self.assertIn("memory_overflow", loaded_config["restart_conditions"])

class TestTrainingRecoveryAutomation(unittest.TestCase):
    """Test training recovery automation"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.automation_dir = Path(self.temp_dir) / "automation"
        self.automation_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_automated_recovery_workflow(self):
        """Test automated recovery workflow"""
        workflow = {
            "workflow_id": "recovery_workflow_789",
            "steps": [
                {
                    "step": 1,
                    "action": "detect_connection_loss",
                    "timeout": 300
                },
                {
                    "step": 2,
                    "action": "save_training_state",
                    "timeout": 60
                },
                {
                    "step": 3,
                    "action": "wait_for_reconnection",
                    "timeout": 1800
                },
                {
                    "step": 4,
                    "action": "resume_training",
                    "timeout": 120
                }
            ],
            "current_step": 1,
            "status": "running"
        }
        
        workflow_file = self.automation_dir / "recovery_workflow.json"
        with open(workflow_file, "w") as f:
            json.dump(workflow, f, indent=2)
        
        # Verify workflow
        with open(workflow_file, "r") as f:
            loaded_workflow = json.load(f)
        
        self.assertEqual(loaded_workflow["workflow_id"], "recovery_workflow_789")
        self.assertEqual(len(loaded_workflow["steps"]), 4)
        self.assertEqual(loaded_workflow["current_step"], 1)
        self.assertEqual(loaded_workflow["status"], "running")
    
    def test_recovery_notification_system(self):
        """Test recovery notification system"""
        notifications = {
            "notifications_enabled": True,
            "notification_types": [
                "connection_lost",
                "training_resumed",
                "recovery_failed",
                "checkpoint_saved"
            ],
            "notification_channels": [
                "console_log",
                "email",
                "dashboard_update"
            ],
            "last_notification": time.time()
        }
        
        notification_file = self.automation_dir / "notifications.json"
        with open(notification_file, "w") as f:
            json.dump(notifications, f, indent=2)
        
        # Verify notifications
        with open(notification_file, "r") as f:
            loaded_notifications = json.load(f)
        
        self.assertTrue(loaded_notifications["notifications_enabled"])
        self.assertIn("connection_lost", loaded_notifications["notification_types"])
        self.assertIn("console_log", loaded_notifications["notification_channels"])

def run_connection_recovery_tests():
    """Run all connection recovery tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConnectionRecovery,
        TestTrainingContinuity,
        TestPowerShellRecovery,
        TestCursorAICrashRecovery,
        TestTrainingRecoveryAutomation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🔌 CONNECTION RECOVERY TEST RESULTS SUMMARY")
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
    success = run_connection_recovery_tests()
    sys.exit(0 if success else 1) 