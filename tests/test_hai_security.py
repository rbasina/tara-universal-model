import pytest
import asyncio
import tempfile
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading

# Import the security components
from tara_universal_model.security.privacy_manager import PrivacyManager, get_privacy_manager
from tara_universal_model.security.resource_monitor import ResourceMonitor, ResourceLimits, get_resource_monitor
from tara_universal_model.security.security_validator import SecurityValidator, get_security_validator

class TestPrivacyManager:
    """Test suite for Privacy Manager"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.privacy_manager = PrivacyManager(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'privacy_manager'):
            self.privacy_manager.shutdown()
    
    def test_privacy_manager_initialization(self):
        """Test privacy manager initialization"""
        assert self.privacy_manager is not None
        assert self.privacy_manager.data_dir.exists()
        assert self.privacy_manager.cipher is not None
        assert len(self.privacy_manager.zero_log_domains) > 0
    
    def test_session_creation(self):
        """Test secure session creation"""
        session_id = self.privacy_manager.create_session("healthcare", "test_user")
        
        assert session_id is not None
        assert len(session_id) == 16  # Expected session ID length
        assert session_id in self.privacy_manager.active_sessions
        
        session = self.privacy_manager.active_sessions[session_id]
        assert session["domain"] == "healthcare"
        assert session["user_id"] == "test_user"
        assert session["zero_log"] == True  # Healthcare is zero-log domain
    
    def test_zero_log_domain_behavior(self):
        """Test zero-logging behavior for sensitive domains"""
        session_id = self.privacy_manager.create_session("healthcare")
        
        # Log interaction in zero-log domain
        test_data = {"sensitive_info": "patient data", "diagnosis": "confidential"}
        result = self.privacy_manager.log_interaction(
            session_id, "diagnosis", test_data
        )
        
        assert result == True
        
        # Retrieve session data
        session_data = self.privacy_manager.get_session_data(session_id)
        
        # Verify no sensitive data is stored
        interaction = session_data["interactions"][0]
        assert "data_encrypted" not in interaction or not interaction["data_encrypted"]
        assert interaction["processed"] == True
    
    def test_regular_domain_encryption(self):
        """Test encryption for regular domains"""
        session_id = self.privacy_manager.create_session("education")
        
        # Log interaction in regular domain
        test_data = {"learning_content": "mathematics lesson", "progress": "good"}
        result = self.privacy_manager.log_interaction(
            session_id, "learning", test_data
        )
        
        assert result == True
        
        # Retrieve session data with decryption
        session_data = self.privacy_manager.get_session_data(session_id, decrypt=True)
        
        # Verify data is properly encrypted and decrypted
        interaction = session_data["interactions"][0]
        assert "data" in interaction
        assert interaction["data"]["learning_content"] == "mathematics lesson"
    
    def test_session_cleanup(self):
        """Test automatic session cleanup"""
        # Create session with zero retention
        session_id = self.privacy_manager.create_session("business")
        assert session_id in self.privacy_manager.active_sessions
        
        # End session (should trigger immediate cleanup for zero retention)
        self.privacy_manager.end_session(session_id)
        
        # For business domain (zero retention), session should be deleted
        assert session_id not in self.privacy_manager.active_sessions
    
    def test_retention_policy_customization(self):
        """Test custom retention policies"""
        # Set custom retention policy
        self.privacy_manager.set_retention_policy("education", 2)  # 2 hours
        
        session_id = self.privacy_manager.create_session("education")
        session = self.privacy_manager.active_sessions[session_id]
        
        assert session["retention_hours"] == 2
    
    def test_emergency_cleanup(self):
        """Test emergency cleanup functionality"""
        # Create multiple sessions
        session1 = self.privacy_manager.create_session("education")
        session2 = self.privacy_manager.create_session("creative")
        
        assert len(self.privacy_manager.active_sessions) == 2
        
        # Emergency cleanup
        self.privacy_manager.emergency_cleanup()
        
        assert len(self.privacy_manager.active_sessions) == 0
    
    def test_privacy_status(self):
        """Test privacy status reporting"""
        status = self.privacy_manager.get_privacy_status()
        
        assert "active_sessions" in status
        assert "zero_log_domains" in status
        assert "retention_policies" in status
        assert "encryption_enabled" in status
        assert status["encryption_enabled"] == True

class TestResourceMonitor:
    """Test suite for Resource Monitor"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limits = ResourceLimits(
            max_cpu_percent=50.0,
            max_memory_mb=1024,
            monitoring_interval=1
        )
        self.resource_monitor = ResourceMonitor(self.limits)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization"""
        assert self.resource_monitor is not None
        assert self.resource_monitor.limits.max_cpu_percent == 50.0
        assert self.resource_monitor.limits.max_memory_mb == 1024
        assert len(self.resource_monitor.usage_history) == 0
    
    def test_resource_usage_collection(self):
        """Test resource usage data collection"""
        usage = self.resource_monitor._collect_resource_usage()
        
        assert usage is not None
        assert hasattr(usage, 'cpu_percent')
        assert hasattr(usage, 'memory_mb')
        assert hasattr(usage, 'network_connections')
        assert hasattr(usage, 'timestamp')
        assert usage.cpu_percent >= 0
        assert usage.memory_mb >= 0
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop"""
        assert not self.resource_monitor.monitoring
        
        self.resource_monitor.start_monitoring()
        assert self.resource_monitor.monitoring
        
        # Wait a bit for monitoring to collect data
        time.sleep(2)
        
        self.resource_monitor.stop_monitoring()
        assert not self.resource_monitor.monitoring
        
        # Should have collected some usage data
        assert len(self.resource_monitor.usage_history) > 0
    
    def test_network_isolation_verification(self):
        """Test network isolation verification"""
        isolation_status = self.resource_monitor.verify_network_isolation()
        
        assert "isolated" in isolation_status
        assert "active_connections" in isolation_status
        assert "listening_ports" in isolation_status
        assert "dns_resolution" in isolation_status
        assert "internet_access" in isolation_status
        
        # Should detect that we're not fully isolated (development environment)
        assert isinstance(isolation_status["isolated"], bool)
    
    def test_performance_metrics_recording(self):
        """Test performance metrics recording"""
        # Record some test metrics
        self.resource_monitor.record_performance_metric(
            "test_response_time", 0.5, {"endpoint": "/test"}
        )
        
        assert "test_response_time" in self.resource_monitor.performance_metrics
        assert len(self.resource_monitor.performance_metrics["test_response_time"]) == 1
        
        metric = self.resource_monitor.performance_metrics["test_response_time"][0]
        assert metric["value"] == 0.5
        assert "timestamp" in metric
        assert "endpoint" in metric
    
    def test_resource_status_reporting(self):
        """Test resource status reporting"""
        status = self.resource_monitor.get_resource_status()
        
        assert "current_usage" in status
        assert "limits" in status
        assert "monitoring_active" in status
        assert "network_isolated" in status
        assert "tracked_processes" in status
        
        assert status["limits"]["max_cpu_percent"] == 50.0
        assert status["limits"]["max_memory_mb"] == 1024
    
    def test_performance_optimization(self):
        """Test performance optimization"""
        optimizations = self.resource_monitor.optimize_performance()
        
        assert "memory_cleanup" in optimizations
        assert "cache_optimization" in optimizations
        assert "process_optimization" in optimizations
        
        # Should complete without errors
        assert optimizations["memory_cleanup"] == True
    
    def test_alert_callback_system(self):
        """Test alert callback system"""
        alerts_received = []
        
        def test_callback(alert_type, data):
            alerts_received.append((alert_type, data))
        
        self.resource_monitor.add_alert_callback(test_callback)
        
        # Trigger an alert manually
        self.resource_monitor._trigger_alert("test_alert", {"test": "data"})
        
        assert len(alerts_received) == 1
        assert alerts_received[0][0] == "test_alert"
        assert alerts_received[0][1]["test"] == "data"
    
    def test_emergency_resource_cleanup(self):
        """Test emergency resource cleanup"""
        # Add some test data
        self.resource_monitor.record_performance_metric("test", 1.0)
        usage = self.resource_monitor._collect_resource_usage()
        self.resource_monitor._update_history(usage)
        
        assert len(self.resource_monitor.performance_metrics["test"]) > 0
        assert len(self.resource_monitor.usage_history) > 0
        
        # Emergency cleanup
        self.resource_monitor.emergency_resource_cleanup()
        
        # Should clear all data
        assert len(self.resource_monitor.performance_metrics["test"]) == 0
        assert len(self.resource_monitor.usage_history) == 0

class TestSecurityValidator:
    """Test suite for Security Validator"""
    
    def setup_method(self):
        """Setup for each test"""
        self.security_validator = SecurityValidator()
    
    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'security_validator'):
            self.security_validator.shutdown()
    
    def test_security_validator_initialization(self):
        """Test security validator initialization"""
        assert self.security_validator is not None
        assert len(self.security_validator.dangerous_patterns) > 0
        assert len(self.security_validator.compiled_patterns) > 0
        assert self.security_validator.secure_temp_dir.exists()
    
    def test_input_sanitization_safe_content(self):
        """Test input sanitization with safe content"""
        safe_text = "Hello, this is a normal message about healthcare."
        sanitized, is_safe = self.security_validator.sanitize_input(safe_text, "healthcare")
        
        assert is_safe == True
        assert sanitized == safe_text  # Should be unchanged
    
    def test_input_sanitization_dangerous_content(self):
        """Test input sanitization with dangerous content"""
        dangerous_text = "<script>alert('xss')</script>Hello world"
        sanitized, is_safe = self.security_validator.sanitize_input(dangerous_text)
        
        assert is_safe == False
        assert "<script>" not in sanitized
        assert "[FILTERED]" in sanitized
        assert "Hello world" in sanitized
    
    def test_input_sanitization_length_limits(self):
        """Test input length limits"""
        # Test normal length
        normal_text = "A" * 1000
        sanitized, is_safe = self.security_validator.sanitize_input(normal_text)
        assert len(sanitized) == 1000
        
        # Test excessive length
        long_text = "A" * 60000
        sanitized, is_safe = self.security_validator.sanitize_input(long_text)
        assert len(sanitized) <= 50000  # Should be truncated
    
    def test_sensitive_domain_extra_sanitization(self):
        """Test extra sanitization for sensitive domains"""
        text_with_pii = "Contact me at john@example.com or call 555-123-4567"
        sanitized, is_safe = self.security_validator.sanitize_input(text_with_pii, "healthcare")
        
        assert "[EMAIL_FILTERED]" in sanitized
        assert "[PHONE_FILTERED]" in sanitized
        assert "john@example.com" not in sanitized
        assert "555-123-4567" not in sanitized
    
    def test_file_access_validation_allowed_paths(self):
        """Test file access validation for allowed paths"""
        # Test access to current directory
        test_file = Path.cwd() / "test.txt"
        is_allowed, reason = self.security_validator.validate_file_access(str(test_file), "read")
        assert is_allowed == True
        
        # Test access to temp directory
        temp_file = Path(tempfile.gettempdir()) / "test.txt"
        is_allowed, reason = self.security_validator.validate_file_access(str(temp_file), "read")
        assert is_allowed == True
    
    def test_file_access_validation_dangerous_paths(self):
        """Test file access validation for dangerous paths"""
        # Test dangerous file extensions
        dangerous_file = "/tmp/malware.exe"
        is_allowed, reason = self.security_validator.validate_file_access(dangerous_file, "read")
        assert is_allowed == False
        assert "Dangerous file extension" in reason
        
        # Test path traversal
        traversal_path = "../../../etc/passwd"
        is_allowed, reason = self.security_validator.validate_file_access(traversal_path, "read")
        assert is_allowed == False
        assert "outside allowed directories" in reason
    
    def test_secure_temp_file_creation(self):
        """Test secure temporary file creation"""
        content = b"test content"
        temp_file = self.security_validator.create_secure_temp_file(
            suffix=".txt", content=content, cleanup_after=60
        )
        
        assert temp_file is not None
        assert Path(temp_file).exists()
        assert Path(temp_file).read_bytes() == content
        
        # File should be tracked for cleanup
        assert temp_file in self.security_validator.temp_files
    
    def test_temp_file_cleanup(self):
        """Test temporary file cleanup"""
        # Create a temp file
        temp_file = self.security_validator.create_secure_temp_file(
            content=b"test", cleanup_after=1  # 1 second cleanup
        )
        
        assert Path(temp_file).exists()
        
        # Wait for cleanup time
        time.sleep(2)
        
        # Trigger cleanup
        cleaned_count = self.security_validator.cleanup_temp_files()
        
        assert cleaned_count >= 1
        assert not Path(temp_file).exists()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client_id = "test_client"
        
        # Should allow initial requests
        is_allowed, remaining = self.security_validator.check_rate_limit(client_id)
        assert is_allowed == True
        assert remaining < self.security_validator.max_requests_per_minute
        
        # Simulate many requests
        for _ in range(self.security_validator.max_requests_per_minute):
            self.security_validator.check_rate_limit(client_id)
        
        # Should now be rate limited
        is_allowed, remaining = self.security_validator.check_rate_limit(client_id)
        assert is_allowed == False
        assert remaining == 0
    
    def test_json_validation_safe(self):
        """Test JSON validation with safe content"""
        safe_json = '{"message": "hello", "domain": "universal"}'
        is_valid, data, error = self.security_validator.validate_json_input(safe_json)
        
        assert is_valid == True
        assert data["message"] == "hello"
        assert data["domain"] == "universal"
        assert error == ""
    
    def test_json_validation_dangerous(self):
        """Test JSON validation with dangerous content"""
        dangerous_json = '{"message": "<script>alert(1)</script>", "domain": "universal"}'
        is_valid, data, error = self.security_validator.validate_json_input(dangerous_json)
        
        assert is_valid == False
        assert "dangerous patterns" in error
    
    def test_json_validation_complex_structure(self):
        """Test JSON validation with overly complex structure"""
        # Create deeply nested JSON
        nested_json = '{"level1": {"level2": {"level3": {"level4": {"level5": {"level6": {"level7": {"level8": {"level9": {"level10": {"level11": "too deep"}}}}}}}}}}}'
        is_valid, data, error = self.security_validator.validate_json_input(nested_json)
        
        assert is_valid == False
        assert "too complex" in error
    
    def test_isolated_process_creation(self):
        """Test isolated process creation"""
        # Test safe command
        success, stdout, stderr = self.security_validator.create_isolated_process(
            ["python", "-c", "print('hello world')"], timeout=10
        )
        
        assert success == True
        assert "hello world" in stdout
        assert stderr == ""
    
    def test_isolated_process_dangerous_command(self):
        """Test isolated process with dangerous command"""
        # Test command with dangerous patterns
        success, stdout, stderr = self.security_validator.create_isolated_process(
            ["python", "-c", "import os; os.system('rm -rf /')"], timeout=10
        )
        
        assert success == False
        assert "dangerous patterns" in stderr
    
    def test_security_status_reporting(self):
        """Test security status reporting"""
        status = self.security_validator.get_security_status()
        
        assert "temp_files_tracked" in status
        assert "isolated_processes" in status
        assert "rate_limit_clients" in status
        assert "cleanup_running" in status
        assert "secure_temp_dir" in status
        assert "max_requests_per_minute" in status
        assert "dangerous_patterns_count" in status
        
        assert status["max_requests_per_minute"] == 60
        assert status["dangerous_patterns_count"] > 0
    
    def test_emergency_security_cleanup(self):
        """Test emergency security cleanup"""
        # Create some test data
        self.security_validator.create_secure_temp_file(content=b"test")
        self.security_validator.check_rate_limit("test_client")
        
        assert len(self.security_validator.temp_files) > 0
        assert len(self.security_validator.request_history) > 0
        
        # Emergency cleanup
        self.security_validator.emergency_security_cleanup()
        
        # Should clear all data
        assert len(self.security_validator.temp_files) == 0
        assert len(self.security_validator.request_history) == 0

class TestHAISecurityIntegration:
    """Test suite for HAI Security Integration"""
    
    def test_global_instance_creation(self):
        """Test global instance creation"""
        privacy_manager = get_privacy_manager()
        resource_monitor = get_resource_monitor()
        security_validator = get_security_validator()
        
        assert privacy_manager is not None
        assert resource_monitor is not None
        assert security_validator is not None
        
        # Should return same instances on subsequent calls
        assert get_privacy_manager() is privacy_manager
        assert get_resource_monitor() is resource_monitor
        assert get_security_validator() is security_validator
    
    def test_integrated_security_workflow(self):
        """Test integrated security workflow"""
        # Get all security components
        privacy_manager = get_privacy_manager()
        resource_monitor = get_resource_monitor()
        security_validator = get_security_validator()
        
        # Create secure session
        session_id = privacy_manager.create_session("healthcare", "test_user")
        
        # Validate and sanitize input
        user_input = "Patient has symptoms of <script>alert('xss')</script> fever"
        sanitized_input, is_safe = security_validator.sanitize_input(user_input, "healthcare")
        
        # Log interaction with privacy protection
        privacy_manager.log_interaction(session_id, "diagnosis", {
            "original_input": user_input,
            "sanitized_input": sanitized_input,
            "is_safe": is_safe
        })
        
        # Check rate limiting
        is_allowed, remaining = security_validator.check_rate_limit("test_client")
        assert is_allowed == True
        
        # Record performance metric
        resource_monitor.record_performance_metric("security_check_time", 0.05)
        
        # Verify everything worked together
        assert session_id in privacy_manager.active_sessions
        assert not is_safe  # XSS should be detected
        assert "[FILTERED]" in sanitized_input
        assert "security_check_time" in resource_monitor.performance_metrics
    
    def test_security_under_load(self):
        """Test security components under load"""
        privacy_manager = get_privacy_manager()
        security_validator = get_security_validator()
        
        # Create multiple sessions and validate inputs concurrently
        import concurrent.futures
        
        def create_session_and_validate(i):
            session_id = privacy_manager.create_session("education", f"user_{i}")
            input_text = f"Test message {i} with potential <script>alert({i})</script> threat"
            sanitized, is_safe = security_validator.sanitize_input(input_text)
            return session_id, sanitized, is_safe
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session_and_validate, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        assert len(results) == 20
        
        # All should detect the XSS
        for session_id, sanitized, is_safe in results:
            assert session_id is not None
            assert not is_safe
            assert "[FILTERED]" in sanitized

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 