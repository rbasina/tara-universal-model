import re
import os
import tempfile
import hashlib
import html
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import uuid
import shutil
from urllib.parse import urlparse
import threading
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """
    Comprehensive Security Validation for TARA Universal Model
    
    Features:
    - Input sanitization for all voice/text inputs
    - Secure temporary file handling
    - Process isolation for model inference
    - XSS and injection attack prevention
    - Content filtering and validation
    """
    
    def __init__(self):
        # Dangerous patterns for input validation
        self.dangerous_patterns = [
            # Script injection patterns
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            
            # SQL injection patterns
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*\s+set',
            
            # Command injection patterns
            r';\s*rm\s+-rf',
            r';\s*del\s+',
            r';\s*format\s+',
            r'&&\s*rm\s+',
            r'\|\s*rm\s+',
            r'`.*`',
            r'\$\(.*\)',
            
            # Path traversal patterns
            r'\.\./.*',
            r'\.\.\\.*',
            r'/etc/passwd',
            r'/etc/shadow',
            r'C:\\Windows\\System32',
            
            # Suspicious file extensions
            r'\.exe\s*$',
            r'\.bat\s*$',
            r'\.cmd\s*$',
            r'\.ps1\s*$',
            r'\.vbs\s*$',
            r'\.scr\s*$'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.dangerous_patterns]
        
        # Allowed domains for sensitive operations
        self.sensitive_domains = {"healthcare", "business", "financial", "mental_health"}
        
        # Secure temp directory
        self.secure_temp_dir = Path(tempfile.gettempdir()) / "tara_secure"
        self.secure_temp_dir.mkdir(exist_ok=True)
        
        # File cleanup tracking
        self.temp_files: Dict[str, Dict] = {}
        self.cleanup_thread = None
        self.cleanup_running = False
        
        # Process isolation tracking
        self.isolated_processes: Dict[str, subprocess.Popen] = {}
        
        # Rate limiting
        self.request_history: Dict[str, List[datetime]] = {}
        self.max_requests_per_minute = 60
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def sanitize_input(self, input_text: str, domain: str = "universal") -> Tuple[str, bool]:
        """
        Sanitize user input for security
        
        Returns:
            Tuple of (sanitized_text, is_safe)
        """
        if not input_text or not isinstance(input_text, str):
            return "", False
        
        original_text = input_text
        
        # Basic HTML escaping
        sanitized = html.escape(input_text)
        
        # Remove null bytes and control characters
        sanitized = sanitized.replace('\x00', '').replace('\r', '').replace('\x08', '')
        
        # Limit length to prevent DoS
        max_length = 10000 if domain in self.sensitive_domains else 50000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Check for dangerous patterns
        is_safe = True
        detected_threats = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(sanitized):
                is_safe = False
                detected_threats.append(self.dangerous_patterns[i])
                # Remove the dangerous content
                sanitized = pattern.sub('[FILTERED]', sanitized)
        
        # Additional sanitization for sensitive domains
        if domain in self.sensitive_domains:
            sanitized = self._extra_sanitization(sanitized)
        
        # Log security events
        if not is_safe:
            self._log_security_event("input_sanitization", {
                "domain": domain,
                "threats_detected": detected_threats,
                "original_length": len(original_text),
                "sanitized_length": len(sanitized)
            })
        
        return sanitized, is_safe
    
    def _extra_sanitization(self, text: str) -> str:
        """Extra sanitization for sensitive domains"""
        # Remove potential data exfiltration patterns
        text = re.sub(r'http[s]?://[^\s]+', '[URL_FILTERED]', text, flags=re.IGNORECASE)
        text = re.sub(r'ftp://[^\s]+', '[FTP_FILTERED]', text, flags=re.IGNORECASE)
        text = re.sub(r'file://[^\s]+', '[FILE_FILTERED]', text, flags=re.IGNORECASE)
        
        # Remove email addresses (potential PII)
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_FILTERED]', text)
        
        # Remove potential phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_FILTERED]', text)
        
        # Remove potential SSN patterns
        text = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_FILTERED]', text)
        
        return text
    
    def validate_file_access(self, file_path: str, operation: str = "read") -> Tuple[bool, str]:
        """
        Validate file access for security
        
        Args:
            file_path: Path to file
            operation: Type of operation (read, write, execute)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            path = Path(file_path).resolve()
            
            # Check if path exists and is within allowed directories
            allowed_dirs = [
                Path.cwd(),
                Path.home() / ".tara",
                self.secure_temp_dir,
                Path(tempfile.gettempdir())
            ]
            
            # Check if path is within allowed directories
            is_allowed_path = any(
                str(path).startswith(str(allowed_dir.resolve()))
                for allowed_dir in allowed_dirs
            )
            
            if not is_allowed_path:
                return False, f"Path outside allowed directories: {path}"
            
            # Check for dangerous file extensions
            dangerous_extensions = {'.exe', '.bat', '.cmd', '.ps1', '.vbs', '.scr', '.msi'}
            if path.suffix.lower() in dangerous_extensions:
                return False, f"Dangerous file extension: {path.suffix}"
            
            # Check file size limits
            if path.exists() and path.is_file():
                file_size = path.stat().st_size
                max_size = 100 * 1024 * 1024  # 100MB limit
                if file_size > max_size:
                    return False, f"File too large: {file_size} bytes"
            
            # Operation-specific checks
            if operation == "write":
                # Ensure parent directory exists and is writable
                parent_dir = path.parent
                if not parent_dir.exists():
                    return False, f"Parent directory does not exist: {parent_dir}"
                if not os.access(parent_dir, os.W_OK):
                    return False, f"Parent directory not writable: {parent_dir}"
            
            elif operation == "execute":
                # Very restrictive for execution
                if not str(path).startswith(str(Path.cwd().resolve())):
                    return False, "Execution only allowed within project directory"
            
            return True, "Access allowed"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def create_secure_temp_file(self, suffix: str = "", prefix: str = "tara_", 
                              content: bytes = None, cleanup_after: int = 1800) -> Optional[str]:
        """
        Create a secure temporary file with automatic cleanup
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Initial content
            cleanup_after: Cleanup after seconds (default 30 minutes)
            
        Returns:
            Path to secure temp file or None if failed
        """
        try:
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}{unique_id}{suffix}"
            file_path = self.secure_temp_dir / filename
            
            # Create file with restricted permissions
            with open(file_path, 'wb') as f:
                if content:
                    f.write(content)
            
            # Set restrictive permissions (owner read/write only)
            if os.name == 'nt':  # Windows
                import stat
                os.chmod(file_path, stat.S_IREAD | stat.S_IWRITE)
            else:  # Unix/Linux
                os.chmod(file_path, 0o600)
            
            # Track for cleanup
            self.temp_files[str(file_path)] = {
                "created_at": datetime.now(),
                "cleanup_at": datetime.now() + timedelta(seconds=cleanup_after),
                "size": file_path.stat().st_size
            }
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {e}")
            return None
    
    def cleanup_temp_files(self, force_all: bool = False) -> int:
        """
        Clean up expired temporary files
        
        Args:
            force_all: If True, cleanup all temp files regardless of expiry
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        current_time = datetime.now()
        files_to_remove = []
        
        for file_path, file_info in self.temp_files.items():
            should_cleanup = force_all or current_time >= file_info["cleanup_at"]
            
            if should_cleanup:
                try:
                    path = Path(file_path)
                    if path.exists():
                        path.unlink()
                        cleaned_count += 1
                    files_to_remove.append(file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
        
        # Remove from tracking
        for file_path in files_to_remove:
            del self.temp_files[file_path]
        
        return cleaned_count
    
    def _start_cleanup_thread(self):
        """Start automatic cleanup thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self.cleanup_running:
            try:
                self.cleanup_temp_files()
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(300)
    
    def create_isolated_process(self, command: List[str], working_dir: str = None, 
                              timeout: int = 300) -> Tuple[bool, str, str]:
        """
        Create an isolated process for model inference
        
        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        process_id = str(uuid.uuid4())[:8]
        
        try:
            # Validate command
            if not command or not isinstance(command, list):
                return False, "", "Invalid command"
            
            # Security check on command
            cmd_str = " ".join(command)
            sanitized_cmd, is_safe = self.sanitize_input(cmd_str)
            if not is_safe:
                return False, "", "Command contains dangerous patterns"
            
            # Set up secure environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            env['TARA_ISOLATED'] = '1'
            
            # Remove potentially dangerous environment variables
            dangerous_env_vars = ['LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'PATH']
            for var in dangerous_env_vars:
                if var in env and var != 'PATH':
                    del env[var]
            
            # Restrict PATH to essential directories only
            if os.name == 'nt':  # Windows
                env['PATH'] = r'C:\Windows\System32;C:\Windows'
            else:  # Unix/Linux
                env['PATH'] = '/usr/bin:/bin'
            
            # Create process with restrictions
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir or str(Path.cwd()),
                env=env,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Track process
            self.isolated_processes[process_id] = process
            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                success = process.returncode == 0
                
                # Log process execution
                self._log_security_event("process_execution", {
                    "process_id": process_id,
                    "command": command[0],  # Only log the executable name
                    "success": success,
                    "return_code": process.returncode,
                    "timeout": timeout
                })
                
                return success, stdout, stderr
                
            except subprocess.TimeoutExpired:
                process.kill()
                return False, "", f"Process timed out after {timeout} seconds"
            
        except Exception as e:
            return False, "", f"Process creation failed: {str(e)}"
        
        finally:
            # Cleanup process tracking
            if process_id in self.isolated_processes:
                del self.isolated_processes[process_id]
    
    def check_rate_limit(self, client_id: str = "default") -> Tuple[bool, int]:
        """
        Check rate limiting for requests
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, requests_remaining)
        """
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        
        # Initialize client history if not exists
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        
        # Clean old requests
        self.request_history[client_id] = [
            req_time for req_time in self.request_history[client_id]
            if req_time > minute_ago
        ]
        
        # Check limit
        current_requests = len(self.request_history[client_id])
        is_allowed = current_requests < self.max_requests_per_minute
        
        if is_allowed:
            # Record this request
            self.request_history[client_id].append(current_time)
            requests_remaining = self.max_requests_per_minute - current_requests - 1
        else:
            requests_remaining = 0
            # Log rate limit violation
            self._log_security_event("rate_limit_exceeded", {
                "client_id": client_id,
                "requests_in_minute": current_requests,
                "limit": self.max_requests_per_minute
            })
        
        return is_allowed, requests_remaining
    
    def validate_json_input(self, json_str: str) -> Tuple[bool, Any, str]:
        """
        Validate and parse JSON input safely
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Tuple of (is_valid, parsed_data, error_message)
        """
        try:
            # Basic sanitization
            sanitized_json, is_safe = self.sanitize_input(json_str)
            if not is_safe:
                return False, None, "JSON contains dangerous patterns"
            
            # Limit JSON size
            max_json_size = 1024 * 1024  # 1MB limit
            if len(sanitized_json) > max_json_size:
                return False, None, f"JSON too large: {len(sanitized_json)} bytes"
            
            # Parse JSON
            parsed_data = json.loads(sanitized_json)
            
            # Validate structure (prevent deeply nested objects)
            if not self._validate_json_structure(parsed_data):
                return False, None, "JSON structure too complex"
            
            return True, parsed_data, ""
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"JSON validation error: {str(e)}"
    
    def _validate_json_structure(self, data: Any, depth: int = 0, max_depth: int = 10) -> bool:
        """Validate JSON structure complexity"""
        if depth > max_depth:
            return False
        
        if isinstance(data, dict):
            if len(data) > 100:  # Limit number of keys
                return False
            for key, value in data.items():
                if not isinstance(key, str) or len(key) > 1000:  # Limit key length
                    return False
                if not self._validate_json_structure(value, depth + 1, max_depth):
                    return False
        
        elif isinstance(data, list):
            if len(data) > 1000:  # Limit array size
                return False
            for item in data:
                if not self._validate_json_structure(item, depth + 1, max_depth):
                    return False
        
        elif isinstance(data, str):
            if len(data) > 10000:  # Limit string length
                return False
        
        return True
    
    def _log_security_event(self, event_type: str, details: Dict):
        """Log security events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        # Log to security log file
        security_log_path = Path.home() / ".tara" / "security.log"
        security_log_path.parent.mkdir(exist_ok=True)
        
        try:
            with open(security_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write security log: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "temp_files_tracked": len(self.temp_files),
            "isolated_processes": len(self.isolated_processes),
            "rate_limit_clients": len(self.request_history),
            "cleanup_running": self.cleanup_running,
            "secure_temp_dir": str(self.secure_temp_dir),
            "max_requests_per_minute": self.max_requests_per_minute,
            "dangerous_patterns_count": len(self.dangerous_patterns)
        }
    
    def emergency_security_cleanup(self):
        """Emergency security cleanup"""
        try:
            # Kill all isolated processes
            for process_id, process in self.isolated_processes.items():
                try:
                    process.kill()
                except Exception:
                    pass
            self.isolated_processes.clear()
            
            # Cleanup all temp files
            self.cleanup_temp_files(force_all=True)
            
            # Clear rate limiting history
            self.request_history.clear()
            
            logger.info("Emergency security cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency security cleanup failed: {e}")
    
    def shutdown(self):
        """Shutdown security validator"""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Final cleanup
        self.emergency_security_cleanup()

# Global security validator instance
_security_validator = None

def get_security_validator() -> SecurityValidator:
    """Get global security validator instance"""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator 