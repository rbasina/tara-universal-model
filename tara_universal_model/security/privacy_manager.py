"""
TARA Universal Model - Privacy Manager
HAI-Enhanced Privacy Protection System

This module implements advanced privacy protection features:
- Local encryption for all user interactions
- Automatic conversation cleanup after session
- Zero-logging mode for sensitive domains
- User-controlled data retention policies
"""

import os
import json
import time
import hashlib
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import tempfile
import shutil

logger = logging.getLogger(__name__)

class PrivacyConfig:
    """HAI Privacy Configuration"""
    
    # Sensitive domains with enhanced privacy
    SENSITIVE_DOMAINS = ["healthcare", "business", "mental_health", "financial"]
    
    # Data retention policies (in minutes)
    DEFAULT_RETENTION = 30  # 30 minutes
    SENSITIVE_RETENTION = 5  # 5 minutes for sensitive domains
    CONVERSATION_RETENTION = 60  # 1 hour for conversation history
    
    # Encryption settings
    ENCRYPTION_ENABLED = True
    KEY_ROTATION_HOURS = 24  # Rotate encryption keys every 24 hours
    
    # Logging policies
    ZERO_LOGGING_DOMAINS = ["healthcare", "mental_health"]
    LOG_LEVEL_SENSITIVE = logging.WARNING  # Minimal logging for sensitive domains

class EncryptionManager:
    """HAI-Enhanced Encryption Manager for Local Data Protection"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.key_file = Path(f".tara_keys_{user_id}.enc")
        self.fernet = self._initialize_encryption()
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption with user-specific key"""
        try:
            if self.key_file.exists():
                # Load existing key
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Hide the key file
                if os.name == 'nt':  # Windows
                    os.system(f'attrib +h "{self.key_file}"')
            
            return Fernet(key)
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            # Fallback to session-only encryption
            return Fernet(Fernet.generate_key())
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            if not PrivacyConfig.ENCRYPTION_ENABLED:
                return data
            
            encrypted = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data  # Fallback to unencrypted
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if not PrivacyConfig.ENCRYPTION_ENABLED:
                return encrypted_data
            
            decoded = base64.b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data  # Fallback to encrypted data
    
    def rotate_key(self):
        """Rotate encryption key for enhanced security"""
        try:
            old_key_file = Path(f".tara_keys_{self.user_id}_backup.enc")
            if self.key_file.exists():
                # Backup old key
                self.key_file.rename(old_key_file)
            
            # Generate new key
            new_key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(new_key)
            
            self.fernet = Fernet(new_key)
            logger.info("🔄 Encryption key rotated successfully")
            
            # Schedule old key cleanup
            if old_key_file.exists():
                os.remove(old_key_file)
                
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")

class ConversationManager:
    """HAI-Enhanced Conversation Management with Privacy Controls"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.encryption_manager = EncryptionManager(user_id)
        self.conversations_dir = Path(f".tara_conversations_{user_id}")
        self.conversations_dir.mkdir(exist_ok=True)
        self.active_sessions = {}
        
        # Hide conversations directory
        if os.name == 'nt':  # Windows
            os.system(f'attrib +h "{self.conversations_dir}"')
    
    def start_session(self, domain: str) -> str:
        """Start a new conversation session"""
        session_id = hashlib.md5(f"{domain}_{time.time()}".encode()).hexdigest()[:12]
        
        session_data = {
            "session_id": session_id,
            "domain": domain,
            "start_time": datetime.now().isoformat(),
            "is_sensitive": domain in PrivacyConfig.SENSITIVE_DOMAINS,
            "messages": [],
            "retention_policy": self._get_retention_policy(domain)
        }
        
        self.active_sessions[session_id] = session_data
        
        # Set up automatic cleanup
        self._schedule_cleanup(session_id, session_data["retention_policy"])
        
        logger.info(f"🔒 Started {'sensitive' if session_data['is_sensitive'] else 'standard'} session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, message: str, response: str, message_type: str = "user"):
        """Add message to conversation with privacy controls"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.active_sessions[session_id]
        domain = session["domain"]
        
        # Check if domain requires zero-logging
        if domain in PrivacyConfig.ZERO_LOGGING_DOMAINS:
            # Don't log the actual content, only metadata
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
                "content_hash": hashlib.sha256(message.encode()).hexdigest()[:16],
                "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
                "domain": domain,
                "privacy_mode": "zero_logging"
            }
        else:
            # Encrypt and store content
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "type": message_type,
                "content": self.encryption_manager.encrypt_data(message),
                "response": self.encryption_manager.encrypt_data(response),
                "domain": domain,
                "privacy_mode": "encrypted"
            }
        
        session["messages"].append(message_data)
        
        # Save to disk for persistence
        self._save_session(session_id)
    
    def get_conversation_history(self, session_id: str, decrypt: bool = True) -> List[Dict]:
        """Get conversation history with privacy controls"""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        messages = session["messages"]
        
        if not decrypt:
            return messages
        
        # Decrypt messages if not in zero-logging mode
        decrypted_messages = []
        for msg in messages:
            if msg.get("privacy_mode") == "zero_logging":
                # Return metadata only
                decrypted_messages.append({
                    "timestamp": msg["timestamp"],
                    "type": msg["type"],
                    "domain": msg["domain"],
                    "content": "[CONTENT PROTECTED - ZERO LOGGING MODE]",
                    "response": "[RESPONSE PROTECTED - ZERO LOGGING MODE]"
                })
            else:
                # Decrypt content
                decrypted_msg = msg.copy()
                if "content" in msg:
                    decrypted_msg["content"] = self.encryption_manager.decrypt_data(msg["content"])
                if "response" in msg:
                    decrypted_msg["response"] = self.encryption_manager.decrypt_data(msg["response"])
                decrypted_messages.append(decrypted_msg)
        
        return decrypted_messages
    
    def end_session(self, session_id: str):
        """End conversation session with cleanup"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session["end_time"] = datetime.now().isoformat()
        
        # Save final state
        self._save_session(session_id)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"🔒 Ended session: {session_id}")
    
    def _get_retention_policy(self, domain: str) -> int:
        """Get retention policy for domain"""
        if domain in PrivacyConfig.SENSITIVE_DOMAINS:
            return PrivacyConfig.SENSITIVE_RETENTION
        return PrivacyConfig.DEFAULT_RETENTION
    
    def _schedule_cleanup(self, session_id: str, retention_minutes: int):
        """Schedule automatic cleanup of session data"""
        # This would typically use a background task scheduler
        # For now, we'll implement cleanup on next access
        pass
    
    def _save_session(self, session_id: str):
        """Save session to disk with encryption"""
        if session_id not in self.active_sessions:
            return
        
        session_file = self.conversations_dir / f"{session_id}.json"
        session_data = self.active_sessions[session_id]
        
        try:
            # Encrypt entire session data
            encrypted_data = self.encryption_manager.encrypt_data(json.dumps(session_data))
            
            with open(session_file, 'w') as f:
                json.dump({"encrypted_session": encrypted_data}, f)
                
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions based on retention policies"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            start_time = datetime.fromisoformat(session["start_time"])
            retention_minutes = session["retention_policy"]
            
            if current_time - start_time > timedelta(minutes=retention_minutes):
                expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
            logger.info(f"🧹 Cleaned up expired session: {session_id}")
    
    def _cleanup_session(self, session_id: str):
        """Permanently delete session data"""
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove session file
        session_file = self.conversations_dir / f"{session_id}.json"
        if session_file.exists():
            try:
                os.remove(session_file)
            except Exception as e:
                logger.error(f"Failed to delete session file {session_id}: {e}")

class PrivacyManager:
    """
    Enhanced Privacy Protection for TARA Universal Model
    
    Features:
    - Local encryption for all user interactions
    - Automatic conversation cleanup after sessions
    - Zero-logging mode for sensitive domains
    - User-controlled data retention policies
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".tara" / "secure"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sensitive domains that require zero-logging
        self.zero_log_domains = {"healthcare", "business", "mental_health", "financial"}
        
        # Default retention policies (in hours)
        self.retention_policies = {
            "healthcare": 0,  # No retention - immediate cleanup
            "business": 0,    # No retention - immediate cleanup
            "mental_health": 0,  # No retention - immediate cleanup
            "financial": 0,   # No retention - immediate cleanup
            "education": 24,  # 24 hours
            "creative": 48,   # 48 hours
            "leadership": 24, # 24 hours
            "universal": 12   # 12 hours
        }
        
        # Initialize encryption
        self._init_encryption()
        
        # Active sessions tracking
        self.active_sessions: Dict[str, Dict] = {}
        
        # Cleanup thread
        self.cleanup_thread = None
        self.cleanup_running = False
        
        # Start automatic cleanup
        self._start_cleanup_thread()
    
    def _init_encryption(self):
        """Initialize encryption with user-specific key"""
        key_file = self.data_dir / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            # Generate new encryption key
            password = os.urandom(32)  # Random password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.encryption_key = key
            
            # Save key securely
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Secure file permissions (Windows)
            if os.name == 'nt':
                import stat
                os.chmod(key_file, stat.S_IREAD | stat.S_IWRITE)
        
        self.cipher = Fernet(self.encryption_key)
    
    def create_session(self, domain: str, user_id: str = "default") -> str:
        """Create a new secure session"""
        session_id = hashlib.sha256(f"{user_id}_{domain}_{time.time()}".encode()).hexdigest()[:16]
        
        session_data = {
            "id": session_id,
            "domain": domain,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "interactions": [],
            "zero_log": domain in self.zero_log_domains,
            "retention_hours": self.retention_policies.get(domain, 12)
        }
        
        self.active_sessions[session_id] = session_data
        return session_id
    
    def log_interaction(self, session_id: str, interaction_type: str, data: Dict[str, Any]) -> bool:
        """Log interaction with privacy controls"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Check if this is a zero-log domain
        if session["zero_log"]:
            # For zero-log domains, only keep minimal metadata
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction_type,
                "domain": session["domain"],
                "processed": True,
                "data_encrypted": False  # No data stored
            }
        else:
            # Encrypt sensitive data
            encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
            
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction_type,
                "domain": session["domain"],
                "data_encrypted": base64.b64encode(encrypted_data).decode(),
                "processed": True
            }
        
        session["interactions"].append(interaction)
        return True
    
    def get_session_data(self, session_id: str, decrypt: bool = True) -> Optional[Dict]:
        """Retrieve session data with decryption"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id].copy()
        
        if decrypt and not session["zero_log"]:
            # Decrypt interaction data
            for interaction in session["interactions"]:
                if "data_encrypted" in interaction and interaction["data_encrypted"]:
                    try:
                        encrypted_data = base64.b64decode(interaction["data_encrypted"])
                        decrypted_data = self.cipher.decrypt(encrypted_data)
                        interaction["data"] = json.loads(decrypted_data.decode())
                        del interaction["data_encrypted"]
                    except Exception as e:
                        interaction["data"] = {"error": "Decryption failed"}
        
        return session
    
    def end_session(self, session_id: str):
        """End session and apply retention policy"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # If zero retention, delete immediately
        if session["retention_hours"] == 0:
            del self.active_sessions[session_id]
            return
        
        # Mark session as ended
        session["ended_at"] = datetime.now().isoformat()
        session["cleanup_at"] = (datetime.now() + timedelta(hours=session["retention_hours"])).isoformat()
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Check if session has cleanup time set
            if "cleanup_at" in session:
                cleanup_time = datetime.fromisoformat(session["cleanup_at"])
                if current_time >= cleanup_time:
                    expired_sessions.append(session_id)
            
            # Also cleanup very old active sessions (safety measure)
            created_time = datetime.fromisoformat(session["created_at"])
            if current_time - created_time > timedelta(hours=48):  # 48 hour max
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        return len(expired_sessions)
    
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
                self.cleanup_expired_sessions()
                self._cleanup_temp_files()
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                # Silent cleanup - don't log errors in privacy mode
                pass
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = Path(tempfile.gettempdir())
        current_time = time.time()
        
        # Clean up TARA-related temp files older than 30 minutes
        for temp_file in temp_dir.glob("tara_*"):
            try:
                if current_time - temp_file.stat().st_mtime > 1800:  # 30 minutes
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
            except Exception:
                pass
    
    def set_retention_policy(self, domain: str, hours: int):
        """Set custom retention policy for domain"""
        self.retention_policies[domain] = hours
        
        # Update existing sessions
        for session in self.active_sessions.values():
            if session["domain"] == domain:
                session["retention_hours"] = hours
                if "ended_at" in session:
                    # Recalculate cleanup time
                    ended_time = datetime.fromisoformat(session["ended_at"])
                    session["cleanup_at"] = (ended_time + timedelta(hours=hours)).isoformat()
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status"""
        return {
            "active_sessions": len(self.active_sessions),
            "zero_log_domains": list(self.zero_log_domains),
            "retention_policies": self.retention_policies.copy(),
            "encryption_enabled": True,
            "cleanup_running": self.cleanup_running,
            "data_directory": str(self.data_dir)
        }
    
    def emergency_cleanup(self):
        """Emergency cleanup - remove all data"""
        # Clear all sessions
        self.active_sessions.clear()
        
        # Clean up temp files
        self._cleanup_temp_files()
        
        # Optionally remove encryption key (uncomment if needed)
        # key_file = self.data_dir / "encryption.key"
        # if key_file.exists():
        #     key_file.unlink()
    
    def shutdown(self):
        """Shutdown privacy manager"""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Final cleanup
        self.cleanup_expired_sessions()
        self._cleanup_temp_files()

# Global privacy manager instance
_privacy_manager = None

def get_privacy_manager() -> PrivacyManager:
    """Get global privacy manager instance"""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager()
    return _privacy_manager 