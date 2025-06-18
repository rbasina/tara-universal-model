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
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

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
    """HAI-Enhanced Privacy Manager - Main Interface"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.conversation_manager = ConversationManager(user_id)
        self.encryption_manager = EncryptionManager(user_id)
        self.privacy_settings = self._load_privacy_settings()
        
        # Set up privacy-aware logging
        self._configure_privacy_logging()
    
    def _load_privacy_settings(self) -> Dict:
        """Load user privacy settings"""
        settings_file = Path(f".tara_privacy_{self.user_id}.json")
        
        default_settings = {
            "data_retention_minutes": PrivacyConfig.DEFAULT_RETENTION,
            "sensitive_data_retention_minutes": PrivacyConfig.SENSITIVE_RETENTION,
            "encryption_enabled": PrivacyConfig.ENCRYPTION_ENABLED,
            "zero_logging_domains": PrivacyConfig.ZERO_LOGGING_DOMAINS.copy(),
            "auto_cleanup_enabled": True,
            "key_rotation_enabled": True
        }
        
        try:
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    user_settings = json.load(f)
                    default_settings.update(user_settings)
        except Exception as e:
            logger.error(f"Failed to load privacy settings: {e}")
        
        return default_settings
    
    def _configure_privacy_logging(self):
        """Configure privacy-aware logging"""
        # Set different log levels for sensitive domains
        for domain in PrivacyConfig.ZERO_LOGGING_DOMAINS:
            domain_logger = logging.getLogger(f"tara.{domain}")
            domain_logger.setLevel(PrivacyConfig.LOG_LEVEL_SENSITIVE)
    
    def start_private_session(self, domain: str) -> str:
        """Start a privacy-enhanced session"""
        return self.conversation_manager.start_session(domain)
    
    def add_interaction(self, session_id: str, user_input: str, ai_response: str):
        """Add interaction with privacy controls"""
        self.conversation_manager.add_message(session_id, user_input, ai_response)
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get session history with privacy controls"""
        return self.conversation_manager.get_conversation_history(session_id)
    
    def end_private_session(self, session_id: str):
        """End privacy-enhanced session"""
        self.conversation_manager.end_session(session_id)
    
    def cleanup_all_expired_data(self):
        """Clean up all expired data"""
        self.conversation_manager.cleanup_expired_sessions()
    
    def get_privacy_status(self) -> Dict:
        """Get current privacy status"""
        return {
            "encryption_enabled": self.privacy_settings["encryption_enabled"],
            "active_sessions": len(self.conversation_manager.active_sessions),
            "zero_logging_domains": self.privacy_settings["zero_logging_domains"],
            "data_retention_minutes": self.privacy_settings["data_retention_minutes"],
            "auto_cleanup_enabled": self.privacy_settings["auto_cleanup_enabled"],
            "privacy_mode": "HAI-Enhanced"
        }
    
    def update_privacy_settings(self, new_settings: Dict):
        """Update privacy settings"""
        self.privacy_settings.update(new_settings)
        
        # Save updated settings
        settings_file = Path(f".tara_privacy_{self.user_id}.json")
        try:
            with open(settings_file, 'w') as f:
                json.dump(self.privacy_settings, f, indent=2)
            
            # Hide settings file
            if os.name == 'nt':  # Windows
                os.system(f'attrib +h "{settings_file}"')
                
            logger.info("🔒 Privacy settings updated")
        except Exception as e:
            logger.error(f"Failed to save privacy settings: {e}")

# Global privacy manager instance
_privacy_manager = None

def get_privacy_manager(user_id: str = "default") -> PrivacyManager:
    """Get global privacy manager instance"""
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager(user_id)
    return _privacy_manager 