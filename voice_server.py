#!/usr/bin/env python3
"""
TARA Voice Server for Frontend Integration
Provides TTS endpoints on port 5000 for tara-ai-companion integration.
This is a consolidated version of multiple voice server scripts.
"""

import asyncio
import logging
import time
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import base64
import re
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import json

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, Field
# TTS System Imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ Edge TTS not available. Install with: pip install edge-tts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️ pyttsx3 not available. Install with: pip install pyttsx3")

import uvicorn

# Import TARA Universal AI Engine
try:
    from tara_universal_model.core import get_universal_engine, AIRequest, AIResponse
    AI_ENGINE_AVAILABLE = True
except ImportError:
    AI_ENGINE_AVAILABLE = False
    print("⚠️ Universal AI Engine not available - using fallback responses")

# Import HAI Security Components
try:
    from tara_universal_model.security.privacy_manager import get_privacy_manager
    from tara_universal_model.security.resource_monitor import get_resource_monitor
    from tara_universal_model.security.security_validator import get_security_validator
    HAI_SECURITY_AVAILABLE = True
except ImportError:
    HAI_SECURITY_AVAILABLE = False
    print("⚠️ HAI Security components not available - using basic security")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HAI Security & Safety Configuration
class HAIConfig:
    """HAI-focused configuration for safety, security, and user-centricity"""
    
    # Rate limiting (requests per minute per IP)
    RATE_LIMIT_PER_MINUTE = 60
    
    # Input validation
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 1
    
    # Security patterns to block
    BLOCKED_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS attempts
        r'javascript:',               # JavaScript injection
        r'data:text/html',           # Data URI attacks
        r'eval\s*\(',                # Code execution attempts
    ]
    
    # Privacy settings
    AUTO_CLEANUP_MINUTES = 30  # Auto-delete temp files after 30 minutes
    LOG_SENSITIVE_DATA = False  # Never log user content
    
    # Fallback behavior
    ENABLE_GRACEFUL_DEGRADATION = True
    MAX_RETRY_ATTEMPTS = 3

# Rate limiting storage
rate_limit_storage = defaultdict(list)

class RateLimiter:
    """HAI-focused rate limiter for user protection"""
    
    @staticmethod
    def is_allowed(client_ip: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        rate_limit_storage[client_ip] = [
            timestamp for timestamp in rate_limit_storage[client_ip]
            if timestamp > minute_ago
        ]
        
        # Check rate limit
        if len(rate_limit_storage[client_ip]) >= HAIConfig.RATE_LIMIT_PER_MINUTE:
            return False
        
        # Add current request
        rate_limit_storage[client_ip].append(now)
        return True

class InputValidator:
    """HAI-focused input validation for safety"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize input text while preserving user intent"""
        if not text:
            return ""
        
        # Remove potentially harmful patterns
        for pattern in HAIConfig.BLOCKED_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def validate_text(text: str) -> tuple[bool, str]:
        """Validate text input with user-friendly feedback"""
        if not text or len(text.strip()) < HAIConfig.MIN_TEXT_LENGTH:
            return False, "Text is too short. Please provide at least 1 character."
        
        if len(text) > HAIConfig.MAX_TEXT_LENGTH:
            return False, f"Text is too long. Maximum {HAIConfig.MAX_TEXT_LENGTH} characters allowed."
        
        # Check for suspicious patterns
        for pattern in HAIConfig.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Input contains potentially harmful content. Please rephrase your request."
        
        return True, "Valid input"

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    domain: str = Field(default="universal", pattern="^(universal|healthcare|business|education|creative|leadership)$")
    voice: Optional[str] = None
    
    @validator('text')
    def validate_and_sanitize_text(cls, v):
        # Sanitize input
        sanitized = InputValidator.sanitize_text(v)
        
        # Validate
        is_valid, message = InputValidator.validate_text(sanitized)
        if not is_valid:
            raise ValueError(message)
        
        return sanitized

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    domain: str = Field(default="universal", pattern="^(universal|healthcare|business|education|creative|leadership)$")
    
    @validator('message')
    def validate_and_sanitize_message(cls, v):
        sanitized = InputValidator.sanitize_text(v)
        is_valid, message = InputValidator.validate_text(sanitized)
        if not is_valid:
            raise ValueError(message)
        return sanitized

# HAI-Enhanced TARA Voice Server
app = FastAPI(
    title="TARA Universal Voice Server",
    description="HAI-Powered Voice Server - Help Anytime, Everywhere",
    version="2.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",  # Original tara-ai-companion
        "http://localhost:2025", "http://127.0.0.1:2025",  # me²TARA Enhanced
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
)

# HAI-Enhanced Domain Configuration with Personality Traits
DOMAIN_VOICES = {
    "universal": {
        "voice": "en-US-AriaNeural",
        "personality": "Friendly, helpful, and adaptable to any situation",
        "fallback_voice": "en-US-JennyNeural"
    },
    "healthcare": {
        "voice": "en-US-AriaNeural", 
        "personality": "Compassionate, professional, and reassuring",
        "fallback_voice": "en-US-SaraNeural"
    },
    "business": {
        "voice": "en-US-JennyNeural",
        "personality": "Professional, confident, and strategic",
        "fallback_voice": "en-US-AriaNeural"
    },
    "education": {
        "voice": "en-US-AriaNeural",
        "personality": "Patient, encouraging, and knowledgeable",
        "fallback_voice": "en-US-JennyNeural"
    },
    "creative": {
        "voice": "en-US-JennyNeural",
        "personality": "Inspiring, imaginative, and enthusiastic",
        "fallback_voice": "en-US-AriaNeural"
    },
    "leadership": {
        "voice": "en-US-GuyNeural",
        "personality": "Authoritative, wise, and motivational",
        "fallback_voice": "en-US-AriaNeural"
    }
}

# Voice mapping for invalid or legacy voice names
VOICE_MAPPING = {
    # Legacy and invalid voice mappings
    "gentle_female": "en-US-JennyNeural",
    "gentle_male": "en-US-GuyNeural",
    "warm_female": "en-US-AriaNeural",
    "professional_female": "en-US-JennyNeural",
    "professional_male": "en-US-ChristopherNeural",
    "friendly_female": "en-US-AriaNeural",
    "friendly_male": "en-US-GuyNeural",
    "calm_female": "en-US-JennyNeural",
    "calm_male": "en-US-GuyNeural",
    "energetic_female": "en-US-AriaNeural",
    "energetic_male": "en-US-GuyNeural",
    # Common voice shortcuts
    "female": "en-US-JennyNeural",
    "male": "en-US-GuyNeural",
    "woman": "en-US-AriaNeural",
    "man": "en-US-GuyNeural",
    "lady": "en-US-JennyNeural",
    "guy": "en-US-GuyNeural",
}

# Valid Edge TTS voices (most commonly used ones)
VALID_EDGE_VOICES = {
    # English (US) voices
    "en-US-AriaNeural": {"gender": "female", "locale": "en-US", "description": "News, conversational"},
    "en-US-JennyNeural": {"gender": "female", "locale": "en-US", "description": "Friendly, considerate"},
    "en-US-GuyNeural": {"gender": "male", "locale": "en-US", "description": "Passionate, energetic"},
    "en-US-AnaNeural": {"gender": "female", "locale": "en-US", "description": "Cheerful, optimistic"},
    "en-US-ChristopherNeural": {"gender": "male", "locale": "en-US", "description": "Reliable, authoritative"},
    "en-US-EricNeural": {"gender": "male", "locale": "en-US", "description": "Rational, calm"},
    "en-US-MichelleNeural": {"gender": "female", "locale": "en-US", "description": "Pleasant, friendly"},
    "en-US-RogerNeural": {"gender": "male", "locale": "en-US", "description": "Lively, energetic"},
    "en-US-SteffanNeural": {"gender": "male", "locale": "en-US", "description": "Rational, professional"},
    
    # English (UK) voices
    "en-GB-SoniaNeural": {"gender": "female", "locale": "en-GB", "description": "British, friendly"},
    "en-GB-RyanNeural": {"gender": "male", "locale": "en-GB", "description": "British, confident"},
    "en-GB-LibbyNeural": {"gender": "female", "locale": "en-GB", "description": "British, warm"},
    "en-GB-MaisieNeural": {"gender": "female", "locale": "en-GB", "description": "British, cheerful"},
    "en-GB-ThomasNeural": {"gender": "male", "locale": "en-GB", "description": "British, professional"},
    
    # Other English variants
    "en-AU-NatashaNeural": {"gender": "female", "locale": "en-AU", "description": "Australian, friendly"},
    "en-AU-WilliamNeural": {"gender": "male", "locale": "en-AU", "description": "Australian, confident"},
    "en-CA-ClaraNeural": {"gender": "female", "locale": "en-CA", "description": "Canadian, warm"},
    "en-CA-LiamNeural": {"gender": "male", "locale": "en-CA", "description": "Canadian, friendly"},
}

def validate_and_map_voice(voice: str, domain: str = "universal") -> str:
    """
    Validate and map voice names to valid Edge TTS voices.
    Returns a valid Edge TTS voice name.
    """
    if not voice:
        # Use domain default if no voice specified
        return DOMAIN_VOICES.get(domain, DOMAIN_VOICES["universal"])["voice"]
    
    # Check if it's already a valid Edge TTS voice
    if voice in VALID_EDGE_VOICES:
        logger.info(f"✅ Using valid Edge TTS voice: {voice}")
        return voice
    
    # Check if it's in our mapping
    if voice.lower() in VOICE_MAPPING:
        mapped_voice = VOICE_MAPPING[voice.lower()]
        logger.info(f"🔄 Mapped '{voice}' to valid Edge TTS voice: {mapped_voice}")
        return mapped_voice
    
    # If voice contains common patterns, try to map it
    voice_lower = voice.lower()
    if "female" in voice_lower or "woman" in voice_lower or "lady" in voice_lower:
        mapped_voice = "en-US-JennyNeural"
        logger.info(f"🔄 Mapped female voice '{voice}' to: {mapped_voice}")
        return mapped_voice
    elif "male" in voice_lower or "man" in voice_lower or "guy" in voice_lower:
        mapped_voice = "en-US-GuyNeural"
        logger.info(f"🔄 Mapped male voice '{voice}' to: {mapped_voice}")
        return mapped_voice
    
    # Try partial matching with valid voices (case insensitive)
    for valid_voice in VALID_EDGE_VOICES.keys():
        if voice.lower() in valid_voice.lower() or valid_voice.lower().endswith(voice.lower()):
            logger.info(f"🔄 Partial match: mapped '{voice}' to: {valid_voice}")
            return valid_voice
    
    # If no mapping found, use domain default and log warning
    default_voice = DOMAIN_VOICES.get(domain, DOMAIN_VOICES["universal"])["voice"]
    logger.warning(f"⚠️ Unknown voice '{voice}', using domain default: {default_voice}")
    return default_voice

# Global TTS system state
tts_systems = {
    "edge_tts": EDGE_TTS_AVAILABLE,
    "pyttsx3": PYTTSX3_AVAILABLE
}

preferred_tts = "edge_tts" if EDGE_TTS_AVAILABLE else "pyttsx3"

# Global storage for file management and caching
temp_files_created = []
audio_cache = {}  # Cache for audio files based on content hash

class FileCleanupManager:
    """HAI-focused file management with automatic cleanup"""
    
    @staticmethod
    def schedule_cleanup(filepath: str):
        """Schedule file for automatic cleanup"""
        temp_files_created.append({
            'path': filepath,
            'created': datetime.now()
        })
    
    @staticmethod
    def cleanup_old_files():
        """Clean up files older than configured time"""
        cutoff_time = datetime.now() - timedelta(minutes=HAIConfig.AUTO_CLEANUP_MINUTES)
        
        for file_info in temp_files_created[:]:
            if file_info['created'] < cutoff_time:
                try:
                    if os.path.exists(file_info['path']):
                        os.remove(file_info['path'])
                        logger.info(f"🧹 Cleaned up old file: {file_info['path']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {file_info['path']}: {e}")
                finally:
                    temp_files_created.remove(file_info)

class AudioCacheManager:
    """Smart content-based caching system for audio files"""
    
    @staticmethod
    def generate_content_hash(text: str, voice: str) -> str:
        """Generate hash based on text content and voice"""
        content = f"{text}||{voice}".encode('utf-8')
        return hashlib.md5(content).hexdigest()[:12]  # Use first 12 chars
    
    @staticmethod
    def get_cached_audio(text: str, voice: str) -> Optional[str]:
        """Check if audio file exists for this content"""
        content_hash = AudioCacheManager.generate_content_hash(text, voice)
        
        if content_hash in audio_cache:
            file_path = audio_cache[content_hash]['path']
            # Check if file still exists
            if os.path.exists(file_path):
                # Update last accessed time
                audio_cache[content_hash]['last_accessed'] = datetime.now()
                filename = os.path.basename(file_path)
                logger.info(f"♻️ Reusing cached audio: {filename}")
                return filename
            else:
                # File was deleted, remove from cache
                del audio_cache[content_hash]
        
        return None
    
    @staticmethod
    def cache_audio_file(text: str, voice: str, file_path: str) -> str:
        """Add audio file to cache"""
        content_hash = AudioCacheManager.generate_content_hash(text, voice)
        filename = os.path.basename(file_path)
        
        audio_cache[content_hash] = {
            'path': file_path,
            'filename': filename,
            'text': text[:100] + "..." if len(text) > 100 else text,  # Store snippet for debugging
            'voice': voice,
            'created': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 1
        }
        
        logger.info(f"💾 Cached new audio: {filename} (hash: {content_hash})")
        return filename
    
    @staticmethod
    def cleanup_cache():
        """Remove old cache entries"""
        cutoff_time = datetime.now() - timedelta(minutes=HAIConfig.AUTO_CLEANUP_MINUTES)
        
        for content_hash in list(audio_cache.keys()):
            cache_entry = audio_cache[content_hash]
            if cache_entry['last_accessed'] < cutoff_time:
                try:
                    if os.path.exists(cache_entry['path']):
                        os.remove(cache_entry['path'])
                        logger.info(f"🧹 Cleaned up cached file: {cache_entry['filename']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup cached file {cache_entry['path']}: {e}")
                finally:
                    del audio_cache[content_hash]

async def synthesize_with_edge_tts(text: str, voice: str, output_path: str) -> bool:
    """HAI-Enhanced Edge TTS synthesis with robust error handling"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        # Verify file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"✅ Edge TTS synthesis successful: {voice}")
            return True
        else:
            logger.error(f"❌ Edge TTS created empty file for voice: {voice}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Edge TTS synthesis failed for {voice}: {str(e)}")
        return False

def synthesize_with_pyttsx3(text: str, output_path: str) -> bool:
    """HAI-Enhanced pyttsx3 synthesis with error handling"""
    try:
        engine = pyttsx3.init()
        
        # Configure voice properties for better quality
        voices = engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        engine.setProperty('rate', 180)  # Moderate speaking rate
        engine.setProperty('volume', 0.9)  # High volume
        
        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        # Verify file creation
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("✅ pyttsx3 synthesis successful")
            return True
        else:
            logger.error("❌ pyttsx3 created empty file")
            return False
            
    except Exception as e:
        logger.error(f"❌ pyttsx3 synthesis failed: {str(e)}")
        return False

# Middleware for rate limiting and security
@app.middleware("http")
async def hai_security_middleware(request: Request, call_next):
    """HAI-focused security and rate limiting middleware"""
    
    # Skip rate limiting for status endpoints
    if request.url.path == "/tts/status":
        response = await call_next(request)
        return response
    
    # Rate limiting
    client_ip = request.client.host
    if not RateLimiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please wait a moment before trying again.",
                "hai_note": "This limit protects both you and the system. TARA is here to help sustainably."
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Cleanup old files and cache periodically
    if len(temp_files_created) > 10:  # Cleanup every 10 requests
        FileCleanupManager.cleanup_old_files()
        AudioCacheManager.cleanup_cache()
    
    return response

@app.options("/tts/status")
async def options_tts_status():
    """Handle OPTIONS preflight for /tts/status"""
    return {"message": "OK"}

@app.get("/tts/status")
async def get_tts_status():
    """HAI-Enhanced TTS status with system health information"""
    
    # Check AI Engine status
    ai_engine_status = "not_available"
    ai_capabilities = {}
    
    if AI_ENGINE_AVAILABLE:
        try:
            ai_engine = await get_universal_engine()
            ai_capabilities = await ai_engine.get_capabilities()
            ai_engine_status = "ready"
        except Exception as e:
            ai_engine_status = "error"
            logger.error(f"AI Engine status check failed: {e}")
    
    return {
        "status": "ready",
        "hai_principle": "Help Anytime, Everywhere",
        "systems": {
            "edge_tts": {
                "available": tts_systems["edge_tts"],
                "description": "High-quality neural voices"
            },
            "pyttsx3": {
                "available": tts_systems["pyttsx3"], 
                "description": "Reliable offline text-to-speech"
            },
            "ai_engine": {
                "available": AI_ENGINE_AVAILABLE,
                "status": ai_engine_status,
                "description": "Universal AI Engine for robust human support",
                "capabilities": ai_capabilities
            }
        },
        "preferred_tts_system": preferred_tts,
        "domains": list(DOMAIN_VOICES.keys()),
        "ai_features": {
            "universal_ai_engine": AI_ENGINE_AVAILABLE,
            "multi_domain_expertise": True,
            "context_aware_responses": True,
            "emergency_protocols": True,
            "personalized_assistance": True
        },
        "safety_features": {
            "rate_limiting": True,
            "input_validation": True,
            "auto_cleanup": True,
            "offline_capable": True,
            "privacy_protection": True
        },
        "version": "2.0.0 HAI-Enhanced with Universal AI Engine",
        "voice_support": {
            "supported_edge_voices": len(VALID_EDGE_VOICES),
            "voice_mapping_enabled": True,
            "voice_aliases": len(VOICE_MAPPING),
            "automatic_mapping": "Invalid voice names are automatically mapped to valid Edge TTS voices"
        },
        "cache_system": {
            "content_based_caching": True,
            "cached_files": len(audio_cache),
            "cache_description": "Identical text+voice combinations reuse the same audio file for efficiency"
        }
    }

@app.get("/tts/voices")
async def get_available_voices():
    """Get all available voices with mapping information"""
    return {
        "valid_edge_voices": {
            voice: {
                "name": voice,
                "gender": info["gender"],
                "locale": info["locale"],
                "description": info["description"]
            }
            for voice, info in VALID_EDGE_VOICES.items()
        },
        "voice_mappings": {
            alias: mapping
            for alias, mapping in VOICE_MAPPING.items()
        },
        "domain_default_voices": {
            domain: {
                "primary": config["voice"],
                "fallback": config["fallback_voice"],
                "personality": config["personality"]
            }
            for domain, config in DOMAIN_VOICES.items()
        },
        "usage_note": "You can use any voice from 'valid_edge_voices' directly, or use aliases from 'voice_mappings' which will be automatically converted to valid voices."
    }

@app.options("/tts/synthesize")
async def options_tts_synthesize():
    """Handle OPTIONS preflight for /tts/synthesize"""
    return {"message": "OK"}

@app.post("/tts/synthesize")
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """HAI-Enhanced speech synthesis with robust fallback system"""
    
    try:
        domain = request.domain
        text = request.text
        
        # Get domain configuration
        domain_config = DOMAIN_VOICES.get(domain, DOMAIN_VOICES["universal"])
        
        # Validate and map voice (handles invalid voices like 'gentle_female')
        raw_voice = request.voice or domain_config["voice"]
        voice = validate_and_map_voice(raw_voice, domain)
        
        # Check cache first - reuse existing audio if same text+voice
        cached_audio = AudioCacheManager.get_cached_audio(text, voice)
        if cached_audio:
            return {
                "success": True,
                "audio_url": f"/audio/{cached_audio}",
                "text_response": text,
                "synthesis_method": f"Cached ({voice})",
                "domain": domain,
                "personality": domain_config["personality"],
                "hai_message": f"TARA efficiently reused cached audio for {domain} domain!",
                "fallback_used": False,
                "cached": True
            }
        
        # Generate content-based filename for new synthesis
        content_hash = AudioCacheManager.generate_content_hash(text, voice)
        audio_file = f"tara_audio_{content_hash}.mp3"
        audio_path = os.path.join(tempfile.gettempdir(), audio_file)
        
        success = False
        synthesis_method = None
        
        # Attempt 1: Edge TTS with primary voice
        if tts_systems["edge_tts"]:
            success = await synthesize_with_edge_tts(text, voice, audio_path)
            if success:
                synthesis_method = f"Edge TTS ({voice})"
            else:
                # Attempt 2: Edge TTS with fallback voice
                fallback_voice = domain_config.get("fallback_voice", "en-US-AriaNeural")
                if fallback_voice != voice:
                    logger.info(f"🔄 Trying fallback voice: {fallback_voice}")
                    success = await synthesize_with_edge_tts(text, fallback_voice, audio_path)
                    if success:
                        synthesis_method = f"Edge TTS ({fallback_voice} - fallback)"
        
        # Attempt 3: pyttsx3 fallback
        if not success and tts_systems["pyttsx3"]:
            # For pyttsx3, use .wav extension
            audio_file = f"tara_audio_{content_hash}.wav"
            audio_path = os.path.join(tempfile.gettempdir(), audio_file)
            success = synthesize_with_pyttsx3(text, audio_path)
            if success:
                synthesis_method = "pyttsx3 (offline fallback)"
        
        # Final fallback: Return text response
        if not success:
            logger.error("❌ All TTS synthesis methods failed")
            return JSONResponse(
                status_code=200,  # Not a server error, graceful degradation
                content={
                    "success": False,
                    "audio_url": None,
                    "text_response": text,
                    "synthesis_method": "text-only (graceful degradation)",
                    "hai_message": "TARA is still here to help! Voice synthesis is temporarily unavailable, but the text response is ready.",
                    "domain": domain,
                    "fallback_used": True
                }
            )
        
        # Cache the audio file for future reuse
        AudioCacheManager.cache_audio_file(text, voice, audio_path)
        
        # Schedule file cleanup (will be handled by cache cleanup)
        FileCleanupManager.schedule_cleanup(audio_path)
        
        return {
            "success": True,
            "audio_url": f"/audio/{audio_file}",
            "text_response": text,
            "synthesis_method": synthesis_method,
            "domain": domain,
            "personality": domain_config["personality"],
            "hai_message": f"TARA is ready to help in the {domain} domain!",
            "fallback_used": synthesis_method != f"Edge TTS ({voice})",
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"❌ Synthesis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Synthesis failed",
                "hai_message": "TARA encountered an issue but is working to resolve it. Please try again.",
                "support": "If this persists, TARA's text responses are still available."
            }
        )

# Frontend compatibility alias
@app.post("/api/synthesize")
async def api_synthesize_alias(request: TTSRequest, background_tasks: BackgroundTasks):
    """Frontend compatibility alias for /tts/synthesize"""
    return await synthesize_speech(request, background_tasks)

@app.post("/chat_with_voice")
async def chat_with_voice(request: ChatRequest, background_tasks: BackgroundTasks):
    """HAI-Enhanced chat with voice response using Universal AI Engine"""
    
    try:
        domain = request.domain
        message = request.message
        
        # Use Universal AI Engine if available
        if AI_ENGINE_AVAILABLE:
            try:
                # Get Universal AI Engine
                ai_engine = await get_universal_engine()
                
                # Create AI request
                ai_request = AIRequest(
                    user_input=message,
                    domain=domain,
                    context={"source": "voice_chat", "timestamp": datetime.now().isoformat()},
                    urgency_level="normal"
                )
                
                # Get AI response
                ai_response = await ai_engine.process_request(ai_request)
                
                response_text = ai_response.response_text
                
                # Enhanced response data
                enhanced_response = {
                    "message": message,
                    "response": response_text,
                    "domain": domain,
                    "confidence": ai_response.confidence,
                    "processing_time": ai_response.processing_time,
                    "suggestions": ai_response.suggestions,
                    "follow_up_questions": ai_response.follow_up_questions,
                    "resources": ai_response.resources,
                    "emotional_tone": ai_response.emotional_tone,
                    "hai_context": ai_response.hai_context,
                    "ai_engine_used": True
                }
                
            except Exception as e:
                logger.error(f"AI Engine error: {e}")
                # Fallback to simple responses
                response_text = await _generate_fallback_response(message, domain)
                enhanced_response = {
                    "message": message,
                    "response": response_text,
                    "domain": domain,
                    "hai_context": f"TARA is engaging with you in {domain} mode (fallback mode)",
                    "ai_engine_used": False
                }
        else:
            # Use fallback responses
            response_text = await _generate_fallback_response(message, domain)
            enhanced_response = {
                "message": message,
                "response": response_text,
                "domain": domain,
                "hai_context": f"TARA is engaging with you in {domain} mode (fallback mode)",
                "ai_engine_used": False
            }
        
        # Create TTS request
        tts_request = TTSRequest(text=response_text, domain=domain)
        
        # Get voice synthesis
        synthesis_result = await synthesize_speech(tts_request, background_tasks)
        
        # Combine response with audio synthesis
        enhanced_response["audio_synthesis"] = synthesis_result
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Chat processing failed",
                "hai_message": "TARA is experiencing a temporary issue but remains committed to helping you. Please try again."
            }
        )

async def _generate_fallback_response(message: str, domain: str) -> str:
    """Generate fallback response when AI Engine is not available"""
    domain_responses = {
        "universal": f"I'm here to help with anything you need. Regarding '{message}', let me provide you with comprehensive assistance.",
        "healthcare": f"I understand your healthcare concern about '{message}'. Let me provide you with helpful, evidence-based information while reminding you to consult with healthcare professionals for medical advice.",
        "business": f"Regarding your business inquiry about '{message}', let me share strategic insights and practical recommendations to help you achieve your goals.",
        "education": f"Great question about '{message}'! Let me break this down in a clear, engaging way that helps you learn and understand the concepts thoroughly.",
        "creative": f"I love your creative thinking about '{message}'! Let me help spark some innovative ideas and approaches you might explore.",
        "leadership": f"Your leadership question about '{message}' is excellent. Let me share some strategic perspectives that can help you guide your team effectively."
    }
    
    return domain_responses.get(domain, domain_responses["universal"])

@app.post("/ai/chat")
async def ai_chat(request: ChatRequest):
    """Direct AI chat without voice synthesis - Core backend AI processing"""
    
    try:
        domain = request.domain
        message = request.message
        
        # Use Universal AI Engine if available
        if AI_ENGINE_AVAILABLE:
            try:
                # Get Universal AI Engine
                ai_engine = await get_universal_engine()
                
                # Create AI request
                ai_request = AIRequest(
                    user_input=message,
                    domain=domain,
                    context={"source": "direct_chat", "timestamp": datetime.now().isoformat()},
                    urgency_level="normal"
                )
                
                # Get AI response
                ai_response = await ai_engine.process_request(ai_request)
                
                return {
                    "success": True,
                    "message": message,
                    "response": ai_response.response_text,
                    "domain": domain,
                    "confidence": ai_response.confidence,
                    "processing_time": ai_response.processing_time,
                    "suggestions": ai_response.suggestions,
                    "follow_up_questions": ai_response.follow_up_questions,
                    "resources": ai_response.resources,
                    "emotional_tone": ai_response.emotional_tone,
                    "hai_context": ai_response.hai_context,
                    "ai_engine_used": True,
                    "backend_type": "universal_ai_engine"
                }
                
            except Exception as e:
                logger.error(f"AI Engine error in direct chat: {e}")
                # Fallback to simple responses
                response_text = await _generate_fallback_response(message, domain)
                return {
                    "success": True,
                    "message": message,
                    "response": response_text,
                    "domain": domain,
                    "hai_context": f"TARA is engaging with you in {domain} mode (fallback mode)",
                    "ai_engine_used": False,
                    "backend_type": "fallback_responses"
                }
        else:
            # Use fallback responses
            response_text = await _generate_fallback_response(message, domain)
            return {
                "success": True,
                "message": message,
                "response": response_text,
                "domain": domain,
                "hai_context": f"TARA is engaging with you in {domain} mode (fallback mode)",
                "ai_engine_used": False,
                "backend_type": "fallback_responses"
            }
        
    except Exception as e:
        logger.error(f"❌ AI Chat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "AI chat processing failed",
                "hai_message": "TARA is experiencing a temporary issue but remains committed to helping you. Please try again."
            }
        )

@app.get("/health")
async def health_check():
    """General health check endpoint for frontend"""
    try:
        # Check TTS systems
        tts_systems_status = []
        if EDGE_TTS_AVAILABLE:
            tts_systems_status.append("edge_tts")
        if PYTTSX3_AVAILABLE:
            tts_systems_status.append("pyttsx3")
        
        # Check AI engine
        ai_status = "available" if AI_ENGINE_AVAILABLE else "fallback"
        
        # Check HAI Security
        security_status = "enabled" if HAI_SECURITY_AVAILABLE else "basic"
        
        return {
            "status": "healthy",
            "server": "TARA Universal Voice Server v2.0.0",
            "tts_systems": tts_systems_status,
            "ai_engine": ai_status,
            "security": security_status,
            "endpoints": {
                "tts": ["/tts/status", "/tts/synthesize"],
                "ai": ["/ai/chat", "/ai/health"],
                "voice": ["/chat_with_voice"],
                "audio": ["/audio/{filename}"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "message": "Health check failed"
            }
        )

@app.get("/ai/health")
async def ai_health_check():
    """AI Engine health check endpoint"""
    
    if not AI_ENGINE_AVAILABLE:
        return {
            "ai_engine_available": False,
            "status": "not_available",
            "message": "Universal AI Engine is not installed or available"
        }
    
    try:
        ai_engine = await get_universal_engine()
        health_status = await ai_engine.health_check()
        
        return {
            "ai_engine_available": True,
            "status": "healthy",
            "health_check": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        return {
            "ai_engine_available": True,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """HAI-Enhanced secure audio file serving"""
    
    try:
        # Security: Validate filename to prevent path traversal
        # Support both old timestamp format and new content-hash format
        if not (re.match(r'^temp_audio_\d+_[a-f0-9]{8}\.(mp3|wav)$', filename) or 
                re.match(r'^tara_audio_[a-f0-9]{12}\.(mp3|wav)$', filename)):
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        file_path = os.path.join(tempfile.gettempdir(), filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Determine media type
        media_type = "audio/mpeg" if filename.endswith('.mp3') else "audio/wav"
        
        return FileResponse(
            file_path,
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Audio serving error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve audio file")

# HAI-Enhanced startup event
@app.on_event("startup")
async def startup_event():
    """HAI-Enhanced startup with comprehensive system initialization"""
    logger.info("🤝 Starting TARA Voice Server with HAI principles...")
    logger.info("🎯 Mission: Help Anytime, Everywhere")
    
    # Initialize TTS systems
    logger.info("🔧 Initializing TTS systems...")
    
    if tts_systems["edge_tts"]:
        logger.info("✅ Edge TTS available - High-quality neural voices ready")
    else:
        logger.warning("⚠️ Edge TTS not available")
    
    if tts_systems["pyttsx3"]:
        logger.info("✅ pyttsx3 available - Reliable offline TTS ready")
    else:
        logger.warning("⚠️ pyttsx3 not available")
    
    available_systems = [name for name, available in tts_systems.items() if available]
    
    if not available_systems:
        logger.error("❌ No TTS systems available! Please install edge-tts or pyttsx3")
        return
    
    logger.info(f"🎤 TTS Systems available: {available_systems}")
    logger.info(f"⭐ Preferred system: {preferred_tts}")
    logger.info(f"🎭 Voice mapping enabled: {len(VALID_EDGE_VOICES)} valid voices, {len(VOICE_MAPPING)} aliases")
    logger.info("🔄 Invalid voice names (like 'gentle_female') will be automatically mapped to valid voices")
    logger.info("💾 Content-based caching enabled: Identical text+voice will reuse same audio file")
    logger.info("♻️ Smart cache management: Reduces file creation and improves performance")
    
    # Initialize HAI Security Components
    if HAI_SECURITY_AVAILABLE:
        try:
            # Initialize privacy manager
            privacy_manager = get_privacy_manager()
            logger.info("🔒 ✅ Privacy Manager initialized - Local encryption & auto-cleanup active")
            
            # Initialize resource monitor
            resource_monitor = get_resource_monitor()
            resource_monitor.start_monitoring()
            logger.info("📊 ✅ Resource Monitor started - CPU/Memory limits enforced")
            
            # Initialize security validator
            security_validator = get_security_validator()
            logger.info("🛡️ ✅ Security Validator initialized - Advanced threat protection active")
            
            logger.info("🤝 ✅ HAI Security Framework fully operational")
        except Exception as e:
            logger.error(f"⚠️ HAI Security initialization failed: {e}")
            logger.warning("⚠️ Continuing with basic security measures")
    else:
        logger.warning("⚠️ HAI Security components not available - using basic security")
    
    # Log HAI features
    logger.info("🛡️ HAI Safety Features Active:")
    logger.info(f"   • Rate limiting: {HAIConfig.RATE_LIMIT_PER_MINUTE} requests/minute")
    logger.info(f"   • Input validation: Max {HAIConfig.MAX_TEXT_LENGTH} characters")
    logger.info(f"   • Auto cleanup: {HAIConfig.AUTO_CLEANUP_MINUTES} minutes")
    logger.info(f"   • Graceful degradation: {HAIConfig.ENABLE_GRACEFUL_DEGRADATION}")
    if HAI_SECURITY_AVAILABLE:
        logger.info(f"   • Advanced privacy protection with local encryption")
        logger.info(f"   • Resource monitoring and limits")
        logger.info(f"   • Enhanced security validation and threat detection")
    
    logger.info("🌟 TARA Voice Server started successfully with HAI enhancement!")
    logger.info("🤝 Ready to help humans anytime, everywhere they need assistance!")

if __name__ == "__main__":
    # HAI-Enhanced server startup
    print("🤝 Starting TARA Universal Backend Server on http://localhost:5000")
    print("🧠 Universal AI Engine Status:", "✅ Available" if AI_ENGINE_AVAILABLE else "⚠️ Not Available (using fallbacks)")
    print("📋 Available endpoints:")
    print("   🔊 Voice Services:")
    print("      • GET  /tts/status - Check TTS & AI system status")
    print("      • GET  /tts/voices - List available voices and mappings")
    print("      • POST /tts/synthesize - Synthesize speech from text")
    print("      • POST /api/synthesize - Frontend compatibility alias")  
    print("      • POST /chat_with_voice - Chat with voice response")
    print("      • GET  /audio/{filename} - Serve audio files")
    print("   🧠 AI Services:")
    print("      • POST /ai/chat - Direct AI chat (no voice)")
    print("      • GET  /ai/health - AI Engine health check")
    print("   🔧 System Services:")
    print("      • GET  /health - General server health check")
    print("🛡️ HAI Safety Features:")
    print(f"   • Rate limiting: {HAIConfig.RATE_LIMIT_PER_MINUTE} req/min per IP")
    print(f"   • Input validation & sanitization")
    print(f"   • Automatic file cleanup ({HAIConfig.AUTO_CLEANUP_MINUTES} min)")
    print(f"   • Smart caching: Identical text+voice reuse same file")
    print(f"   • Multi-level fallback system")
    print(f"   • Universal AI Engine with 6 domain experts")
    print("🚀 Ready for tara-ai-companion frontend integration!")
    print("🌟 Mission: Robust Backend to Support All Human Needs Through AI!")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5000,
        log_level="info"
    ) 