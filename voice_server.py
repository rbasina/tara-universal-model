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
from typing import Dict, Any, Optional
import base64

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TARA Voice Server",
    description="TTS server for TARA AI companion integration",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3005", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS manager
tts_manager = None

class TTSManager:
    """TTS manager using available TTS systems."""
    
    def __init__(self):
        self.available_systems = {}
        self.preferred_system = None
        self.temp_dir = Path(tempfile.gettempdir()) / "tara_audio"
        self.temp_dir.mkdir(exist_ok=True)
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize available TTS systems."""
        logger.info("Initializing TTS systems...")
        
        # Try Edge TTS first
        try:
            import edge_tts
            self.available_systems['edge_tts'] = {
                'module': edge_tts,
                'voices': {
                    'healthcare': 'en-US-AriaNeural',
                    'business': 'en-US-JennyNeural', 
                    'education': 'en-US-AriaNeural',
                    'creative': 'en-US-JennyNeural',
                    'leadership': 'en-US-GuyNeural',
                    'universal': 'en-US-AriaNeural'
                }
            }
            self.preferred_system = 'edge_tts'
            logger.info("✅ Edge TTS available")
        except ImportError:
            logger.warning("❌ Edge TTS not available")
        
        # Try pyttsx3 as fallback
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                self.available_systems['pyttsx3'] = {
                    'module': pyttsx3,
                    'engine': engine,
                    'voices': voices
                }
                if not self.preferred_system:
                    self.preferred_system = 'pyttsx3'
                logger.info("✅ pyttsx3 available")
            else:
                logger.warning("❌ pyttsx3 no voices found")
        except Exception as e:
            logger.warning(f"❌ pyttsx3 error: {e}")
        
        logger.info(f"TTS Systems available: {list(self.available_systems.keys())}")
        logger.info(f"Preferred system: {self.preferred_system}")
    
    async def synthesize_speech(self, text: str, domain: str = "universal") -> Dict[str, Any]:
        """Synthesize speech from text."""
        if not self.available_systems:
            return {"success": False, "error": "No TTS systems available"}
        
        start_time = time.time()
        
        try:
            if self.preferred_system == 'edge_tts':
                return await self._synthesize_edge_tts(text, domain, start_time)
            elif self.preferred_system == 'pyttsx3':
                return await self._synthesize_pyttsx3(text, domain, start_time)
            else:
                return {"success": False, "error": "No preferred TTS system"}
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _synthesize_edge_tts(self, text: str, domain: str, start_time: float) -> Dict[str, Any]:
        """Synthesize using Edge TTS."""
        try:
            edge_tts = self.available_systems['edge_tts']['module']
            voices = self.available_systems['edge_tts']['voices']
            voice = voices.get(domain, voices['universal'])
            
            # Create temporary file
            audio_file = self.temp_dir / f"temp_audio_{int(time.time() * 1000)}.mp3"
            
            # Generate speech
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(audio_file))
            
            # Check if file was created and has content
            if not audio_file.exists() or audio_file.stat().st_size == 0:
                # Try fallback voice
                fallback_voice = 'en-US-AriaNeural'
                logger.warning(f"Voice {voice} failed, trying fallback {fallback_voice}")
                audio_file = self.temp_dir / f"temp_audio_{int(time.time() * 1000)}_fallback.mp3"
                communicate = edge_tts.Communicate(text, fallback_voice)
                await communicate.save(str(audio_file))
                voice = fallback_voice
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "audio_url": f"http://localhost:5000/audio/{audio_file.name}",
                "processing_time": processing_time,
                "voice": voice,
                "system": "edge_tts"
            }
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            # Try fallback to pyttsx3 if available
            if 'pyttsx3' in self.available_systems:
                logger.info("Falling back to pyttsx3")
                return await self._synthesize_pyttsx3(text, domain, start_time)
            return {"success": False, "error": str(e)}
    
    async def _synthesize_pyttsx3(self, text: str, domain: str, start_time: float) -> Dict[str, Any]:
        """Synthesize using pyttsx3."""
        try:
            engine = self.available_systems['pyttsx3']['engine']
            
            # Create temporary file
            audio_file = self.temp_dir / f"temp_audio_{int(time.time() * 1000)}.wav"
            
            # Configure voice properties
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.9)
            
            # Generate speech
            engine.save_to_file(text, str(audio_file))
            engine.runAndWait()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "audio_url": f"http://localhost:5000/audio/{audio_file.name}",
                "processing_time": processing_time,
                "system": "pyttsx3"
            }
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get TTS system status."""
        return {
            "status": "ready" if self.available_systems else "no_tts_available",
            "edge_tts_available": "edge_tts" in self.available_systems,
            "pyttsx3_available": "pyttsx3" in self.available_systems,
            "preferred_system": self.preferred_system,
            "domains": ["healthcare", "business", "education", "creative", "leadership", "universal"]
        }

# Initialize TTS manager
@app.on_event("startup")
async def startup_event():
    global tts_manager
    logger.info("Starting TARA Voice Server...")
    tts_manager = TTSManager()
    logger.info("TARA Voice Server started successfully!")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "TARA Voice Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/tts/synthesize", "/tts/status", "/chat_with_voice"]
    }

# TTS Status endpoint
@app.get("/tts/status")
async def tts_status():
    """Get TTS system status."""
    try:
        if not tts_manager:
            return {"success": False, "error": "TTS manager not initialized"}
        
        status = tts_manager.get_status()
        return {
            "success": True,
            **status
        }
    except Exception as e:
        logger.error(f"TTS status error: {e}")
        return {"success": False, "error": str(e)}

# TTS Synthesize endpoint
@app.post("/tts/synthesize")
async def synthesize_speech_endpoint(request: Request):
    """
    Synthesize speech from text.
    
    Request body:
    {
        "text": "Hello, I'm TARA!",
        "domain": "universal"
    }
    """
    try:
        request_data = await request.json()
        text = request_data.get("text", "")
        domain = request_data.get("domain", "universal")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if not tts_manager:
            raise HTTPException(status_code=503, detail="TTS manager not initialized")
        
        result = await tts_manager.synthesize_speech(text, domain)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat with Voice endpoint
@app.post("/chat_with_voice")
async def chat_with_voice_endpoint(request: Request):
    """
    Chat with TARA and get voice response.
    
    Request body:
    {
        "message": "How are you today?",
        "domain": "universal",
        "voice_response": true
    }
    """
    try:
        request_data = await request.json()
        message = request_data.get("message", "")
        domain = request_data.get("domain", "universal")
        voice_response = request_data.get("voice_response", True)
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Simple TARA response (you can enhance this)
        responses = {
            "healthcare": "I'm here to help with your health and wellness needs. How can I assist you today?",
            "business": "I'm ready to help you with your business objectives. What would you like to discuss?",
            "education": "I'm excited to help you learn something new! What topic interests you?",
            "creative": "Let's explore your creative side! What inspiring project are you working on?",
            "leadership": "I'm here to support your leadership journey. What challenges are you facing?",
            "universal": "Hello! I'm TARA, your AI companion. I'm here to help with whatever you need!"
        }
        
        text_response = responses.get(domain, responses["universal"])
        
        result = {
            "success": True,
            "text_response": text_response,
            "domain": domain,
            "processing_time": 0.1
        }
        
        # Add voice synthesis if requested
        if voice_response and tts_manager:
            voice_result = await tts_manager.synthesize_speech(text_response, domain)
            if voice_result["success"]:
                result["audio_url"] = voice_result["audio_url"]
                result["voice_processing_time"] = voice_result["processing_time"]
            else:
                result["voice_error"] = voice_result["error"]
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat with voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Audio file serving endpoint
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files."""
    try:
        audio_file = tts_manager.temp_dir / filename
        if not audio_file.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(audio_file),
            media_type="audio/mpeg" if filename.endswith('.mp3') else "audio/wav",
            filename=filename
        )
    except Exception as e:
        logger.error(f"Audio serving error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)}
    )

if __name__ == "__main__":
    print("🎤 Starting TARA Voice Server on http://localhost:5000")
    print("📋 Available endpoints:")
    print("   • GET  /tts/status - Check TTS system status")
    print("   • POST /tts/synthesize - Synthesize speech from text")
    print("   • POST /chat_with_voice - Chat with voice response")
    print("   • GET  /audio/{filename} - Serve audio files")
    print("🚀 Ready for tara-ai-companion integration!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    ) 