"""
TTS Integration for TARA Universal Model
Supports multiple TTS systems with intelligent fallback
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TTSManager:
    """Manages Text-to-Speech for TARA with multiple backend support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TTS Manager with configuration."""
        self.config = config or {}
        self.available_systems = {}
        self.preferred_system = None
        self.voice_cache = {}
        
        # Default configuration
        self.default_config = {
            "edge_tts": {
                "voices": {
                    "healthcare": "en-US-AriaNeural",      # Gentle, caring
                    "business": "en-US-JennyNeural",       # Professional
                    "education": "en-US-AriaNeural",       # Patient, clear
                    "creative": "en-GB-SoniaNeural",       # Expressive
                    "leadership": "en-US-JennyNeural",     # Authoritative
                    "universal": "en-US-AriaNeural"        # Friendly default
                },
                "rate": "+0%",
                "pitch": "+0Hz"
            },
            "pyttsx3": {
                "rate": 180,
                "volume": 0.9,
                "voice_index": 0  # Use first available voice
            }
        }
        
        # Initialize available systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize and test available TTS systems."""
        logger.info("Initializing TTS systems...")
        
        # Test Edge TTS
        try:
            import edge_tts
            self.available_systems['edge_tts'] = {
                'module': edge_tts,
                'status': 'available',
                'speed': 'fast',
                'quality': 'high',
                'offline': False
            }
            if not self.preferred_system:
                self.preferred_system = 'edge_tts'
            logger.info("✅ Edge TTS available")
        except ImportError:
            logger.warning("❌ Edge TTS not available")
        
        # Test pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                self.available_systems['pyttsx3'] = {
                    'module': pyttsx3,
                    'engine': engine,
                    'voices': voices,
                    'status': 'available',
                    'speed': 'very_fast',
                    'quality': 'medium',
                    'offline': True
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
    
    async def synthesize_speech(
        self, 
        text: str, 
        domain: str = "universal",
        system: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> Tuple[bool, Optional[bytes], Dict[str, Any]]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            domain: TARA domain (healthcare, business, etc.)
            system: Preferred TTS system (None for auto)
            emotion: Emotion hint for advanced TTS
            
        Returns:
            (success, audio_data, metadata)
        """
        start_time = time.time()
        
        # Choose TTS system
        tts_system = system or self.preferred_system
        
        if not tts_system or tts_system not in self.available_systems:
            # Fallback to any available system
            if self.available_systems:
                tts_system = list(self.available_systems.keys())[0]
            else:
                return False, None, {"error": "No TTS systems available"}
        
        # Generate speech based on system
        try:
            if tts_system == 'edge_tts':
                success, audio_data, metadata = await self._synthesize_edge_tts(text, domain, emotion)
            elif tts_system == 'pyttsx3':
                success, audio_data, metadata = await self._synthesize_pyttsx3(text, domain)
            else:
                return False, None, {"error": f"Unknown TTS system: {tts_system}"}
            
            # Add timing metadata
            generation_time = time.time() - start_time
            metadata.update({
                "system": tts_system,
                "domain": domain,
                "generation_time": generation_time,
                "text_length": len(text)
            })
            
            if success:
                logger.info(f"TTS generated: {len(text)} chars in {generation_time:.2f}s using {tts_system}")
            
            return success, audio_data, metadata
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return False, None, {"error": str(e), "system": tts_system}
    
    async def _synthesize_edge_tts(
        self, 
        text: str, 
        domain: str, 
        emotion: Optional[str] = None
    ) -> Tuple[bool, Optional[bytes], Dict[str, Any]]:
        """Synthesize using Edge TTS."""
        try:
            edge_tts = self.available_systems['edge_tts']['module']
            
            # Select voice based on domain
            voice_config = self.default_config['edge_tts']['voices']
            voice = voice_config.get(domain, voice_config['universal'])
            
            # Generate audio
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return True, audio_data, {
                "voice": voice,
                "format": "mp3",
                "sample_rate": 24000
            }
            
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return False, None, {"error": str(e)}
    
    async def _synthesize_pyttsx3(
        self, 
        text: str, 
        domain: str
    ) -> Tuple[bool, Optional[bytes], Dict[str, Any]]:
        """Synthesize using pyttsx3."""
        try:
            system_info = self.available_systems['pyttsx3']
            engine = system_info['engine']
            voices = system_info['voices']
            
            # Configure engine
            config = self.default_config['pyttsx3']
            engine.setProperty('rate', config['rate'])
            engine.setProperty('volume', config['volume'])
            
            # Select voice (prefer female voices)
            voice_index = config['voice_index']
            if voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
            
            # Generate to temporary file (pyttsx3 limitation)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            
            # Read audio data
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
            
            return True, audio_data, {
                "voice": voices[voice_index].name if voice_index < len(voices) else "default",
                "format": "wav",
                "sample_rate": 22050
            }
            
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return False, None, {"error": str(e)}
    
    def get_available_systems(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available TTS systems."""
        return self.available_systems.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current TTS system status."""
        return {
            "available_systems": list(self.available_systems.keys()),
            "preferred_system": self.preferred_system,
            "total_systems": len(self.available_systems),
            "systems_info": {
                name: {
                    "status": info.get("status", "unknown"),
                    "speed": info.get("speed", "unknown"),
                    "quality": info.get("quality", "unknown"),
                    "offline": info.get("offline", False)
                }
                for name, info in self.available_systems.items()
            }
        }

# Global TTS manager instance
tts_manager = None

def get_tts_manager() -> TTSManager:
    """Get or create global TTS manager instance."""
    global tts_manager
    if tts_manager is None:
        tts_manager = TTSManager()
    return tts_manager

async def synthesize_tara_speech(
    text: str,
    domain: str = "universal",
    emotion: Optional[str] = None
) -> Tuple[bool, Optional[bytes], Dict[str, Any]]:
    """
    Convenience function to synthesize speech for TARA.
    
    Args:
        text: Text to synthesize
        domain: TARA domain
        emotion: Optional emotion hint
        
    Returns:
        (success, audio_data, metadata)
    """
    manager = get_tts_manager()
    return await manager.synthesize_speech(text, domain, emotion=emotion) 