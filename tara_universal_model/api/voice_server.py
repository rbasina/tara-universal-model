#!/usr/bin/env python3
"""
TARA Voice Server Module Entry Point
This file serves as a module entry point for the voice server.
"""

import sys
import os

# Add the parent directory to sys.path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import and run the voice server
from voice_server import app

if __name__ == "__main__":
    import uvicorn
    print("🎤 Starting TARA Voice Server on http://localhost:5000")
    print("📋 Available endpoints:")
    print("   • GET  /tts/status - Check TTS system status")
    print("   • POST /tts/synthesize - Synthesize speech from text")
    print("   • POST /chat_with_voice - Chat with voice response")
    print("   • GET  /audio/{filename} - Serve audio files")
    print("🚀 Ready for tara-ai-companion integration!")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info") 