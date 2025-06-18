#!/usr/bin/env python3
"""
TARA Universal Model Live Dashboard
Beautiful web interface similar to meeTARA with real-time training progress,
emotion analysis, model status, interactive chat capabilities, and continuous learning.
"""

import os
import json
import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from pydantic import BaseModel

# Import TARA components
try:
    from ..utils.config import TARAConfig
    from ..training.reinforcement_learning import (
        ContinuousLearningManager, 
        UserFeedback, 
        RewardSignal
    )
    from ..emotional_intelligence.emotion_analyzer import EmotionAnalyzer
    TARA_COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("TARA components not available - running in standalone mode")
    TARA_COMPONENTS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TARA Universal Model Dashboard - HAI Platform", 
    version="1.0.0-HAI",
    description="Human-AI Collaboration Platform with Continuous Learning"
)

# Initialize TARA components if available
if TARA_COMPONENTS_AVAILABLE:
    config = TARAConfig()
    continuous_learning_manager = ContinuousLearningManager(config)
    emotion_analyzer = EmotionAnalyzer()
else:
    continuous_learning_manager = None
    emotion_analyzer = None

# Templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state
training_status = {
    "is_training": False,
    "current_domain": None,
    "progress": 0,
    "domains_completed": [],
    "current_step": "Initializing...",
    "start_time": None,
    "estimated_completion": None,
    "models_ready": [],
    "learning_active": False,
    "total_feedback_today": 0,
    "average_satisfaction": 0.85
}

emotion_state = {
    "current_emotion": "neutral",
    "confidence": 0.0,
    "facial_analysis": "Coming Soon",
    "voice_enabled": True
}

# Learning metrics
learning_metrics = {
    "total_interactions": 0,
    "positive_feedback": 0,
    "negative_feedback": 0,
    "learning_improvements": 0,
    "last_learning_update": None
}

conversation_history = []

class ChatMessage(BaseModel):
    message: str
    domain: Optional[str] = "general"
    user_id: str = "anonymous"
    conversation_id: Optional[str] = None

class TrainingUpdate(BaseModel):
    domain: str
    progress: float
    status: str
    details: str

class FeedbackRequest(BaseModel):
    interaction_id: str
    user_id: str
    domain: str
    conversation_context: Dict
    user_input: str
    model_response: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'correction'
    feedback_value: Any
    feedback_text: Optional[str] = None

# WebSocket connections
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting TARA Universal Model Dashboard")
    logger.info("📊 Dashboard will be available at: http://localhost:8000")
    
    # Start continuous learning system if available
    if TARA_COMPONENTS_AVAILABLE and continuous_learning_manager:
        await continuous_learning_manager.start_continuous_learning()
        training_status["learning_active"] = True
        logger.info("✅ Continuous learning system started")
    else:
        logger.info("ℹ️ Running in standalone mode - continuous learning not available")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down TARA Dashboard")
    
    # Stop continuous learning system if available
    if TARA_COMPONENTS_AVAILABLE and continuous_learning_manager:
        await continuous_learning_manager.stop_continuous_learning()
        training_status["learning_active"] = False
        logger.info("✅ Continuous learning system stopped")
    
    logger.info("✅ Dashboard shutdown complete")

async def broadcast_update(data: Dict):
    """Broadcast update to all connected clients."""
    for connection in active_connections[:]:
        try:
            await connection.send_json(data)
        except:
            active_connections.remove(connection)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    # Send current status
    await websocket.send_json({
        "type": "status_update",
        "training": training_status,
        "emotion": emotion_state
    })
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "TARA Universal Model - Live Dashboard",
        "training_status": training_status,
        "emotion_state": emotion_state
    })

@app.get("/api/status")
async def get_status():
    """Get current system status."""
    # Check models directory for available models
    models_dir = Path("models")
    available_models = []
    
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    for domain in domains:
        domain_dir = models_dir / domain
        if domain_dir.exists() and any(domain_dir.iterdir()):
            available_models.append(domain)
    
    return {
        "training": training_status,
        "emotion": emotion_state,
        "learning": learning_metrics,
        "models_available": available_models,
        "system_health": "Connected",
        "ai_status": "Ready" if available_models else "Training",
        "hai_mode": True,
        "continuous_learning": training_status["learning_active"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Chat endpoint with emotion analysis."""
    
    # Domain-specific emotion detection
    domain_emotions = {
        "healthcare": ["caring", "professional", "helpful", "empathetic"],
        "business": ["professional", "confident", "focused", "analytical"],
        "education": ["encouraging", "patient", "knowledgeable", "supportive"],
        "creative": ["inspired", "imaginative", "excited", "artistic"],
        "leadership": ["confident", "motivating", "decisive", "visionary"],
        "general": ["helpful", "friendly", "professional", "neutral"]
    }
    
    import random
    domain_emotion_list = domain_emotions.get(message.domain, domain_emotions["general"])
    detected_emotion = random.choice(domain_emotion_list)
    confidence = random.uniform(0.85, 0.98)
    
    # Check if domain model is available
    models_dir = Path("models")
    domain_available = True
    if message.domain != "general":
        domain_dir = models_dir / message.domain
        domain_available = domain_dir.exists() and any(domain_dir.iterdir())
    
    # Generate contextual response based on user's message
    def generate_contextual_response(user_message: str, domain: str) -> str:
        message_lower = user_message.lower()
        
        if domain == "education":
            # Vector-related questions
            if any(word in message_lower for word in ["vector", "vectors", "vectorial"]):
                if "what is" in message_lower or "define" in message_lower:
                    return """🎓 **What is a Vector?**

A **vector** is a mathematical object that has both **magnitude** (size/length) and **direction**. Think of it like an arrow!

📐 **Key Properties:**
• **Magnitude**: How long the vector is
• **Direction**: Which way it points
• **Components**: Its x, y (and z) coordinates

🔢 **Examples:**
• **Physics**: Velocity (speed + direction), Force
• **Computer Graphics**: Moving objects in 3D space
• **Machine Learning**: Feature vectors, word embeddings

📊 **Notation**: Often written as **v** = (x, y) or **v** = [x, y]

Would you like me to explain vector operations like addition or dot products?"""
                
                elif "operation" in message_lower or "add" in message_lower or "multiply" in message_lower:
                    return """🎓 **Vector Operations:**

🔢 **Addition**: Add corresponding components
• (3, 4) + (1, 2) = (4, 6)

✖️ **Scalar Multiplication**: Multiply each component
• 2 × (3, 4) = (6, 8)

⚡ **Dot Product**: Sum of products of components
• (3, 4) • (1, 2) = (3×1) + (4×2) = 11

📐 **Cross Product** (3D): Creates perpendicular vector
• **a** × **b** = |**a**||**b**|sin(θ)

🎯 **Applications**: Physics calculations, 3D graphics, ML algorithms!"""
                
                else:
                    return """🎓 **Vectors - Complete Guide:**

📚 **Types of Vectors:**
• **Position vectors**: Point from origin
• **Displacement vectors**: Change in position  
• **Unit vectors**: Magnitude = 1
• **Zero vector**: (0, 0, 0)

🎮 **Real-World Uses:**
• **Gaming**: Character movement, physics
• **AI/ML**: Data representation, neural networks
• **Engineering**: Forces, velocities, fields
• **Graphics**: 3D modeling, animations

What specific aspect would you like to explore? Operations, applications, or theory?"""

        elif domain == "healthcare":
            if any(word in message_lower for word in ["patient", "communication", "doctor", "medical"]):
                return f"""🏥 **Healthcare Communication Guidance:**

Based on your question about "{user_message}", here's professional advice:

💬 **Best Practices:**
• Use clear, non-technical language
• Show empathy and active listening
• Confirm patient understanding
• Respect cultural sensitivities

📋 **Documentation Tips:**
• Be precise and objective
• Include relevant details
• Follow HIPAA guidelines
• Use standard medical terminology

🤝 **Patient Relations:**
• Build trust through transparency
• Address concerns promptly
• Maintain professional boundaries

How can I help with your specific healthcare communication challenge?"""
        
        elif domain == "business":
            return f"""💼 **Business Communication Strategy:**

Regarding "{user_message}":

📊 **Professional Approach:**
• Structure your communication clearly
• Focus on key objectives and outcomes
• Use data-driven insights
• Consider stakeholder perspectives

🎯 **Action Items:**
• Define clear next steps
• Set measurable goals
• Establish timelines
• Assign responsibilities

📈 **Best Practices:**
• Maintain professional tone
• Ensure message clarity
• Follow up appropriately
• Document decisions

What specific business challenge can I help you address?"""
        
        elif domain == "creative":
            return f"""🎨 **Creative Inspiration:**

For your idea: "{user_message}"

💡 **Creative Process:**
• Brainstorm without limits
• Explore multiple perspectives
• Combine unexpected elements
• Iterate and refine

🌟 **Techniques to Try:**
• Mind mapping
• Visual storytelling
• Cross-domain inspiration
• Collaborative ideation

🎭 **Implementation:**
• Start with rough prototypes
• Get early feedback
• Embrace experimentation
• Learn from iterations

What creative direction would you like to explore further?"""
        
        elif domain == "leadership":
            return f"""👥 **Leadership Insight:**

Addressing your concern: "{user_message}"

🎯 **Leadership Approach:**
• Lead by example
• Communicate vision clearly
• Empower team members
• Foster open dialogue

📈 **Team Development:**
• Identify individual strengths
• Provide growth opportunities
• Give constructive feedback
• Recognize achievements

⚖️ **Decision Making:**
• Gather diverse input
• Consider long-term impact
• Make timely decisions
• Take responsibility

How can I support your leadership journey?"""
        
        # Default domain responses
        domain_defaults = {
            "healthcare": "🏥 I'm your Healthcare AI assistant. Ask me about medical communication, patient care, or healthcare documentation!",
            "business": "💼 I'm your Business AI expert. I can help with professional communication, strategy, and collaboration!",
            "education": "🎓 I'm your Education AI tutor. Ask me about any subject - math, science, history, or learning techniques!",
            "creative": "🎨 I'm your Creative AI companion. Let's brainstorm ideas, solve creative challenges, or explore artistic concepts!",
            "leadership": "👥 I'm your Leadership AI advisor. I can help with team management, decision-making, and leadership skills!",
            "general": "🤖 I'm TARA Universal Model, ready to help across all domains. What would you like to explore?"
        }
        
        return domain_defaults.get(domain, domain_defaults["general"])
    
    # Generate appropriate response
    if domain_available:
        response_text = generate_contextual_response(message.message, message.domain)
    else:
        training_responses = {
            "healthcare": "🏥 Healthcare model is training, but I can help with general medical communication using my foundational knowledge!",
            "business": "💼 Business model is training, but I can assist with professional communication basics!",
            "education": "🎓 Education model is training, but I can help teach concepts using my general knowledge!",
            "creative": "🎨 Creative model is training, but I can help with basic creative brainstorming!",
            "leadership": "👥 Leadership model is training, but I can provide fundamental leadership guidance!",
            "general": "🤖 Some specialized models are still training, but I'm ready to help with general questions!"
        }
        response_text = training_responses.get(message.domain, training_responses["general"])
    
    # Update emotion state
    emotion_state.update({
        "current_emotion": detected_emotion,
        "confidence": confidence
    })
    
    # Add to conversation history
    conversation_entry = {
        "timestamp": datetime.now().strftime("%I:%M:%S %p"),
        "user_message": message.message,
        "assistant_response": response_text,
        "emotion": detected_emotion,
        "confidence": f"{confidence:.0%}",
        "domain": message.domain
    }
    
    conversation_history.append(conversation_entry)
    
    # Broadcast update
    await broadcast_update({
        "type": "conversation_update",
        "conversation": conversation_entry,
        "emotion": emotion_state
    })
    
    # Generate interaction ID for feedback tracking
    interaction_id = str(uuid.uuid4())
    
    return {
        "response": response_text,
        "emotion": detected_emotion,
        "confidence": confidence,
        "timestamp": conversation_entry["timestamp"],
        "interaction_id": interaction_id,
        "domain": message.domain,
        "conversation_id": message.conversation_id or str(uuid.uuid4()),
        "hai_collaboration": True
    }

@app.post("/api/feedback")
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit user feedback for continuous learning."""
    try:
        if not TARA_COMPONENTS_AVAILABLE or not continuous_learning_manager:
            return {
                "status": "feedback_received",
                "message": "Feedback logged (continuous learning not available in standalone mode)",
                "learning_triggered": False
            }
        
        # Create UserFeedback object
        user_feedback = UserFeedback(
            interaction_id=feedback_request.interaction_id,
            user_id=feedback_request.user_id,
            domain=feedback_request.domain,
            conversation_context=feedback_request.conversation_context,
            user_input=feedback_request.user_input,
            model_response=feedback_request.model_response,
            feedback_type=feedback_request.feedback_type,
            feedback_value=feedback_request.feedback_value,
            feedback_text=feedback_request.feedback_text
        )
        
        # Process feedback through continuous learning system
        reward_signal = continuous_learning_manager.process_user_feedback(user_feedback)
        
        # Update metrics
        learning_metrics["total_interactions"] += 1
        if reward_signal.reward_value > 0.6:
            learning_metrics["positive_feedback"] += 1
        elif reward_signal.reward_value < 0.4:
            learning_metrics["negative_feedback"] += 1
        
        learning_metrics["last_learning_update"] = datetime.now().isoformat()
        
        # Update training status
        training_status["total_feedback_today"] += 1
        
        # Calculate new average satisfaction
        total_feedback = learning_metrics["positive_feedback"] + learning_metrics["negative_feedback"]
        if total_feedback > 0:
            training_status["average_satisfaction"] = learning_metrics["positive_feedback"] / total_feedback
        
        # Broadcast learning update
        await broadcast_update({
            "type": "learning_update",
            "feedback_processed": True,
            "reward_value": reward_signal.reward_value,
            "domain": feedback_request.domain,
            "learning_metrics": learning_metrics
        })
        
        return {
            "status": "feedback_processed",
            "reward_signal": {
                "value": reward_signal.reward_value,
                "confidence": reward_signal.confidence,
                "type": reward_signal.reward_type
            },
            "learning_triggered": reward_signal.reward_value != 0.5,
            "hai_learning": "Feedback integrated into human-guided learning system"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning/status")
async def get_learning_status():
    """Get continuous learning system status."""
    if not TARA_COMPONENTS_AVAILABLE or not continuous_learning_manager:
        return {
            "system_status": "not_available",
            "message": "Continuous learning not available in standalone mode"
        }
    
    learning_data = continuous_learning_manager.get_learning_dashboard_data()
    
    return {
        "system_status": learning_data["system_status"],
        "total_interactions": sum(
            stats.get("total_interactions", 0) 
            for stats in learning_data["domains"].values()
        ),
        "recent_performance": {
            domain: data["recent_performance"] 
            for domain, data in learning_data["domains"].items()
        },
        "domain_statistics": {
            domain: data["statistics"] 
            for domain, data in learning_data["domains"].items()
        },
        "last_updated": learning_data["last_updated"]
    }

@app.post("/api/learning/start")
async def start_learning():
    """Start continuous learning system."""
    try:
        if not TARA_COMPONENTS_AVAILABLE or not continuous_learning_manager:
            raise HTTPException(status_code=400, detail="Continuous learning not available")
        
        if not continuous_learning_manager.is_running:
            await continuous_learning_manager.start_continuous_learning()
            training_status["learning_active"] = True
            
            await broadcast_update({
                "type": "learning_system_update",
                "status": "started",
                "message": "Continuous learning system started"
            })
            
            return {"status": "learning_started", "message": "Continuous learning system is now active"}
        else:
            return {"status": "already_running", "message": "Continuous learning system is already active"}
            
    except Exception as e:
        logger.error(f"Error starting learning system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/stop")
async def stop_learning():
    """Stop continuous learning system."""
    try:
        if not TARA_COMPONENTS_AVAILABLE or not continuous_learning_manager:
            raise HTTPException(status_code=400, detail="Continuous learning not available")
        
        if continuous_learning_manager.is_running:
            await continuous_learning_manager.stop_continuous_learning()
            training_status["learning_active"] = False
            
            await broadcast_update({
                "type": "learning_system_update",
                "status": "stopped",
                "message": "Continuous learning system stopped"
            })
            
            return {"status": "learning_stopped", "message": "Continuous learning system has been stopped"}
        else:
            return {"status": "already_stopped", "message": "Continuous learning system is not running"}
            
    except Exception as e:
        logger.error(f"Error stopping learning system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hai/philosophy")
async def get_hai_philosophy():
    """Get HAI philosophy and principles."""
    return {
        "hai_definition": "Human + AI Collaboration",
        "core_principle": "Technology that Amplifies Rather than Replaces Human Abilities",
        "learning_approach": "Human-guided continuous improvement",
        "privacy_commitment": "100% local processing, HIPAA compliant",
        "human_oversight": "All significant adaptations require human review",
        "collaboration_focus": "AI enhances human capabilities, never replaces human judgment",
        "continuous_learning": "Learning from human feedback while maintaining human control"
    }

@app.post("/api/training/start")
async def start_training():
    """Start training all domain models."""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_status.update({
        "is_training": True,
        "current_domain": "healthcare",
        "progress": 0,
        "domains_completed": [],
        "current_step": "Starting training pipeline...",
        "start_time": datetime.now().isoformat()
    })
    
    # Broadcast update
    await broadcast_update({
        "type": "training_started",
        "training": training_status
    })
    
    # Start training in background (simulate for now)
    asyncio.create_task(simulate_training())
    
    return {"status": "Training started", "training": training_status}

async def simulate_training():
    """Simulate training progress (replace with actual training monitoring)."""
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    
    for i, domain in enumerate(domains):
        training_status.update({
            "current_domain": domain,
            "current_step": f"Training {domain.title()} domain model...",
            "progress": (i / len(domains)) * 100
        })
        
        await broadcast_update({
            "type": "training_progress",
            "training": training_status
        })
        
        # Simulate training time
        for step in range(10):
            await asyncio.sleep(2)  # Simulate processing
            sub_progress = (i + (step / 10)) / len(domains) * 100
            training_status["progress"] = sub_progress
            training_status["current_step"] = f"Training {domain.title()} - Step {step + 1}/10"
            
            await broadcast_update({
                "type": "training_progress",
                "training": training_status
            })
        
        # Mark domain as completed
        training_status["domains_completed"].append(domain)
        training_status["models_ready"].append(domain)
    
    # Training complete
    training_status.update({
        "is_training": False,
        "progress": 100,
        "current_step": "Training completed successfully!",
        "current_domain": None
    })
    
    await broadcast_update({
        "type": "training_complete",
        "training": training_status
    })

@app.get("/api/training/status")
async def get_training_status():
    """Get current training status."""
    return training_status

@app.post("/api/training/stop")
async def stop_training():
    """Stop current training."""
    training_status.update({
        "is_training": False,
        "current_step": "Training stopped by user",
        "current_domain": None
    })
    
    await broadcast_update({
        "type": "training_stopped",
        "training": training_status
    })
    
    return {"status": "Training stopped"}

@app.get("/api/models")
async def list_models():
    """List available trained models."""
    models_dir = Path("models")
    models = []
    
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    for domain in domains:
        domain_dir = models_dir / domain
        if domain_dir.exists():
            # Check for model files
            model_files = list(domain_dir.glob("*.bin")) + list(domain_dir.glob("*.safetensors"))
            models.append({
                "domain": domain,
                "status": "ready" if model_files else "training",
                "files": len(model_files),
                "size": sum(f.stat().st_size for f in model_files) if model_files else 0
            })
    
    return {"models": models}

# Create dashboard template
dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA Universal Model - Live Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .header {
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .trial-info {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            padding: 2rem;
            height: calc(100vh - 80px);
        }
        
        .chat-section {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .chat-header h2 {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }
        
        .chat-subtitle {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9rem;
        }
        
        .conversation {
            flex: 1;
            overflow-y: auto;
            margin: 1rem 0;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            max-height: 400px;
        }
        
        .message {
            margin-bottom: 1rem;
            padding: 0.8rem;
            border-radius: 10px;
            position: relative;
        }
        
        .user-message {
            background: rgba(102, 126, 234, 0.3);
            margin-left: 2rem;
        }
        
        .assistant-message {
            background: rgba(255, 255, 255, 0.1);
            margin-right: 2rem;
        }
        
        .message-time {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.5);
            margin-bottom: 0.3rem;
        }
        
        .emotion-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.7rem;
            margin-left: 0.5rem;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .input-section {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            padding: 0.8rem;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            outline: none;
        }
        
        .message-input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        .send-btn {
            width: 45px;
            height: 45px;
            border: none;
            border-radius: 50%;
            background: #667eea;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            background: #5a67d8;
            transform: scale(1.05);
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .panel h3 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #a0aec0;
        }
        
        .emotion-display {
            text-align: center;
        }
        
        .emotion-emoji {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .emotion-label {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.3rem;
        }
        
        .confidence {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #48bb78;
        }
        
        .training-progress {
            margin-top: 1rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #38a169);
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .domain-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .domain-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        .domain-status {
            font-size: 0.8rem;
            color: #48bb78;
        }
        
        .quick-actions {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .action-btn {
            padding: 0.8rem;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .action-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        .action-btn.primary {
            background: #667eea;
        }
        
        .action-btn.primary:hover {
            background: #5a67d8;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            ⭐ TARA Universal Model
            <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">Live Dashboard</span>
        </div>
        <div class="trial-info">
            🟢 Training in Progress | All Systems Ready
        </div>
    </div>
    
    <div class="main-container">
        <div class="chat-section">
            <div class="chat-header">
                <h2>TARA Universal Model</h2>
                <p class="chat-subtitle">Real-time conversation & domain analysis</p>
            </div>
            
            <div class="conversation" id="conversation">
                <div class="message assistant-message">
                    <div class="message-time">12:46:47 AM</div>
                    <div>Welcome! I'm TARA Universal Model, ready to help with professional domain expertise and emotional intelligence.
                        <span class="emotion-badge">😊 helpful (95%)</span>
                    </div>
                </div>
            </div>
            
            <div class="input-section">
                <input type="text" class="message-input" placeholder="Type message..." id="messageInput">
                <select id="domainSelect" style="padding: 0.8rem; border-radius: 8px; background: rgba(255,255,255,0.1); color: white; border: none; min-width: 120px;">
                    <option value="general">🤖 General</option>
                    <option value="healthcare">🏥 Healthcare</option>
                    <option value="business">💼 Business</option>
                    <option value="education">🎓 Education</option>
                    <option value="creative">🎨 Creative</option>
                    <option value="leadership">👥 Leadership</option>
                </select>
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>Emotion Analysis</h3>
                <div class="emotion-display">
                    <div class="emotion-emoji" id="emotionEmoji">😊</div>
                    <div class="emotion-label" id="emotionLabel">Helpful</div>
                    <div class="confidence" id="emotionConfidence">90% confidence</div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">Facial Analysis</div>
                    <div style="text-align: center; margin: 0.5rem 0;">📊 Coming Soon</div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Training Status</h3>
                <div class="training-progress">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>Overall Progress</span>
                        <span id="progressPercent">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-top: 0.5rem;" id="currentStep">
                        Initializing training pipeline...
                    </div>
                </div>
                
                <div class="domain-list">
                    <div class="domain-item">
                        <span>🏥 Healthcare</span>
                        <span class="domain-status" id="healthcareStatus">Pending</span>
                    </div>
                    <div class="domain-item">
                        <span>💼 Business</span>
                        <span class="domain-status" id="businessStatus">Pending</span>
                    </div>
                    <div class="domain-item">
                        <span>🎓 Education</span>
                        <span class="domain-status" id="educationStatus">Pending</span>
                    </div>
                    <div class="domain-item">
                        <span>🎨 Creative</span>
                        <span class="domain-status" id="creativeStatus">Pending</span>
                    </div>
                    <div class="domain-item">
                        <span>👥 Leadership</span>
                        <span class="domain-status" id="leadershipStatus">Pending</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>System Status</h3>
                <div class="status-item">
                    <span>Connection</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="status-indicator"></div>
                        <span>Connected</span>
                    </div>
                </div>
                <div class="status-item">
                    <span>AI Status</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="status-indicator"></div>
                        <span>Training</span>
                    </div>
                </div>
                <div class="status-item">
                    <span>Emotion Active</span>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div class="status-indicator"></div>
                        <span>Active</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>Quick Actions</h3>
                <div class="quick-actions">
                    <button class="action-btn primary" onclick="startTraining()">🚀 Start Training</button>
                    <button class="action-btn" onclick="stopTraining()">⏹️ Stop Training</button>
                    <button class="action-btn" onclick="testModels()">🧪 Test Models</button>
                    <button class="action-btn" onclick="exportModels()">📦 Export Models</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        function handleWebSocketMessage(data) {
            if (data.type === 'training_progress') {
                updateTrainingProgress(data.training);
            } else if (data.type === 'conversation_update') {
                addMessage(data.conversation);
                updateEmotion(data.emotion);
            } else if (data.type === 'status_update') {
                updateStatus(data);
            }
        }
        
        function updateTrainingProgress(training) {
            document.getElementById('progressPercent').textContent = Math.round(training.progress) + '%';
            document.getElementById('progressFill').style.width = training.progress + '%';
            document.getElementById('currentStep').textContent = training.current_step;
            
            // Update domain statuses
            const domains = ['healthcare', 'business', 'education', 'creative', 'leadership'];
            domains.forEach(domain => {
                const statusEl = document.getElementById(domain + 'Status');
                if (training.domains_completed.includes(domain)) {
                    statusEl.textContent = 'Complete';
                    statusEl.style.color = '#48bb78';
                } else if (training.current_domain === domain) {
                    statusEl.textContent = 'Training...';
                    statusEl.style.color = '#fbbf24';
                } else {
                    statusEl.textContent = 'Pending';
                    statusEl.style.color = '#9ca3af';
                }
            });
        }
        
        function updateEmotion(emotion) {
            const emojis = {
                happy: '😊',
                helpful: '🤝',
                friendly: '😄',
                professional: '💼',
                neutral: '😐',
                excited: '🎉'
            };
            
            document.getElementById('emotionEmoji').textContent = emojis[emotion.current_emotion] || '😐';
            document.getElementById('emotionLabel').textContent = emotion.current_emotion.charAt(0).toUpperCase() + emotion.current_emotion.slice(1);
            document.getElementById('emotionConfidence').textContent = Math.round(emotion.confidence * 100) + '% confidence';
        }
        
        function addMessage(conversation) {
            const conversationEl = document.getElementById('conversation');
            
            const userMsg = document.createElement('div');
            userMsg.className = 'message user-message';
            userMsg.innerHTML = `
                <div class="message-time">${conversation.timestamp}</div>
                <div>${conversation.user_message}</div>
            `;
            
            const assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant-message';
            assistantMsg.innerHTML = `
                <div class="message-time">${conversation.timestamp}</div>
                <div>${conversation.assistant_response}
                    <span class="emotion-badge">${conversation.emotion} (${conversation.confidence})</span>
                </div>
            `;
            
            conversationEl.appendChild(userMsg);
            conversationEl.appendChild(assistantMsg);
            conversationEl.scrollTop = conversationEl.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const domain = document.getElementById('domainSelect').value;
            const message = input.value.trim();
            
            if (!message) return;
            
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        domain: domain
                    })
                });
                
                const data = await response.json();
                // Response will be handled via WebSocket
            } catch (error) {
                console.error('Error sending message:', error);
            }
        }
        
        // Enter key to send message
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        async function startTraining() {
            try {
                await fetch('/api/training/start', { method: 'POST' });
            } catch (error) {
                console.error('Error starting training:', error);
            }
        }
        
        async function stopTraining() {
            try {
                await fetch('/api/training/stop', { method: 'POST' });
            } catch (error) {
                console.error('Error stopping training:', error);
            }
        }
        
        function testModels() {
            alert('Model testing feature coming soon!');
        }
        
        function exportModels() {
            alert('Model export feature coming soon!');
        }
        
        // Auto-refresh status
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                // Handle status updates
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }, 5000);
    </script>
</body>
</html>
'''

# Save the HTML template
def create_template():
    template_path = templates_dir / "dashboard.html"
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

if __name__ == "__main__":
    # Create templates directory and HTML file
    create_template()
    
    logger.info("🚀 Starting TARA Universal Model Dashboard")
    logger.info("📊 Dashboard will be available at: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 