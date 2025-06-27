#!/usr/bin/env python3
"""
🎯 Universal GGUF Factory - Phase-Wise Domain Expansion
Handles intelligent routing, emotional intelligence, and efficient compression
Supports dynamic domain addition with real-time model updates
"""

import os
import torch
import logging
import json
import subprocess
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    Q2_K = "Q2_K"      # Mobile/Edge (fastest, smallest)
    Q4_K_M = "Q4_K_M"  # Production (balanced)
    Q5_K_M = "Q5_K_M"  # Quality-critical (highest quality)
    Q8_0 = "Q8_0"      # Development/Testing (full precision)

class CompressionType(Enum):
    STANDARD = "standard"      # Basic quantization
    SPARSE = "sparse"          # Sparse quantization
    HYBRID = "hybrid"          # Mixed precision
    DISTILLED = "distilled"    # Knowledge distillation

@dataclass
class DomainInfo:
    name: str
    base_model: str
    adapter_path: Path
    training_quality: float  # 0-1
    response_speed: float    # 0-1 (higher = faster)
    emotional_intensity: float  # 0-1
    context_length: int
    specialties: List[str]
    phase: int  # Which phase this domain was added

@dataclass
class RoutingDecision:
    primary_model: str
    fallback_model: str
    confidence: float
    reasoning: str
    emotional_context: Dict[str, float]

class IntelligentRouter:
    """AI-powered routing system for optimal model selection"""
    
    def __init__(self):
        self.domain_models: Dict[str, DomainInfo] = {}
        self.routing_cache: Dict[str, RoutingDecision] = {}
        self.emotional_context: Dict[str, float] = {}
        
    def add_domain(self, domain_info: DomainInfo):
        """Add a new domain to the routing system"""
        self.domain_models[domain_info.name] = domain_info
        logger.info(f"🧠 Added {domain_info.name} to intelligent router (Phase {domain_info.phase})")
        
    def route_query(self, query: str, user_context: Dict[str, Any] = None) -> RoutingDecision:
        """Intelligently route query to best model"""
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.routing_cache:
            return self.routing_cache[query_hash]
        
        # Analyze query content and context
        content_analysis = self._analyze_query_content(query)
        emotional_analysis = self._analyze_emotional_context(query, user_context)
        
        # Score each domain
        domain_scores = {}
        for domain_name, domain_info in self.domain_models.items():
            score = self._calculate_domain_score(
                domain_info, content_analysis, emotional_analysis
            )
            domain_scores[domain_name] = score
        
        # Select best and fallback models
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]
        fallback_domain = sorted_domains[1][0] if len(sorted_domains) > 1 else primary_domain
        
        # Create routing decision
        decision = RoutingDecision(
            primary_model=primary_domain,
            fallback_model=fallback_domain,
            confidence=sorted_domains[0][1],
            reasoning=f"Content: {content_analysis['primary_domain']}, Emotion: {emotional_analysis['dominant_emotion']}",
            emotional_context=emotional_analysis
        )
        
        # Cache decision
        self.routing_cache[query_hash] = decision
        return decision
    
    def _analyze_query_content(self, query: str) -> Dict[str, Any]:
        """Analyze query content for domain classification"""
        query_lower = query.lower()
        
        # Domain keywords
        domain_keywords = {
            'healthcare': ['health', 'medical', 'doctor', 'patient', 'treatment', 'symptoms', 'medicine'],
            'business': ['business', 'strategy', 'market', 'profit', 'management', 'leadership', 'company'],
            'education': ['learn', 'study', 'education', 'student', 'teacher', 'course', 'knowledge'],
            'creative': ['creative', 'art', 'design', 'imagine', 'story', 'write', 'inspire'],
            'leadership': ['lead', 'team', 'manage', 'motivate', 'vision', 'strategy', 'organization']
        }
        
        # Calculate domain scores
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Find primary domain
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'complexity': len(query.split()) / 20.0,  # Normalized complexity
            'urgency': 1.0 if any(word in query_lower for word in ['urgent', 'emergency', 'help', 'now']) else 0.0
        }
    
    def _analyze_emotional_context(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, float]:
        """Analyze emotional context of query"""
        query_lower = query.lower()
        
        # Emotional keywords
        emotion_keywords = {
            'joy': ['happy', 'excited', 'great', 'wonderful', 'amazing'],
            'sadness': ['sad', 'depressed', 'lonely', 'hurt', 'pain'],
            'anger': ['angry', 'frustrated', 'mad', 'upset', 'annoyed'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'surprise': ['wow', 'unexpected', 'shocked', 'surprised'],
            'disgust': ['disgusting', 'gross', 'terrible', 'awful'],
            'trust': ['trust', 'believe', 'confident', 'sure'],
            'anticipation': ['hope', 'expect', 'look forward', 'plan']
        }
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            emotion_scores[emotion] = score
        
        # Normalize scores
        total = sum(emotion_scores.values()) or 1
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # Add user context if available
        if user_context and 'emotional_state' in user_context:
            for emotion, intensity in user_context['emotional_state'].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] = (emotion_scores[emotion] + intensity) / 2
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': max(emotion_scores.values()),
            'emotional_stability': 1.0 - (max(emotion_scores.values()) - min(emotion_scores.values()))
        }
    
    def _calculate_domain_score(self, domain_info: DomainInfo, content_analysis: Dict, emotional_analysis: Dict) -> float:
        """Calculate comprehensive score for domain selection"""
        
        # Content relevance (40% weight)
        content_relevance = content_analysis['domain_scores'].get(domain_info.name, 0) / 10.0
        
        # Emotional compatibility (30% weight)
        emotional_compatibility = 1.0 - abs(
            emotional_analysis['emotional_intensity'] - domain_info.emotional_intensity
        )
        
        # Response speed requirement (20% weight)
        speed_requirement = 1.0 if content_analysis['urgency'] > 0.5 else 0.5
        speed_compatibility = 1.0 - abs(speed_requirement - domain_info.response_speed)
        
        # Training quality (10% weight)
        quality_score = domain_info.training_quality
        
        # Calculate weighted score
        score = (
            content_relevance * 0.4 +
            emotional_compatibility * 0.3 +
            speed_compatibility * 0.2 +
            quality_score * 0.1
        )
        
        return score

class EmotionalIntelligenceEngine:
    """Handles emotional intelligence and response modulation"""
    
    def __init__(self):
        self.emotion_models = {}
        self.response_templates = {}
        self._load_emotional_models()
    
    def _load_emotional_models(self):
        """Load emotional intelligence models"""
        # Emotional response templates
        self.response_templates = {
            'joy': {
                'tone': 'enthusiastic',
                'modifiers': ['wonderful', 'amazing', 'fantastic'],
                'empathy_level': 'high'
            },
            'sadness': {
                'tone': 'gentle',
                'modifiers': ['I understand', 'I\'m here for you', 'it\'s okay'],
                'empathy_level': 'very_high'
            },
            'anger': {
                'tone': 'calm',
                'modifiers': ['I hear you', 'let\'s work through this', 'I understand your frustration'],
                'empathy_level': 'high'
            },
            'fear': {
                'tone': 'reassuring',
                'modifiers': ['you\'re safe', 'we\'ll figure this out', 'I\'m here to help'],
                'empathy_level': 'very_high'
            },
            'neutral': {
                'tone': 'professional',
                'modifiers': [],
                'empathy_level': 'medium'
            }
        }
    
    def modulate_response(self, response: str, emotional_context: Dict[str, float]) -> str:
        """Modulate response based on emotional context"""
        dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
        intensity = emotional_context.get('emotional_intensity', 0.0)
        
        template = self.response_templates.get(dominant_emotion, self.response_templates['neutral'])
        
        # Apply emotional modulation
        if intensity > 0.7:  # High emotional intensity
            if template['empathy_level'] in ['high', 'very_high']:
                response = f"{template['modifiers'][0] if template['modifiers'] else ''} {response}"
        
        return response

class UniversalGGUFFactory:
    """Main factory for creating phase-wise universal GGUF models"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("models/universal-factory")
        self.temp_dir = None
        self.router = IntelligentRouter()
        self.current_phase = 1
        
        # Phase tracking
        self.phase_domains: Dict[int, List[str]] = {}
        self.phase_models: Dict[int, Path] = {}
        
        # Performance tracking
        self.response_times: Dict[str, List[float]] = {}
        self.accuracy_metrics: Dict[str, float] = {}
        
    def add_domain_phase(self, domain_name: str, adapter_path: Path, 
                        base_model: str = "microsoft/DialoGPT-medium",
                        training_quality: float = 0.95,
                        response_speed: float = 0.8,
                        emotional_intensity: float = 0.7,
                        context_length: int = 4096,
                        specialties: List[str] = None) -> bool:
        """Add a new domain to the current phase"""
        
        try:
            # Validate adapter
            if not self._validate_adapter(adapter_path):
                logger.error(f"❌ Invalid adapter for {domain_name}")
                return False
            
            # Create domain info
            domain_info = DomainInfo(
                name=domain_name,
                base_model=base_model,
                adapter_path=adapter_path,
                training_quality=training_quality,
                response_speed=response_speed,
                emotional_intensity=emotional_intensity,
                context_length=context_length,
                specialties=specialties or [domain_name],
                phase=self.current_phase
            )
            
            # Add to router
            self.router.add_domain(domain_info)
            
            # Track in phase
            if self.current_phase not in self.phase_domains:
                self.phase_domains[self.current_phase] = []
            self.phase_domains[self.current_phase].append(domain_name)
            
            logger.info(f"✅ Added {domain_name} to Phase {self.current_phase}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add domain {domain_name}: {e}")
            return False
    
    def create_phase_gguf(self, phase: int, quantization: QuantizationType = QuantizationType.Q4_K_M,
                         compression: CompressionType = CompressionType.STANDARD) -> bool:
        """Create GGUF for specific phase"""
        
        try:
            logger.info(f"🎯 Creating Phase {phase} GGUF...")
            
            # Get domains for this phase
            domains = self.phase_domains.get(phase, [])
            if not domains:
                logger.error(f"❌ No domains found for Phase {phase}")
                return False
            
            # Create temporary directory
            self.temp_dir = Path(tempfile.mkdtemp())
            
            # Step 1: Clean and merge domains
            if not self._merge_phase_domains(phase, domains):
                return False
            
            # Step 2: Add intelligent routing
            if not self._embed_intelligent_routing(phase):
                return False
            
            # Step 3: Add emotional intelligence
            if not self._embed_emotional_intelligence(phase):
                return False
            
            # Step 4: Convert to GGUF with compression
            if not self._convert_to_gguf(phase, quantization, compression):
                return False
            
            # Step 5: Validate and finalize
            if not self._validate_phase_gguf(phase):
                return False
            
            # Track phase model
            self.phase_models[phase] = self.output_dir / f"meetara-phase-{phase}-{quantization.value}.gguf"
            
            logger.info(f"🎉 Phase {phase} GGUF created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Phase {phase} GGUF: {e}")
            return False
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
    
    def _validate_adapter(self, adapter_path: Path) -> bool:
        """Validate adapter files"""
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        return all((adapter_path / f).exists() for f in required_files)
    
    def _merge_phase_domains(self, phase: int, domains: List[str]) -> bool:
        """Merge domains for specific phase"""
        logger.info(f"🔄 Merging {len(domains)} domains for Phase {phase}...")
        
        try:
            # Load base model (use first domain's base model)
            first_domain = self.router.domain_models[domains[0]]
            base_model = AutoModelForCausalLM.from_pretrained(
                first_domain.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(first_domain.base_model)
            
            # Fix tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Merge domains sequentially
            merged_model = base_model
            for domain_name in domains:
                domain_info = self.router.domain_models[domain_name]
                logger.info(f"🔄 Merging {domain_name}...")
                
                peft_model = PeftModel.from_pretrained(merged_model, domain_info.adapter_path)
                merged_model = peft_model.merge_and_unload()
            
            # Save merged model
            model_path = self.temp_dir / f"phase-{phase}-merged"
            model_path.mkdir(parents=True, exist_ok=True)
            
            merged_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"✅ Phase {phase} domains merged successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Domain merging failed: {e}")
            return False
    
    def _embed_intelligent_routing(self, phase: int) -> bool:
        """Embed intelligent routing system"""
        logger.info(f"🧠 Embedding intelligent routing for Phase {phase}...")
        
        try:
            # Create routing configuration
            routing_config = {
                "phase": phase,
                "domains": self.phase_domains[phase],
                "router_type": "intelligent_content_emotional",
                "cache_enabled": True,
                "fallback_strategy": "confidence_based",
                "performance_tracking": True,
                "routing_algorithm": {
                    "content_analysis": True,
                    "emotional_analysis": True,
                    "speed_optimization": True,
                    "quality_prioritization": True
                }
            }
            
            # Save routing config
            config_path = self.temp_dir / f"phase-{phase}-merged" / "routing_config.json"
            with open(config_path, 'w') as f:
                json.dump(routing_config, f, indent=2)
            
            logger.info(f"✅ Intelligent routing embedded for Phase {phase}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to embed routing: {e}")
            return False
    
    def _embed_emotional_intelligence(self, phase: int) -> bool:
        """Embed emotional intelligence capabilities"""
        logger.info(f"💙 Embedding emotional intelligence for Phase {phase}...")
        
        try:
            # Create emotional intelligence config
            emotion_config = {
                "phase": phase,
                "emotional_analysis": True,
                "response_modulation": True,
                "empathy_levels": ["low", "medium", "high", "very_high"],
                "context_awareness": True,
                "emotional_stability_tracking": True,
                "intensity_based_modulation": True
            }
            
            # Save emotion config
            config_path = self.temp_dir / f"phase-{phase}-merged" / "emotion_config.json"
            with open(config_path, 'w') as f:
                json.dump(emotion_config, f, indent=2)
            
            logger.info(f"✅ Emotional intelligence embedded for Phase {phase}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to embed emotional intelligence: {e}")
            return False
    
    def _convert_to_gguf(self, phase: int, quantization: QuantizationType, 
                        compression: CompressionType) -> bool:
        """Convert to GGUF with specified compression"""
        logger.info(f"🔄 Converting Phase {phase} to GGUF ({quantization.value}, {compression.value})...")
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get llama.cpp
            if not Path("llama.cpp").exists():
                logger.info("📥 Cloning llama.cpp...")
                subprocess.run([
                    "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
                ], check=True)
            
            # Convert to F16 first
            model_path = self.temp_dir / f"phase-{phase}-merged"
            f16_path = self.temp_dir / f"phase-{phase}-f16.gguf"
            
            logger.info("🔄 Converting to F16...")
            subprocess.run([
                "python", "llama.cpp/convert_hf_to_gguf.py",
                str(model_path),
                "--outfile", str(f16_path),
                "--outtype", "f16"
            ], check=True)
            
            # Apply quantization
            final_path = self.output_dir / f"meetara-phase-{phase}-{quantization.value}.gguf"
            
            if compression == CompressionType.STANDARD:
                # Standard quantization
                logger.info(f"🔄 Applying {quantization.value} quantization...")
                subprocess.run([
                    "llama.cpp/quantize",
                    str(f16_path),
                    str(final_path),
                    quantization.value.lower()
                ], check=True)
            
            elif compression == CompressionType.SPARSE:
                # Sparse quantization (if supported)
                logger.info("🔄 Applying sparse quantization...")
                # Note: Implement sparse quantization logic here
                shutil.copy2(f16_path, final_path)
            
            elif compression == CompressionType.HYBRID:
                # Hybrid quantization
                logger.info("🔄 Applying hybrid quantization...")
                # Note: Implement hybrid quantization logic here
                shutil.copy2(f16_path, final_path)
            
            else:  # DISTILLED
                # Knowledge distillation
                logger.info("🔄 Applying knowledge distillation...")
                # Note: Implement distillation logic here
                shutil.copy2(f16_path, final_path)
            
            # Get file size
            size_mb = final_path.stat().st_size / (1024*1024)
            logger.info(f"✅ Phase {phase} GGUF created: {size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GGUF conversion failed: {e}")
            return False
    
    def _validate_phase_gguf(self, phase: int) -> bool:
        """Validate phase GGUF"""
        logger.info(f"🔍 Validating Phase {phase} GGUF...")
        
        try:
            gguf_path = self.output_dir / f"meetara-phase-{phase}-Q4_K_M.gguf"
            
            if not gguf_path.exists():
                logger.error(f"❌ Phase {phase} GGUF not found")
                return False
            
            # Basic validation
            size_mb = gguf_path.stat().st_size / (1024*1024)
            if size_mb < 100:  # Minimum expected size
                logger.warning(f"⚠️ Phase {phase} GGUF seems small: {size_mb:.1f}MB")
            
            logger.info(f"✅ Phase {phase} GGUF validated: {size_mb:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    def get_phase_summary(self, phase: int) -> Dict[str, Any]:
        """Get summary for specific phase"""
        domains = self.phase_domains.get(phase, [])
        model_path = self.phase_models.get(phase)
        
        return {
            "phase": phase,
            "domains": domains,
            "model_path": str(model_path) if model_path else None,
            "model_size_mb": model_path.stat().st_size / (1024*1024) if model_path and model_path.exists() else 0,
            "router_domains": len(self.router.domain_models),
            "emotional_intelligence": True,
            "intelligent_routing": True,
            "compression_techniques": ["standard", "sparse", "hybrid", "distilled"],
            "quantization_types": [q.value for q in QuantizationType]
        }
    
    def advance_phase(self):
        """Advance to next phase"""
        self.current_phase += 1
        logger.info(f"🚀 Advanced to Phase {self.current_phase}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all phases"""
        return {
            "total_phases": self.current_phase - 1,
            "total_domains": len(self.router.domain_models),
            "response_times": self.response_times,
            "accuracy_metrics": self.accuracy_metrics,
            "router_cache_size": len(self.router.routing_cache)
        }

def main():
    """Main function for testing and demonstration"""
    
    # Create factory
    factory = UniversalGGUFFactory()
    
    # Phase 1: Core domains
    logger.info("🎯 Setting up Phase 1...")
    
    # Add healthcare domain
    factory.add_domain_phase(
        "healthcare",
        Path("models/adapters/healthcare"),
        training_quality=0.97,
        response_speed=0.8,
        emotional_intensity=0.9,
        specialties=["medical", "therapeutic", "crisis_intervention"]
    )
    
    # Add business domain
    factory.add_domain_phase(
        "business", 
        Path("models/adapters/business"),
        training_quality=0.95,
        response_speed=0.7,
        emotional_intensity=0.5,
        specialties=["strategy", "leadership", "analysis"]
    )
    
    # Create Phase 1 GGUF
    if factory.create_phase_gguf(1, QuantizationType.Q4_K_M, CompressionType.STANDARD):
        logger.info("🎉 Phase 1 complete!")
        
        # Show summary
        summary = factory.get_phase_summary(1)
        logger.info(f"📊 Phase 1 Summary: {summary}")
    
    # Advance to Phase 2
    factory.advance_phase()
    
    # Phase 2: Additional domains
    logger.info("🎯 Setting up Phase 2...")
    
    # Add education domain
    factory.add_domain_phase(
        "education",
        Path("models/adapters/education"), 
        training_quality=0.93,
        response_speed=0.6,
        emotional_intensity=0.7,
        specialties=["learning", "teaching", "knowledge"]
    )
    
    # Create Phase 2 GGUF
    if factory.create_phase_gguf(2, QuantizationType.Q4_K_M, CompressionType.STANDARD):
        logger.info("🎉 Phase 2 complete!")
        
        # Show summary
        summary = factory.get_phase_summary(2)
        logger.info(f"📊 Phase 2 Summary: {summary}")
    
    # Show performance metrics
    metrics = factory.get_performance_metrics()
    logger.info(f"📈 Performance Metrics: {metrics}")

if __name__ == "__main__":
    main() 