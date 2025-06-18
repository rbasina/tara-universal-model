# TARA Universal Model - Cursor AI Setup Prompt

## Context & Background

I'm Ramesh, working on TARA AI - a privacy-first conversational AI with emotional intelligence. We've separated the codebase into two repositories:

1. **tara-ai-companion** - Main application (UI, integrations, deployment)
2. **tara-universal-model** - Model training & serving (this repository)

## Current Status

✅ **Completed:**
- Created `tara-universal-model` directory structure
- Basic package structure with `__init__.py` files
- Directory layout: `tara_universal_model/`, `models/`, `data/`, `scripts/`, `configs/`, etc.
- Some core Python modules started but need completion

❌ **Needs Completion:**
- All Python module implementations
- Training scripts and CLI
- Model serving API
- Configuration files
- Documentation
- Integration with tara-ai-companion

## Project Goals

### Cost-Effective Training Framework
- **Budget**: $750-$3,000 total (vs. $3M enterprise solutions)
- **Method**: LoRA/QLoRA fine-tuning on free base models
- **Base Models**: Llama2, Qwen, Phi-3 (all free)
- **Data**: 5,000 quality samples per domain (synthetic generation)

### Professional Domains
1. **Healthcare** - Patient communication, HIPAA compliance
2. **Business** - Strategy, leadership, professional communication  
3. **Education** - Tutoring, exam prep, learning support
4. **Creative** - Writing, brainstorming, artistic collaboration
5. **Leadership** - Executive coaching, team management

### Technical Architecture
```
TARA Universal Model
├── Base Model (Llama2/Qwen/Phi-3)
├── Emotional Intelligence Layer (Proprietary)
├── Domain Expert Routing (Proprietary) 
├── Professional Context Switching
└── Safety & Compliance Layer
```

## Required Implementation

### 1. Core Python Modules

**`tara_universal_model/serving/model.py`**
- Main `TARAUniversalModel` class
- Integration with emotion detection and domain routing
- Chat interface with professional context switching
- Local-only processing (privacy-first)

**`tara_universal_model/emotional_intelligence/detector.py`**
- Text-based emotion detection using transformers
- Professional emotion categories (stress, confidence, etc.)
- Voice emotion detection (placeholder for future)
- Emotion trajectory analysis

**`tara_universal_model/domain_experts/router.py`**
- Keyword-based domain classification
- Professional context switching
- Cross-domain insights
- Safety validation per domain

**`tara_universal_model/training/cli.py`**
- Command-line training interface
- LoRA/QLoRA configuration
- Synthetic data generation integration
- Progress tracking and logging

**`tara_universal_model/utils/config.py`**
- Configuration management for all components
- Domain-specific settings
- Training parameters
- Serving configuration

### 2. Essential Scripts

**`scripts/download_models.py`**
- Download free base models (Llama2, Qwen, Phi-3)
- Model verification and setup
- Cost estimation display

**`scripts/train_domain.py`**
- Domain-specific training workflows
- Data preparation and tokenization
- LoRA adapter training

**`scripts/serve_model.py`**
- FastAPI server for model serving
- Health checks and monitoring
- Rate limiting and safety

### 3. Data Generation

**`tara_universal_model/utils/data_generator.py`**
- Synthetic conversation generation
- Domain-specific templates
- Professional scenario simulation
- Quality validation

### 4. Configuration Files

**`configs/domains/`**
- YAML configs for each professional domain
- System prompts and expertise areas
- Safety guidelines and compliance rules

**`configs/training/`**
- LoRA/QLoRA parameters per domain
- Training hyperparameters
- Data generation settings

### 5. Integration Layer

**`tara_universal_model/integration/adapter.py`**
- Backward compatibility with tara-ai-companion
- API interface for existing codebase
- Gradual migration support

## Key Requirements

### Privacy & Security
- 100% local processing, no external API calls
- HIPAA compliance for healthcare domain
- Data encryption and secure storage

### Cost Optimization
- Use only free base models
- Efficient LoRA/QLoRA training (90% memory reduction)
- Cloud GPU rental optimization ($200-800 total)

### Professional Quality
- Domain-specific expertise and terminology
- Appropriate emotional responses
- Safety guardrails and compliance

### Easy Integration
- Simple Python package installation
- CLI tools for training and serving
- Clean API for tara-ai-companion integration

## Success Criteria

1. **Training Pipeline**: Complete domain training in <2 hours per domain
2. **Model Quality**: Professional-grade responses with emotional intelligence
3. **Cost Target**: Total investment under $3,000
4. **Integration**: Seamless connection with tara-ai-companion
5. **Privacy**: Zero external data transmission

## Next Steps

1. **Complete all Python module implementations**
2. **Create comprehensive training scripts**
3. **Build model serving API**
4. **Generate domain-specific training data**
5. **Test integration with tara-ai-companion**
6. **Document deployment and usage**

## File Structure Reference

```
tara-universal-model/
├── tara_universal_model/
│   ├── __init__.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── model.py (main TARAUniversalModel class)
│   │   └── api.py (FastAPI server)
│   ├── emotional_intelligence/
│   │   ├── __init__.py
│   │   └── detector.py (emotion detection)
│   ├── domain_experts/
│   │   ├── __init__.py
│   │   └── router.py (domain routing)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── cli.py (training CLI)
│   │   └── trainer.py (custom trainer)
│   └── utils/
│       ├── __init__.py
│       ├── config.py (configuration)
│       └── data_generator.py (synthetic data)
├── scripts/
│   ├── download_models.py
│   ├── train_domain.py
│   └── serve_model.py
├── configs/
│   ├── domains/
│   └── training/
├── models/ (downloaded base models)
├── data/ (training data)
├── README.md
├── requirements.txt
└── setup.py
```

## Request

Please help me complete this TARA Universal Model implementation with:
1. All missing Python module code
2. Training and serving scripts
3. Configuration files
4. Integration layer
5. Documentation and examples

Focus on cost-effectiveness, privacy, and professional quality. The goal is a complete, production-ready model training framework that costs under $3,000 total. 