# Technical Context - TARA Universal Model

**📅 Created**: June 22, 2025  
**🔄 Last Updated**: June 25, 2025  
**🎯 Tech Status**: Repository Restructuring - Backend Focus Optimization

## Technologies Used

### **Core AI/ML Stack**
- **PyTorch 2.7.1+cpu** - Primary deep learning framework
- **Transformers 4.52.4** - Hugging Face transformers for model architecture
- **PEFT 0.15.2** - Parameter-Efficient Fine-Tuning for domain adaptation
- **Microsoft Models**: DialoGPT-medium (primary base model), Phi-3.5-mini-instruct (secondary base)

### **GGUF Optimization Stack**
- **llama.cpp** - GGUF model conversion and quantization
- **Q4_K_M Quantization** - Primary compression method (4-bit keys, mixed precision)
- **ctransformers** - Alternative inference library for GGUF models
- **gguf-py** - Python utilities for GGUF file manipulation

### **Voice Integration Stack**
- **SpeechBrain** - Speech recognition, emotion detection, speaker recognition
- **Edge TTS** - Primary text-to-speech engine
- **pyttsx3** - Fallback TTS system
- **Audio Processing** - Real-time voice analysis

### **Security & Privacy Stack**
- **Fernet Encryption** - Local data encryption
- **AES-256** - Industry-standard encryption protocols
- **Local Storage** - No cloud dependencies for sensitive data
- **GDPR/HIPAA Compliance** - Privacy regulation adherence

## Development Setup

### **System Requirements**
- **OS**: Windows 10.0.26100 (Primary development environment)
- **RAM**: Minimum 8GB, Recommended 16GB+ for model training
- **Storage**: SSD recommended for model loading speed
- **CPU**: Multi-core processor for parallel domain training

### **Development Environment**
```bash
# Current Working Directory
C:\Users\rames\Documents\tara-universal-model

# Virtual Environment (Conda/Miniconda)
conda activate base  # Primary environment

# Key Directories
├── configs/           # Configuration files
├── data/              # Training and processed data
├── models/            # Model checkpoints and adapters
├── scripts/           # Training and utility scripts
├── tara_universal_model/  # Main package
└── docs/              # Documentation (lifecycle + memory-bank)
```

### **Configuration Files**
- **config.yaml** - Main system configuration
- **universal_domains.yaml** - Domain-specific training configurations
- **model_mapping.json** - Model routing and selection
- **requirements.txt** - Python dependencies

## GGUF Optimization Focus

### **Current GGUF Models**
- **meetara-universal-model-1.0.gguf** (4.6GB) - Primary production model
- **Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf** (4.6GB) - Business/Analysis base
- **Phi-3.5-mini-instruct-Q4_K_M.gguf** (2.2GB) - Programming/Technical base
- **qwen2.5-3b-instruct-q4_0.gguf** (1.9GB) - Creative/Multilingual base
- **llama-3.2-1b-instruct-q4_0.gguf** (0.7GB) - Quick/Mobile base

### **Quantization Methods**
| Method | Precision | File Size | Quality | Use Case |
|--------|-----------|-----------|---------|----------|
| Q4_K_M | 4-bit | Smallest | Good | Production default |
| Q5_K_M | 5-bit | Medium | Better | Quality-critical domains |
| Q2_K | 2-bit | Smallest | Reduced | Mobile/Edge deployment |
| Q8_0 | 8-bit | Largest | Best | Development/Testing |

### **GGUF Optimization Strategy**
- **Baseline**: DialoGPT-medium (345M parameters) with LoRA adapters
- **Compression**: 98.3% size reduction from potential 40GB+ approach
- **Domain Integration**: 5 specialized domains in single GGUF file
- **Research Focus**: Further optimization without quality degradation

## Essential Scripts

### **Training Scripts**
- **train_domain.py** - Single domain training with DialoGPT-medium
- **train_all_domains.py** - Parallel training of multiple domains
- **train_meetara_universal_model.py** - Complete training orchestration

### **GGUF Conversion Scripts**
- **create_clean_gguf.py** - Basic GGUF creation with corruption prevention
- **fix_meetara_gguf.py** - Repair corrupted GGUF files

### **Monitoring Scripts**
- **monitor_training.py** - Training progress tracking
- **simple_web_monitor.py** - Web-based monitoring dashboard
- **training_recovery.py** - Automatic training recovery system
- **monitor_and_resume_training.ps1** - PowerShell script for system-level monitoring

### **Recovery System**
- **training_recovery.py** - Monitors and resumes interrupted training
- **training_recovery_state.json** - Saves training state for recovery
- **resume_training.bat** - Quick recovery script for manual execution
- **domain_optimization_dashboard.html** - Web dashboard with recovery button

### **Utility Scripts**
- **download_models.py** - Base model acquisition
- **download_datasets.py** - Training data preparation
- **serve_model.py** - Local model serving for testing

## Technical Constraints

### **1. Local Processing Constraint**
**Requirement**: All AI inference must happen locally
**Impact**: 
- Model size limitations based on local hardware
- No cloud-based model APIs allowed for sensitive domains
- Requires efficient model optimization and quantization

### **2. Memory Management Constraint**
**Requirement**: Support long conversations without memory exhaustion
**Impact**:
- Efficient context window management
- Conversation history compression
- Dynamic model loading/unloading

### **3. Storage Optimization Constraint**
**Requirement**: Minimize repository and deployment size
**Impact**:
- GGUF quantization optimization
- Removal of redundant files and directories
- Efficient organization of training artifacts

### **4. Privacy Compliance Constraint**
**Requirement**: GDPR, HIPAA, CCPA compliance
**Impact**:
- Data encryption requirements
- User consent management
- Audit trail maintenance
- Right to deletion implementation

## Technical Architecture Decisions

### **1. Base Model Decision**
**Choice**: DialoGPT-medium (345M parameters)
**Rationale**:
- Excellent conversation capabilities
- Efficient fine-tuning with LoRA
- Small enough for deployment on consumer hardware
- Strong performance with domain-specific training

### **2. GGUF Format Decision**
**Choice**: Q4_K_M quantization as default
**Rationale**:
- 4-bit precision offers good balance of size and quality
- Mixed precision maintains performance on critical operations
- Widely supported by inference engines
- Proven stability in production environments

### **3. Training Strategy Decision**
**Choice**: LoRA adapters (15.32% trainable parameters)
**Rationale**:
- Parameter-efficient fine-tuning
- Domain specialization without full model retraining
- Faster training cycles
- Smaller adapter files

### **4. Repository Focus Decision**
**Choice**: Backend-only repository
**Rationale**:
- Clear separation of concerns
- Focus on model training and optimization
- Simplified integration with MeeTARA frontend
- Better organization of technical components

## Performance Targets

### **GGUF Optimization Targets**
- **File Size**: Further 10-15% reduction without quality loss
- **Loading Speed**: <5 seconds on target hardware
- **Inference Speed**: <1 second response time
- **Memory Usage**: <2GB RAM during operation

### **Training Efficiency Targets**
- **Domain Training**: <2 hours per domain
- **Sample Efficiency**: 2000 samples per domain
- **Quality Threshold**: 97%+ improvement over base model
- **Validation Success**: 100% clean responses

## Development Workflow

### **Repository Organization**
```
tara-universal-model/
├── configs/              # Configuration files
├── data/                 # Training data
│   ├── raw/              # Raw training data
│   └── processed/        # Processed training data
├── models/               # Model files
│   ├── adapters/         # LoRA adapters
│   ├── gguf/             # GGUF models
│   └── [domain]/         # Domain-specific models
├── scripts/              # Training and utility scripts
│   ├── train_*.py        # Training scripts
│   ├── create_*.py       # GGUF creation scripts
│   └── utility scripts   # Monitoring, download, etc.
├── tara_universal_model/ # Core package
└── docs/                 # Documentation
    ├── memory-bank/      # Session continuity
    └── [lifecycle]/      # Project documentation
```

### **Script Organization**
- **Training**: Domain-specific and universal training
- **Conversion**: GGUF creation and optimization
- **Monitoring**: Training progress and validation
- **Utilities**: Data preparation and model management

### **Documentation Standards**
- **Memory Bank**: Session continuity for Cursor AI
- **Script Documentation**: Purpose and usage of each script
- **Model Documentation**: Capabilities and integration guide
- **Architecture Documentation**: System design and decisions

---

**💻 Technical Status**: Dependencies validated, training infrastructure operational  
**🔧 Development Environment**: Windows PowerShell + Conda + VS Code + Cursor AI  
**🎯 Architecture Focus**: Local-first, privacy-preserving, multi-domain coordination  
**⚡ Performance Target**: 504% intelligence amplification through Trinity Architecture 