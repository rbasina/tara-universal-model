# TARA Universal Model - Essential Scripts

This folder contains backups of the essential training and monitoring scripts for the TARA Universal Model project.

## Training Scripts

### 1. parameterized_train_domains.py
- **Purpose**: Train specific domains with specified base models
- **Usage**: `python scripts/training/parameterized_train_domains.py --domains domain1 domain2 --model "model_name"`
- **Example**: `python scripts/training/parameterized_train_domains.py --domains education creative leadership --model "Qwen/Qwen2.5-3B-Instruct"`
- **Best for**: Domain-specific model optimization phase

### 2. train_meetara_universal_model.py
- **Purpose**: Comprehensive Trinity Architecture training across all domains
- **Usage**: `python scripts/train_meetara_universal_model.py [--phase N] [--domain domain_name] [--samples N] [--style style_name]`
- **Example**: `python scripts/train_meetara_universal_model.py`
- **Best for**: Full Trinity Architecture implementation (Phase 1-4)

### 3. train_qwen_domains.py
- **Purpose**: Train domains specifically with Qwen2.5-3B-Instruct model
- **Usage**: `python scripts/train_qwen_domains.py --domains domain1 domain2`
- **Example**: `python scripts/train_qwen_domains.py --domains education creative leadership`
- **Best for**: Reasoning-focused domains

## Monitoring Scripts

### 1. simple_web_monitor.py
- **Purpose**: Web-based dashboard for monitoring training progress
- **Usage**: `python scripts/monitoring/simple_web_monitor.py`
- **Access**: http://localhost:8001/
- **Features**: Shows training progress, domain adapters, and Python processes

## Domain Model Assignments

### Conversation-focused domains (DialoGPT-medium)
- Healthcare
- Business

### Reasoning-focused domains (Qwen2.5-3B-Instruct)
- Education
- Creative
- Leadership

## Training Process

1. Train conversation-focused domains with DialoGPT-medium
2. Train reasoning-focused domains with Qwen2.5-3B-Instruct
3. Convert trained models to GGUF format
4. Create specialized GGUF files per domain family
5. Deploy to MeeTARA

## Note
These scripts have been backed up on June 25, 2025 to preserve the essential training and monitoring functionality. 