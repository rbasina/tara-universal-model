# Root Backup Scripts

This directory contains a backup of all original Python scripts from the root scripts directory before reorganization.

## Purpose
- Preserve original scripts for reference
- Maintain a snapshot of the codebase before reorganization
- Provide fallback in case of issues with reorganized scripts

## Organization
The scripts in this directory were organized into the following categories in the main scripts directory:

1. **Conversion Scripts** (`scripts/conversion/`)
   - Scripts for GGUF model creation and conversion
   - All scripts with `create_*.py` pattern

2. **Monitoring Scripts** (`scripts/monitoring/`)
   - Scripts for monitoring training progress
   - Web interfaces and dashboards
   - Scripts: `monitor_training.py`, `simple_web_monitor.py`, `training_monitor_dashboard.py`, `watch_training.py`, `web_monitor.py`

3. **Training Scripts** (`scripts/training/`)
   - Scripts for training models across domains
   - Scripts: `train_*.py`, `test_*.py`, `demo_reinforcement_learning.py`

4. **Utility Scripts** (`scripts/utilities/`)
   - Helper scripts for various tasks
   - Scripts: `backup_training_data.py`, `download_*.py`, `fix_meetara_gguf.py`, `serve_model.py`

## Date of Backup
June 25, 2025

## Project Phase
Phase 1: Arc Reactor Foundation Training - ACTIVE

# TARA Universal Model - Scripts Analysis

This document provides a detailed analysis of the scripts in the TARA Universal Model project's root scripts directory.

## Script Categories

### Training Scripts
- **train_meetara_universal_model.py** - Main script implementing the Trinity Architecture (Phase 1-4)
- **train_all_domains.py** - Trains all domains with default models
- **train_domain.py** - Trains a single domain
- **train_qwen_domains.py** - Trains domains with Qwen2.5-3B-Instruct
- **train_qwen_simple.py** - Simplified training with Qwen model
- **auto_train_remaining_domains.py** - Automatically trains domains that haven't been trained yet
- **enhanced_training_runner.py** - Runner for enhanced training with production validation

### GGUF Creation Scripts
- **create_clean_gguf.py** - Creates clean GGUF models
- **create_combined_universal_gguf.py** - Creates combined universal GGUF
- **create_hierarchical_gguf.py** - Creates hierarchical GGUF structure
- **create_meetara_universal.py** - Creates MeeTARA universal model
- **create_meetara_universal_1_0.py** - Creates MeeTARA universal model v1.0
- **create_meetara_universal_combo.py** - Creates MeeTARA universal combo model
- **create_tara_gguf.py** - Creates TARA GGUF model
- **create_universal_embedded_gguf.py** - Creates universal embedded GGUF
- **create_working_meetara_gguf.py** - Creates working MeeTARA GGUF
- **fix_meetara_gguf.py** - Fixes issues in MeeTARA GGUF models

### Monitoring Scripts
- **monitor_training.py** - Monitors training progress
- **simple_web_monitor.py** - Simple web-based training monitor (port 8001)
- **web_monitor.py** - Web-based training monitor (port 8000)
- **watch_training.py** - Command-line training watcher
- **training_monitor_dashboard.py** - Dashboard for training monitoring

### Utility Scripts
- **backup_training_data.py** - Backs up training data
- **download_datasets.py** - Downloads required datasets
- **download_models.py** - Downloads base models
- **download_qwen_model.py** - Downloads Qwen model specifically
- **serve_model.py** - Serves a model for inference

### Testing Scripts
- **test_domain_training.py** - Tests domain training
- **test_phase2_intelligence.py** - Tests Phase 2 (Perplexity Intelligence)
- **demo_reinforcement_learning.py** - Demonstrates reinforcement learning

## Essential Scripts

The following scripts are considered essential for the core functionality of the TARA Universal Model:

1. **train_meetara_universal_model.py** - Main Trinity Architecture implementation
2. **parameterized_train_domains.py** - Flexible domain-specific training (in training subfolder)
3. **simple_web_monitor.py** - Training monitoring dashboard
4. **create_meetara_universal_1_0.py** - Creates final GGUF model

## Cleanup Recommendation

Scripts that can be moved to subdirectories:

1. Move GGUF creation scripts to `scripts/conversion/` directory
2. Move monitoring scripts to `scripts/monitoring/` directory
3. Move utility scripts to `scripts/utilities/` directory
4. Move training scripts to `scripts/training/` directory

## Usage Guide

### Training Workflow
1. Use `parameterized_train_domains.py` for domain-specific training
2. Use `train_meetara_universal_model.py` for full Trinity Architecture
3. Monitor with `simple_web_monitor.py`
4. Create GGUF models with `create_meetara_universal_1_0.py`

### Monitoring
- Web dashboard: `simple_web_monitor.py` (port 8001)
- Command line: `watch_training.py`

### Model Creation
- Universal model: `create_meetara_universal_1_0.py`
- Domain-specific: `create_hierarchical_gguf.py` 