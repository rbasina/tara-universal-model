# TARA Universal Model - Training Recovery System

This directory contains scripts for monitoring and recovering training sessions for the TARA Universal Model.

## Overview

The training recovery system provides automatic detection and resumption of interrupted training sessions. It works by:

1. Monitoring active training processes
2. Saving checkpoints and progress information
3. Detecting interruptions (system sleep, crashes, etc.)
4. Automatically resuming training from the last checkpoint

## Key Components

- **training_recovery.py**: Core recovery system that monitors and resumes training
- **monitor_training.py**: Simple script for monitoring training progress
- **test_recovery.py**: Test script for the recovery system

## Usage

### Basic Training Recovery

To start training with recovery support:

```bash
python scripts/monitoring/training_recovery.py --domains education,creative,leadership --model Qwen2.5-3B-Instruct
```

### Auto-Resume After Interruption

If training is interrupted, you can resume from the last checkpoint:

```bash
python scripts/monitoring/training_recovery.py --auto_resume
```

### Static Dashboard

The project uses a static HTML dashboard that can be opened directly:

```
.\open_dashboard.ps1
```

This will open the dashboard in your default browser. No server is required.

### Recovery from PowerShell

You can also use the PowerShell script to monitor and resume training:

```
.\monitor_and_resume_training.ps1
```

## Command Line Options

```
--domains          Comma-separated list of domains to train
--model           Model to use for training
--check_interval  Interval between checks in seconds (default: 300)
--max_runtime     Maximum runtime in hours before saving state (default: 6)
--auto_resume     Automatically resume training if interrupted
```

## Recovery State

The system maintains a recovery state file (`training_recovery_state.json`) that contains:

- Domain information
- Model being used
- Training start time
- Last checkpoint location
- Last check timestamp

## Testing

To test the recovery system:

```bash
python scripts/monitoring/test_recovery.py
```

This will create a mock recovery state and test the recovery functionality. 