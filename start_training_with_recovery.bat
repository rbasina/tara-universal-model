@echo off
echo TARA Universal Model - Training with Recovery System
echo This script will start training with automatic recovery capabilities

REM Set domains and model
set DOMAINS=education,creative,leadership
set MODEL=Qwen/Qwen2.5-3B-Instruct
set MAX_RUNTIME=6

echo Starting training for domains: %DOMAINS%
echo Using model: %MODEL%
echo Maximum runtime per session: %MAX_RUNTIME% hours

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Start the training recovery system
python scripts/monitoring/training_recovery.py --domains %DOMAINS% --model %MODEL% --max_runtime %MAX_RUNTIME%

echo Training session completed or interrupted
echo To resume training, run: resume_training.bat 