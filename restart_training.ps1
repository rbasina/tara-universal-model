# TARA Universal Model - Training Restart Script
# This script will restart training for all domains with fixed configuration

Write-Host "TARA Universal Model - Training Restart"
Write-Host "========================================"
Write-Host "This script will restart training for all domains with the fixed configuration"

# Configuration
$domains = "education,creative,leadership"
$model = "Qwen/Qwen2.5-3B-Instruct"

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory | Out-Null
}

# Create log file
$logFile = "logs/restart_training.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"$timestamp - Starting training restart" | Out-File -FilePath $logFile

# Check for existing checkpoints
Write-Host "Checking for existing checkpoints..."
$educationCheckpoints = Get-ChildItem -Path "models/adapters/education" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue
$creativeCheckpoints = Get-ChildItem -Path "models/adapters/creative" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue
$leadershipCheckpoints = Get-ChildItem -Path "models/adapters/leadership" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue

if ($educationCheckpoints -or $creativeCheckpoints -or $leadershipCheckpoints) {
    Write-Host "Found existing checkpoints:"
    if ($educationCheckpoints) { Write-Host "- Education: $($educationCheckpoints.Count) checkpoints" }
    if ($creativeCheckpoints) { Write-Host "- Creative: $($creativeCheckpoints.Count) checkpoints" }
    if ($leadershipCheckpoints) { Write-Host "- Leadership: $($leadershipCheckpoints.Count) checkpoints" }
    
    Write-Host "Will resume from latest checkpoints"
    "$timestamp - Will resume from existing checkpoints" | Out-File -FilePath $logFile -Append
    
    # Start training with auto checkpoint detection
    Write-Host "Starting training with auto checkpoint detection..."
    "$timestamp - Starting training with command: python scripts/training/parameterized_train_domains.py --domains $domains --model $model --resume_from_checkpoint auto" | Out-File -FilePath $logFile -Append
    
    # Start the process
    Start-Process -FilePath "python" -ArgumentList "scripts/training/parameterized_train_domains.py --domains $domains --model $model --resume_from_checkpoint auto" -NoNewWindow
} else {
    Write-Host "No existing checkpoints found, starting fresh training"
    "$timestamp - No existing checkpoints found, starting fresh training" | Out-File -FilePath $logFile -Append
    
    # Start fresh training
    Write-Host "Starting fresh training..."
    "$timestamp - Starting training with command: python scripts/training/parameterized_train_domains.py --domains $domains --model $model" | Out-File -FilePath $logFile -Append
    
    # Start the process
    Start-Process -FilePath "python" -ArgumentList "scripts/training/parameterized_train_domains.py --domains $domains --model $model" -NoNewWindow
}

Write-Host "Training restart initiated. Check logs/domain_training.log for progress."
"$timestamp - Training restart initiated" | Out-File -FilePath $logFile -Append 