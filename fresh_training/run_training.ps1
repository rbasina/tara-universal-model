# TARA Universal Model - Fresh Training Script
# Runs training from scratch to avoid optimizer mismatch issues

Write-Host "TARA Universal Model - Fresh Training"
Write-Host "====================================="

# Create directories
if (-not (Test-Path "fresh_training/models")) {
    New-Item -Path "fresh_training/models" -ItemType Directory -Force | Out-Null
}

if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory -Force | Out-Null
}

# Kill any existing training processes
$existingProcesses = Get-Process | Where-Object { $_.ProcessName -like "*python*" } | 
                    Where-Object { $_.CommandLine -like "*train_*domains.py*" -or $_.CommandLine -like "*parameterized_train_domains.py*" }

if ($existingProcesses) {
    Write-Host "Stopping existing training processes..."
    $existingProcesses | ForEach-Object { 
        try {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        } catch {
            # Ignore errors
        }
    }
    Start-Sleep -Seconds 2
}

# Configuration
$domains = "education,creative,leadership"
$model = "Qwen/Qwen2.5-3B-Instruct"
$epochs = 3
$batchSize = 2

# Log start
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "$timestamp - Starting fresh training for domains: $domains"
Write-Host "Using model: $model"
Write-Host "Epochs: $epochs, Batch Size: $batchSize"

# Start training
Write-Host "Starting training..."
$command = "python fresh_training/train_fresh.py --domains $domains --model $model --epochs $epochs --batch-size $batchSize"
Write-Host "Command: $command"

# Execute
Invoke-Expression $command

Write-Host "Training complete. Check logs/fresh_training.log for details." 