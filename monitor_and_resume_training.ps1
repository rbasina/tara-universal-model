# TARA Universal Model - Training Monitor and Resume Script
# This script monitors training progress and automatically resumes it after system restarts

# Configuration
$domains = "education,creative,leadership"
$model = "Qwen/Qwen2.5-3B-Instruct"
$maxRuntimeHours = 6
$checkInterval = 300  # 5 minutes
$logFile = "logs/training_monitor.log"
$trainingStateFile = "training_recovery_state.json"

# Ensure log directory exists
if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory | Out-Null
}

function Write-Log {
    param (
        [string]$Message
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $logFile -Append
    Write-Host "$timestamp - $Message"
}

function Is-TrainingRunning {
    # Check if training process is running by looking for python processes
    $pythonProcesses = Get-Process -Name python -ErrorAction SilentlyContinue
    
    if ($pythonProcesses) {
        # Check if any of them are training processes
        foreach ($process in $pythonProcesses) {
            $processInfo = Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)" | Select-Object CommandLine
            if ($processInfo -and $processInfo.CommandLine -like "*train*") {
                return $true
            }
        }
    }
    
    # Check training state file as backup method
    if (Test-Path $trainingStateFile) {
        $stateFileTime = (Get-Item $trainingStateFile).LastWriteTime
        $timeSinceUpdate = (Get-Date) - $stateFileTime
        
        # If state file was updated in the last 10 minutes, consider training active
        if ($timeSinceUpdate.TotalMinutes -lt 10) {
            return $true
        }
    }
    
    return $false
}

function Start-TrainingRecovery {
    param (
        [switch]$Resume
    )
    
    if ($Resume) {
        Write-Log "Resuming training from previous checkpoint"
        Start-Process -FilePath "python" -ArgumentList "scripts/monitoring/training_recovery.py --auto_resume" -NoNewWindow
    }
    else {
        Write-Log "Starting new training session with domains: $domains, model: $model"
        Start-Process -FilePath "python" -ArgumentList "scripts/monitoring/training_recovery.py --domains $domains --model $model --max_runtime $maxRuntimeHours" -NoNewWindow
    }
    
    # Wait for training to start
    $attempts = 0
    while (-not (Is-TrainingRunning) -and $attempts -lt 10) {
        Write-Log "Waiting for training to start... (attempt $($attempts + 1))"
        Start-Sleep -Seconds 10
        $attempts++
    }
    
    if (Is-TrainingRunning) {
        Write-Log "Training started successfully"
        # Open the static dashboard
        Start-Process "domain_optimization_dashboard.html"
        return $true
    }
    else {
        Write-Log "Failed to start training after multiple attempts"
        return $false
    }
}

function Get-TrainingProgress {
    if (Test-Path $trainingStateFile) {
        try {
            $trainingState = Get-Content $trainingStateFile | ConvertFrom-Json
            return $trainingState
        }
        catch {
            Write-Log "Error reading training state file: $_"
            return $null
        }
    }
    else {
        return $null
    }
}

function Create-ScheduledTask {
    Write-Log "Creating scheduled task to resume training after restart"
    
    $action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\monitor_and_resume_training.ps1`" -Resume"
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    # Remove existing task if it exists
    Unregister-ScheduledTask -TaskName "TARA_Training_Resume" -Confirm:$false -ErrorAction SilentlyContinue
    
    # Register new task
    Register-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings -TaskName "TARA_Training_Resume" -Description "Resume TARA Universal Model training after system restart"
    
    Write-Log "Scheduled task created successfully"
}

function Monitor-Training {
    $startTime = Get-Date
    $endTime = $startTime.AddHours($maxRuntimeHours)
    
    Write-Log "Starting training monitoring until $endTime"
    
    try {
        while ((Get-Date) -lt $endTime) {
            if (-not (Is-TrainingRunning)) {
                Write-Log "Training process not detected, attempting to resume"
                
                $resumeSuccess = Start-TrainingRecovery -Resume
                if (-not $resumeSuccess) {
                    Write-Log "Failed to resume training, exiting monitoring"
                    break
                }
            }
            
            $progress = Get-TrainingProgress
            if ($progress) {
                Write-Log "Current training progress: $($progress | ConvertTo-Json -Compress)"
            }
            
            Write-Log "Training running, next check in $($checkInterval / 60) minutes"
            Start-Sleep -Seconds $checkInterval
        }
        
        Write-Log "Reached maximum runtime of $maxRuntimeHours hours"
        Write-Log "Training state saved. To resume, run: .\monitor_and_resume_training.ps1 -Resume"
    }
    catch {
        Write-Log "Error in monitoring: $_"
    }
}

# Main script
Write-Log "="*60
Write-Log "TARA Universal Model - Training Monitor and Resume"
Write-Log "="*60

# Check if resuming from previous run
$resumeMode = $args -contains "-Resume"

if ($resumeMode) {
    Write-Log "Running in resume mode"
    Start-TrainingRecovery -Resume
}
else {
    Write-Log "Running in new training mode"
    Write-Log "Domains: $domains"
    Write-Log "Model: $model"
    Write-Log "Max runtime: $maxRuntimeHours hours"
    
    # Create scheduled task for auto-resume after restart
    Create-ScheduledTask
    
    # Start training with recovery
    Start-TrainingRecovery
}

# Monitor training
Monitor-Training

Write-Log "Monitor script completed" 