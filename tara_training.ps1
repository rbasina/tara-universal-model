# TARA Universal Model - Unified Training Management Script
# This script combines all training management functions into a single tool
# Usage:
#   .\tara_training.ps1 -Action start     # Start new training
#   .\tara_training.ps1 -Action resume    # Resume interrupted training
#   .\tara_training.ps1 -Action monitor   # Monitor active training
#   .\tara_training.ps1 -Action dashboard # Open the training dashboard

param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "resume", "monitor", "dashboard")]
    [string]$Action = "dashboard",
    
    [Parameter(Mandatory=$false)]
    [string]$Domains = "education,creative,leadership",
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "Qwen/Qwen2.5-3B-Instruct",
    
    [Parameter(Mandatory=$false)]
    [int]$MaxRuntimeHours = 6,
    
    [Parameter(Mandatory=$false)]
    [switch]$Resilient = $false
)

# Configuration
$trainingStateFile = "training_recovery_state.json"
$logFile = "logs/tara_training.log"
$taskName = "TaraModelTrainingRestart"

# Ensure logs directory exists
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
            if ($processInfo -and ($processInfo.CommandLine -like "*train*" -or $processInfo.CommandLine -like "*domain*")) {
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

function Start-Training {
    param (
        [switch]$Resume,
        [switch]$Resilient
    )
    
    Write-Log "========================================"
    Write-Log "TARA Universal Model - Training Manager"
    Write-Log "========================================"
    
    # Check for existing checkpoints if resuming
    if ($Resume) {
        Write-Log "Attempting to resume training from previous checkpoint"
        
        if ($Resilient) {
            Write-Log "Starting resilient training recovery"
            Start-Process -FilePath "python" -ArgumentList "scripts/monitoring/training_recovery.py --auto_resume" -NoNewWindow
        } else {
            # Check for existing checkpoints
            Write-Log "Checking for existing checkpoints..."
            $educationCheckpoints = Get-ChildItem -Path "models/adapters/education" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue
            $creativeCheckpoints = Get-ChildItem -Path "models/adapters/creative" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue
            $leadershipCheckpoints = Get-ChildItem -Path "models/adapters/leadership" -Filter "checkpoint-*" -Directory -ErrorAction SilentlyContinue

            if ($educationCheckpoints -or $creativeCheckpoints -or $leadershipCheckpoints) {
                Write-Log "Found existing checkpoints:"
                if ($educationCheckpoints) { Write-Log "- Education: $($educationCheckpoints.Count) checkpoints" }
                if ($creativeCheckpoints) { Write-Log "- Creative: $($creativeCheckpoints.Count) checkpoints" }
                if ($leadershipCheckpoints) { Write-Log "- Leadership: $($leadershipCheckpoints.Count) checkpoints" }
                
                # Start training with auto checkpoint detection
                Write-Log "Starting training with auto checkpoint detection..."
                Start-Process -FilePath "python" -ArgumentList "scripts/training/parameterized_train_domains.py --domains $Domains --model $Model --resume_from_checkpoint auto" -NoNewWindow
            } else {
                Write-Log "No existing checkpoints found, starting fresh training"
                Start-Process -FilePath "python" -ArgumentList "scripts/training/parameterized_train_domains.py --domains $Domains --model $Model" -NoNewWindow
            }
        }
    } else {
        # Start new training
        Write-Log "Starting new training session"
        Write-Log "Domains: $Domains"
        Write-Log "Model: $Model"
        Write-Log "Max runtime: $MaxRuntimeHours hours"
        Write-Log "Resilient mode: $Resilient"
        
        if ($Resilient) {
            # Check for psutil
            $psutilInstalled = python -c "import pkgutil; print('installed' if pkgutil.find_loader('psutil') else 'missing')" 2>$null
            if ($psutilInstalled -ne "installed") {
                Write-Log "Installing psutil package for memory management..."
                pip install psutil | Out-File -FilePath $logFile -Append
            }
            
            # Use restart_domains.py for resilient training
            Write-Log "Starting training with resilience to interruptions..."
            Start-Process -FilePath "python" -ArgumentList "scripts/training/restart_domains.py --domains $Domains --model $Model" -NoNewWindow
            
            # Create scheduled task for auto-restart
            Create-ScheduledTask
        } else {
            # Use training_recovery.py for standard training with recovery
            Write-Log "Starting training with standard recovery..."
            Start-Process -FilePath "python" -ArgumentList "scripts/monitoring/training_recovery.py --domains $Domains --model $Model --max_runtime $MaxRuntimeHours" -NoNewWindow
        }
    }
    
    # Wait a moment for training to start
    Start-Sleep -Seconds 5
    
    # Open dashboard
    Write-Log "Opening training dashboard..."
    Start-Process "domain_optimization_dashboard.html"
    
    Write-Log "Training initiated. To monitor progress, run: .\tara_training.ps1 -Action monitor"
}

function Create-ScheduledTask {
    Write-Log "Creating scheduled task for automatic restart after system reboot"
    
    $taskExists = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($taskExists) {
        Write-Log "Updating existing scheduled task..."
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$PSScriptRoot\tara_training.ps1`" -Action resume -Resilient"
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings
    
    Write-Log "Scheduled task created successfully"
}

function Monitor-Training {
    $startTime = Get-Date
    $endTime = $startTime.AddHours($MaxRuntimeHours)
    
    Write-Log "Starting training monitoring until $endTime"
    
    try {
        while ((Get-Date) -lt $endTime) {
            if (-not (Is-TrainingRunning)) {
                Write-Log "Training process not detected, attempting to resume"
                
                Start-Training -Resume
                Start-Sleep -Seconds 30
                
                if (-not (Is-TrainingRunning)) {
                    Write-Log "Failed to resume training, exiting monitoring"
                    break
                }
            }
            
            if (Test-Path $trainingStateFile) {
                try {
                    $trainingState = Get-Content $trainingStateFile | ConvertFrom-Json
                    Write-Log "Current training progress: $($trainingState | ConvertTo-Json -Compress)"
                }
                catch {
                    Write-Log "Error reading training state file: $_"
                }
            }
            
            Write-Log "Training running, next check in 5 minutes"
            Start-Sleep -Seconds 300
        }
        
        Write-Log "Reached maximum runtime of $MaxRuntimeHours hours"
        Write-Log "Training state saved. To resume, run: .\tara_training.ps1 -Action resume"
    }
    catch {
        Write-Log "Error in monitoring: $_"
    }
}

function Open-Dashboard {
    Write-Host "Opening TARA Universal Model Training Dashboard..."
    Start-Process "domain_optimization_dashboard.html"
    Write-Host "Dashboard opened in your default browser."
}

# Main execution
switch ($Action) {
    "start" {
        Start-Training -Resilient:$Resilient
    }
    "resume" {
        Start-Training -Resume -Resilient:$Resilient
    }
    "monitor" {
        Monitor-Training
    }
    "dashboard" {
        Open-Dashboard
    }
} 