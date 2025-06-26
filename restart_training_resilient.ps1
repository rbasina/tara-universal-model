# TARA Universal Model - Resilient Training Restart Script
# This script will restart training with improved resilience to Cursor AI restarts

Write-Host "TARA Universal Model - Resilient Training Restart"
Write-Host "=================================================="
Write-Host "This script will restart training with improved resilience to Cursor AI restarts"

# Configuration
$domains = "education,creative,leadership"
$model = "Qwen/Qwen2.5-3B-Instruct"

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -Path "logs" -ItemType Directory | Out-Null
}

# Create log file
$logFile = "logs/restart_training_resilient.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"$timestamp - Starting resilient training restart" | Out-File -FilePath $logFile

# Check if psutil is installed
Write-Host "Checking for required Python packages..."
$psutilInstalled = python -c "import pkgutil; print('installed' if pkgutil.find_loader('psutil') else 'missing')" 2>$null
if ($psutilInstalled -ne "installed") {
    Write-Host "Installing psutil package for memory management..."
    pip install psutil | Out-File -FilePath $logFile -Append
}

# Check for existing Python processes
Write-Host "Checking for existing training processes..."
$existingProcess = Get-Process | Where-Object { $_.ProcessName -like "*python*" } | 
                  Where-Object { $_.CommandLine -like "*parameterized_train_domains.py*" } | 
                  Select-Object -First 1

if ($existingProcess) {
    Write-Host "Found existing training process (PID: $($existingProcess.Id))"
    $choice = Read-Host "Do you want to stop it and restart training? (y/n)"
    
    if ($choice -eq "y") {
        Write-Host "Stopping existing process..."
        Stop-Process -Id $existingProcess.Id -Force
        Start-Sleep -Seconds 2
    } else {
        Write-Host "Exiting without restarting training"
        exit
    }
}

# Check for training state
Write-Host "Checking for existing training state..."
$stateFile = "training_state/overall_training_state.json"

if (Test-Path $stateFile) {
    try {
        $state = Get-Content $stateFile | ConvertFrom-Json
        
        if ($state.status -eq "in_progress") {
            $pendingDomains = $state.pending_domains
            
            if ($pendingDomains -and $pendingDomains.Count -gt 0) {
                Write-Host "Found previous training run with pending domains: $($pendingDomains -join ', ')"
                Write-Host "Already completed: $($state.completed_domains -join ', ')"
                
                $choice = Read-Host "Do you want to resume with pending domains? (y/n)"
                
                if ($choice -eq "y") {
                    $domains = $pendingDomains -join ","
                    Write-Host "Will resume with domains: $domains"
                }
            }
        }
    }
    catch {
        Write-Host "Failed to read training state: $_"
    }
}

# Start the resilient training script
Write-Host "Starting training with resilience to Cursor AI restarts..."
"$timestamp - Starting training with command: python scripts/training/restart_domains.py --domains $domains --model $model" | Out-File -FilePath $logFile -Append

# Start the process
Start-Process -FilePath "python" -ArgumentList "scripts/training/restart_domains.py --domains $domains --model $model" -NoNewWindow

Write-Host "Resilient training restart initiated. Check logs/restart_domains.log for progress."
"$timestamp - Resilient training restart initiated" | Out-File -FilePath $logFile -Append

# Create a scheduled task to restart training after system restart
Write-Host "Setting up automatic restart after system reboot..."

$taskName = "TaraModelTrainingRestart"
$taskExists = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue

if ($taskExists) {
    Write-Host "Updating existing scheduled task for automatic restart..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$PSScriptRoot\restart_training_resilient.ps1`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings

Write-Host "Scheduled task created for automatic restart after system reboot"
"$timestamp - Scheduled task created for automatic restart" | Out-File -FilePath $logFile -Append 