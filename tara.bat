@echo off
REM TARA Universal Model - Unified Command Interface
REM This batch file provides a simple interface for training and monitoring

if "%1"=="" (
    echo TARA Universal Model - Command Interface
    echo =====================================
    echo Usage:
    echo   tara start [domains]    - Start new training for specified domains
    echo   tara resume [domains]   - Resume interrupted training
    echo   tara monitor   - Monitor active training
    echo   tara dashboard - Open the training dashboard
    echo   tara fresh [domains]    - Force fresh training (ignore checkpoints)
    echo   tara education - Resume Education domain (tracking from step 134)
    echo   tara leadership - Start Leadership domain with ultra-optimized settings
    echo   tara parallel  - Train multiple domains in parallel
    echo   tara status    - Check training status
    echo   tara fast [domain]      - Run fast training with optimized settings
    
    echo.
    echo Opening dashboard by default...
    start "" domain_optimization_dashboard.html
    goto :EOF
)

if "%1"=="start" (
    echo Starting TARA Universal Model training...
    if "%2"=="" (
        echo Error: Please specify domain(s) to train
        echo Example: tara start education
        echo Example: tara start education,creative,leadership
        goto :EOF
    )
    python scripts/training/parameterized_train_domains.py --domains %2
    goto :EOF
)

if "%1"=="resume" (
    echo Resuming interrupted training...
    if "%2"=="" (
        echo Resuming all domains: education,creative,leadership
        python scripts/training/parameterized_train_domains.py --domains education,creative,leadership
    ) else (
        echo Resuming specified domains: %2
        python scripts/training/parameterized_train_domains.py --domains %2
    )
    goto :EOF
)

if "%1"=="monitor" (
    echo Starting training monitor...
    python scripts/monitoring/monitor_training.py
    goto :EOF
)

if "%1"=="dashboard" (
    echo Opening training dashboard...
    start "" domain_optimization_dashboard.html
    goto :EOF
)

if "%1"=="fresh" (
    echo Starting fresh training (ignoring checkpoints)...
    if "%2"=="" (
        echo Error: Please specify domain(s) for fresh training
        echo Example: tara fresh education
        echo Example: tara fresh education,creative,leadership
        goto :EOF
    )
    echo Starting fresh training for %2 (ignoring checkpoints)...
    python scripts/training/parameterized_train_domains.py --domains %2 --force_fresh
    goto :EOF
)

if "%1"=="education" (
    echo Starting Education domain training from checkpoint 134...
    python scripts/training/parameterized_train_domains.py --domains education --max_retries 1
    goto :EOF
)

if "%1"=="leadership" (
    echo Starting Leadership domain training with ultra-optimized settings...
    echo Using minimal memory settings (reduced batch size, sequence length, and LoRA params)
    python scripts/training/parameterized_train_domains.py --domains leadership --force_fresh --max_retries 1 --max_steps 200 --batch_size 1 --seq_length 32 --lora_r 2
    goto :EOF
)

if "%1"=="parallel" (
    echo Starting parallel domain training...
    python scripts/training/auto_train_remaining_domains.py
    goto :EOF
)

if "%1"=="status" (
    echo Checking training status...
    python scripts/monitoring/monitor_training.py --status
    goto :EOF
)

if "%1"=="fast" (
    echo Starting fast training with optimized settings...
    echo This will train with reduced parameters for quicker results
    
    REM Check if domain is specified
    if "%2"=="" (
        echo Please specify a domain: creative, education, leadership, healthcare, or business
        echo Example: tara fast creative
        goto :EOF
    )
    
    echo Starting fast training for %2 with optimized settings...
    python scripts/training/parameterized_train_domains.py --domains %2 --force_fresh --max_retries 1 --max_steps 200
    goto :EOF
)

echo Unknown command: %1
echo Run 'tara' without arguments to see available commands 