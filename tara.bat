@echo off
REM TARA Universal Model - Unified Command Interface
REM This batch file forwards commands to the PowerShell script

if "%1"=="" (
    echo TARA Universal Model - Command Interface
    echo =====================================
    echo Usage:
    echo   tara start     - Start new training
    echo   tara resume    - Resume interrupted training
    echo   tara monitor   - Monitor active training
    echo   tara dashboard - Open the training dashboard
    echo.
    echo Opening dashboard by default...
    powershell -ExecutionPolicy Bypass -File "%~dp0tara_training.ps1" -Action dashboard
    goto :EOF
)

if "%1"=="start" (
    powershell -ExecutionPolicy Bypass -File "%~dp0tara_training.ps1" -Action start %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="resume" (
    powershell -ExecutionPolicy Bypass -File "%~dp0tara_training.ps1" -Action resume %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="monitor" (
    powershell -ExecutionPolicy Bypass -File "%~dp0tara_training.ps1" -Action monitor %2 %3 %4 %5 %6 %7 %8 %9
) else if "%1"=="dashboard" (
    powershell -ExecutionPolicy Bypass -File "%~dp0tara_training.ps1" -Action dashboard
) else (
    echo Unknown command: %1
    echo Valid commands: start, resume, monitor, dashboard
) 