# TARA Universal Model - Remove Duplicate Scripts
# This script removes Python scripts from the root scripts folder that are duplicated in category subfolders

Write-Host "TARA Universal Model - Remove Duplicate Scripts"
Write-Host "=============================================="
Write-Host ""

# Define the duplicates we found
$duplicates = @(
    # Conversion Scripts
    "create_clean_gguf.py",
    "create_combined_universal_gguf.py",
    "create_hierarchical_gguf.py",
    "create_meetara_universal.py",
    "create_meetara_universal_1_0.py",
    "create_meetara_universal_combo.py",
    "create_tara_gguf.py",
    "create_universal_embedded_gguf.py",
    "create_working_meetara_gguf.py",
    
    # Monitoring Scripts
    "monitor_training.py",
    "simple_web_monitor.py",
    "watch_training.py",
    "web_monitor.py",
    
    # Training Scripts
    "demo_reinforcement_learning.py",
    "test_domain_training.py",
    "test_phase2_intelligence.py",
    "train_all_domains.py",
    "train_domain.py",
    "train_meetara_universal_model.py",
    "train_qwen_domains.py",
    "train_qwen_simple.py",
    
    # Utilities Scripts
    "backup_training_data.py",
    "download_datasets.py",
    "download_models.py",
    "download_qwen_model.py",
    "fix_meetara_gguf.py",
    "serve_model.py"
)

# Count variables
$totalRemoved = 0
$totalFailed = 0

Write-Host "The following files will be removed from the root scripts folder:"
foreach ($file in $duplicates) {
    Write-Host "  - $file" -ForegroundColor Yellow
}
Write-Host ""

# Automatically proceed with deletion
Write-Host "Removing duplicate files..." -ForegroundColor Cyan

foreach ($file in $duplicates) {
    $filePath = Join-Path "scripts" $file
    
    if (Test-Path $filePath) {
        try {
            Remove-Item -Path $filePath -Force
            Write-Host "  Removed: $file" -ForegroundColor Green
            $totalRemoved++
        }
        catch {
            Write-Host "  Failed to remove: $file - $($_.Exception.Message)" -ForegroundColor Red
            $totalFailed++
        }
    }
    else {
        Write-Host "  File not found: $file" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Summary:"
Write-Host "--------"
Write-Host "Total files removed: $totalRemoved" -ForegroundColor Green
if ($totalFailed -gt 0) {
    Write-Host "Failed to remove: $totalFailed" -ForegroundColor Red
}

Write-Host ""
Write-Host "Script completed." 