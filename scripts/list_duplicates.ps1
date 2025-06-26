# TARA Universal Model - List Duplicate Scripts
# This script lists Python scripts that exist in both the root scripts folder
# and their respective category subfolders

Write-Host "TARA Universal Model - List Duplicate Scripts"
Write-Host "============================================="
Write-Host ""

# Define the categories and their folders
$categories = @{
    "training" = @("train_*.py", "test_*.py", "demo_*.py", "auto_train_*.py")
    "monitoring" = @("monitor_*.py", "watch_*.py", "web_*.py", "simple_web_*.py")
    "conversion" = @("create_*.py")
    "utilities" = @("download_*.py", "backup_*.py", "fix_*.py", "serve_*.py")
}

# Count variables
$totalDuplicates = 0

# Process each category
foreach ($category in $categories.Keys) {
    $categoryFolder = Join-Path "scripts" $category
    
    Write-Host "Checking $category scripts..."
    $categoryDuplicates = 0
    
    # Process each pattern in this category
    foreach ($pattern in $categories[$category]) {
        # Find files in root scripts folder matching the pattern
        $rootFiles = Get-ChildItem -Path "scripts" -Filter $pattern -File
        
        foreach ($rootFile in $rootFiles) {
            $fileName = $rootFile.Name
            $subfolderFile = Join-Path $categoryFolder $fileName
            
            # Check if the file exists in the subfolder
            if (Test-Path $subfolderFile) {
                Write-Host "  DUPLICATE: $fileName" -ForegroundColor Yellow
                $totalDuplicates++
                $categoryDuplicates++
            }
        }
    }
    
    Write-Host "  Found $categoryDuplicates duplicates in $category" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Summary:"
Write-Host "--------"
Write-Host "Total duplicates found: $totalDuplicates" -ForegroundColor Green
Write-Host ""
Write-Host "To safely remove these duplicates, you can run:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts/cleanup_duplicates.ps1"
Write-Host "" 