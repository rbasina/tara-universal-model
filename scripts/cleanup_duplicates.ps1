# TARA Universal Model - Duplicate Script Cleanup
# This script removes duplicate Python scripts from the root scripts folder
# that are already properly organized in subfolders

Write-Host "TARA Universal Model - Duplicate Script Cleanup"
Write-Host "================================================"
Write-Host ""

# Define the categories and their folders
$categories = @{
    "training" = @("train_*.py", "test_*.py", "demo_*.py", "auto_train_*.py")
    "monitoring" = @("monitor_*.py", "watch_*.py", "web_*.py", "simple_web_*.py")
    "conversion" = @("create_*.py")
    "utilities" = @("download_*.py", "backup_*.py", "fix_*.py", "serve_*.py")
}

# Count variables
$totalFiles = 0
$safeToDelete = 0
$notSafeToDelete = 0

# Process each category
foreach ($category in $categories.Keys) {
    $categoryFolder = Join-Path "scripts" $category
    
    Write-Host "Checking $category scripts..."
    
    # Process each pattern in this category
    foreach ($pattern in $categories[$category]) {
        # Find files in root scripts folder matching the pattern
        $rootFiles = Get-ChildItem -Path "scripts" -Filter $pattern -File
        
        foreach ($rootFile in $rootFiles) {
            $totalFiles++
            $fileName = $rootFile.Name
            $subfolderFile = Join-Path $categoryFolder $fileName
            
            # Check if the file exists in the subfolder
            if (Test-Path $subfolderFile) {
                # Compare file content to ensure they're identical
                $rootContent = Get-Content -Path $rootFile.FullName -Raw
                $subfolderContent = Get-Content -Path $subfolderFile -Raw
                
                if ($rootContent -eq $subfolderContent) {
                    Write-Host "  [SAFE TO DELETE] $fileName - Identical copy exists in $category folder" -ForegroundColor Green
                    $safeToDelete++
                } else {
                    Write-Host "  [WARNING] $fileName - File exists in $category folder but content differs" -ForegroundColor Yellow
                    $notSafeToDelete++
                }
            } else {
                Write-Host "  [KEEP] $fileName - Not found in $category folder" -ForegroundColor Red
                $notSafeToDelete++
            }
        }
    }
    
    Write-Host ""
}

Write-Host "Summary:"
Write-Host "--------"
Write-Host "Total files checked: $totalFiles"
Write-Host "Safe to delete: $safeToDelete" -ForegroundColor Green
Write-Host "Not safe to delete: $notSafeToDelete" -ForegroundColor Yellow
Write-Host ""

# Ask for confirmation to delete the duplicate files
if ($safeToDelete -gt 0) {
    $confirmation = Read-Host "Do you want to delete the $safeToDelete duplicate files? (y/n)"
    
    if ($confirmation -eq 'y') {
        Write-Host "Deleting duplicate files..." -ForegroundColor Cyan
        
        # Process each category again to delete files
        foreach ($category in $categories.Keys) {
            $categoryFolder = Join-Path "scripts" $category
            
            # Process each pattern in this category
            foreach ($pattern in $categories[$category]) {
                # Find files in root scripts folder matching the pattern
                $rootFiles = Get-ChildItem -Path "scripts" -Filter $pattern -File
                
                foreach ($rootFile in $rootFiles) {
                    $fileName = $rootFile.Name
                    $subfolderFile = Join-Path $categoryFolder $fileName
                    
                    # Check if the file exists in the subfolder and has identical content
                    if (Test-Path $subfolderFile) {
                        $rootContent = Get-Content -Path $rootFile.FullName -Raw
                        $subfolderContent = Get-Content -Path $subfolderFile -Raw
                        
                        if ($rootContent -eq $subfolderContent) {
                            Remove-Item -Path $rootFile.FullName -Force
                            Write-Host "  Deleted: $fileName" -ForegroundColor Green
                        }
                    }
                }
            }
        }
        
        Write-Host "Deletion complete!" -ForegroundColor Cyan
    } else {
        Write-Host "No files were deleted." -ForegroundColor Yellow
    }
} else {
    Write-Host "No duplicate files found that are safe to delete." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Script completed." 