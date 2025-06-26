# TARA Universal Model - Move MD Files
# This script moves the README.md file to the docs folder and removes duplicate MD files from the root directory

Write-Host "TARA Universal Model - Move MD Files"
Write-Host "===================================="
Write-Host ""

# Define the files to check
$mdFiles = @(
    "COMPREHENSIVE_DOMAIN_CATEGORIZATION.md",
    "DOMAIN_MODEL_CATEGORIZATION.md",
    "GGUF_PARENT_DOMAIN_STRATEGY.md",
    "README.md"
)

# Define the target directory for README.md
$docsDir = "docs"

# Count variables
$totalMoved = 0
$totalRemoved = 0

# First, copy the README.md to the docs directory if it doesn't exist there
$readmePath = Join-Path (Get-Location).Path "README.md"
$docsReadmePath = Join-Path $docsDir "README.md"

if (Test-Path $readmePath) {
    if (-not (Test-Path $docsReadmePath)) {
        try {
            Copy-Item -Path $readmePath -Destination $docsReadmePath -Force
            Write-Host "  Copied: README.md to $docsDir/README.md" -ForegroundColor Green
            $totalMoved++
        }
        catch {
            Write-Host "  Failed to copy README.md: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  README.md already exists in $docsDir directory" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  README.md not found in root directory" -ForegroundColor Yellow
}

# Now check for duplicate MD files and remove them if they exist in the docs/2-architecture directory
$archDir = Join-Path $docsDir "2-architecture"

Write-Host ""
Write-Host "Checking for duplicate MD files..."

foreach ($file in $mdFiles) {
    # Skip README.md as we've already handled it
    if ($file -eq "README.md") {
        continue
    }
    
    $rootPath = Join-Path (Get-Location).Path $file
    $archPath = Join-Path $archDir $file
    
    if (Test-Path $rootPath) {
        if (Test-Path $archPath) {
            # Compare file content to ensure they're identical
            $rootContent = Get-Content -Path $rootPath -Raw
            $archContent = Get-Content -Path $archPath -Raw
            
            if ($rootContent -eq $archContent) {
                try {
                    Remove-Item -Path $rootPath -Force
                    Write-Host "  Removed duplicate: $file (identical copy exists in $archDir)" -ForegroundColor Green
                    $totalRemoved++
                }
                catch {
                    Write-Host "  Failed to remove $file: $($_.Exception.Message)" -ForegroundColor Red
                }
            }
            else {
                Write-Host "  WARNING: $file exists in $archDir but content differs" -ForegroundColor Yellow
                
                # Ask if user wants to overwrite the architecture file with root file
                $overwrite = Read-Host "  Do you want to overwrite the architecture file with the root file? (y/n)"
                if ($overwrite -eq 'y') {
                    try {
                        Copy-Item -Path $rootPath -Destination $archPath -Force
                        Remove-Item -Path $rootPath -Force
                        Write-Host "  Updated $archPath and removed $file from root" -ForegroundColor Green
                        $totalRemoved++
                    }
                    catch {
                        Write-Host "  Failed to update/remove $file: $($_.Exception.Message)" -ForegroundColor Red
                    }
                }
            }
        }
        else {
            # File doesn't exist in architecture directory, so move it there
            try {
                Copy-Item -Path $rootPath -Destination $archPath -Force
                Remove-Item -Path $rootPath -Force
                Write-Host "  Moved: $file to $archDir" -ForegroundColor Green
                $totalMoved++
            }
            catch {
                Write-Host "  Failed to move $file: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    else {
        Write-Host "  $file not found in root directory" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Summary:"
Write-Host "--------"
Write-Host "Total files moved: $totalMoved" -ForegroundColor Green
Write-Host "Total duplicates removed: $totalRemoved" -ForegroundColor Green

Write-Host ""
Write-Host "Script completed." 