# Cursor AI Troubleshooting Guide

## When Cursor Freezes or Won't Open

### Step 1: Check Configuration Files First
**Before assuming file volume issues**, check these files:

#### .cursorignore Issues
```powershell
# Check for malformed entries
Get-Content .cursorignore | Select-String -Pattern "node_modules"

# Look for duplicates or syntax errors
Get-Content .cursorignore
```

**Common Problems:**
- Duplicate entries (e.g., `node_modules/` listed multiple times)
- Invalid patterns or syntax
- Conflicting ignore rules

**Quick Fix:**
```powershell
# Remove corrupted ignore file
Remove-Item .cursorignore

# Test opening
code .

# Recreate clean ignore file if needed
```

#### Other Configuration Files
- `.vscode/settings.json` - Can cause conflicts
- `tsconfig.json` - Large include/exclude arrays
- `package.json` - Malformed scripts or dependencies

### Step 2: Compare with Working Projects
If you have similar projects that work:
- Compare file structures
- Compare configuration files
- Check for unique files in the problematic project

### Step 3: File Volume Analysis (Last Resort)
Only if configuration fixes don't work:

```powershell
# Count files by directory
Get-ChildItem -Directory | ForEach-Object { 
    $count = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "$($_.Name): $count files" 
} | Sort-Object { [int]($_ -split ': ')[1] -replace ' files' } -Descending
```

**File Volume Guidelines:**
- **< 1,000 files**: Should work perfectly
- **1,000-5,000 files**: Usually fine, might be slow
- **5,000-10,000 files**: May cause performance issues
- **> 10,000 files**: Likely to cause problems

### Step 4: Clean Heavy Directories
If file volume IS the issue:

```powershell
# Remove node_modules (can be reinstalled)
Remove-Item node_modules -Recurse -Force

# Remove build outputs
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .next -Recurse -Force -ErrorAction SilentlyContinue

# Remove cache directories
Remove-Item .cache -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .tmp -Recurse -Force -ErrorAction SilentlyContinue
```

## Diagnostic Checklist

### ✅ Quick Wins (Try First)
- [ ] Remove/rename `.cursorignore`
- [ ] Check `.vscode/settings.json` for conflicts
- [ ] Compare with working similar projects
- [ ] Restart Cursor completely

### ✅ Configuration Analysis
- [ ] Validate all ignore files
- [ ] Check TypeScript configuration
- [ ] Review package.json for issues
- [ ] Look for lock files or cache corruption

### ✅ File Volume Analysis (If Needed)
- [ ] Count files by directory
- [ ] Identify largest directories
- [ ] Remove unnecessary heavy folders
- [ ] Create clean workspace alternative

### ✅ Nuclear Options (Last Resort)
- [ ] Remove all node_modules recursively
- [ ] Clear all cache directories
- [ ] Create minimal project subset
- [ ] Reinstall Cursor AI

## Prevention Tips

### 1. Maintain Clean .cursorignore
```
# Good practices
node_modules/          # Not node_modules/ node_modules/
build/                 # Clear, single entries
*.log                  # Use wildcards appropriately
```

### 2. Regular Cleanup
```powershell
# Weekly cleanup script
Remove-Item node_modules -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .next -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item *.log -Force -ErrorAction SilentlyContinue
```

### 3. Monitor Project Size
```powershell
# Check project health
Get-ChildItem -Recurse | Measure-Object | Select-Object Count
```

**Healthy project**: < 2,000 files for optimal Cursor performance

## Common Error Patterns

### "Window Not Responding"
- **90% of time**: Configuration file corruption
- **10% of time**: Actual file volume overload

### Slow Indexing
- Usually file volume related
- Check for large binary files being indexed
- Verify ignore patterns are working

### Frequent Crashes
- Memory issues from large projects
- Corrupted workspace settings
- Extension conflicts

## Success Indicators
- Cursor opens within 5-10 seconds
- File tree loads completely
- Search functionality works
- No memory warnings in task manager

**Remember**: Configuration issues are far more common than file volume issues!

**Last Updated**: June 22, 2025 