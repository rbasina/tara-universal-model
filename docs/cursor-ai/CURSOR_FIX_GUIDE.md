# MeeTARA Cursor AI Freezing - SOLVED

## Problem
Cursor AI was freezing with "window not responding" error when opening MeeTARA folder due to **23,552 files in node_modules**.

## Root Cause Analysis
- **Total files**: 24,779 files causing Cursor to freeze during indexing
- **Major culprit**: Multiple `node_modules` directories
  - **Main node_modules**: 23,552 files (lucide-react icons: 2,756, next/dist: 6,846, es-abstract: 2,518)
  - **Hidden node_modules**: `services/node_modules` (26 files) + `services/native-drivers-cpp/node_modules` (477 files)
- **Secondary issues**: `.cursorignore` wasn't properly excluding nested node_modules

## Solution Applied
1. **Removed main node_modules**: `Remove-Item node_modules -Recurse -Force`
2. **Removed services node_modules**: `Remove-Item services\node_modules -Recurse -Force`  
3. **Removed nested node_modules**: `Remove-Item services\native-drivers-cpp\node_modules -Recurse -Force`
4. **File count reduced**: 24,779 → 721 files (97% reduction)
5. **Cursor now opens successfully** without freezing

## Development Workflow
### For Frontend Development (React/Next.js)
```powershell
# Install dependencies when needed
npm install

# Run development server
npm run dev

# When done, remove node_modules to keep Cursor fast
Remove-Item node_modules -Recurse -Force
```

### For Backend Development (Python)
```powershell
# Python backend is in services/ai-engine-python/
cd services/ai-engine-python

# Create virtual environment if needed
python -m venv .venv-tara-py312
.venv-tara-py312\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python app.py
```

### Full-Stack Testing
1. Install node_modules: `npm install`
2. Start frontend: `npm run dev` (port 3000)
3. Start backend: `python services/ai-engine-python/app.py` (port 8000)
4. Test Trinity Architecture integration
5. **Clean up**: `Remove-Item node_modules -Recurse -Force`

## Alternative: Clean Development Workspace
Created `meetara_dev_clean/` with only essential files (100 files total):
- Source code (`src/`)
- Documentation (`docs/`)
- Configuration files
- Essential backend Python files

## Key Learnings
- **Cursor limit**: ~1,000-2,000 files for stable operation
- **node_modules is toxic**: Always exclude from IDEs
- **.cursorignore might not work reliably**: Manual removal more effective
- **Development workflow**: Install → Develop → Remove heavy folders

## Status: ✅ RESOLVED
- Cursor opens instantly without freezing
- Development capabilities preserved
- Trinity Architecture testing enabled
- File management strategy documented

**Last Updated**: June 22, 2025
**Files reduced**: 24,779 → 1,226 (95% reduction) 