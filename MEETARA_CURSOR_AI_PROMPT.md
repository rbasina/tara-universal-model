# MeeTARA Project - New Cursor AI Session Prompt

## 🚨 CRITICAL ISSUE CONTEXT
**MeeTARA Project** - Trinity Complete HAI Companion with Cursor AI freezing issue and lost dependencies

### **PRIMARY PROBLEM**
- **Location**: `C:\Users\rames\Documents\meetara`
- **Issue**: Cursor AI freezes when trying to open the MeeTARA folder
- **Root Cause**: 10.4GB of GGUF model files overwhelming Cursor indexing
- **Dependencies**: LOST - Need complete frontend and backend package restoration
- **Status**: Trinity Complete (June 20, 2025) but inaccessible due to technical issues

---

## 🎯 IMMEDIATE EMERGENCY ACTIONS

### **1. CURSOR FREEZING FIX (TOP PRIORITY)**
#### Create .cursorignore file to exclude GGUF models:

```bash
# Navigate to MeeTARA directory
cd C:\Users\rames\Documents\meetara

# Create .cursorignore file
echo "# Exclude large GGUF model files from Cursor indexing" > .cursorignore
echo "*.gguf" >> .cursorignore
echo "models/" >> .cursorignore
echo "**/*.gguf" >> .cursorignore
echo "**/*model*/*.gguf" >> .cursorignore
echo "llama/" >> .cursorignore
echo "phi/" >> .cursorignore
echo "qwen/" >> .cursorignore
echo "# Cache and temporary files" >> .cursorignore
echo "node_modules/" >> .cursorignore
echo ".next/" >> .cursorignore
echo "dist/" >> .cursorignore
echo "__pycache__/" >> .cursorignore
echo "*.pyc" >> .cursorignore
echo ".env" >> .cursorignore
```

#### GGUF Files Identified (10.4GB total):
- `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.92 GB)
- `Phi-3.5-mini-instruct-Q4_K_M.gguf` (2.39 GB) 
- `qwen2.5-3b-instruct-q4_0.gguf` (1.99 GB)
- Additional model files (~1.1 GB)

### **2. ALTERNATIVE ACCESS METHODS**
If Cursor still freezes after .cursorignore:

#### Option A: Move GGUF files temporarily
```bash
# Create backup directory
mkdir C:\Users\rames\Documents\meetara_models_backup

# Move GGUF files
move "C:\Users\rames\Documents\meetara\**\*.gguf" "C:\Users\rames\Documents\meetara_models_backup\"
```

#### Option B: Command-line project access
```bash
# Use terminal to navigate and work
cd C:\Users\rames\Documents\meetara
code . --disable-extensions
```

---

## 🔧 DEPENDENCY RESTORATION

### **Frontend Dependencies (React/Next.js)**
```bash
# Navigate to MeeTARA project
cd C:\Users\rames\Documents\meetara

# Check for package.json
Get-Content package.json

# Install Node.js dependencies
npm install

# Or if using yarn
yarn install

# Common MeeTARA frontend dependencies
npm install react react-dom next
npm install @emotion/react @emotion/styled
npm install @mui/material @mui/icons-material
npm install axios
npm install socket.io-client
npm install react-speech-recognition
npm install @types/react @types/node typescript
```

### **Backend Dependencies (Python 3.12)**
```bash
# Navigate to MeeTARA project
cd C:\Users\rames\Documents\meetara

# Create Python 3.12 virtual environment
python -m venv .venv-tara-p3.12

# Activate virtual environment
.venv-tara-p3.12\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.12.x

# Check for requirements.txt or Python files
Get-Content requirements.txt

# Install Python dependencies in virtual environment
pip install fastapi uvicorn websockets
pip install openai anthropic
pip install edge-tts pyttsx3
pip install speechrecognition pyaudio
pip install pandas numpy
pip install python-dotenv
pip install asyncio aiofiles
pip install transformers torch
pip install llama-cpp-python

# If requirements.txt exists
pip install -r requirements.txt
```

### **Voice/Audio Dependencies**
```bash
# Ensure virtual environment is activated
.venv-tara-p3.12\Scripts\activate

# Voice processing
pip install edge-tts pyttsx3
pip install speechrecognition
pip install pyaudio
pip install sounddevice

# If pyaudio fails, try:
pip install pipwin
pipwin install pyaudio
```

---

## 🏗️ MEETARA PROJECT ARCHITECTURE

### **Trinity Architecture Components**
Based on MeeTARA Vision (Trinity Complete - June 20, 2025):

1. **Tony Stark's Arc Reactor** - Core processing engine
2. **Perplexity Intelligence** - Context-aware reasoning
3. **Einstein's E=mc²** - 504% intelligence amplification

### **Expected Project Structure**
```
meetara/
├── frontend/              # React/Next.js frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── next.config.js
├── backend/               # Python FastAPI backend
│   ├── api/
│   ├── models/
│   ├── services/
│   └── requirements.txt
├── models/                # GGUF model files (CAUSING CURSOR FREEZE)
│   ├── llama/
│   ├── phi/
│   └── qwen/
├── docs/                  # Documentation
│   ├── 1-vision/
│   └── 2-architecture/
├── voice/                 # Voice processing
├── websocket/            # Real-time communication
└── .cursorignore         # FIX FOR CURSOR FREEZING
```

### **Port Configuration**
- **Frontend**: Port 2025
- **WebSocket**: Port 8765  
- **HTTP API**: Port 8766
- **Voice Server**: Port 5000

---

## 🚀 SYSTEM RESTORATION STEPS

### **Step 1: Fix Cursor Access**
1. Create .cursorignore file (commands above)
2. Restart Cursor AI
3. Try opening `C:\Users\rames\Documents\meetara`
4. If still freezing, move GGUF files temporarily

### **Step 2: Identify Project Structure**
```bash
cd C:\Users\rames\Documents\meetara
dir
Get-Content package.json
Get-Content requirements.txt
```

### **Step 3: Restore Dependencies**
```bash
# Frontend (if package.json exists)
npm install

# Backend (if requirements.txt exists)
pip install -r requirements.txt

# Or install common dependencies manually
```

### **Step 4: Test System**
```bash
# Test frontend
npm run dev

# Test backend
python app.py
# or
uvicorn main:app --host 0.0.0.0 --port 8766

# Test voice server
python voice_server.py
```

---

## 💡 MEETARA CONTEXT

### **HAI Philosophy**
*"Replace every AI app with ONE intelligent companion"*

### **Trinity Complete Status** (June 20, 2025)
- **Arc Reactor**: Core processing operational
- **Perplexity Intelligence**: Context-aware reasoning active
- **Einstein Fusion**: 504% amplification proven

### **Key Features**
- **Emotional Intelligence**: Therapeutic relationship
- **Multi-modal**: Voice + Text + Context
- **Professional Adaptation**: Automatically detects context
- **Privacy-First**: Local processing
- **Universal Companion**: All-in-one AI assistant

---

## 🎯 SUCCESS METRICS

### **Immediate Success Indicators**
- [ ] Cursor AI can open MeeTARA folder without freezing
- [ ] Frontend dependencies restored and running
- [ ] Backend dependencies restored and running  
- [ ] Voice system operational
- [ ] WebSocket connections working
- [ ] Trinity architecture components accessible

### **Full System Success**
- [ ] Frontend loads on port 2025
- [ ] Backend API responds on port 8766
- [ ] WebSocket connection active on port 8765
- [ ] Voice server running on port 5000
- [ ] All Trinity components integrated and functional

---

## ⚠️ KNOWN ISSUES & SOLUTIONS

### **Cursor Freezing Issue**
- **Cause**: 10.4GB of GGUF model files
- **Solution**: .cursorignore file excluding *.gguf files
- **Backup Plan**: Temporarily move model files

### **Common Dependency Issues**
- **pyaudio**: Use `pipwin install pyaudio` if pip fails
- **Node modules**: Delete node_modules folder and reinstall
- **Python path**: Ensure correct Python environment

### **Port Conflicts**
- **Frontend (2025)**: Check if port is in use
- **Backend (8766)**: Kill existing processes if needed
- **Voice (5000)**: Ensure no conflicts with other services

---

## 🎬 FIRST ACTIONS CHECKLIST

```
□ Navigate to C:\Users\rames\Documents\meetara
□ Create .cursorignore file with GGUF exclusions
□ Restart Cursor AI and test folder access
□ Identify project structure (package.json, requirements.txt)
□ Install frontend dependencies (npm install)
□ Install backend dependencies (pip install)
□ Test each component individually
□ Validate Trinity architecture components
□ Document current system state
□ Plan next development steps
```

---

## 🔧 EMERGENCY COMMANDS

### **If Cursor Still Freezes**
```bash
# Move all GGUF files temporarily
mkdir C:\Users\rames\Documents\meetara_models_backup
robocopy "C:\Users\rames\Documents\meetara" "C:\Users\rames\Documents\meetara_models_backup" *.gguf /S /MOVE
```

### **Quick System Check**
```bash
# Check what's running on key ports
netstat -an | findstr ":2025"
netstat -an | findstr ":8766"
netstat -an | findstr ":8765"
netstat -an | findstr ":5000"
```

### **Dependency Verification**
```bash
# Python packages
pip list | findstr -i "fastapi\|torch\|transformers\|edge-tts"

# Node packages
npm list --depth=0
```

---

**🎯 MISSION**: Restore MeeTARA Trinity Complete system to full operational status
**📅 CONTEXT**: Trinity achieved June 20, 2025 - system inaccessible due to technical issues
**🚀 PRIORITY**: Fix Cursor access → Restore dependencies → Validate Trinity components

**START HERE**: Create .cursorignore file first, then restart Cursor and test access! 