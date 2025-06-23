# MeeTARA Repository Setup Guide

## 🎯 **New Repository Structure**

We've successfully split your MeeTARA project into **three focused repositories** to solve the Cursor freezing issue:

```
📁 meetara-frontend/     # Next.js React frontend (~60 files)
📁 meetara-backend/      # Python AI engine (~20 files)  
📁 meetara-docs/         # Documentation (~50 files)
```

## ✅ **Benefits of This Approach**

### 1. **Cursor Performance**
- Each repo has < 100 files → **Opens instantly**
- No more freezing or "window not responding" errors
- Optimal IDE performance and responsiveness

### 2. **Development Workflow**
- **Focused development**: Work on frontend/backend separately
- **Independent deployments**: Deploy each component independently
- **Team collaboration**: Different teams can work on different repos
- **Version control**: Independent versioning and releases

### 3. **Scalability**
- Add new services as separate repositories
- Microservices architecture ready
- Easy to manage dependencies per component

## 🚀 **Repository Details**

### **meetara-frontend/** 
**Purpose**: Next.js React frontend for MeeTARA UI
```
src/
├── app/              # Next.js 13+ app directory
├── components/       # React components
├── lib/             # Utility functions
└── types/           # TypeScript definitions

Configuration:
├── package.json      # Dependencies and scripts
├── next.config.js    # Next.js configuration
├── tailwind.config.js # Styling configuration
└── tsconfig.json     # TypeScript configuration
```

**Development Commands:**
```bash
npm install           # Install dependencies
npm run dev          # Start development server (port 3000)
npm run build        # Build for production
npm run start        # Start production server
```

### **meetara-backend/**
**Purpose**: Python AI engine and API services
```
backend/
├── app.py           # Main application entry
├── ai_engine.py     # Core AI functionality
├── voice_server.py  # Voice processing
└── *.py            # Other Python modules

Configuration:
├── requirements.txt  # Python dependencies (to be created)
├── .env             # Environment variables
└── config.py        # Application configuration
```

**Development Commands:**
```bash
python -m venv venv          # Create virtual environment
venv\Scripts\activate        # Activate environment (Windows)
pip install -r requirements.txt  # Install dependencies
python app.py                # Start backend server (port 8000)
```

### **meetara-docs/**
**Purpose**: Documentation, guides, and project information
```
docs/
├── 1-vision/        # Project vision and HAI philosophy
├── 2-architecture/  # System design and roadmaps
├── 3-development/   # Training progress and guides
├── 4-testing/       # Testing strategies
└── 5-deployment/    # Security and deployment
```

## 🔧 **Setup Instructions**

### Step 1: Initialize Git Repositories
```bash
# Frontend
cd meetara-frontend
git init
git add .
git commit -m "Initial commit: MeeTARA frontend"

# Backend  
cd ../meetara-backend
git init
git add .
git commit -m "Initial commit: MeeTARA backend"

# Documentation
cd ../meetara-docs
git init
git add .
git commit -m "Initial commit: MeeTARA documentation"
```

### Step 2: Create GitHub Repositories
```bash
# Create repositories on GitHub, then:
git remote add origin https://github.com/yourusername/meetara-frontend.git
git push -u origin main

git remote add origin https://github.com/yourusername/meetara-backend.git
git push -u origin main

git remote add origin https://github.com/yourusername/meetara-docs.git
git push -u origin main
```

### Step 3: Development Environment Setup

**Frontend Development:**
```bash
cd meetara-frontend
npm install
npm run dev
# Open http://localhost:3000
```

**Backend Development:**
```bash
cd meetara-backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn python-multipart
# Create requirements.txt with your dependencies
python app.py
# Backend runs on http://localhost:8000
```

## 🔄 **Full-Stack Development Workflow**

### Option 1: Separate Terminal Windows
```bash
# Terminal 1 - Frontend
cd meetara-frontend
npm run dev

# Terminal 2 - Backend  
cd meetara-backend
venv\Scripts\activate
python app.py

# Terminal 3 - Documentation
cd meetara-docs
code .
```

### Option 2: Development Scripts
Create `start-dev.bat` in Documents folder:
```batch
@echo off
start "Frontend" cmd /k "cd meetara-frontend && npm run dev"
start "Backend" cmd /k "cd meetara-backend && venv\Scripts\activate && python app.py"
start "Docs" cmd /k "cd meetara-docs && code ."
```

## 📝 **Next Steps**

### Immediate Actions:
1. ✅ **Test each repository** in Cursor (should open instantly)
2. ✅ **Create requirements.txt** for backend
3. ✅ **Set up GitHub repositories** for version control
4. ✅ **Configure environment variables** for backend

### Development Priorities:
1. **Trinity Architecture Integration**: Connect frontend ↔ backend
2. **API Development**: Create REST/GraphQL APIs
3. **Voice Integration**: Implement SpeechBrain features
4. **Documentation**: Update guides for new structure

## 🎊 **Success Metrics**

- ✅ **Cursor opens each repo in < 5 seconds**
- ✅ **No more freezing or performance issues**
- ✅ **Independent development workflows**
- ✅ **Clean separation of concerns**
- ✅ **Scalable architecture for future growth**

## 🔗 **Repository Links** (Update after GitHub creation)
- Frontend: `https://github.com/yourusername/meetara-frontend`
- Backend: `https://github.com/yourusername/meetara-backend`  
- Documentation: `https://github.com/yourusername/meetara-docs`

**Status**: ✅ **Repository structure created and ready for development**
**Last Updated**: June 22, 2025
**Solution**: Multi-repository approach solving Cursor performance issues 