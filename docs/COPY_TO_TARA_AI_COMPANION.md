# 📁 Files to Copy to tara-ai-companion Project

## 🎯 **INTEGRATION FILES READY**

The following files have been created and are ready to be copied to your **tara-ai-companion** project:

---

## 📋 **COPY INSTRUCTIONS**

### **1. Copy Voice API Client**
```bash
# Copy from: docs/voice-api.ts
# Copy to: tara-ai-companion/apps/web-ui/src/lib/voice-api.ts
```

### **2. Copy Voice React Hook**
```bash
# Copy from: docs/useVoice.ts  
# Copy to: tara-ai-companion/apps/web-ui/src/hooks/useVoice.ts
```

### **3. Copy Integration Guide**
```bash
# Copy from: docs/CURSOR_TARA_AI_COMPANION_PROMPT.md
# Copy to: tara-ai-companion/VOICE_INTEGRATION_PROMPT.md
```

---

## 🚀 **QUICK START COMMANDS**

### **For Windows PowerShell:**
```powershell
# Navigate to tara-ai-companion project
cd ..\tara-ai-companion\apps\web-ui

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path "src\lib"
New-Item -ItemType Directory -Force -Path "src\hooks"

# Copy files (adjust paths as needed)
Copy-Item "..\..\tara-universal-model\docs\voice-api.ts" "src\lib\voice-api.ts"
Copy-Item "..\..\tara-universal-model\docs\useVoice.ts" "src\hooks\useVoice.ts"
Copy-Item "..\..\tara-universal-model\docs\CURSOR_TARA_AI_COMPANION_PROMPT.md" "VOICE_INTEGRATION_PROMPT.md"
```

### **For Linux/Mac:**
```bash
# Navigate to tara-ai-companion project
cd ../tara-ai-companion/apps/web-ui

# Create directories if they don't exist
mkdir -p src/lib src/hooks

# Copy files (adjust paths as needed)
cp ../../tara-universal-model/docs/voice-api.ts src/lib/voice-api.ts
cp ../../tara-universal-model/docs/useVoice.ts src/hooks/useVoice.ts
cp ../../tara-universal-model/docs/CURSOR_TARA_AI_COMPANION_PROMPT.md VOICE_INTEGRATION_PROMPT.md
```

---

## 🎯 **NEXT STEPS AFTER COPYING**

### **1. Start TARA Voice Server**
```bash
# In tara-universal-model directory
python -m tara_universal_model.api.voice_server
```

### **2. Start Frontend Development**
```bash
# In tara-ai-companion/apps/web-ui directory
npm run dev
```

### **3. Open Cursor AI with Integration Prompt**
1. Open `VOICE_INTEGRATION_PROMPT.md` in Cursor AI
2. Use the prompt to guide voice integration into MeeTARA.tsx
3. Follow the step-by-step integration instructions

---

## 📚 **INTEGRATION OVERVIEW**

### **What You'll Be Integrating:**

1. **Voice API Client** (`src/lib/voice-api.ts`)
   - TypeScript client for TARA voice server
   - Handles TTS synthesis and chat with voice
   - Domain-specific voice configurations
   - Error handling and status checking

2. **Voice React Hook** (`src/hooks/useVoice.ts`)
   - Complete state management for voice functionality
   - Audio playback controls
   - Server availability checking
   - Error handling and loading states

3. **Integration Target** (`src/components/MeeTARA.tsx`)
   - Add voice toggle button
   - Integrate voice chat functionality
   - Add audio playback for TARA responses
   - Add voice status indicators

---

## 🎨 **EXPECTED RESULT**

After integration, users will be able to:

✅ **Toggle voice on/off** in MeeTARA chat header  
✅ **Hear TARA responses** automatically when voice is enabled  
✅ **Replay any TARA message** with voice  
✅ **Switch voice domains** (Healthcare, Business, Education, Creative, Leadership, Universal)  
✅ **See voice status** (enabled, playing, loading, error)  
✅ **Graceful fallback** to text-only when voice server unavailable  

---

## 🔧 **TESTING CHECKLIST**

After integration, test these scenarios:

- [ ] Voice toggle button works
- [ ] TARA responses are spoken automatically
- [ ] Message replay buttons work
- [ ] Domain switching changes voice personality
- [ ] Error handling shows appropriate messages
- [ ] Loading states are visible
- [ ] Voice works with existing text chat
- [ ] Graceful fallback when voice server offline

---

## 🎉 **SUCCESS CRITERIA**

**Integration is complete when:**
- Voice toggle appears in MeeTARA header
- TARA speaks responses automatically when voice enabled
- Users can replay any TARA message
- Voice works seamlessly alongside text chat
- Error handling provides clear feedback
- Loading states are visible during voice generation

---

**🚀 Ready to bring TARA's voice to the frontend! Copy the files and start integrating!** 