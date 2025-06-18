#!/usr/bin/env python3
"""
Simple TARA Training Monitor Dashboard
Real-time monitoring of TARA model training progress
"""

import os
import json
import time
import glob
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import psutil

app = FastAPI(title="TARA Training Monitor", version="1.0.0")

# Setup templates
template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 TARA Training Monitor</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.1); 
            padding: 30px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .status-card { 
            background: rgba(255,255,255,0.2); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 15px 0; 
        }
        .progress-bar { 
            width: 100%; 
            height: 25px; 
            background: rgba(255,255,255,0.3); 
            border-radius: 12px; 
            overflow: hidden; 
        }
        .progress-fill { 
            height: 100%; 
            background: linear-gradient(90deg, #4CAF50, #8BC34A); 
            transition: width 0.3s ease; 
        }
        .metric { 
            display: inline-block; 
            margin: 10px 15px; 
            text-align: center; 
        }
        .metric-value { 
            font-size: 2em; 
            font-weight: bold; 
            display: block; 
        }
        .metric-label { 
            font-size: 0.9em; 
            opacity: 0.8; 
        }
        .process-list { 
            background: rgba(0,0,0,0.3); 
            padding: 15px; 
            border-radius: 8px; 
            font-family: monospace; 
        }
        .refresh-btn { 
            background: #4CAF50; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            font-size: 16px; 
        }
        .refresh-btn:hover { 
            background: #45a049; 
        }
        .domain-status {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .domain-card {
            background: rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .domain-card.active {
            background: rgba(76, 175, 80, 0.3);
            border: 2px solid #4CAF50;
        }
        .domain-card.completed {
            background: rgba(76, 175, 80, 0.2);
        }
    </style>
    <script>
        function refreshData() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status-data').innerHTML = JSON.stringify(data, null, 2);
                    updateProgressBar(data.overall_progress || 0);
                    updateDomainStatus(data.domains || {});
                });
        }
        
        function updateProgressBar(progress) {
            const progressBar = document.querySelector('.progress-fill');
            if (progressBar) {
                progressBar.style.width = progress + '%';
            }
        }
        
        function updateDomainStatus(domains) {
            const container = document.querySelector('.domain-status');
            if (container && domains) {
                container.innerHTML = '';
                for (const [domain, status] of Object.entries(domains)) {
                    const card = document.createElement('div');
                    card.className = `domain-card ${status.status}`;
                    card.innerHTML = `
                        <h3>${domain.charAt(0).toUpperCase() + domain.slice(1)}</h3>
                        <p>Status: ${status.status}</p>
                        <p>Progress: ${status.progress}%</p>
                    `;
                    container.appendChild(card);
                }
            }
        }
        
        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
        
        // Initial load
        window.onload = refreshData;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 TARA Universal Model Training Monitor</h1>
            <p>Real-time monitoring of AI model training progress</p>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
        </div>
        
        <div class="status-card">
            <h2>📊 Overall Training Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-value" id="progress-percent">0%</span>
                <span class="metric-label">Complete</span>
            </div>
            <div class="metric">
                <span class="metric-value" id="active-processes">-</span>
                <span class="metric-label">Active Processes</span>
            </div>
            <div class="metric">
                <span class="metric-value" id="current-domain">-</span>
                <span class="metric-label">Current Domain</span>
            </div>
        </div>
        
        <div class="status-card">
            <h2>🎯 Domain Training Status</h2>
            <div class="domain-status">
                <!-- Domain cards will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="status-card">
            <h2>🔍 System Status</h2>
            <div class="process-list">
                <pre id="status-data">Loading...</pre>
            </div>
        </div>
        
        <div class="status-card">
            <h2>📈 Training Metrics</h2>
            <div class="metric">
                <span class="metric-value">5</span>
                <span class="metric-label">Total Domains</span>
            </div>
            <div class="metric">
                <span class="metric-value">1000</span>
                <span class="metric-label">Samples/Domain</span>
            </div>
            <div class="metric">
                <span class="metric-value">~7.5h</span>
                <span class="metric-label">Est. Total Time</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

def get_training_status():
    """Get current training status by analyzing files and processes."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "overall_progress": 0,
        "current_domain": None,
        "active_processes": 0,
        "domains": {
            "healthcare": {"status": "pending", "progress": 0},
            "business": {"status": "pending", "progress": 0},
            "education": {"status": "pending", "progress": 0},
            "creative": {"status": "pending", "progress": 0},
            "leadership": {"status": "pending", "progress": 0}
        },
        "latest_files": [],
        "python_processes": []
    }
    
    # Check for Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            if proc.info['name'] == 'python.exe':
                python_processes.append({
                    'pid': proc.info['pid'],
                    'memory_mb': round(proc.info['memory_info'].rss / 1024 / 1024, 1),
                    'cpu_percent': proc.info['cpu_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    status["python_processes"] = python_processes
    status["active_processes"] = len(python_processes)
    
    # Check latest synthetic data files
    data_dir = Path("data/synthetic")
    if data_dir.exists():
        json_files = list(data_dir.glob("*.json"))
        if json_files:
            latest_files = sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            status["latest_files"] = [
                {
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                }
                for f in latest_files
            ]
            
            # Determine current domain from latest file
            if latest_files:
                latest_name = latest_files[0].name
                for domain in status["domains"].keys():
                    if domain in latest_name:
                        status["current_domain"] = domain
                        status["domains"][domain]["status"] = "active"
                        break
    
    # Check for completed models
    models_dir = Path("models/adapters")
    if models_dir.exists():
        for domain in status["domains"].keys():
            domain_dir = models_dir / domain
            if domain_dir.exists() and any(domain_dir.iterdir()):
                status["domains"][domain]["status"] = "completed"
                status["domains"][domain]["progress"] = 100
    
    # Calculate overall progress
    completed_domains = sum(1 for d in status["domains"].values() if d["status"] == "completed")
    status["overall_progress"] = (completed_domains / 5) * 100
    
    return status

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    return HTMLResponse(content=template_content)

@app.get("/api/status")
async def get_status():
    """API endpoint for training status."""
    return JSONResponse(content=get_training_status())

if __name__ == "__main__":
    print("🚀 Starting TARA Training Monitor Dashboard")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔄 Auto-refreshes every 10 seconds")
    uvicorn.run(app, host="0.0.0.0", port=8000) 