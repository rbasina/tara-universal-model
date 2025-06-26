#!/usr/bin/env python3
"""
TARA Universal Model - Simple Training Monitor
Provides a clean, efficient dashboard for monitoring training progress
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
BASE_MODEL_DIR = "models"

def get_domain_status():
    """Check which domain adapters have been created"""
    adapters_dir = Path("models/adapters")
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    
    status = {}
    for domain in domains:
        domain_files = list(adapters_dir.glob(f"{domain}*"))
        if domain_files:
            files = []
            for domain_dir in domain_files:
                if domain_dir.is_dir():
                    files.extend(list(domain_dir.glob("*")))
            
            if files:
                has_model = any(f.name in ["pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"] for f in files)
                has_config = any(f.name in ["config.json", "adapter_config.json"] for f in files)
                status[domain] = {
                    "status": "completed" if (has_model and has_config) else "in_progress",
                    "files": len(files)
                }
            else:
                status[domain] = {"status": "empty", "files": 0}
        else:
            status[domain] = {"status": "not_started", "files": 0}
    
    return status

def get_training_state():
    """Get training state from state files"""
    training_state_dir = Path("training_state")
    if not training_state_dir.exists():
        return {}
    
    # Check overall training state
    overall_state_file = training_state_dir / "overall_training_state.json"
    if not overall_state_file.exists():
        return {}
    
    try:
        with open(overall_state_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading training state: {e}")
        return {}

def get_python_processes():
    """Check running Python processes (Windows)"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                processes = []
                for line in lines[1:]:
                    parts = [p.strip('"') for p in line.split('","')]
                    if len(parts) >= 5:
                        memory_kb = parts[4].replace(',', '').replace(' K', '')
                        processes.append({
                            "pid": parts[1],
                            "memory_mb": round(int(memory_kb) / 1024, 1)
                        })
                return processes
        return []
    except Exception as e:
        logger.error(f"Error checking processes: {e}")
        return []

def get_training_status():
    """Get comprehensive training status"""
    domain_status = get_domain_status()
    training_state = get_training_state()
    processes = get_python_processes()
    
    # Update domain status with training state information
    if training_state:
        completed_domains = training_state.get("completed_domains", [])
        current_domain = training_state.get("current_domain")
        pending_domains = training_state.get("pending_domains", [])
        
        for domain in domain_status:
            if domain in completed_domains:
                domain_status[domain]["status"] = "completed"
            elif domain == current_domain:
                domain_status[domain]["status"] = "in_progress"
            elif domain in pending_domains:
                domain_status[domain]["status"] = "pending"
    
    # Calculate metrics
    completed = sum(1 for info in domain_status.values() if info["status"] == "completed")
    total_memory_mb = sum(p.get("memory_mb", 0) for p in processes)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domains": domain_status,
        "processes": processes,
        "progress": {
            "completed": completed,
            "total": 5,
            "percentage": (completed / 5) * 100
        },
        "system": {
            "total_memory_mb": total_memory_mb,
            "active_processes": len(processes),
            "training_active": total_memory_mb > 100
        },
        "training_state": training_state
    }

@app.route('/')
def dashboard():
    """Main dashboard page with inline HTML"""
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA Universal Model - Training Monitor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(90deg, #4b6cb7, #182848);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-content {
            flex: 1;
        }
        .header h1 {
            font-size: 1.8rem;
            margin-bottom: 5px;
        }
        .header p {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            margin-left: 20px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background: #4cd137;
            box-shadow: 0 0 10px #4cd137;
        }
        .status-idle {
            background: #e84118;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .progress-container {
            display: grid;
            grid-template-columns: 1fr 200px;
            gap: 20px;
            margin-bottom: 20px;
        }
        .progress-bar-container {
            height: 10px;
            background: #f0f0f0;
            border-radius: 5px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #4b6cb7, #182848);
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        .domains-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 15px;
        }
        .domain-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #ddd;
        }
        .domain-card h3 {
            font-size: 1.1rem;
            margin-bottom: 10px;
        }
        .domain-card.completed {
            border-left-color: #4cd137;
        }
        .domain-card.in_progress {
            border-left-color: #fbc531;
        }
        .domain-card.not_started {
            border-left-color: #dcdde1;
        }
        .stats {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }
        .stat {
            background: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            flex: 1;
            min-width: 120px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4b6cb7;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #666;
            margin-top: 5px;
        }
        .domain-info {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        .domain-model {
            color: #666;
        }
        .domain-progress {
            font-weight: bold;
        }
        .btn {
            background: #4b6cb7;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #3a5a9f;
        }
        .btn-refresh {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .refresh-icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid white;
            border-radius: 50%;
            border-top-color: transparent;
        }
        .refresh-active {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .auto-refresh {
            background: #f0f2f5;
            border-radius: 20px;
            padding: 5px 10px;
            font-size: 0.8rem;
            display: inline-block;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>TARA Universal Model</h1>
                <p>Training Progress Monitor</p>
                <div class="auto-refresh">Auto-refreshes every 5 seconds</div>
            </div>
            <div id="status-indicator" class="status-indicator">
                <div id="status-dot" class="status-dot"></div>
                <span id="status-text">Checking...</span>
            </div>
        </div>

        <div class="card">
            <h2>📊 Training Progress</h2>
            <div class="progress-container">
                <div>
                    <div class="progress-bar-container">
                        <div id="progress-bar" class="progress-bar" style="width: 0%"></div>
                    </div>
                    <div id="progress-text">Loading...</div>
                </div>
                <button id="refresh-btn" class="btn btn-refresh">
                    <span class="refresh-icon" id="refresh-icon"></span>
                    Refresh Data
                </button>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div id="completed-domains" class="stat-value">-</div>
                    <div class="stat-label">Completed Domains</div>
                </div>
                <div class="stat">
                    <div id="memory-usage" class="stat-value">-</div>
                    <div class="stat-label">Memory Usage</div>
                </div>
                <div class="stat">
                    <div id="active-processes" class="stat-value">-</div>
                    <div class="stat-label">Active Processes</div>
                </div>
                <div class="stat">
                    <div id="last-updated" class="stat-value">-</div>
                    <div class="stat-label">Last Updated</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Domain Status</h2>
            <div id="domains-grid" class="domains-grid">
                <div class="domain-card">
                    <h3>Loading domains...</h3>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Elements
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const completedDomains = document.getElementById('completed-domains');
        const memoryUsage = document.getElementById('memory-usage');
        const activeProcesses = document.getElementById('active-processes');
        const lastUpdated = document.getElementById('last-updated');
        const domainsGrid = document.getElementById('domains-grid');
        const refreshBtn = document.getElementById('refresh-btn');
        const refreshIcon = document.getElementById('refresh-icon');

        // API endpoint
        const apiUrl = '/api/status';

        // Load data function
        async function loadData() {
            refreshIcon.classList.add('refresh-active');
            
            try {
                const response = await fetch(apiUrl);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
                showError();
            } finally {
                refreshIcon.classList.remove('refresh-active');
            }
        }

        // Update dashboard with data
        function updateDashboard(data) {
            // Update status indicator
            const isActive = data.system.training_active;
            statusDot.className = 'status-dot ' + (isActive ? 'status-active' : 'status-idle');
            statusText.textContent = isActive ? 'Training Active' : 'Training Idle';
            
            // Update progress
            const progress = data.progress.percentage;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}% Complete (${data.progress.completed}/${data.progress.total} domains)`;
            
            // Update stats
            completedDomains.textContent = `${data.progress.completed}/${data.progress.total}`;
            memoryUsage.textContent = `${Math.round(data.system.total_memory_mb)} MB`;
            activeProcesses.textContent = data.system.active_processes;
            
            // Format time as HH:MM:SS
            const time = data.timestamp.split(' ')[1];
            lastUpdated.textContent = time;
            
            // Update domains grid
            const domains = data.domains;
            
            if (domains) {
                domainsGrid.innerHTML = '';
                
                // Define base models for domains
                const baseModels = {
                    "healthcare": "DialoGPT-medium",
                    "business": "DialoGPT-medium",
                    "education": "Qwen2.5-3B-Instruct",
                    "creative": "Qwen2.5-3B-Instruct",
                    "leadership": "Qwen2.5-3B-Instruct"
                };
                
                // Get current domain from training state
                let currentDomain = "";
                if (data.training_state && data.training_state.current_domain) {
                    currentDomain = data.training_state.current_domain;
                }
                
                Object.entries(domains).forEach(([domain, info]) => {
                    const domainCard = document.createElement('div');
                    domainCard.className = `domain-card ${info.status}`;
                    
                    // Get base model for this domain
                    let baseModel = baseModels[domain] || "Unknown";
                    
                    // Format status
                    let statusText = info.status.replace('_', ' ');
                    statusText = statusText.charAt(0).toUpperCase() + statusText.slice(1);
                    
                    // Add "Current" indicator if this is the active domain
                    if (domain === currentDomain) {
                        statusText += " (Active)";
                    }
                    
                    domainCard.innerHTML = `
                        <h3>${domain.charAt(0).toUpperCase() + domain.slice(1)}</h3>
                        <div class="domain-info">
                            <span class="domain-model">${baseModel}</span>
                            <span class="domain-progress">${statusText}</span>
                        </div>
                    `;
                    
                    domainsGrid.appendChild(domainCard);
                });
            }
        }

        // Show error state
        function showError() {
            statusDot.className = 'status-dot status-idle';
            statusText.textContent = 'Connection Error';
            progressText.textContent = 'Failed to load data. Check if the server is running.';
        }

        // Event listeners
        refreshBtn.addEventListener('click', loadData);

        // Initial load
        loadData();

        // Auto-refresh every 5 seconds
        setInterval(loadData, 5000);
    </script>
</body>
</html>'''
    return Response(html, mimetype='text/html')

@app.route('/api/status')
def api_status():
    """API endpoint for training status"""
    return jsonify(get_training_status())

@app.route('/status')
def status():
    """Simple status endpoint for monitoring"""
    return jsonify(get_training_status())

@app.route('/recover_training', methods=['POST'])
def recover_training():
    """Endpoint to trigger training recovery"""
    try:
        # Check if recovery state exists
        recovery_file = Path("training_recovery_state.json")
        if not recovery_file.exists():
            return jsonify({
                "success": False,
                "error": "No recovery state found. Training may not have been interrupted."
            })
        
        # Load recovery state
        with open(recovery_file, "r") as f:
            state = json.load(f)
        
        # Start recovery process
        recovery_script = Path("scripts/monitoring/training_recovery.py")
        if not recovery_script.exists():
            return jsonify({
                "success": False,
                "error": "Recovery script not found at scripts/monitoring/training_recovery.py"
            })
        
        # Run recovery script with --auto_resume flag
        subprocess.Popen([sys.executable, str(recovery_script), "--auto_resume"])
        
        # Get checkpoint information
        domains = state.get("domains", "").split(",")
        checkpoints = state.get("checkpoints", {})
        checkpoints_found = sum(1 for cp in checkpoints.values() if cp)
        
        return jsonify({
            "success": True,
            "domains": domains,
            "model": state.get("model", "unknown"),
            "checkpoints_found": checkpoints_found,
            "message": f"Recovery initiated for {len(domains)} domains with {checkpoints_found} checkpoints"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    print("🌐 Starting TARA Universal Model Training Monitor")
    print("🚀 Dashboard available at: http://localhost:8001/")
    app.run(host='0.0.0.0', port=8001, debug=False) 