#!/usr/bin/env python3
"""
TARA Universal Model - Simple Web Training Monitor
Real-time web dashboard for monitoring training progress at http://localhost:8001/
"""

from flask import Flask, jsonify, Response
import os
import json
import glob
from datetime import datetime
from pathlib import Path
import subprocess

app = Flask(__name__)

def check_adapter_status():
    """Check which domain adapters have been created"""
    adapters_dir = Path("models/adapters")
    domains = ["healthcare", "business", "education", "creative", "leadership"]
    
    status = {}
    for domain in domains:
        domain_dir = adapters_dir / domain
        if domain_dir.exists():
            files = list(domain_dir.glob("*"))
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

def check_training_data():
    """Check recent training data generation"""
    data_dir = Path("data/synthetic")
    if not data_dir.exists():
        return {}
    
    today = datetime.now().strftime("%Y%m%d")
    recent_files = {}
    
    for domain in ["healthcare", "business", "education", "creative", "leadership"]:
        pattern = f"{domain}_train_{today}_*.json"
        files = list(data_dir.glob(pattern))
        if files:
            latest = max(files, key=lambda x: x.stat().st_mtime)
            recent_files[domain] = {
                "file": latest.name,
                "size_mb": round(latest.stat().st_size / (1024*1024), 2),
                "modified": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S")
            }
    
    return recent_files

def check_python_processes():
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
    except Exception:
        return []

def get_training_status():
    """Get comprehensive training status"""
    adapter_status = check_adapter_status()
    training_data = check_training_data()
    processes = check_python_processes()
    
    completed = sum(1 for info in adapter_status.values() if info["status"] == "completed")
    total_memory_mb = sum(p.get("memory_mb", 0) for p in processes)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adapters": adapter_status,
        "training_data": training_data,
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
        }
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #667eea;
            margin: 0;
            font-size: 2.5em;
        }
        .status-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
        }
        .domain-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .domain-card {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .domain-card.completed {
            border-color: #28a745;
            background: #d4edda;
        }
        .domain-card.in_progress {
            border-color: #ffc107;
            background: #fff3cd;
        }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .auto-refresh {
            display: block;
            text-align: center;
            margin-top: 10px;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 TARA Universal Model</h1>
            <p>Simple Training Monitor Dashboard</p>
            <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
            <div class="auto-refresh">Auto-refreshes every 10 seconds</div>
        </div>
        <div id="content">Loading...</div>
    </div>
    <script>
        function loadData() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const domainsHtml = Object.entries(data.adapters).map(([domain, info]) => `
                        <div class="domain-card ${info.status}">
                            <h3>${domain.charAt(0).toUpperCase() + domain.slice(1)}</h3>
                            <p>Status: ${info.status.replace('_', ' ')}</p>
                            <p>Files: ${info.files}</p>
                        </div>
                    `).join('');
                    
                    document.getElementById('content').innerHTML = `
                        <div class="status-card">
                            <h2>📊 Overall Progress</h2>
                            <div class="metric">
                                <div class="metric-value">${data.progress.percentage.toFixed(1)}%</div>
                                <div class="metric-label">Complete</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.progress.completed}/${data.progress.total}</div>
                                <div class="metric-label">Domains</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.system.total_memory_mb.toFixed(1)} MB</div>
                                <div class="metric-label">Memory Usage</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.system.training_active ? '🟢 ACTIVE' : '🔴 IDLE'}</div>
                                <div class="metric-label">Training Status</div>
                            </div>
                        </div>
                        <div class="status-card">
                            <h2>🎯 Domain Status</h2>
                            <div class="domain-grid">${domainsHtml}</div>
                        </div>
                        <div class="status-card">
                            <h2>⏱️ Last Updated</h2>
                            <p>${data.timestamp}</p>
                        </div>
                    `;
                })
                .catch(err => {
                    document.getElementById('content').innerHTML = `
                        <div class="status-card">
                            <h2>⚠️ Error</h2>
                            <p>Failed to load data. Please try again.</p>
                        </div>
                    `;
                    console.error(err);
                });
        }
        
        // Load data immediately
        loadData();
        
        // Auto-refresh every 10 seconds
        setInterval(loadData, 10000);
    </script>
</body>
</html>'''
    return Response(html, mimetype='text/html')

@app.route('/api/status')
def api_status():
    """API endpoint for training status"""
    status_data = get_training_status()
    
    # Add additional fields needed by the dashboard
    domain_config = get_domain_config()
    model_mapping = get_model_mapping()
    all_base_models = get_all_base_models()
    
    # Prepare data for model availability
    available_models_info = {}
    for model_name in all_base_models:
        available_models_info[model_name] = True  # Mark as available
    
    # Add explicitly defined models if not already found in GGUF folder
    explicit_models = ["DialoGPT-medium", "Phi-3.5-mini-instruct", "Qwen2.5-3B-Instruct"]
    for model in explicit_models:
        if model not in available_models_info:
            available_models_info[model] = os.path.exists(os.path.join(BASE_MODEL_DIR, model))
    
    # Get domain statuses
    domain_statuses = []
    for domain, details in domain_config.items():
        base_model_for_domain = details.get("base_model", "Unknown")
        adapter_trained = check_adapter_status(domain, base_model_for_domain)
        
        status_text = "Training Complete" if adapter_trained else "Adapter Missing"
        if not adapter_trained and base_model_for_domain not in all_base_models:
            status_text = f"Base Model Missing ({base_model_for_domain})"
        
        domain_statuses.append({
            "domain": domain,
            "base_model": base_model_for_domain,
            "adapter_trained": adapter_trained,
            "status": status_text
        })
    
    # Combine all data
    status_data.update({
        "domain_statuses": domain_statuses,
        "model_mapping": model_mapping,
        "available_base_models": available_models_info,
        "all_base_models_found_in_gguf": all_base_models
    })
    
    return jsonify(status_data)

if __name__ == '__main__':
    print("🌐 Starting TARA Universal Model Simple Web Monitor")
    print("🚀 Dashboard available at: http://localhost:8001/")
    print("ℹ️  Note: Full dashboard is running on port 8000")
    app.run(host='0.0.0.0', port=8001, debug=False) 