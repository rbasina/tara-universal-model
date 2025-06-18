#!/usr/bin/env python3
"""
TARA Universal Model - Web Training Monitor
Real-time web dashboard for monitoring training progress at http://localhost:8000/
"""

from flask import Flask, render_template, jsonify
import os
import json
import glob
import time
from datetime import datetime
from pathlib import Path
import subprocess
import threading

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
                # Check for key files that indicate successful training
                has_model = any(f.name in ["pytorch_model.bin", "adapter_model.bin", "adapter_model.safetensors"] for f in files)
                has_config = any(f.name in ["config.json", "adapter_config.json"] for f in files)
                status[domain] = {
                    "status": "completed" if (has_model and has_config) else "in_progress",
                    "files": len(files),
                    "file_list": [f.name for f in files[:10]]  # Show first 10 files
                }
            else:
                status[domain] = {"status": "empty", "files": 0, "file_list": []}
        else:
            status[domain] = {"status": "not_started", "files": 0, "file_list": []}
    
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
                "modified": datetime.fromtimestamp(latest.stat().st_mtime).strftime("%H:%M:%S"),
                "full_path": str(latest)
            }
    
    return recent_files

def check_training_summaries():
    """Check recent training summary files"""
    summaries = glob.glob("training_summary_*.json")
    if not summaries:
        return None
    
    latest = max(summaries, key=lambda x: os.path.getmtime(x))
    
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        return {
            "file": latest,
            "timestamp": data.get("timestamp", "unknown"),
            "successful_domains": data.get("successful_domains", []),
            "failed_domains": data.get("failed_domains", []),
            "total_time_hours": data.get("total_time_hours", 0),
            "details": data
        }
    except Exception as e:
        return {"error": str(e)}

def check_python_processes():
    """Check running Python processes (Windows)"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Header + data
                processes = []
                for line in lines[1:]:  # Skip header
                    parts = [p.strip('"') for p in line.split('","')]
                    if len(parts) >= 5:
                        memory_kb = parts[4].replace(',', '').replace(' K', '')
                        processes.append({
                            "name": parts[0],
                            "pid": parts[1],
                            "memory_kb": memory_kb,
                            "memory_mb": round(int(memory_kb) / 1024, 1)
                        })
                return processes
        return []
    except Exception as e:
        return [{"error": str(e)}]

def get_training_status():
    """Get comprehensive training status"""
    adapter_status = check_adapter_status()
    training_data = check_training_data()
    summary = check_training_summaries()
    processes = check_python_processes()
    
    # Calculate progress
    completed = sum(1 for info in adapter_status.values() if info["status"] == "completed")
    in_progress = sum(1 for info in adapter_status.values() if info["status"] in ["in_progress", "empty"])
    
    # Total memory usage
    total_memory_mb = sum(p.get("memory_mb", 0) for p in processes if isinstance(p.get("memory_mb"), (int, float)))
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "adapters": adapter_status,
        "training_data": training_data,
        "summary": summary,
        "processes": processes,
        "progress": {
            "completed": completed,
            "in_progress": in_progress,
            "total": 5,
            "percentage": (completed / 5) * 100
        },
        "system": {
            "total_memory_mb": total_memory_mb,
            "active_processes": len(processes),
            "training_active": total_memory_mb > 50  # Assume training if using >50MB
        }
    }

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for training status"""
    return jsonify(get_training_status())

@app.route('/api/logs')
def api_logs():
    """API endpoint for recent logs"""
    try:
        # Get recent log files
        log_files = glob.glob("logs/*.log")
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
            with open(latest_log, 'r') as f:
                # Get last 50 lines
                lines = f.readlines()[-50:]
                return jsonify({"logs": lines, "file": latest_log})
        return jsonify({"logs": [], "file": None})
    except Exception as e:
        return jsonify({"error": str(e), "logs": [], "file": None})

if __name__ == '__main__':
    # Create templates directory and HTML template
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''<!DOCTYPE html>
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
        .header p {
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status-card h3 {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.3em;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #28a745, #20c997);
            height: 100%;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }
        .domain-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .domain-item:last-child {
            border-bottom: none;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-completed { background: #d4edda; color: #155724; }
        .status-in_progress { background: #fff3cd; color: #856404; }
        .status-empty { background: #f8d7da; color: #721c24; }
        .status-not_started { background: #e2e3e5; color: #383d41; }
        .process-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #f1f3f4;
            border-radius: 6px;
            margin: 5px 0;
        }
        .memory-high { background: #ffe0e0; }
        .memory-medium { background: #fff0e0; }
        .memory-low { background: #e0f0e0; }
        .refresh-info {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-size: 1.1em;
            padding: 20px;
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-bottom: 20px;
        }
        .hai-quote {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 TARA Universal Model</h1>
            <p>Training Monitor Dashboard</p>
        </div>
        
        <div class="hai-quote">
            <strong>"Technology that Amplifies Rather than Replaces Abilities"</strong><br>
            Building HAI across Healthcare, Business, Education, Creative & Leadership domains
        </div>
        
        <div class="timestamp" id="timestamp">Loading...</div>
        
        <div id="content" class="loading">
            Loading training status...
        </div>
        
        <div class="refresh-info">
            Auto-refreshing every 10 seconds
        </div>
    </div>

    <script>
        function formatBytes(bytes) {
            if (bytes === 0) return '0 MB';
            const mb = bytes / (1024 * 1024);
            return mb.toFixed(1) + ' MB';
        }
        
        function getStatusBadge(status) {
            const badges = {
                'completed': '✅ COMPLETED', 
                'in_progress': '🔄 IN PROGRESS',
                'empty': '⏳ WAITING',
                'not_started': '❌ NOT STARTED'
            };
            return badges[status] || '❓ UNKNOWN';
        }
        
        function getMemoryClass(memoryMb) {
            if (memoryMb > 200) return 'memory-high';
            if (memoryMb > 50) return 'memory-medium';
            return 'memory-low';
        }
        
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('timestamp').textContent = 
                        'Last Updated: ' + data.timestamp;
                    
                    const content = document.getElementById('content');
                    content.innerHTML = `
                        <div class="status-grid">
                            <div class="status-card">
                                <h3>📊 Training Progress</h3>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${data.progress.percentage}%">
                                        ${data.progress.percentage.toFixed(1)}%
                                    </div>
                                </div>
                                <p>
                                    <strong>${data.progress.completed}/5</strong> domains completed<br>
                                    <strong>${data.progress.in_progress}/5</strong> domains in progress
                                </p>
                                ${data.progress.completed === 5 ? 
                                    '<p style="color: #28a745; font-weight: bold;">🎉 All domains completed!</p>' : 
                                    '<p style="color: #667eea;">⏳ Training in progress...</p>'
                                }
                            </div>
                            
                            <div class="status-card">
                                <h3>📁 Domain Adapters</h3>
                                ${Object.entries(data.adapters).map(([domain, info]) => `
                                    <div class="domain-item">
                                        <span><strong>${domain.charAt(0).toUpperCase() + domain.slice(1)}</strong></span>
                                        <span class="status-badge status-${info.status}">
                                            ${getStatusBadge(info.status)}
                                        </span>
                                    </div>
                                    ${info.files > 0 ? `<div style="font-size: 0.8em; color: #666; margin-left: 10px;">${info.files} files</div>` : ''}
                                `).join('')}
                            </div>
                            
                            <div class="status-card">
                                <h3>🐍 Python Processes</h3>
                                <p><strong>Total Memory:</strong> ${data.system.total_memory_mb.toFixed(1)} MB</p>
                                <p><strong>Active Processes:</strong> ${data.system.active_processes}</p>
                                ${data.processes.map(proc => `
                                    <div class="process-item ${getMemoryClass(proc.memory_mb)}">
                                        <span><strong>PID ${proc.pid}</strong></span>
                                        <span>${proc.memory_mb} MB</span>
                                    </div>
                                `).join('')}
                                ${data.system.training_active ? 
                                    '<p style="color: #28a745;">🚀 Training actively running!</p>' : 
                                    '<p style="color: #dc3545;">⚠️ Low activity detected</p>'
                                }
                            </div>
                            
                            <div class="status-card">
                                <h3>📄 Training Data</h3>
                                ${Object.keys(data.training_data).length === 0 ? 
                                    '<p>No recent training data found</p>' :
                                    Object.entries(data.training_data).map(([domain, info]) => `
                                        <div class="domain-item">
                                            <span><strong>${domain.charAt(0).toUpperCase() + domain.slice(1)}</strong></span>
                                            <span>${info.size_mb} MB (${info.modified})</span>
                                        </div>
                                    `).join('')
                                }
                            </div>
                        </div>
                        
                        ${data.summary ? `
                            <div class="status-card">
                                <h3>📈 Latest Training Summary</h3>
                                <p><strong>File:</strong> ${data.summary.file}</p>
                                <p><strong>Timestamp:</strong> ${data.summary.timestamp}</p>
                                <p><strong>Successful:</strong> ${data.summary.successful_domains.length} domains</p>
                                <p><strong>Failed:</strong> ${data.summary.failed_domains.length} domains</p>
                                ${data.summary.failed_domains.length > 0 ? 
                                    `<p><strong>Failed Domains:</strong> ${data.summary.failed_domains.join(', ')}</p>` : ''
                                }
                            </div>
                        ` : ''}
                    `;
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                    document.getElementById('content').innerHTML = 
                        '<div class="loading">Error loading data. Retrying...</div>';
                });
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every 10 seconds
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>'''
    
    # Write the template file
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("🌐 Starting TARA Universal Model Web Monitor")
    print("🚀 Dashboard available at: http://localhost:8000/")
    print("📊 Real-time training progress monitoring")
    print("⏱️  Auto-refreshing every 10 seconds")
    print("\nPress Ctrl+C to stop the web server")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False) 