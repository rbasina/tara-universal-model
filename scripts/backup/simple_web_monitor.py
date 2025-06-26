import os
import json
import glob
from flask import Flask, render_template, jsonify
import subprocess
from datetime import datetime

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'templates'))

# Configuration paths - adjust as necessary for your project structure
BASE_MODEL_DIR = "models"
ADAPTER_DIR = "models"
CONFIG_DIR = "configs"

# --- HELPER FUNCTIONS ---

def get_domain_config():
    """Loads the universal domain configuration from YAML."""
    config_path = os.path.join(CONFIG_DIR, "universal_domains.yaml")
    try:
        with open(config_path, 'r') as f:
            # A simple way to parse YAML without an extra dependency
            # For more complex YAML, consider PyYAML
            content = f.read()
            config = {}
            current_domain = None
            for line in content.split('\n'):
                line = line.strip()
                if line.endswith(':') and not line.startswith(' '):
                    current_domain = line[:-1]
                    config[current_domain] = {}
                elif current_domain and ':' in line:
                    key, value = line.split(':', 1)
                    config[current_domain][key.strip()] = value.strip().strip("'\"") # Corrected strip
            return config
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        return {}
    except Exception as e:
        print(f"Error reading domain config: {e}")
        return {}

def get_model_mapping():
    """Loads the model mapping from JSON."""
    mapping_path = os.path.join(CONFIG_DIR, "model_mapping.json")
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {mapping_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {mapping_path}. Check file format.")
        return {}
    except Exception as e:
        print(f"Error reading model mapping: {e}")
        return {}

def check_adapter_status(domain, base_model_name):
    """Checks for the existence of an adapter model for a given domain and base model."""
    expected_adapter_path = os.path.join(ADAPTER_DIR, domain, f"{base_model_name}_adapter")
    return os.path.exists(expected_adapter_path)

def get_all_base_models():
    """Dynamically gets all available base models from the models/gguf directory."""
    base_models = set()
    gguf_path = os.path.join(BASE_MODEL_DIR, "gguf")
    if os.path.exists(gguf_path):
        for model_file in os.listdir(gguf_path):
            if model_file.endswith(".gguf"):
                # Extract model name before .gguf and before any versioning (e.g., -1.0)
                name_parts = model_file.replace(".gguf", "").split('-')
                base_name = name_parts[0] # Take the first part as base name
                if base_name:
                    base_models.add(base_name)
    return sorted(list(base_models))


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    domain_config = get_domain_config()
    model_mapping = get_model_mapping()
    all_base_models = get_all_base_models()

    # Get domain statuses
    domain_statuses = []
    adapter_statuses = {}
    for domain, details in domain_config.items():
        base_model_for_domain = details.get("base_model", "Unknown")
        adapter_trained = check_adapter_status(domain, base_model_for_domain)
        
        # Determine if the domain training is "complete" based on adapter existence
        # and if the assigned base model is recognized.
        status_text = "Training Complete" if adapter_trained and base_model_for_domain in all_base_models else "Training Pending / Adapter Missing"
        if not adapter_trained:
            status_text = "Adapter Missing"
        elif base_model_for_domain not in all_base_models:
            status_text = f"Base Model Missing ({base_model_for_domain})"
        
        domain_statuses.append({
            "domain": domain,
            "base_model": base_model_for_domain,
            "adapter_trained": adapter_trained,
            "status": status_text
        })
        
        # Add to adapter_statuses in the format expected by the dashboard
        adapter_statuses[domain] = {
            "status": "completed" if adapter_trained else "not_started",
            "files": 1 if adapter_trained else 0,
            "file_list": []
        }

    # Prepare data for model availability
    available_models_info = {}
    for model_name in all_base_models:
        available_models_info[model_name] = True # Mark as available

    # Add explicitly defined models if not already found in GGUF folder
    # This covers models like DialoGPT-medium, Phi-3.5-mini-instruct which might not be GGUF yet
    explicit_models = ["DialoGPT-medium", "Phi-3.5-mini-instruct", "Qwen2.5-3B-Instruct"]
    for model in explicit_models:
        if model not in available_models_info:
            available_models_info[model] = os.path.exists(os.path.join(BASE_MODEL_DIR, model)) # Check if directory exists

    # Get Python processes
    processes = []
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, shell=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = [p.strip('"') for p in line.split('","')]
                    if len(parts) >= 5:
                        memory_kb = parts[4].replace(',', '').replace(' K', '')
                        try:
                            memory_mb = round(int(memory_kb) / 1024, 1)
                        except ValueError:
                            memory_mb = 0
                        processes.append({
                            "pid": parts[1],
                            "memory_mb": memory_mb
                        })
    except Exception as e:
        print(f"Error getting processes: {e}")

    # Calculate progress
    completed_domains = sum(1 for info in adapter_statuses.values() if info["status"] == "completed")
    in_progress_domains = sum(1 for info in adapter_statuses.values() if info["status"] == "in_progress")
    
    # Get training data
    training_data = {}
    data_dir = os.path.join("data", "synthetic")
    if os.path.exists(data_dir):
        for domain in ["healthcare", "business", "education", "creative", "leadership"]:
            pattern = os.path.join(data_dir, f"{domain}_train_*.json")
            files = glob.glob(pattern)
            if files:
                latest = max(files, key=os.path.getmtime)
                size_mb = round(os.path.getsize(latest) / (1024*1024), 2)
                modified = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%H:%M:%S")
                training_data[domain] = {
                    "file": os.path.basename(latest),
                    "size_mb": size_mb,
                    "modified": modified
                }

    # Total memory usage
    total_memory_mb = sum(p.get("memory_mb", 0) for p in processes)
    
    # Return complete response structure
    return jsonify({
        "domain_statuses": domain_statuses,
        "model_mapping": model_mapping,
        "available_base_models": available_models_info,
        "all_base_models_found_in_gguf": all_base_models,
        # Additional fields required by the dashboard
        "adapters": adapter_statuses,
        "processes": processes,
        "progress": {
            "completed": completed_domains,
            "in_progress": in_progress_domains,
            "total": 5,
            "percentage": (completed_domains / 5) * 100 if completed_domains > 0 else 0
        },
        "system": {
            "total_memory_mb": total_memory_mb,
            "active_processes": len(processes),
            "training_active": total_memory_mb > 50
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_data": training_data
    })

if __name__ == '__main__':
    # Ensure the templates directory exists for Flask to find HTML files
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create a simple dashboard.html if it doesn't exist
    dashboard_html_path = os.path.join('templates', 'dashboard.html')
    if not os.path.exists(dashboard_html_path):
        with open(dashboard_html_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA Universal Model Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 900px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
        h1, h2 { color: #0056b3; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 20px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 20px; }
        .status-card { background-color: #e9f5ff; border-left: 5px solid #007bff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: transform 0.2s ease-in-out; }
        .status-card:hover { transform: translateY(-5px); }
        .status-card h3 { margin-top: 0; color: #007bff; }
        .status-card p { margin: 5px 0; font-size: 0.95em; }
        .status-card .status { font-weight: bold; color: #28a745; } /* Green for complete */
        .status-card .status.pending { color: #ffc107; } /* Amber for pending */
        .status-card .status.missing { color: #dc3545; } /* Red for missing */
        .model-list ul { list-style: none; padding: 0; }
        .model-list li { background: #f9f9f9; padding: 10px 15px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; border-radius: 5px; margin-bottom: 8px; }
        .model-list li:last-child { border-bottom: none; }
        .model-list .model-name { font-weight: bold; }
        .model-list .status-indicator { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-left: 10px; }
        .model-list .status-indicator.available { background-color: #28a745; }
        .model-list .status-indicator.unavailable { background-color: #dc3545; }
        .json-output { background-color: #eef; padding: 15px; border-radius: 8px; white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', Courier, monospace; font-size: 0.85em; max-height: 400px; overflow-y: auto; border: 1px solid #ccc; }
        .footer { text-align: center; margin-top: 40px; font-size: 0.8em; color: #777; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TARA Universal Model Dashboard</h1>
        <p>This dashboard provides a real-time overview of the training status for different domains and available models.</p>

        <h2>Domain Training Status</h2>
        <div class="status-grid" id="domainStatus">
            <!-- Domain statuses will be loaded here by JavaScript -->
        </div>

        <h2>Available Base Models (GGUF & Explicit)</h2>
        <div class="model-list">
            <ul id="availableModels">
                <!-- Available models will be loaded here by JavaScript -->
            </ul>
        </div>

        <h2>Raw API Data</h2>
        <pre class="json-output" id="rawData">
            <!-- Raw JSON data will be displayed here -->
        </pre>

        <div class="footer">
            <p>Dashboard updates every 15 seconds. Full dashboard on port 8000.</p>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log("Fetched data:", data); // For debugging

                // Update Domain Status
                const domainStatusDiv = document.getElementById('domainStatus');
                domainStatusDiv.innerHTML = ''; // Clear previous content
                if (data.domain_statuses && data.domain_statuses.length > 0) {
                    data.domain_statuses.forEach(domain => {
                        const card = document.createElement('div');
                        card.className = 'status-card';
                        let statusClass = '';
                        if (domain.status.includes('Complete')) {
                            statusClass = 'status'; // Green
                        } else if (domain.status.includes('Pending')) {
                            statusClass = 'status pending'; // Amber
                        } else if (domain.status.includes('Missing')) {
                            statusClass = 'status missing'; // Red
                        }

                        card.innerHTML = `
                            <h3>${domain.domain}</h3>
                            <p>Base Model: <strong>${domain.base_model}</strong></p>
                            <p>Adapter Trained: <span class="${domain.adapter_trained ? '' : 'missing'}">${domain.adapter_trained ? 'Yes' : 'No'}</span></p>
                            <p>Overall Status: <span class="${statusClass}">${domain.status}</span></p>
                        `;
                        domainStatusDiv.appendChild(card);
                    });
                } else {
                    domainStatusDiv.innerHTML = '<p>No domain status information available.</p>';
                }

                // Update Available Models
                const availableModelsList = document.getElementById('availableModels');
                availableModelsList.innerHTML = ''; // Clear previous content
                if (data.available_base_models) {
                    const modelsArray = Object.entries(data.available_base_models).sort(); // Sort alphabetically
                    modelsArray.forEach(([modelName, available]) => {
                        const listItem = document.createElement('li');
                        const statusIndicatorClass = available ? 'available' : 'unavailable';
                        listItem.innerHTML = `
                            <span class="model-name">${modelName}</span>
                            <span class="status-indicator ${statusIndicatorClass}"></span>
                        `;
                        availableModelsList.appendChild(listItem);
                    });
                } else {
                    availableModelsList.innerHTML = '<p>No available model information.</p>';
                }

                // Display Raw Data
                document.getElementById('rawData').textContent = JSON.stringify(data, null, 2);

            } catch (error) {
                console.error('Error fetching status:', error);
                document.getElementById('rawData').textContent = 'Error fetching data: ' + error.message;
            }
        }

        // Fetch status immediately and then every 15 seconds
        fetchStatus();
        setInterval(fetchStatus, 15000);
    </script>
</body>
</html>
""")
    print("🌐 Starting TARA Universal Model Simple Web Monitor")
    print("🚀 Dashboard available at: http://localhost:8001/")
    print("ℹ️  Note: Full dashboard is running on port 8000")
    app.run(host='0.0.0.0', port=8001, debug=False) 