import os
import json
import subprocess
import glob
from flask import Flask, render_template_string, request, redirect, url_for
import yaml

app = Flask(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
STATUS_FILE = os.path.join(PROJECT_ROOT, 'dashboard_status.json')
TRAINING_STATE_DIR = os.path.join(PROJECT_ROOT, 'training_state')
DOMAIN_CONFIG = os.path.join(PROJECT_ROOT, 'configs/domain_model_mapping.yaml')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TARA Universal Model - Simple Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
        .container { max-width: 1100px; margin: 30px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #0001; padding: 24px; }
        h1 { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px 8px; border-bottom: 1px solid #eee; text-align: left; }
        th { background: #f0f0f0; }
        tr.completed { background: #eaffea; }
        tr.in_progress { background: #fffbe6; }
        tr.not_started { background: #f7f7f7; }
        .status { font-weight: bold; }
        .status.completed { color: #27ae60; }
        .status.in_progress { color: #f39c12; }
        .status.not_started { color: #888; }
        .footer { margin-top: 30px; font-size: 0.95em; color: #888; }
        .action-btn { padding: 5px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.95em; margin-left: 8px; }
        .restart-btn { background: #e67e22; color: #fff; }
        .resume-btn { background: #2980b9; color: #fff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TARA Universal Model - Domain Training Dashboard</h1>
        <p>All domains visualized for human access. <b>Auto-refreshes every 30 seconds.</b></p>
        <table>
            <tr>
                <th>Domain</th>
                <th>Current Model</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Model Size</th>
                <th>Action</th>
            </tr>
            {% for d in domains %}
            <tr class="{{ d.status }}">
                <td>{{ d.name }}</td>
                <td>{{ d.current_model }}</td>
                <td class="status {{ d.status }}">{{ d.status.replace('_', ' ').title() }}</td>
                <td>{{ d.progress }}</td>
                <td>{{ d.model_size or '-' }}</td>
                <td>
                    {% if d.status != 'completed' %}
                        <form method="post" action="/action">
                            <input type="hidden" name="domain" value="{{ d.name }}">
                            {% if d.status == 'in_progress' %}
                                <button class="action-btn restart-btn" name="action" value="restart">Restart</button>
                            {% else %}
                                <button class="action-btn resume-btn" name="action" value="resume">Resume</button>
                            {% endif %}
                        </form>
                    {% else %}
                        -
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        <div class="footer">
            <b>Total Model+Speech+STT+RMS Size:</b> {{ total_size }} GB &nbsp; | &nbsp; <b>Domains:</b> {{ domains|length }}
        </div>
    </div>
</body>
</html>
'''

def get_model_size_gb(model_name):
    # Try to find the model file in models/gguf or models/ by name
    model_dirs = [os.path.join(PROJECT_ROOT, 'models/gguf'), os.path.join(PROJECT_ROOT, 'models')]
    for d in model_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if model_name.lower() in f.lower() and f.endswith(('.gguf', '.bin', '.pt')):
                    size_gb = os.path.getsize(os.path.join(root, f)) / (1024**3)
                    return f"{size_gb:.1f}"
    return None

def get_training_state(domain):
    # Look for a training state file for the domain
    state_file = os.path.join(TRAINING_STATE_DIR, f"{domain.lower()}_training_state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_dashboard_status():
    # Load domain mapping
    with open(DOMAIN_CONFIG, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    domain_models = config.get('domain_models', {})
    domains = []
    total_size = 0.0
    for domain, model in domain_models.items():
        state = get_training_state(domain)
        if state:
            steps = state.get('current_step', 0)
            total_steps = state.get('total_steps', 400)
            completed = state.get('completed', False)
            if completed or steps >= total_steps:
                status = 'completed'
                progress = f"100% ({total_steps}/{total_steps})"
            elif steps > 0:
                status = 'in_progress'
                percent = int(steps / total_steps * 100)
                progress = f"{percent}% ({steps}/{total_steps})"
            else:
                status = 'not_started'
                progress = "0%"
        else:
            status = 'not_started'
            progress = "0%"
        model_size = get_model_size_gb(model)
        if model_size:
            try:
                total_size += float(model_size)
            except Exception:
                pass
        domains.append({
            'name': domain.replace('_', ' ').title(),
            'current_model': model,
            'status': status,
            'progress': progress,
            'model_size': model_size
        })
    return {
        'domains': domains,
        'total_size_gb': f"{total_size:.1f}"
    }

@app.route('/', methods=['GET'])
def dashboard():
    # Always generate the latest dashboard_status.json
    status = get_dashboard_status()
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2)
    return render_template_string(HTML_TEMPLATE, domains=status['domains'], total_size=status['total_size_gb'])

@app.route('/action', methods=['POST'])
def domain_action():
    domain = request.form.get('domain')
    action = request.form.get('action')
    print(f"[ACTION] {action.title()} requested for domain: {domain}")
    # Example: call a shell script or training command
    # subprocess.Popen(["python", "scripts/training/restart_domains.py", "--domain", domain, f"--{action}"])
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False) 