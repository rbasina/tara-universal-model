#!/usr/bin/env python3
"""
🔌 Connection Recovery System for TARA Universal Model
Handles Cursor AI disconnections and training interruptions automatically
"""

import time
import json
import threading
import subprocess
import signal
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import logging
from typing import Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/connection_recovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConnectionRecoverySystem:
    """Robust connection recovery system for training continuity"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.recovery_dir = self.project_root / "connection_recovery"
        self.recovery_dir.mkdir(exist_ok=True)
        
        # State files
        self.heartbeat_file = self.recovery_dir / "heartbeat.txt"
        self.connection_state_file = self.recovery_dir / "connection_state.json"
        self.training_state_file = self.recovery_dir / "training_state.json"
        self.recovery_trigger_file = self.recovery_dir / "recovery_trigger.json"
        
        # Recovery settings
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # 5 minutes
        self.max_recovery_attempts = 5
        self.recovery_delay = 60  # seconds
        
        # Monitoring state
        self.monitoring_active = False
        self.heartbeat_thread = None
        self.recovery_thread = None
        
        # Initialize state
        self._initialize_state()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_state(self):
        """Initialize recovery state files"""
        # Initialize connection state
        if not self.connection_state_file.exists():
            connection_state = {
                "last_heartbeat": time.time(),
                "connection_active": True,
                "recovery_attempts": 0,
                "last_recovery_time": None,
                "total_recoveries": 0
            }
            self._save_json(self.connection_state_file, connection_state)
        
        # Initialize recovery trigger
        if not self.recovery_trigger_file.exists():
            recovery_trigger = {
                "connection_lost": False,
                "loss_time": None,
                "recovery_triggered": False,
                "max_recovery_attempts": self.max_recovery_attempts,
                "recovery_delay": self.recovery_delay
            }
            self._save_json(self.recovery_trigger_file, recovery_trigger)
    
    def start_monitoring(self):
        """Start connection monitoring"""
        logger.info("🔌 Starting connection recovery monitoring...")
        self.monitoring_active = True
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        # Start recovery monitoring thread
        self.recovery_thread = threading.Thread(target=self._recovery_monitor_loop, daemon=True)
        self.recovery_thread.start()
        
        logger.info("✅ Connection monitoring started")
    
    def stop_monitoring(self):
        """Stop connection monitoring"""
        logger.info("🛑 Stopping connection recovery monitoring...")
        self.monitoring_active = False
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5)
        
        logger.info("✅ Connection monitoring stopped")
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while self.monitoring_active:
            try:
                # Update heartbeat
                self._update_heartbeat()
                
                # Update connection state
                self._update_connection_state()
                
                # Sleep for heartbeat interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"❌ Heartbeat loop error: {e}")
                time.sleep(self.heartbeat_interval)
    
    def _recovery_monitor_loop(self):
        """Recovery monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for connection loss
                if self._detect_connection_loss():
                    logger.warning("⚠️ Connection loss detected!")
                    self._trigger_recovery()
                
                # Check for recovery completion
                self._check_recovery_status()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Recovery monitor error: {e}")
                time.sleep(60)
    
    def _update_heartbeat(self):
        """Update heartbeat timestamp"""
        current_time = time.time()
        self.heartbeat_file.write_text(str(current_time))
        
        # Update connection state
        connection_state = self._load_json(self.connection_state_file)
        connection_state["last_heartbeat"] = current_time
        connection_state["connection_active"] = True
        self._save_json(self.connection_state_file, connection_state)
    
    def _update_connection_state(self):
        """Update connection state"""
        connection_state = self._load_json(self.connection_state_file)
        connection_state["last_update"] = time.time()
        self._save_json(self.connection_state_file, connection_state)
    
    def _detect_connection_loss(self) -> bool:
        """Detect if connection is lost"""
        if not self.heartbeat_file.exists():
            return True
        
        try:
            last_heartbeat = float(self.heartbeat_file.read_text().strip())
            current_time = time.time()
            time_diff = current_time - last_heartbeat
            
            return time_diff > self.connection_timeout
            
        except (ValueError, FileNotFoundError):
            return True
    
    def _trigger_recovery(self):
        """Trigger recovery process"""
        recovery_trigger = self._load_json(self.recovery_trigger_file)
        connection_state = self._load_json(self.connection_state_file)
        
        # Check if recovery should be triggered
        if recovery_trigger["recovery_triggered"]:
            return
        
        if connection_state["recovery_attempts"] >= self.max_recovery_attempts:
            logger.error(f"❌ Max recovery attempts ({self.max_recovery_attempts}) reached")
            return
        
        # Update recovery trigger
        recovery_trigger["connection_lost"] = True
        recovery_trigger["loss_time"] = time.time()
        recovery_trigger["recovery_triggered"] = True
        self._save_json(self.recovery_trigger_file, recovery_trigger)
        
        # Update connection state
        connection_state["connection_active"] = False
        connection_state["recovery_attempts"] += 1
        connection_state["last_recovery_time"] = time.time()
        connection_state["total_recoveries"] += 1
        self._save_json(self.connection_state_file, connection_state)
        
        logger.warning(f"🔄 Triggering recovery attempt {connection_state['recovery_attempts']}")
        
        # Start recovery process
        self._start_recovery_process()
    
    def _start_recovery_process(self):
        """Start the recovery process"""
        try:
            # Save current training state
            self._save_training_state()
            
            # Create recovery script
            recovery_script = self._create_recovery_script()
            
            # Execute recovery
            self._execute_recovery(recovery_script)
            
        except Exception as e:
            logger.error(f"❌ Recovery process error: {e}")
    
    def _save_training_state(self):
        """Save current training state"""
        try:
            # Look for training state files
            training_state_files = [
                self.project_root / "training_state.json",
                self.project_root / "training_recovery_state.json",
                self.project_root / "training_state" / "overall_training_state.json"
            ]
            
            for state_file in training_state_files:
                if state_file.exists():
                    # Copy to recovery directory
                    backup_file = self.recovery_dir / f"training_state_backup_{int(time.time())}.json"
                    backup_file.write_text(state_file.read_text())
                    logger.info(f"✅ Training state backed up: {backup_file}")
                    break
            
            # Create recovery training state
            recovery_training_state = {
                "recovery_time": time.time(),
                "recovery_attempt": self._load_json(self.connection_state_file)["recovery_attempts"],
                "original_state_files": [str(f) for f in training_state_files if f.exists()],
                "cursor_session": {
                    "session_id": f"recovery_session_{int(time.time())}",
                    "recovery_triggered": True
                }
            }
            
            self._save_json(self.training_state_file, recovery_training_state)
            
        except Exception as e:
            logger.error(f"❌ Failed to save training state: {e}")
    
    def _create_recovery_script(self) -> Path:
        """Create PowerShell recovery script"""
        recovery_script = self.recovery_dir / "recovery.ps1"
        
        script_content = f"""
# TARA Universal Model - Connection Recovery Script
# Generated: {datetime.now().isoformat()}

Write-Host "🔄 TARA Universal Model - Connection Recovery" -ForegroundColor Yellow
Write-Host "Recovery Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" -ForegroundColor Cyan

# Set environment variables
$env:TARA_RECOVERY_MODE = "true"
$env:TARA_RECOVERY_TIME = "{int(time.time())}"
$env:TARA_RECOVERY_ATTEMPT = "{self._load_json(self.connection_state_file)['recovery_attempts']}"

# Navigate to project directory
Set-Location "{self.project_root}"

# Check if training state exists
$trainingStateFile = "training_recovery_state.json"
if (Test-Path $trainingStateFile) {{
    Write-Host "✅ Found training state file" -ForegroundColor Green
    $trainingState = Get-Content $trainingStateFile | ConvertFrom-Json
    Write-Host "Current Domain: $($trainingState.current_domain)" -ForegroundColor Cyan
    Write-Host "Current Step: $($trainingState.current_step)" -ForegroundColor Cyan
}} else {{
    Write-Host "⚠️ No training state file found" -ForegroundColor Yellow
}}

# Resume training
Write-Host "🚀 Resuming training..." -ForegroundColor Green

# Check for domain-specific training scripts
$domains = @("healthcare", "business", "education", "creative", "leadership")
foreach ($domain in $domains) {{
    $checkpointDir = "models/$domain/checkpoint-*"
    if (Test-Path $checkpointDir) {{
        $latestCheckpoint = Get-ChildItem $checkpointDir | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Write-Host "Found checkpoint for $domain`: $($latestCheckpoint.Name)" -ForegroundColor Green
    }}
}}

# Resume with parameterized trainer
if (Test-Path "scripts/training/parameterized_train_domains.py") {{
    Write-Host "🔄 Resuming with parameterized trainer..." -ForegroundColor Green
    python scripts/training/parameterized_train_domains.py --resume --recovery-mode
}} elseif (Test-Path "scripts/training/train_meetara_universal_model.py") {{
    Write-Host "🔄 Resuming with universal trainer..." -ForegroundColor Green
    python scripts/training/train_meetara_universal_model.py --resume --recovery-mode
}} else {{
    Write-Host "❌ No training script found" -ForegroundColor Red
}}

Write-Host "✅ Recovery script completed" -ForegroundColor Green
"""
        
        recovery_script.write_text(script_content)
        logger.info(f"✅ Recovery script created: {recovery_script}")
        
        return recovery_script
    
    def _execute_recovery(self, recovery_script: Path):
        """Execute recovery script"""
        try:
            # Execute PowerShell script
            cmd = [
                "powershell.exe",
                "-ExecutionPolicy", "Bypass",
                "-File", str(recovery_script)
            ]
            
            logger.info(f"🔄 Executing recovery script: {recovery_script}")
            
            # Run in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
                
                if process.returncode == 0:
                    logger.info("✅ Recovery script executed successfully")
                    if stdout:
                        logger.info(f"Recovery output: {stdout}")
                else:
                    logger.error(f"❌ Recovery script failed: {stderr}")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("⚠️ Recovery script timed out")
                
        except Exception as e:
            logger.error(f"❌ Failed to execute recovery script: {e}")
    
    def _check_recovery_status(self):
        """Check recovery status"""
        recovery_trigger = self._load_json(self.recovery_trigger_file)
        
        if recovery_trigger["recovery_triggered"]:
            # Check if recovery is complete
            if self._detect_connection_restored():
                logger.info("✅ Connection restored, marking recovery complete")
                self._mark_recovery_complete()
    
    def _detect_connection_restored(self) -> bool:
        """Detect if connection is restored"""
        return not self._detect_connection_loss()
    
    def _mark_recovery_complete(self):
        """Mark recovery as complete"""
        recovery_trigger = self._load_json(self.recovery_trigger_file)
        recovery_trigger["recovery_triggered"] = False
        recovery_trigger["connection_lost"] = False
        recovery_trigger["recovery_completed"] = True
        recovery_trigger["completion_time"] = time.time()
        self._save_json(self.recovery_trigger_file, recovery_trigger)
        
        # Update connection state
        connection_state = self._load_json(self.connection_state_file)
        connection_state["connection_active"] = True
        connection_state["last_recovery_completion"] = time.time()
        self._save_json(self.connection_state_file, connection_state)
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_monitoring()
        sys.exit(0)
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            if file_path.exists():
                return json.loads(file_path.read_text())
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load JSON from {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save JSON file"""
        try:
            file_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"❌ Failed to save JSON to {file_path}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get recovery system status"""
        connection_state = self._load_json(self.connection_state_file)
        recovery_trigger = self._load_json(self.recovery_trigger_file)
        
        return {
            "monitoring_active": self.monitoring_active,
            "connection_active": connection_state.get("connection_active", False),
            "last_heartbeat": connection_state.get("last_heartbeat", 0),
            "recovery_attempts": connection_state.get("recovery_attempts", 0),
            "total_recoveries": connection_state.get("total_recoveries", 0),
            "connection_lost": recovery_trigger.get("connection_lost", False),
            "recovery_triggered": recovery_trigger.get("recovery_triggered", False),
            "recovery_completed": recovery_trigger.get("recovery_completed", False)
        }

class TrainingContinuityManager:
    """Manages training continuity during connection issues"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.continuity_dir = self.project_root / "training_continuity"
        self.continuity_dir.mkdir(exist_ok=True)
        
        # State files
        self.progress_file = self.continuity_dir / "progress.json"
        self.checkpoint_file = self.continuity_dir / "checkpoint_status.json"
        self.resume_file = self.continuity_dir / "resume_data.json"
    
    def save_training_progress(self, domain: str, step: int, total_steps: int, 
                             training_loss: float, learning_rate: float):
        """Save training progress"""
        progress_data = {
            "domain": domain,
            "current_step": step,
            "total_steps": total_steps,
            "progress_percent": (step / total_steps) * 100,
            "training_loss": training_loss,
            "learning_rate": learning_rate,
            "last_update": time.time(),
            "checkpoint_path": f"models/{domain}/checkpoint-{step}"
        }
        
        self._save_json(self.progress_file, progress_data)
        logger.info(f"✅ Training progress saved: {domain} at step {step}")
    
    def save_checkpoint_status(self, domain: str, checkpoint_path: Path, 
                             validation_score: float = None):
        """Save checkpoint status"""
        checkpoint_data = {
            "domain": domain,
            "checkpoint_path": str(checkpoint_path),
            "validation_score": validation_score,
            "saved_time": time.time(),
            "file_size_mb": self._get_file_size_mb(checkpoint_path)
        }
        
        self._save_json(self.checkpoint_file, checkpoint_data)
        logger.info(f"✅ Checkpoint status saved: {checkpoint_path}")
    
    def create_resume_data(self, domain: str, step: int, checkpoint_path: Path):
        """Create resume data for recovery"""
        resume_data = {
            "resume_domain": domain,
            "resume_step": step,
            "checkpoint_path": str(checkpoint_path),
            "resume_time": time.time(),
            "recovery_mode": True
        }
        
        self._save_json(self.resume_file, resume_data)
        logger.info(f"✅ Resume data created: {domain} at step {step}")
    
    def get_resume_data(self) -> Optional[Dict[str, Any]]:
        """Get resume data"""
        if self.resume_file.exists():
            return self._load_json(self.resume_file)
        return None
    
    def clear_resume_data(self):
        """Clear resume data after successful recovery"""
        if self.resume_file.exists():
            self.resume_file.unlink()
            logger.info("✅ Resume data cleared")
    
    def _get_file_size_mb(self, path: Path) -> float:
        """Get file size in MB"""
        try:
            if path.is_file():
                return path.stat().st_size / (1024 * 1024)
            elif path.is_dir():
                total_size = 0
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                return total_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            if file_path.exists():
                return json.loads(file_path.read_text())
            return {}
        except Exception as e:
            logger.error(f"❌ Failed to load JSON from {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save JSON file"""
        try:
            file_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"❌ Failed to save JSON to {file_path}: {e}")

def main():
    """Main function for connection recovery system"""
    project_root = Path.cwd()
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize recovery system
    recovery_system = ConnectionRecoverySystem(project_root)
    continuity_manager = TrainingContinuityManager(project_root)
    
    try:
        # Start monitoring
        recovery_system.start_monitoring()
        
        logger.info("🔌 Connection recovery system running...")
        logger.info("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(10)
            
            # Print status every 5 minutes
            if int(time.time()) % 300 == 0:
                status = recovery_system.get_status()
                logger.info(f"📊 Status: {status}")
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        recovery_system.stop_monitoring()

if __name__ == "__main__":
    main() 