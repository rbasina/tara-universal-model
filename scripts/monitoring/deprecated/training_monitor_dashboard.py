#!/usr/bin/env python3
"""
Real-Time Training Monitor Dashboard
Provides live updates on TARA Universal Model training progress.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import sys

# Clear screen function
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class TrainingDashboard:
    """Real-time training dashboard with live updates."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.progress_file = "training_progress.json"
        self.log_file = "logs/enhanced_training.log"
        
    def run_dashboard(self):
        """Run the real-time dashboard."""
        print("🎬 TARA Universal Model - Training Dashboard")
        print("🔄 Updates every 10 seconds | Press Ctrl+C to exit")
        print("=" * 80)
        
        try:
            while True:
                clear_screen()
                self._display_header()
                self._display_training_progress()
                self._display_system_status()
                self._display_recent_logs()
                self._display_footer()
                
                time.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")
    
    def _display_header(self):
        """Display dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("🎯 TARA UNIVERSAL MODEL - TRAINING DASHBOARD")
        print(f"🕒 {now} | 🔄 Auto-updating every 10 seconds")
        print("=" * 80)
    
    def _display_training_progress(self):
        """Display current training progress."""
        print("\n📊 TRAINING PROGRESS")
        print("-" * 40)
        
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file) as f:
                    progress = json.load(f)
                
                domains = progress.get("domains", {})
                
                # Summary stats
                completed = sum(1 for d in domains.values() if d["status"] == "completed")
                training = sum(1 for d in domains.values() if d["status"] == "training")
                pending = sum(1 for d in domains.values() if d["status"] == "pending")
                
                print(f"✅ Completed: {completed}/5 domains")
                print(f"🔄 Training:  {training}/5 domains")
                print(f"⏳ Pending:   {pending}/5 domains")
                print()
                
                # Individual domain status
                for domain, info in domains.items():
                    status = info.get("status", "unknown")
                    emoji = {"completed": "✅", "training": "🔄", "pending": "⏳"}.get(status, "❓")
                    
                    model_count = len(info.get("model_files", []))
                    validation = info.get("validation_results", {})
                    
                    if validation:
                        score = validation.get("overall_score", 0)
                        ready = validation.get("production_ready", False)
                        ready_text = "READY" if ready else f"SCORE: {score:.2f}"
                    else:
                        ready_text = "NO VALIDATION"
                    
                    print(f"{emoji} {domain.upper():12} | Models: {model_count:2} | {ready_text}")
                
            else:
                print("⏳ Training progress file not found - training may be starting...")
                
        except Exception as e:
            print(f"❌ Error reading progress: {e}")
    
    def _display_system_status(self):
        """Display system resource usage."""
        print("\n💻 SYSTEM STATUS")
        print("-" * 40)
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print(f"🖥️  CPU Usage:    {cpu_percent:5.1f}%")
            print(f"💾 Memory Usage: {memory.percent:5.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
            print(f"💽 Disk Usage:   {disk.percent:5.1f}% ({disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB)")
            
            # Check for GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"🎮 GPU Memory:   {gpu_memory:.1f}GB available")
                else:
                    print("🎮 GPU:          Not available (CPU training)")
            except:
                print("🎮 GPU:          Status unknown")
                
            # Check backend server
            # Backend server on port 5000 no longer needed - using embedded GGUF
            print("🌐 Backend:      ✅ Embedded GGUF integration (no port needed)")
                
        except Exception as e:
            print(f"❌ System status error: {e}")
    
    def _display_recent_logs(self):
        """Display recent training logs."""
        print("\n📝 RECENT TRAINING LOGS")
        print("-" * 40)
        
        try:
            if Path(self.log_file).exists():
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Get last 10 lines
                recent_lines = lines[-10:] if len(lines) >= 10 else lines
                
                for line in recent_lines:
                    # Clean and format log line
                    clean_line = line.strip()
                    if clean_line:
                        # Extract timestamp and message
                        if " - " in clean_line:
                            parts = clean_line.split(" - ", 2)
                            if len(parts) >= 3:
                                timestamp = parts[0][-8:]  # Last 8 chars (HH:MM:SS)
                                level = parts[1]
                                message = parts[2]
                                
                                # Color-code by level
                                level_emoji = {
                                    "INFO": "ℹ️",
                                    "WARNING": "⚠️", 
                                    "ERROR": "❌",
                                    "DEBUG": "🔍"
                                }.get(level, "📝")
                                
                                print(f"{timestamp} {level_emoji} {message[:60]}...")
                            else:
                                print(f"📝 {clean_line[:70]}...")
                        else:
                            print(f"📝 {clean_line[:70]}...")
            else:
                print("⏳ Training log file not found - training may be starting...")
                
        except Exception as e:
            print(f"❌ Log reading error: {e}")
    
    def _display_footer(self):
        """Display dashboard footer."""
        print("\n" + "=" * 80)
        print("🔄 Dashboard auto-updates every 10 seconds")
        print("📊 Training progress saved to: training_progress.json")
        print("📝 Full logs available in: logs/enhanced_training.log")
        print("🛑 Press Ctrl+C to stop dashboard")

def main():
    """Main entry point."""
    dashboard = TrainingDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 