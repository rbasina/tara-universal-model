#!/usr/bin/env python3
"""
TARA Universal Model Training Watcher
Continuously monitors training progress with periodic updates
"""

import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_monitor():
    """Run the training monitor and return the output"""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/monitor_training.py"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error running monitor: {e}"

def main():
    """Main watching loop"""
    print("🔍 TARA Training Watcher Started")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    update_interval = 30  # seconds
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Clear screen and show header
            clear_screen()
            print("🔍 TARA Universal Model Training Watcher")
            print("=" * 60)
            print(f"🔄 Update #{iteration} | Interval: {update_interval}s")
            print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            print()
            
            # Run monitoring
            monitor_output = run_monitor()
            print(monitor_output)
            
            print("=" * 60)
            print(f"⏳ Next update in {update_interval} seconds... (Ctrl+C to stop)")
            
            # Wait for next update
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Training watcher stopped by user")
        print("Training processes continue running in background")
    except Exception as e:
        print(f"\n\n❌ Error in watcher: {e}")

if __name__ == "__main__":
    main() 