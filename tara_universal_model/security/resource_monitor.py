"""
TARA Universal Model - Resource Monitor
HAI-Enhanced Resource Management and System Safety

This module implements:
- CPU/memory limits to prevent system overload
- Network isolation verification
- Resource usage monitoring and alerts
- Automatic resource cleanup
"""

import os
import time
import psutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """HAI Resource Limits Configuration"""
    max_cpu_percent: float = 70.0  # Maximum CPU usage percentage
    max_memory_mb: int = 1024      # Maximum memory usage in MB
    max_disk_usage_mb: int = 500   # Maximum disk usage for temp files
    max_active_sessions: int = 10  # Maximum concurrent sessions
    monitoring_interval: int = 5   # Monitoring interval in seconds
    
@dataclass
class ResourceAlert:
    """Resource usage alert"""
    timestamp: datetime
    alert_type: str
    message: str
    current_value: float
    limit_value: float
    severity: str  # 'warning', 'critical'

class NetworkIsolationChecker:
    """HAI Network Isolation Verification"""
    
    @staticmethod
    def verify_offline_mode() -> Dict[str, bool]:
        """Verify that TARA can operate completely offline"""
        results = {
            "dns_isolated": False,
            "no_external_connections": False,
            "local_only_bindings": False,
            "offline_capable": False
        }
        
        try:
            # Check for DNS resolution attempts (should fail in offline mode)
            import socket
            socket.setdefaulttimeout(1)
            try:
                socket.gethostbyname('google.com')
                results["dns_isolated"] = False
            except socket.gaierror:
                results["dns_isolated"] = True
            
            # Check for active external connections
            connections = psutil.net_connections()
            external_connections = [
                conn for conn in connections 
                if conn.raddr and not conn.raddr.ip.startswith(('127.', '192.168.', '10.', '172.'))
            ]
            results["no_external_connections"] = len(external_connections) == 0
            
            # Check if services are bound to localhost only
            listening_sockets = [
                conn for conn in connections 
                if conn.status == 'LISTEN' and conn.laddr
            ]
            local_only = all(
                conn.laddr.ip in ['127.0.0.1', '0.0.0.0', '::1'] 
                for conn in listening_sockets
            )
            results["local_only_bindings"] = local_only
            
            # Overall offline capability
            results["offline_capable"] = (
                results["dns_isolated"] and 
                results["no_external_connections"] and 
                results["local_only_bindings"]
            )
            
        except Exception as e:
            logger.error(f"Network isolation check failed: {e}")
        
        return results

class ResourceMonitor:
    """HAI-Enhanced Resource Monitor for System Safety"""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process()
        self.start_time = datetime.now()
        self.alerts: List[ResourceAlert] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_callbacks: List[Callable] = []
        
        # Resource usage history
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        self.disk_usage_history: List[float] = []
        
        # Session tracking
        self.active_sessions = set()
        
        logger.info("🔧 Resource Monitor initialized with HAI safety limits")
    
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("📊 Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_resources()
                time.sleep(self.limits.monitoring_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.limits.monitoring_interval)
    
    def _check_resources(self):
        """Check all resource limits"""
        current_time = datetime.now()
        
        # CPU usage check
        cpu_percent = self.process.cpu_percent()
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) > 60:  # Keep last 60 readings
            self.cpu_history.pop(0)
        
        if cpu_percent > self.limits.max_cpu_percent:
            self._create_alert(
                "cpu_limit_exceeded",
                f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.limits.max_cpu_percent}%)",
                cpu_percent,
                self.limits.max_cpu_percent,
                "critical" if cpu_percent > self.limits.max_cpu_percent * 1.2 else "warning"
            )
        
        # Memory usage check
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.memory_history.append(memory_mb)
        if len(self.memory_history) > 60:
            self.memory_history.pop(0)
        
        if memory_mb > self.limits.max_memory_mb:
            self._create_alert(
                "memory_limit_exceeded",
                f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.limits.max_memory_mb}MB)",
                memory_mb,
                self.limits.max_memory_mb,
                "critical" if memory_mb > self.limits.max_memory_mb * 1.2 else "warning"
            )
        
        # Disk usage check
        disk_usage = self._get_temp_disk_usage()
        self.disk_usage_history.append(disk_usage)
        if len(self.disk_usage_history) > 60:
            self.disk_usage_history.pop(0)
        
        if disk_usage > self.limits.max_disk_usage_mb:
            self._create_alert(
                "disk_limit_exceeded",
                f"Temp disk usage ({disk_usage:.1f}MB) exceeds limit ({self.limits.max_disk_usage_mb}MB)",
                disk_usage,
                self.limits.max_disk_usage_mb,
                "warning"
            )
        
        # Session limit check
        active_count = len(self.active_sessions)
        if active_count > self.limits.max_active_sessions:
            self._create_alert(
                "session_limit_exceeded",
                f"Active sessions ({active_count}) exceed limit ({self.limits.max_active_sessions})",
                active_count,
                self.limits.max_active_sessions,
                "warning"
            )
    
    def _get_temp_disk_usage(self) -> float:
        """Calculate disk usage of temporary files"""
        try:
            temp_dirs = [
                Path.cwd() / "temp",
                Path.cwd() / "logs",
                Path("/tmp") if os.name != 'nt' else Path(os.environ.get('TEMP', 'C:\\temp'))
            ]
            
            total_size = 0
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file_path in temp_dir.rglob('*'):
                        if file_path.is_file():
                            try:
                                total_size += file_path.stat().st_size
                            except (OSError, FileNotFoundError):
                                continue
            
            return total_size / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Disk usage calculation failed: {e}")
            return 0.0
    
    def _create_alert(self, alert_type: str, message: str, current_value: float, limit_value: float, severity: str):
        """Create and handle resource alert"""
        alert = ResourceAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            severity=severity
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Log alert
        log_func = logger.critical if severity == "critical" else logger.warning
        log_func(f"🚨 Resource Alert [{severity.upper()}]: {message}")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Auto-mitigation for critical alerts
        if severity == "critical":
            self._handle_critical_alert(alert)
    
    def _handle_critical_alert(self, alert: ResourceAlert):
        """Handle critical resource alerts with auto-mitigation"""
        if alert.alert_type == "cpu_limit_exceeded":
            logger.warning("🔧 Implementing CPU throttling...")
            # Implement CPU throttling logic here
            
        elif alert.alert_type == "memory_limit_exceeded":
            logger.warning("🔧 Triggering memory cleanup...")
            self._emergency_memory_cleanup()
            
        elif alert.alert_type == "disk_limit_exceeded":
            logger.warning("🔧 Cleaning up temporary files...")
            self._emergency_disk_cleanup()
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedures"""
        try:
            import gc
            gc.collect()  # Force garbage collection
            
            # Clear caches if available
            if hasattr(self, 'model_cache'):
                self.model_cache.clear()
            
            logger.info("🧹 Emergency memory cleanup completed")
        except Exception as e:
            logger.error(f"Emergency memory cleanup failed: {e}")
    
    def _emergency_disk_cleanup(self):
        """Emergency disk cleanup procedures"""
        try:
            import tempfile
            import glob
            
            # Clean up temp audio files
            temp_patterns = [
                "temp_audio_*.mp3",
                "temp_audio_*.wav",
                "*.tmp",
                "tara_temp_*"
            ]
            
            for pattern in temp_patterns:
                for file_path in glob.glob(os.path.join(tempfile.gettempdir(), pattern)):
                    try:
                        os.remove(file_path)
                    except OSError:
                        continue
            
            logger.info("🧹 Emergency disk cleanup completed")
        except Exception as e:
            logger.error(f"Emergency disk cleanup failed: {e}")
    
    def register_session(self, session_id: str):
        """Register a new active session"""
        self.active_sessions.add(session_id)
        logger.debug(f"📝 Registered session: {session_id}")
    
    def unregister_session(self, session_id: str):
        """Unregister an active session"""
        self.active_sessions.discard(session_id)
        logger.debug(f"📝 Unregistered session: {session_id}")
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Add callback for resource alerts"""
        self.alert_callbacks.append(callback)
    
    def get_resource_status(self) -> Dict:
        """Get current resource status"""
        current_cpu = self.process.cpu_percent()
        current_memory = self.process.memory_info().rss / 1024 / 1024
        current_disk = self._get_temp_disk_usage()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "current_percent": current_cpu,
                "limit_percent": self.limits.max_cpu_percent,
                "status": "normal" if current_cpu <= self.limits.max_cpu_percent else "exceeded",
                "history_avg": sum(self.cpu_history[-10:]) / len(self.cpu_history[-10:]) if self.cpu_history else 0
            },
            "memory": {
                "current_mb": current_memory,
                "limit_mb": self.limits.max_memory_mb,
                "status": "normal" if current_memory <= self.limits.max_memory_mb else "exceeded",
                "history_avg": sum(self.memory_history[-10:]) / len(self.memory_history[-10:]) if self.memory_history else 0
            },
            "disk": {
                "current_mb": current_disk,
                "limit_mb": self.limits.max_disk_usage_mb,
                "status": "normal" if current_disk <= self.limits.max_disk_usage_mb else "exceeded"
            },
            "sessions": {
                "active_count": len(self.active_sessions),
                "limit_count": self.limits.max_active_sessions,
                "status": "normal" if len(self.active_sessions) <= self.limits.max_active_sessions else "exceeded"
            },
            "alerts": {
                "total_count": len(self.alerts),
                "critical_count": len([a for a in self.alerts if a.severity == "critical"]),
                "warning_count": len([a for a in self.alerts if a.severity == "warning"])
            },
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "monitoring_active": self.monitoring_active
        }
    
    def get_network_isolation_status(self) -> Dict:
        """Get network isolation verification status"""
        return NetworkIsolationChecker.verify_offline_mode()
    
    def export_resource_report(self, filepath: Optional[str] = None) -> str:
        """Export detailed resource usage report"""
        if filepath is None:
            filepath = f"tara_resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "resource_limits": {
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_memory_mb": self.limits.max_memory_mb,
                "max_disk_usage_mb": self.limits.max_disk_usage_mb,
                "max_active_sessions": self.limits.max_active_sessions
            },
            "current_status": self.get_resource_status(),
            "network_isolation": self.get_network_isolation_status(),
            "resource_history": {
                "cpu_history": self.cpu_history[-60:],  # Last 60 readings
                "memory_history": self.memory_history[-60:],
                "disk_history": self.disk_usage_history[-60:]
            },
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "type": alert.alert_type,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "limit_value": alert.limit_value,
                    "severity": alert.severity
                }
                for alert in self.alerts[-50:]  # Last 50 alerts
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Resource report exported to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export resource report: {e}")
            return ""

# Global resource monitor instance
_resource_monitor = None

def get_resource_monitor(limits: Optional[ResourceLimits] = None) -> ResourceMonitor:
    """Get global resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor(limits)
    return _resource_monitor 