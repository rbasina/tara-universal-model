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
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import json
import socket
import subprocess

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource usage limits"""
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 2048
    max_disk_io_mb: int = 100
    max_network_connections: int = 10
    monitoring_interval: int = 5  # seconds

@dataclass
class ResourceUsage:
    """Current resource usage"""
    cpu_percent: float
    memory_mb: float
    disk_io_mb: float
    network_connections: int
    timestamp: datetime
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "disk_io_mb": self.disk_io_mb,
            "network_connections": self.network_connections,
            "timestamp": self.timestamp.isoformat(),
            "process_count": self.process_count
        }

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
    """
    Resource Management and Monitoring for TARA Universal Model
    
    Features:
    - CPU/Memory limits to prevent system overload
    - Network isolation verification for complete offline mode
    - Performance monitoring and optimization
    - Process isolation and management
    """
    
    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource usage history
        self.usage_history: List[ResourceUsage] = []
        self.max_history_size = 1000
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Process tracking
        self.tara_processes: List[psutil.Process] = []
        self.main_process = psutil.Process()
        
        # Network isolation status
        self.network_isolated = False
        self.allowed_connections = set()
        
        # Performance metrics
        self.performance_metrics = {
            "ai_response_times": [],
            "voice_synthesis_times": [],
            "model_load_times": [],
            "memory_peaks": [],
            "cpu_peaks": []
        }
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
    
    def _monitor_worker(self):
        """Background monitoring worker"""
        while self.monitoring:
            try:
                usage = self._collect_resource_usage()
                self._check_limits(usage)
                self._update_history(usage)
                time.sleep(self.limits.monitoring_interval)
            except Exception as e:
                self._trigger_alert("monitoring_error", {"error": str(e)})
                time.sleep(self.limits.monitoring_interval)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / 1024 / 1024
        
        # Disk I/O (approximate)
        disk_io = psutil.disk_io_counters()
        disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024 if disk_io else 0
        
        # Network connections
        network_connections = len(psutil.net_connections())
        
        # Process count
        process_count = len(psutil.pids())
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_io_mb=disk_io_mb,
            network_connections=network_connections,
            timestamp=datetime.now(),
            process_count=process_count
        )
    
    def _check_limits(self, usage: ResourceUsage):
        """Check if resource usage exceeds limits"""
        alerts = []
        
        if usage.cpu_percent > self.limits.max_cpu_percent:
            alerts.append(("cpu_limit_exceeded", {
                "current": usage.cpu_percent,
                "limit": self.limits.max_cpu_percent
            }))
        
        if usage.memory_mb > self.limits.max_memory_mb:
            alerts.append(("memory_limit_exceeded", {
                "current": usage.memory_mb,
                "limit": self.limits.max_memory_mb
            }))
        
        if usage.network_connections > self.limits.max_network_connections and not self.network_isolated:
            alerts.append(("network_connections_exceeded", {
                "current": usage.network_connections,
                "limit": self.limits.max_network_connections
            }))
        
        # Trigger alerts
        for alert_type, data in alerts:
            self._trigger_alert(alert_type, data)
    
    def _update_history(self, usage: ResourceUsage):
        """Update usage history"""
        self.usage_history.append(usage)
        
        # Maintain history size
        if len(self.usage_history) > self.max_history_size:
            self.usage_history = self.usage_history[-self.max_history_size:]
        
        # Update performance metrics
        self._update_performance_metrics(usage)
    
    def _update_performance_metrics(self, usage: ResourceUsage):
        """Update performance metrics"""
        # Track CPU peaks
        if usage.cpu_percent > 50:
            self.performance_metrics["cpu_peaks"].append({
                "value": usage.cpu_percent,
                "timestamp": usage.timestamp.isoformat()
            })
        
        # Track memory peaks
        if usage.memory_mb > 1000:
            self.performance_metrics["memory_peaks"].append({
                "value": usage.memory_mb,
                "timestamp": usage.timestamp.isoformat()
            })
        
        # Limit metrics history
        for metric in self.performance_metrics:
            if len(self.performance_metrics[metric]) > 100:
                self.performance_metrics[metric] = self.performance_metrics[metric][-100:]
    
    def verify_network_isolation(self) -> Dict[str, Any]:
        """Verify complete offline mode"""
        isolation_status = {
            "isolated": True,
            "active_connections": [],
            "listening_ports": [],
            "dns_resolution": False,
            "internet_access": False
        }
        
        try:
            # Check active network connections
            connections = psutil.net_connections()
            external_connections = [
                conn for conn in connections 
                if conn.status == 'ESTABLISHED' and 
                conn.raddr and 
                not self._is_local_address(conn.raddr.ip)
            ]
            
            isolation_status["active_connections"] = [
                f"{conn.laddr.ip}:{conn.laddr.port} -> {conn.raddr.ip}:{conn.raddr.port}"
                for conn in external_connections
            ]
            
            if external_connections:
                isolation_status["isolated"] = False
            
            # Check listening ports
            listening = [
                conn for conn in connections 
                if conn.status == 'LISTEN'
            ]
            
            isolation_status["listening_ports"] = [
                f"{conn.laddr.ip}:{conn.laddr.port}"
                for conn in listening
            ]
            
            # Test DNS resolution (should fail in isolated mode)
            try:
                socket.gethostbyname("google.com")
                isolation_status["dns_resolution"] = True
                isolation_status["isolated"] = False
            except socket.gaierror:
                isolation_status["dns_resolution"] = False
            
            # Test internet access (should fail in isolated mode)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(("8.8.8.8", 53))
                sock.close()
                if result == 0:
                    isolation_status["internet_access"] = True
                    isolation_status["isolated"] = False
            except Exception:
                pass
            
        except Exception as e:
            isolation_status["error"] = str(e)
            isolation_status["isolated"] = False
        
        self.network_isolated = isolation_status["isolated"]
        return isolation_status
    
    def _is_local_address(self, ip: str) -> bool:
        """Check if IP address is local"""
        local_ranges = [
            "127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.", 
            "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", 
            "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", 
            "172.29.", "172.30.", "172.31.", "::1", "localhost"
        ]
        return any(ip.startswith(prefix) for prefix in local_ranges)
    
    def limit_process_resources(self, pid: int = None):
        """Apply resource limits to process"""
        try:
            process = psutil.Process(pid) if pid else self.main_process
            
            # Set CPU affinity (limit to specific cores if needed)
            # process.cpu_affinity([0, 1])  # Limit to first 2 cores
            
            # Set process priority (lower priority)
            if os.name == 'nt':  # Windows
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:  # Unix/Linux
                process.nice(10)
            
            # Track process
            if process not in self.tara_processes:
                self.tara_processes.append(process)
            
            return True
            
        except Exception as e:
            self._trigger_alert("process_limit_error", {"error": str(e), "pid": pid})
            return False
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance for TARA"""
        optimizations = {
            "memory_cleanup": False,
            "cache_optimization": False,
            "process_optimization": False,
            "disk_cleanup": False
        }
        
        try:
            # Memory cleanup
            import gc
            gc.collect()
            optimizations["memory_cleanup"] = True
            
            # Process optimization
            for process in self.tara_processes:
                if process.is_running():
                    self.limit_process_resources(process.pid)
            optimizations["process_optimization"] = True
            
            # Cache optimization (clear old cache files)
            cache_dir = Path.home() / ".tara" / "cache"
            if cache_dir.exists():
                current_time = time.time()
                for cache_file in cache_dir.rglob("*"):
                    if cache_file.is_file() and current_time - cache_file.stat().st_mtime > 3600:  # 1 hour
                        cache_file.unlink()
            optimizations["cache_optimization"] = True
            
        except Exception as e:
            self._trigger_alert("optimization_error", {"error": str(e)})
        
        return optimizations
    
    def record_performance_metric(self, metric_type: str, value: float, metadata: Dict = None):
        """Record custom performance metric"""
        if metric_type not in self.performance_metrics:
            self.performance_metrics[metric_type] = []
        
        metric_data = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            metric_data.update(metadata)
        
        self.performance_metrics[metric_type].append(metric_data)
        
        # Limit metric history
        if len(self.performance_metrics[metric_type]) > 100:
            self.performance_metrics[metric_type] = self.performance_metrics[metric_type][-100:]
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        current_usage = self._collect_resource_usage()
        
        return {
            "current_usage": current_usage.to_dict(),
            "limits": {
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_memory_mb": self.limits.max_memory_mb,
                "max_disk_io_mb": self.limits.max_disk_io_mb,
                "max_network_connections": self.limits.max_network_connections
            },
            "monitoring_active": self.monitoring,
            "network_isolated": self.network_isolated,
            "tracked_processes": len(self.tara_processes),
            "history_size": len(self.usage_history)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.usage_history:
            return {"error": "No usage history available"}
        
        # Calculate averages
        avg_cpu = sum(u.cpu_percent for u in self.usage_history) / len(self.usage_history)
        avg_memory = sum(u.memory_mb for u in self.usage_history) / len(self.usage_history)
        
        # Find peaks
        max_cpu = max(u.cpu_percent for u in self.usage_history)
        max_memory = max(u.memory_mb for u in self.usage_history)
        
        return {
            "monitoring_period": {
                "start": self.usage_history[0].timestamp.isoformat(),
                "end": self.usage_history[-1].timestamp.isoformat(),
                "duration_minutes": len(self.usage_history) * self.limits.monitoring_interval / 60
            },
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(avg_memory, 2)
            },
            "peaks": {
                "cpu_percent": max_cpu,
                "memory_mb": max_memory
            },
            "performance_metrics": self.performance_metrics,
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score (0-100)"""
        if not self.usage_history:
            return 0.0
        
        # Score based on resource utilization vs limits
        avg_cpu = sum(u.cpu_percent for u in self.usage_history) / len(self.usage_history)
        avg_memory = sum(u.memory_mb for u in self.usage_history) / len(self.usage_history)
        
        cpu_efficiency = max(0, 100 - (avg_cpu / self.limits.max_cpu_percent * 100))
        memory_efficiency = max(0, 100 - (avg_memory / self.limits.max_memory_mb * 100))
        
        return round((cpu_efficiency + memory_efficiency) / 2, 2)
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, data: Dict):
        """Trigger alert to all callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, data)
            except Exception:
                pass  # Silent failure for callbacks
    
    def emergency_resource_cleanup(self):
        """Emergency resource cleanup"""
        try:
            # Terminate non-essential TARA processes
            for process in self.tara_processes[:]:
                if process.is_running() and process.pid != os.getpid():
                    try:
                        process.terminate()
                        self.tara_processes.remove(process)
                    except Exception:
                        pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear performance metrics
            for metric in self.performance_metrics:
                self.performance_metrics[metric] = []
            
            # Clear usage history
            self.usage_history = []
            
        except Exception as e:
            self._trigger_alert("emergency_cleanup_error", {"error": str(e)})

# Global resource monitor instance
_resource_monitor = None

def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor 