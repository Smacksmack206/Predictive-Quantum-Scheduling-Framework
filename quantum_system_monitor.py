#!/usr/bin/env python3
"""
Quantum System Monitor
Real-time monitoring of quantum performance, error rates, and resource utilization
"""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import psutil

@dataclass
class QuantumMetrics:
    """Quantum system metrics snapshot"""
    timestamp: float
    system_type: str
    qubit_count: int
    active_circuits: int
    execution_rate: float  # executions per second
    error_rate: float  # percentage
    success_rate: float  # percentage
    average_execution_time: float  # seconds
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    thermal_state: float
    quantum_volume: int
    entanglement_operations: int
    gate_fidelity: float

@dataclass
class AlertCondition:
    """Alert condition configuration"""
    metric_name: str
    threshold_value: float
    comparison: str  # 'greater', 'less', 'equal'
    severity: str  # 'info', 'warning', 'critical'
    enabled: bool
    callback: Optional[Callable] = None

class QuantumSystemMonitor:
    """
    Real-time quantum system monitoring
    Tracks performance, errors, and resource utilization
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds
        
        # Alert system
        self.alert_conditions = []
        self.alert_callbacks = []
        self.alerts_triggered = []
        
        # Performance baselines
        self.performance_baselines = {}
        
        # Resource tracking
        self.resource_utilization = {
            'peak_memory_mb': 0.0,
            'peak_cpu_percent': 0.0,
            'peak_gpu_percent': 0.0,
            'total_executions': 0,
            'total_errors': 0
        }
        
        # Initialize default alert conditions
        self._setup_default_alerts()
        
        print("📊 QuantumSystemMonitor initialized")
        print(f"📈 Monitoring history: {history_size} snapshots")
    
    def start_monitoring(self, quantum_controller=None):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
        
        self.quantum_controller = quantum_controller
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("🔍 Quantum system monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        print("🔍 Quantum system monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                if metrics:
                    # Store metrics
                    self.metrics_history.append(metrics)
                    
                    # Update resource tracking
                    self._update_resource_tracking(metrics)
                    
                    # Check alert conditions
                    self._check_alert_conditions(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(5.0)  # Longer delay on error
    
    def _collect_metrics(self) -> Optional[QuantumMetrics]:
        """Collect current quantum system metrics"""
        try:
            # Get system metrics
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            cpu_usage = psutil.cpu_percent()
            
            # Get GPU usage (simplified)
            gpu_usage = self._get_gpu_usage()
            
            # Get thermal state
            thermal_state = self._get_thermal_state()
            
            # Get quantum-specific metrics
            quantum_metrics = self._get_quantum_metrics()
            
            metrics = QuantumMetrics(
                timestamp=time.time(),
                system_type=quantum_metrics.get('system_type', 'unknown'),
                qubit_count=quantum_metrics.get('qubit_count', 0),
                active_circuits=quantum_metrics.get('active_circuits', 0),
                execution_rate=quantum_metrics.get('execution_rate', 0.0),
                error_rate=quantum_metrics.get('error_rate', 0.0),
                success_rate=quantum_metrics.get('success_rate', 100.0),
                average_execution_time=quantum_metrics.get('average_execution_time', 0.0),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=gpu_usage,
                thermal_state=thermal_state,
                quantum_volume=quantum_metrics.get('quantum_volume', 0),
                entanglement_operations=quantum_metrics.get('entanglement_operations', 0),
                gate_fidelity=quantum_metrics.get('gate_fidelity', 1.0)
            )
            
            return metrics
            
        except Exception as e:
            print(f"❌ Failed to collect metrics: {e}")
            return None
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage"""
        try:
            # Try to get GPU usage (simplified)
            # In practice, this would use proper GPU monitoring libraries
            return 0.0  # Placeholder
        except:
            return 0.0
    
    def _get_thermal_state(self) -> float:
        """Get thermal state"""
        try:
            import subprocess
            result = subprocess.run(['sysctl', 'machdep.xcpm.cpu_thermal_state'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.split(':')[-1].strip())
        except:
            pass
        
        return 25.0  # Default safe temperature
    
    def _get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum-specific metrics"""
        if not hasattr(self, 'quantum_controller') or not self.quantum_controller:
            return {}
        
        try:
            # Get status from quantum controller
            status = self.quantum_controller.get_quantum_system_status()
            active_system = status.get('active_system')
            
            if not active_system:
                return {}
            
            active_status = status.get(active_system, {})
            
            # Calculate derived metrics
            quantum_metrics = {
                'system_type': active_system,
                'qubit_count': active_status.get('qubit_count', 0),
                'active_circuits': 1 if active_status.get('active', False) else 0,
                'execution_rate': self._calculate_execution_rate(),
                'error_rate': self._calculate_error_rate(),
                'success_rate': self._calculate_success_rate(),
                'average_execution_time': self._calculate_average_execution_time(),
                'quantum_volume': self._calculate_quantum_volume(active_status.get('qubit_count', 0)),
                'entanglement_operations': self._count_entanglement_operations(),
                'gate_fidelity': self._estimate_gate_fidelity()
            }
            
            return quantum_metrics
            
        except Exception as e:
            print(f"❌ Failed to get quantum metrics: {e}")
            return {}
    
    def _calculate_execution_rate(self) -> float:
        """Calculate quantum execution rate"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Look at recent executions
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 snapshots
        
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        if time_span <= 0:
            return 0.0
        
        # Estimate executions (simplified)
        estimated_executions = len(recent_metrics)
        return estimated_executions / time_span
    
    def _calculate_error_rate(self) -> float:
        """Calculate quantum error rate"""
        # Simplified error rate calculation
        # In practice, this would track actual quantum errors
        
        if len(self.metrics_history) < 5:
            return 0.0
        
        # Estimate error rate based on system performance
        recent_metrics = list(self.metrics_history)[-5:]
        avg_execution_time = np.mean([m.average_execution_time for m in recent_metrics])
        
        # Higher execution times might indicate more errors
        if avg_execution_time > 0.1:  # 100ms
            return min(5.0, avg_execution_time * 50)  # Max 5% error rate
        
        return 1.0  # Default 1% error rate
    
    def _calculate_success_rate(self) -> float:
        """Calculate quantum success rate"""
        error_rate = self._calculate_error_rate()
        return max(0.0, 100.0 - error_rate)
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time"""
        if len(self.metrics_history) < 3:
            return 0.05  # Default 50ms
        
        recent_metrics = list(self.metrics_history)[-5:]
        execution_times = [m.average_execution_time for m in recent_metrics if m.average_execution_time > 0]
        
        if execution_times:
            return np.mean(execution_times)
        
        return 0.05
    
    def _calculate_quantum_volume(self, qubit_count: int) -> int:
        """Calculate quantum volume"""
        if qubit_count <= 0:
            return 0
        
        # Simplified quantum volume calculation
        # Real quantum volume depends on gate fidelity, connectivity, etc.
        return min(2 ** qubit_count, 2 ** 20)  # Cap at 2^20
    
    def _count_entanglement_operations(self) -> int:
        """Count entanglement operations"""
        # Simplified count - in practice would track actual entanglement operations
        return len(self.metrics_history) * 2  # Estimate
    
    def _estimate_gate_fidelity(self) -> float:
        """Estimate gate fidelity"""
        # Simplified fidelity estimation based on error rate
        error_rate = self._calculate_error_rate()
        return max(0.9, 1.0 - (error_rate / 100.0))
    
    def _update_resource_tracking(self, metrics: QuantumMetrics):
        """Update resource utilization tracking"""
        self.resource_utilization['peak_memory_mb'] = max(
            self.resource_utilization['peak_memory_mb'],
            metrics.memory_usage_mb
        )
        
        self.resource_utilization['peak_cpu_percent'] = max(
            self.resource_utilization['peak_cpu_percent'],
            metrics.cpu_usage_percent
        )
        
        self.resource_utilization['peak_gpu_percent'] = max(
            self.resource_utilization['peak_gpu_percent'],
            metrics.gpu_usage_percent
        )
        
        self.resource_utilization['total_executions'] += max(0, int(metrics.execution_rate))
        self.resource_utilization['total_errors'] += max(0, int(metrics.error_rate))
    
    def _setup_default_alerts(self):
        """Setup default alert conditions"""
        default_alerts = [
            AlertCondition(
                metric_name='error_rate',
                threshold_value=5.0,
                comparison='greater',
                severity='warning',
                enabled=True
            ),
            AlertCondition(
                metric_name='error_rate',
                threshold_value=10.0,
                comparison='greater',
                severity='critical',
                enabled=True
            ),
            AlertCondition(
                metric_name='memory_usage_mb',
                threshold_value=8192.0,  # 8GB
                comparison='greater',
                severity='warning',
                enabled=True
            ),
            AlertCondition(
                metric_name='thermal_state',
                threshold_value=80.0,
                comparison='greater',
                severity='critical',
                enabled=True
            ),
            AlertCondition(
                metric_name='success_rate',
                threshold_value=90.0,
                comparison='less',
                severity='warning',
                enabled=True
            )
        ]
        
        self.alert_conditions.extend(default_alerts)
    
    def _check_alert_conditions(self, metrics: QuantumMetrics):
        """Check alert conditions against current metrics"""
        for condition in self.alert_conditions:
            if not condition.enabled:
                continue
            
            # Get metric value
            metric_value = getattr(metrics, condition.metric_name, None)
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if condition.comparison == 'greater':
                triggered = metric_value > condition.threshold_value
            elif condition.comparison == 'less':
                triggered = metric_value < condition.threshold_value
            elif condition.comparison == 'equal':
                triggered = abs(metric_value - condition.threshold_value) < 0.001
            
            if triggered:
                self._trigger_alert(condition, metrics, metric_value)
    
    def _trigger_alert(self, condition: AlertCondition, metrics: QuantumMetrics, metric_value: float):
        """Trigger an alert"""
        alert = {
            'timestamp': metrics.timestamp,
            'condition': condition.metric_name,
            'severity': condition.severity,
            'threshold': condition.threshold_value,
            'actual_value': metric_value,
            'system_type': metrics.system_type,
            'message': f"{condition.metric_name} {condition.comparison} {condition.threshold_value} (actual: {metric_value:.2f})"
        }
        
        self.alerts_triggered.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts_triggered) > 100:
            self.alerts_triggered = self.alerts_triggered[-50:]
        
        # Call callback if provided
        if condition.callback:
            try:
                condition.callback(alert)
            except Exception as e:
                print(f"❌ Alert callback error: {e}")
        
        # Print alert
        severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(condition.severity, '❓')
        print(f"{severity_icon} ALERT: {alert['message']}")
    
    def add_alert_condition(self, condition: AlertCondition):
        """Add custom alert condition"""
        self.alert_conditions.append(condition)
        print(f"📢 Added alert condition: {condition.metric_name} {condition.comparison} {condition.threshold_value}")
    
    def get_current_metrics(self) -> Optional[QuantumMetrics]:
        """Get most recent metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[QuantumMetrics]:
        """Get metrics history"""
        if duration_seconds is None:
            return list(self.metrics_history)
        
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_summary(self, duration_seconds: float = 300.0) -> Dict[str, Any]:
        """Get performance summary for specified duration"""
        recent_metrics = self.get_metrics_history(duration_seconds)
        
        if not recent_metrics:
            return {'error': 'No metrics available'}
        
        # Calculate summary statistics
        execution_rates = [m.execution_rate for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        execution_times = [m.average_execution_time for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        
        summary = {
            'time_period_seconds': duration_seconds,
            'metrics_count': len(recent_metrics),
            'system_type': recent_metrics[-1].system_type if recent_metrics else 'unknown',
            'qubit_count': recent_metrics[-1].qubit_count if recent_metrics else 0,
            
            # Performance metrics
            'average_execution_rate': np.mean(execution_rates),
            'peak_execution_rate': np.max(execution_rates),
            'average_error_rate': np.mean(error_rates),
            'peak_error_rate': np.max(error_rates),
            'average_success_rate': np.mean(success_rates),
            'minimum_success_rate': np.min(success_rates),
            
            # Timing metrics
            'average_execution_time': np.mean(execution_times),
            'fastest_execution_time': np.min(execution_times),
            'slowest_execution_time': np.max(execution_times),
            
            # Resource metrics
            'average_memory_usage_mb': np.mean(memory_usage),
            'peak_memory_usage_mb': np.max(memory_usage),
            'current_memory_usage_mb': recent_metrics[-1].memory_usage_mb,
            
            # Alert summary
            'alerts_triggered': len([a for a in self.alerts_triggered 
                                   if a['timestamp'] >= time.time() - duration_seconds]),
            'critical_alerts': len([a for a in self.alerts_triggered 
                                  if a['timestamp'] >= time.time() - duration_seconds 
                                  and a['severity'] == 'critical'])
        }
        
        return summary
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        return self.resource_utilization.copy()
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts_triggered[-count:] if self.alerts_triggered else []
    
    def export_metrics(self, filename: str, duration_seconds: Optional[float] = None):
        """Export metrics to JSON file"""
        metrics_to_export = self.get_metrics_history(duration_seconds)
        
        # Convert to serializable format
        export_data = {
            'export_timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'metrics_count': len(metrics_to_export),
            'metrics': [asdict(m) for m in metrics_to_export]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Exported {len(metrics_to_export)} metrics to {filename}")
    
    def reset_monitoring(self):
        """Reset monitoring data"""
        self.metrics_history.clear()
        self.alerts_triggered.clear()
        self.resource_utilization = {
            'peak_memory_mb': 0.0,
            'peak_cpu_percent': 0.0,
            'peak_gpu_percent': 0.0,
            'total_executions': 0,
            'total_errors': 0
        }
        
        print("🔄 Monitoring data reset")

if __name__ == "__main__":
    print("🧪 Testing QuantumSystemMonitor")
    
    # Initialize monitor
    monitor = QuantumSystemMonitor(history_size=100)
    
    # Add custom alert
    custom_alert = AlertCondition(
        metric_name='execution_rate',
        threshold_value=10.0,
        comparison='greater',
        severity='info',
        enabled=True
    )
    monitor.add_alert_condition(custom_alert)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Let it run for a few seconds
    print("📊 Monitoring for 5 seconds...")
    time.sleep(5)
    
    # Get current metrics
    current_metrics = monitor.get_current_metrics()
    if current_metrics:
        print(f"📈 Current metrics: {current_metrics.system_type}, {current_metrics.qubit_count} qubits")
    
    # Get performance summary
    summary = monitor.get_performance_summary(duration_seconds=10.0)
    print(f"📊 Performance summary: {summary}")
    
    # Get resource utilization
    resources = monitor.get_resource_utilization()
    print(f"💾 Resource utilization: {resources}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("🎉 QuantumSystemMonitor test completed!")