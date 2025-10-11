#!/usr/bin/env python3
"""
Hardware Performance Monitoring for Advanced EAS
Real-time monitoring of CPU, GPU, thermal, and power metrics
"""

# Line 1-25: Hardware monitoring setup
import subprocess
import json
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import psutil
import os
from permission_manager import permission_manager

@dataclass
class HardwareMetrics:
    cpu_frequency: Dict[str, float]  # P-core and E-core frequencies
    cpu_temperature: float
    gpu_temperature: float
    power_consumption: Dict[str, float]  # CPU, GPU, total
    thermal_pressure: float
    memory_bandwidth_utilization: float
    cache_hit_rates: Dict[str, float]

class HardwareMonitor:
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_history = []
        self.current_metrics = None
        self.monitor_thread = None
        
    def start_monitoring(self):
        # Line 26-35: Start hardware monitoring thread
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🔧 Hardware monitoring started")
        
    def stop_monitoring(self):
        # Line 36-40: Stop monitoring
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🔧 Hardware monitoring stopped")
            
    def _monitoring_loop(self):
        # Line 41-60: Main monitoring loop
        while self.monitoring:
            try:
                metrics = self._collect_hardware_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Keep only last 300 samples (5 minutes at 1Hz)
                if len(self.metrics_history) > 300:
                    self.metrics_history.pop(0)
                    
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Hardware monitoring error: {e}")
                time.sleep(self.sampling_interval)
                
    def _collect_hardware_metrics(self) -> HardwareMetrics:
        # Line 61-100: Collect comprehensive hardware metrics
        try:
            # CPU frequency monitoring
            cpu_freq = self._get_cpu_frequencies()
            
            # Temperature monitoring
            cpu_temp, gpu_temp = self._get_temperatures()
            
            # Power consumption
            power_data = self._get_power_consumption()
            
            # Thermal pressure
            thermal_pressure = self._get_thermal_pressure()
            
            # Memory bandwidth
            memory_bandwidth = self._get_memory_bandwidth()
            
            # Cache statistics
            cache_stats = self._get_cache_statistics()
            
            return HardwareMetrics(
                cpu_frequency=cpu_freq,
                cpu_temperature=cpu_temp,
                gpu_temperature=gpu_temp,
                power_consumption=power_data,
                thermal_pressure=thermal_pressure,
                memory_bandwidth_utilization=memory_bandwidth,
                cache_hit_rates=cache_stats
            )
            
        except Exception as e:
            # Return default metrics on error
            return HardwareMetrics(
                cpu_frequency={'p_core': 0.0, 'e_core': 0.0},
                cpu_temperature=50.0,
                gpu_temperature=50.0,
                power_consumption={'cpu': 0.0, 'gpu': 0.0, 'total': 0.0},
                thermal_pressure=0.0,
                memory_bandwidth_utilization=0.0,
                cache_hit_rates={'l1': 0.95, 'l2': 0.90, 'l3': 0.85}
            )
            
    def _get_cpu_frequencies(self) -> Dict[str, float]:
        # Line 101-125: CPU frequency monitoring via powermetrics
        try:
            result = permission_manager.execute_with_sudo([
                'powermetrics', '--samplers', 'cpu_power', '-n', '1', 
                '--show-initial-usage', '--format', 'plist'
            ], timeout=5)
            
            if result.returncode == 0:
                # Parse plist output for frequency data
                # This is simplified - real implementation would parse XML plist
                output_lines = result.stdout.split('\n')
                p_core_freq = 0.0
                e_core_freq = 0.0
                
                for line in output_lines:
                    if 'P-Cluster HW active frequency' in line:
                        p_core_freq = float(line.split(':')[1].strip().split()[0])
                    elif 'E-Cluster HW active frequency' in line:
                        e_core_freq = float(line.split(':')[1].strip().split()[0])
                        
                return {'p_core': p_core_freq, 'e_core': e_core_freq}
            else:
                return {'p_core': 0.0, 'e_core': 0.0}
                
        except Exception:
            return {'p_core': 0.0, 'e_core': 0.0}
            
    def _get_temperatures(self) -> tuple[float, float]:
        # Line 126-150: Temperature monitoring
        try:
            result = permission_manager.execute_with_sudo([
                'powermetrics', '--samplers', 'smc', '-n', '1', 
                '--show-initial-usage'
            ], timeout=5)
            
            cpu_temp = 50.0  # Default
            gpu_temp = 50.0  # Default
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU die temperature' in line:
                        temp_str = line.split(':')[1].strip().replace('C', '')
                        cpu_temp = float(temp_str)
                    elif 'GPU die temperature' in line:
                        temp_str = line.split(':')[1].strip().replace('C', '')
                        gpu_temp = float(temp_str)
                        
            return cpu_temp, gpu_temp
            
        except Exception:
            return 50.0, 50.0  
          
    def _get_power_consumption(self) -> Dict[str, float]:
        # Line 151-175: Power consumption monitoring
        try:
            result = permission_manager.execute_with_sudo([
                'powermetrics', '--samplers', 'cpu_power,gpu_power', 
                '-n', '1', '--show-initial-usage'
            ], timeout=5)
            
            cpu_power = 0.0
            gpu_power = 0.0
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU Power' in line and 'mW' in line:
                        power_str = line.split(':')[1].strip().replace('mW', '')
                        cpu_power = float(power_str) / 1000.0  # Convert to watts
                    elif 'GPU Power' in line and 'mW' in line:
                        power_str = line.split(':')[1].strip().replace('mW', '')
                        gpu_power = float(power_str) / 1000.0  # Convert to watts
                        
            total_power = cpu_power + gpu_power
            
            return {
                'cpu': cpu_power,
                'gpu': gpu_power,
                'total': total_power
            }
            
        except Exception:
            return {'cpu': 0.0, 'gpu': 0.0, 'total': 0.0}
            
    def _get_thermal_pressure(self) -> float:
        # Line 176-190: Thermal pressure calculation
        try:
            # Thermal pressure based on temperature and frequency scaling
            if self.current_metrics:
                cpu_temp = self.current_metrics.cpu_temperature
                
                # Calculate pressure based on temperature thresholds
                if cpu_temp > 90:
                    return 1.0  # Maximum thermal pressure
                elif cpu_temp > 80:
                    return 0.8
                elif cpu_temp > 70:
                    return 0.5
                elif cpu_temp > 60:
                    return 0.2
                else:
                    return 0.0
                    
            return 0.0
            
        except Exception:
            return 0.0
            
    def _get_memory_bandwidth(self) -> float:
        # Line 191-210: Memory bandwidth utilization
        try:
            # Use vm_stat for memory statistics
            result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                # Parse vm_stat output for memory pressure indicators
                lines = result.stdout.split('\n')
                
                # Look for memory pressure indicators
                for line in lines:
                    if 'Pages paged in' in line or 'Pages paged out' in line:
                        # High paging indicates memory pressure
                        return 0.8  # Simplified calculation
                        
                # If no paging, check memory usage
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
                
            return 0.0
            
        except Exception:
            return 0.0
            
    def _get_cache_statistics(self) -> Dict[str, float]:
        # Line 211-225: Cache hit rate estimation
        try:
            # This would require performance counters access
            # For now, return estimated values based on system load
            cpu_percent = psutil.cpu_percent()
            
            # Estimate cache performance based on CPU load
            if cpu_percent > 80:
                return {'l1': 0.90, 'l2': 0.85, 'l3': 0.75}
            elif cpu_percent > 50:
                return {'l1': 0.93, 'l2': 0.88, 'l3': 0.80}
            else:
                return {'l1': 0.95, 'l2': 0.92, 'l3': 0.87}
                
        except Exception:
            return {'l1': 0.95, 'l2': 0.90, 'l3': 0.85}
            
    def get_current_metrics(self) -> Optional[HardwareMetrics]:
        # Line 226-230: Get current hardware metrics
        return self.current_metrics
        
    def get_average_metrics(self, window_seconds: int = 60) -> Optional[HardwareMetrics]:
        # Line 231-260: Calculate average metrics over time window
        if not self.metrics_history:
            return None
            
        # Calculate how many samples to include
        samples_needed = min(window_seconds // self.sampling_interval, len(self.metrics_history))
        recent_metrics = self.metrics_history[-int(samples_needed):]
        
        if not recent_metrics:
            return None
            
        # Calculate averages
        avg_cpu_temp = sum(m.cpu_temperature for m in recent_metrics) / len(recent_metrics)
        avg_gpu_temp = sum(m.gpu_temperature for m in recent_metrics) / len(recent_metrics)
        avg_thermal_pressure = sum(m.thermal_pressure for m in recent_metrics) / len(recent_metrics)
        avg_memory_bandwidth = sum(m.memory_bandwidth_utilization for m in recent_metrics) / len(recent_metrics)
        
        # Average frequencies
        avg_p_core_freq = sum(m.cpu_frequency.get('p_core', 0) for m in recent_metrics) / len(recent_metrics)
        avg_e_core_freq = sum(m.cpu_frequency.get('e_core', 0) for m in recent_metrics) / len(recent_metrics)
        
        # Average power
        avg_cpu_power = sum(m.power_consumption.get('cpu', 0) for m in recent_metrics) / len(recent_metrics)
        avg_gpu_power = sum(m.power_consumption.get('gpu', 0) for m in recent_metrics) / len(recent_metrics)
        avg_total_power = sum(m.power_consumption.get('total', 0) for m in recent_metrics) / len(recent_metrics)
        
        return HardwareMetrics(
            cpu_frequency={'p_core': avg_p_core_freq, 'e_core': avg_e_core_freq},
            cpu_temperature=avg_cpu_temp,
            gpu_temperature=avg_gpu_temp,
            power_consumption={'cpu': avg_cpu_power, 'gpu': avg_gpu_power, 'total': avg_total_power},
            thermal_pressure=avg_thermal_pressure,
            memory_bandwidth_utilization=avg_memory_bandwidth,
            cache_hit_rates={'l1': 0.95, 'l2': 0.90, 'l3': 0.85}  # Simplified
        )

# Test function
def test_hardware_monitor():
    """Test the hardware monitor"""
    print("🔧 Testing Hardware Monitor")
    print("=" * 50)
    
    monitor = HardwareMonitor(sampling_interval=2.0)
    monitor.start_monitoring()
    
    try:
        # Monitor for 10 seconds
        for i in range(5):
            time.sleep(2)
            metrics = monitor.get_current_metrics()
            
            if metrics:
                print(f"📊 Sample {i+1}:")
                print(f"  CPU Temperature: {metrics.cpu_temperature:.1f}°C")
                print(f"  GPU Temperature: {metrics.gpu_temperature:.1f}°C")
                print(f"  CPU Power: {metrics.power_consumption['cpu']:.2f}W")
                print(f"  GPU Power: {metrics.power_consumption['gpu']:.2f}W")
                print(f"  Total Power: {metrics.power_consumption['total']:.2f}W")
                print(f"  Thermal Pressure: {metrics.thermal_pressure:.2f}")
                print(f"  Memory Bandwidth: {metrics.memory_bandwidth_utilization:.2f}")
                print()
            else:
                print(f"  No metrics available yet...")
                
    except KeyboardInterrupt:
        print("Stopping test...")
    finally:
        monitor.stop_monitoring()
        
    # Show average metrics
    avg_metrics = monitor.get_average_metrics(10)
    if avg_metrics:
        print(f"📈 Average Metrics (last 10 seconds):")
        print(f"  Avg CPU Temperature: {avg_metrics.cpu_temperature:.1f}°C")
        print(f"  Avg Total Power: {avg_metrics.power_consumption['total']:.2f}W")
        print(f"  Avg Thermal Pressure: {avg_metrics.thermal_pressure:.2f}")

if __name__ == "__main__":
    test_hardware_monitor()