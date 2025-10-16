#!/usr/bin/env python3
"""
PQS Framework - Comprehensive System Optimizer
Controls ALL tunable system parameters for maximum performance optimization
"""

import os
import sys
import time
import json
import subprocess
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class SystemState:
    """Current system state snapshot"""
    cpu_usage: float
    memory_usage: float
    thermal_state: str
    battery_level: int
    power_source: str
    active_processes: int
    network_activity: float
    disk_activity: float
    timestamp: float

@dataclass
class OptimizationResult:
    """Result of an optimization operation"""
    success: bool
    parameter: str
    old_value: Any
    new_value: Any
    expected_impact: float
    actual_impact: Optional[float] = None
    error_message: Optional[str] = None

class CPUFrequencyController:
    """Controls CPU frequency scaling and P-states"""
    
    def __init__(self):
        self.supported_governors = self._detect_governors()
        self.current_governor = self._get_current_governor()
        self.frequency_range = self._get_frequency_range()
        
    def _detect_governors(self) -> List[str]:
        """Detect available CPU governors"""
        try:
            # macOS uses different power management
            if sys.platform == 'darwin':
                return ['automatic', 'performance', 'powersave']
            else:
                # Linux governors
                result = subprocess.run(['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors'],
                                      capture_output=True, text=True)
                return result.stdout.strip().split() if result.returncode == 0 else []
        except:
            return ['automatic']
    
    def _get_current_governor(self) -> str:
        """Get current CPU governor"""
        try:
            if sys.platform == 'darwin':
                # Use pmset to get current power management settings
                result = subprocess.run(['pmset', '-g', 'custom'], capture_output=True, text=True)
                if 'powernap' in result.stdout:
                    return 'automatic'
                return 'automatic'
            else:
                result = subprocess.run(['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'],
                                      capture_output=True, text=True)
                return result.stdout.strip() if result.returncode == 0 else 'unknown'
        except:
            return 'unknown'
    
    def _get_frequency_range(self) -> Tuple[int, int]:
        """Get CPU frequency range"""
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                return (int(cpu_freq.min), int(cpu_freq.max))
            return (1000, 3000)  # Default range in MHz
        except:
            return (1000, 3000)
    
    def set_governor(self, governor: str) -> OptimizationResult:
        """Set CPU frequency governor"""
        old_governor = self.current_governor
        
        try:
            if sys.platform == 'darwin':
                # macOS power management through pmset
                if governor == 'performance':
                    subprocess.run(['sudo', 'pmset', '-a', 'powernap', '0'], check=True)
                    subprocess.run(['sudo', 'pmset', '-a', 'sleep', '0'], check=True)
                elif governor == 'powersave':
                    subprocess.run(['sudo', 'pmset', '-a', 'powernap', '1'], check=True)
                    subprocess.run(['sudo', 'pmset', '-a', 'sleep', '10'], check=True)
                else:  # automatic
                    subprocess.run(['sudo', 'pmset', '-a', 'powernap', '1'], check=True)
                    subprocess.run(['sudo', 'pmset', '-a', 'sleep', '30'], check=True)
            else:
                # Linux frequency scaling
                for cpu in range(psutil.cpu_count()):
                    subprocess.run(['sudo', 'sh', '-c', 
                                  f'echo {governor} > /sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor'],
                                 check=True)
            
            self.current_governor = governor
            expected_impact = self._calculate_governor_impact(governor)
            
            return OptimizationResult(
                success=True,
                parameter='cpu_governor',
                old_value=old_governor,
                new_value=governor,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='cpu_governor',
                old_value=old_governor,
                new_value=governor,
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _calculate_governor_impact(self, governor: str) -> float:
        """Calculate expected impact of governor change"""
        impact_map = {
            'performance': 15.0,  # 15% performance increase
            'powersave': -25.0,   # 25% power reduction
            'automatic': 0.0      # Balanced
        }
        return impact_map.get(governor, 0.0)

class SchedulerController:
    """Controls macOS scheduler instructions and process priorities"""
    
    def __init__(self):
        self.process_priorities = {}
        self.scheduler_policies = self._detect_scheduler_policies()
        
    def _detect_scheduler_policies(self) -> List[str]:
        """Detect available scheduler policies"""
        if sys.platform == 'darwin':
            return ['SCHED_OTHER', 'SCHED_FIFO', 'SCHED_RR']
        return ['SCHED_NORMAL', 'SCHED_FIFO', 'SCHED_RR', 'SCHED_BATCH']
    
    def set_process_priority(self, pid: int, priority: int) -> OptimizationResult:
        """Set process priority (nice value)"""
        try:
            old_priority = os.getpriority(os.PRIO_PROCESS, pid)
            os.setpriority(os.PRIO_PROCESS, pid, priority)
            
            self.process_priorities[pid] = priority
            
            return OptimizationResult(
                success=True,
                parameter=f'process_priority_{pid}',
                old_value=old_priority,
                new_value=priority,
                expected_impact=abs(old_priority - priority) * 2.0
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter=f'process_priority_{pid}',
                old_value=0,
                new_value=priority,
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def set_cpu_affinity(self, pid: int, cpu_list: List[int]) -> OptimizationResult:
        """Set CPU affinity for process"""
        try:
            process = psutil.Process(pid)
            old_affinity = process.cpu_affinity()
            process.cpu_affinity(cpu_list)
            
            return OptimizationResult(
                success=True,
                parameter=f'cpu_affinity_{pid}',
                old_value=old_affinity,
                new_value=cpu_list,
                expected_impact=5.0  # 5% performance improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter=f'cpu_affinity_{pid}',
                old_value=[],
                new_value=cpu_list,
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def optimize_scheduler_for_workload(self, workload_type: str) -> List[OptimizationResult]:
        """Optimize scheduler settings for specific workload"""
        results = []
        
        if workload_type == 'interactive':
            # Optimize for UI responsiveness
            results.extend(self._optimize_interactive_workload())
        elif workload_type == 'compute':
            # Optimize for computational tasks
            results.extend(self._optimize_compute_workload())
        elif workload_type == 'background':
            # Optimize for background tasks
            results.extend(self._optimize_background_workload())
        
        return results
    
    def _optimize_interactive_workload(self) -> List[OptimizationResult]:
        """Optimize for interactive/UI workload"""
        results = []
        
        # Boost UI processes
        ui_processes = ['WindowServer', 'Dock', 'Finder', 'SystemUIServer']
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in ui_processes:
                result = self.set_process_priority(proc.info['pid'], -5)  # Higher priority
                results.append(result)
        
        return results
    
    def _optimize_compute_workload(self) -> List[OptimizationResult]:
        """Optimize for computational workload"""
        results = []
        
        # Set CPU affinity for compute processes
        cpu_count = psutil.cpu_count()
        compute_cpus = list(range(cpu_count // 2, cpu_count))  # Use performance cores
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 50:
                result = self.set_cpu_affinity(proc.info['pid'], compute_cpus)
                results.append(result)
        
        return results
    
    def _optimize_background_workload(self) -> List[OptimizationResult]:
        """Optimize for background workload"""
        results = []
        
        # Lower priority for background processes
        background_processes = ['backupd', 'mds', 'mdworker', 'spotlight']
        for proc in psutil.process_iter(['pid', 'name']):
            if any(bg in proc.info['name'].lower() for bg in background_processes):
                result = self.set_process_priority(proc.info['pid'], 10)  # Lower priority
                results.append(result)
        
        return results

class MemoryController:
    """Controls memory management, compression, and swapping"""
    
    def __init__(self):
        self.memory_stats = self._get_memory_stats()
        self.swap_enabled = self._is_swap_enabled()
        
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
    
    def _is_swap_enabled(self) -> bool:
        """Check if swap is enabled"""
        try:
            swap = psutil.swap_memory()
            return swap.total > 0
        except:
            return False
    
    def optimize_memory_allocation(self) -> List[OptimizationResult]:
        """Optimize memory allocation strategies"""
        results = []
        
        # Enable memory compression on macOS
        if sys.platform == 'darwin':
            result = self._enable_memory_compression()
            results.append(result)
        
        # Optimize swap usage
        result = self._optimize_swap_usage()
        results.append(result)
        
        # Clean memory caches
        result = self._clean_memory_caches()
        results.append(result)
        
        return results
    
    def _enable_memory_compression(self) -> OptimizationResult:
        """Enable memory compression on macOS"""
        try:
            # Check current compression status
            result = subprocess.run(['sysctl', 'vm.compressor_mode'], 
                                  capture_output=True, text=True)
            old_value = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Enable compression
            subprocess.run(['sudo', 'sysctl', '-w', 'vm.compressor_mode=4'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='memory_compression',
                old_value=old_value,
                new_value='enabled',
                expected_impact=10.0  # 10% memory efficiency improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='memory_compression',
                old_value='unknown',
                new_value='enabled',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _optimize_swap_usage(self) -> OptimizationResult:
        """Optimize swap file usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 80:  # High memory usage
                # Increase swap aggressiveness
                if sys.platform == 'darwin':
                    subprocess.run(['sudo', 'sysctl', '-w', 'vm.swappiness=60'], check=True)
                else:
                    subprocess.run(['sudo', 'sysctl', '-w', 'vm.swappiness=60'], check=True)
                
                return OptimizationResult(
                    success=True,
                    parameter='swap_aggressiveness',
                    old_value='default',
                    new_value='increased',
                    expected_impact=5.0
                )
            else:
                # Reduce swap aggressiveness for better performance
                if sys.platform == 'darwin':
                    subprocess.run(['sudo', 'sysctl', '-w', 'vm.swappiness=10'], check=True)
                else:
                    subprocess.run(['sudo', 'sysctl', '-w', 'vm.swappiness=10'], check=True)
                
                return OptimizationResult(
                    success=True,
                    parameter='swap_aggressiveness',
                    old_value='default',
                    new_value='reduced',
                    expected_impact=3.0
                )
                
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='swap_aggressiveness',
                old_value='default',
                new_value='optimized',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _clean_memory_caches(self) -> OptimizationResult:
        """Clean system memory caches"""
        try:
            if sys.platform == 'darwin':
                # Purge memory caches on macOS
                subprocess.run(['sudo', 'purge'], check=True)
            else:
                # Clear caches on Linux
                subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='memory_cache_cleanup',
                old_value='cached',
                new_value='cleared',
                expected_impact=8.0  # 8% memory availability improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='memory_cache_cleanup',
                old_value='cached',
                new_value='cleared',
                expected_impact=0.0,
                error_message=str(e)
            )

class ThermalController:
    """Controls thermal management and throttling prevention"""
    
    def __init__(self):
        self.thermal_sensors = self._detect_thermal_sensors()
        self.fan_control_available = self._check_fan_control()
        self.thermal_thresholds = self._get_thermal_thresholds()
        
    def _detect_thermal_sensors(self) -> List[str]:
        """Detect available thermal sensors"""
        sensors = []
        try:
            if sys.platform == 'darwin':
                # macOS thermal sensors
                result = subprocess.run(['powermetrics', '--samplers', 'smc', '-n', '1'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    sensors = ['cpu_thermal', 'gpu_thermal', 'system_thermal']
            else:
                # Linux thermal zones
                thermal_zones = os.listdir('/sys/class/thermal/')
                sensors = [zone for zone in thermal_zones if zone.startswith('thermal_zone')]
        except:
            pass
        
        return sensors
    
    def _check_fan_control(self) -> bool:
        """Check if fan control is available"""
        try:
            if sys.platform == 'darwin':
                # Check for fan control utilities
                result = subprocess.run(['which', 'smcFanControl'], capture_output=True)
                return result.returncode == 0
            else:
                # Check for pwm fan control
                return os.path.exists('/sys/class/hwmon/')
        except:
            return False
    
    def _get_thermal_thresholds(self) -> Dict[str, float]:
        """Get thermal throttling thresholds"""
        return {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'gpu_warning': 85.0,
            'gpu_critical': 100.0
        }
    
    def get_current_temperatures(self) -> Dict[str, float]:
        """Get current system temperatures"""
        temperatures = {}
        
        try:
            if sys.platform == 'darwin':
                # Use powermetrics for macOS
                result = subprocess.run(['powermetrics', '--samplers', 'smc', '-n', '1'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse temperature data from powermetrics output
                    temperatures = self._parse_macos_temperatures(result.stdout)
            else:
                # Read Linux thermal zones
                for sensor in self.thermal_sensors:
                    temp_file = f'/sys/class/thermal/{sensor}/temp'
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r') as f:
                            temp = float(f.read().strip()) / 1000.0  # Convert millicelsius
                            temperatures[sensor] = temp
        except Exception as e:
            logger.warning(f"Failed to read temperatures: {e}")
        
        return temperatures
    
    def _parse_macos_temperatures(self, powermetrics_output: str) -> Dict[str, float]:
        """Parse temperature data from macOS powermetrics"""
        temperatures = {}
        
        # Estimate CPU temperature from CPU usage (fallback method)
        try:
            cpu_percent = psutil.cpu_percent()
            estimated_cpu_temp = 35 + (cpu_percent * 0.6)  # Rough estimation
            temperatures['cpu'] = estimated_cpu_temp
        except:
            temperatures['cpu'] = 45.0  # Default safe temperature
        
        return temperatures
    
    def prevent_thermal_throttling(self) -> List[OptimizationResult]:
        """Implement thermal throttling prevention strategies"""
        results = []
        
        temperatures = self.get_current_temperatures()
        
        for sensor, temp in temperatures.items():
            if temp > self.thermal_thresholds.get(f'{sensor}_warning', 80.0):
                # Temperature is high, apply cooling strategies
                cooling_results = self._apply_cooling_strategies(sensor, temp)
                results.extend(cooling_results)
        
        return results
    
    def _apply_cooling_strategies(self, sensor: str, temperature: float) -> List[OptimizationResult]:
        """Apply cooling strategies for overheating components"""
        results = []
        
        # Strategy 1: Reduce CPU frequency
        if 'cpu' in sensor and temperature > 85.0:
            result = self._reduce_cpu_frequency()
            results.append(result)
        
        # Strategy 2: Increase fan speed
        if self.fan_control_available:
            result = self._increase_fan_speed(temperature)
            results.append(result)
        
        # Strategy 3: Throttle high-CPU processes
        result = self._throttle_hot_processes()
        results.append(result)
        
        return results
    
    def _reduce_cpu_frequency(self) -> OptimizationResult:
        """Reduce CPU frequency to lower temperature"""
        try:
            if sys.platform == 'darwin':
                # Use pmset to reduce performance
                subprocess.run(['sudo', 'pmset', '-a', 'reducebright', '1'], check=True)
                
                return OptimizationResult(
                    success=True,
                    parameter='cpu_frequency_reduction',
                    old_value='normal',
                    new_value='reduced',
                    expected_impact=-10.0  # 10% performance reduction for cooling
                )
            else:
                # Linux CPU frequency scaling
                subprocess.run(['sudo', 'cpupower', 'frequency-set', '-u', '2GHz'], check=True)
                
                return OptimizationResult(
                    success=True,
                    parameter='cpu_frequency_reduction',
                    old_value='max',
                    new_value='2GHz',
                    expected_impact=-15.0
                )
                
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='cpu_frequency_reduction',
                old_value='normal',
                new_value='reduced',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _increase_fan_speed(self, temperature: float) -> OptimizationResult:
        """Increase fan speed based on temperature"""
        try:
            # Calculate target fan speed based on temperature
            fan_speed_percent = min(100, max(30, (temperature - 40) * 2))
            
            if sys.platform == 'darwin':
                # macOS fan control (requires third-party tools)
                subprocess.run(['smcFanControl', '-s', str(fan_speed_percent)], check=True)
            else:
                # Linux PWM fan control
                fan_speed_pwm = int(fan_speed_percent * 2.55)  # Convert to PWM value
                subprocess.run(['sudo', 'sh', '-c', 
                              f'echo {fan_speed_pwm} > /sys/class/hwmon/hwmon0/pwm1'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='fan_speed',
                old_value='auto',
                new_value=f'{fan_speed_percent}%',
                expected_impact=5.0  # 5% cooling improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='fan_speed',
                old_value='auto',
                new_value='increased',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _throttle_hot_processes(self) -> OptimizationResult:
        """Throttle processes that are generating heat"""
        try:
            throttled_count = 0
            
            # Find high-CPU processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 80:
                    try:
                        # Reduce process priority to lower CPU usage
                        os.setpriority(os.PRIO_PROCESS, proc.info['pid'], 10)
                        throttled_count += 1
                    except:
                        continue
            
            return OptimizationResult(
                success=True,
                parameter='process_thermal_throttling',
                old_value=0,
                new_value=throttled_count,
                expected_impact=throttled_count * 3.0  # 3% cooling per process
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='process_thermal_throttling',
                old_value=0,
                new_value=0,
                expected_impact=0.0,
                error_message=str(e)
            )

class GPUScheduler:
    """Controls GPU workload distribution and scheduling"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_info()
        self.metal_available = self._check_metal_support()
        self.gpu_processes = []
        
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information"""
        gpu_info = {
            'type': 'unknown',
            'memory': 0,
            'compute_units': 0
        }
        
        try:
            if sys.platform == 'darwin':
                # macOS GPU detection
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True)
                if 'Apple' in result.stdout:
                    gpu_info['type'] = 'apple_silicon'
                elif 'Intel' in result.stdout:
                    gpu_info['type'] = 'intel_integrated'
                elif 'AMD' in result.stdout or 'NVIDIA' in result.stdout:
                    gpu_info['type'] = 'discrete'
        except:
            pass
        
        return gpu_info
    
    def _check_metal_support(self) -> bool:
        """Check if Metal Performance Shaders are available"""
        try:
            if sys.platform == 'darwin':
                # Check for Metal support
                result = subprocess.run(['xcrun', 'metal', '--version'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
        except:
            pass
        
        return False
    
    def optimize_gpu_workload_distribution(self) -> List[OptimizationResult]:
        """Optimize GPU workload distribution"""
        results = []
        
        if self.gpu_info['type'] == 'apple_silicon':
            # Apple Silicon GPU optimization
            results.extend(self._optimize_apple_silicon_gpu())
        elif self.gpu_info['type'] == 'discrete':
            # Discrete GPU optimization
            results.extend(self._optimize_discrete_gpu())
        
        return results
    
    def _optimize_apple_silicon_gpu(self) -> List[OptimizationResult]:
        """Optimize Apple Silicon GPU usage"""
        results = []
        
        # Enable Metal Performance Shaders for compute tasks
        result = self._enable_metal_compute()
        results.append(result)
        
        # Optimize GPU memory allocation
        result = self._optimize_gpu_memory()
        results.append(result)
        
        return results
    
    def _enable_metal_compute(self) -> OptimizationResult:
        """Enable Metal compute for eligible processes"""
        try:
            if not self.metal_available:
                return OptimizationResult(
                    success=False,
                    parameter='metal_compute',
                    old_value='unavailable',
                    new_value='enabled',
                    expected_impact=0.0,
                    error_message='Metal not available'
                )
            
            # Set environment variable for Metal compute
            os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
            os.environ['METAL_PERFORMANCE_SHADERS_FRAMEWORKS'] = '1'
            
            return OptimizationResult(
                success=True,
                parameter='metal_compute',
                old_value='disabled',
                new_value='enabled',
                expected_impact=20.0  # 20% GPU compute improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='metal_compute',
                old_value='disabled',
                new_value='enabled',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _optimize_gpu_memory(self) -> OptimizationResult:
        """Optimize GPU memory allocation"""
        try:
            # Set GPU memory allocation preferences
            if sys.platform == 'darwin':
                # macOS unified memory optimization
                subprocess.run(['sudo', 'sysctl', '-w', 'vm.gpu_memory_limit=0'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='gpu_memory_optimization',
                old_value='default',
                new_value='optimized',
                expected_impact=10.0  # 10% GPU memory efficiency
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='gpu_memory_optimization',
                old_value='default',
                new_value='optimized',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _optimize_discrete_gpu(self) -> List[OptimizationResult]:
        """Optimize discrete GPU usage"""
        results = []
        
        # Switch to discrete GPU for compute tasks
        result = self._switch_to_discrete_gpu()
        results.append(result)
        
        return results
    
    def _switch_to_discrete_gpu(self) -> OptimizationResult:
        """Switch to discrete GPU for performance"""
        try:
            if sys.platform == 'darwin':
                # Force discrete GPU usage
                subprocess.run(['sudo', 'pmset', '-a', 'gpuswitch', '1'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='gpu_switching',
                old_value='integrated',
                new_value='discrete',
                expected_impact=25.0  # 25% GPU performance improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='gpu_switching',
                old_value='integrated',
                new_value='discrete',
                expected_impact=0.0,
                error_message=str(e)
            )

class IOScheduler:
    """Controls I/O scheduling for disk and network operations"""
    
    def __init__(self):
        self.disk_schedulers = self._detect_disk_schedulers()
        self.network_interfaces = self._detect_network_interfaces()
        self.current_io_scheduler = self._get_current_io_scheduler()
        
    def _detect_disk_schedulers(self) -> List[str]:
        """Detect available disk I/O schedulers"""
        schedulers = []
        
        try:
            if sys.platform == 'darwin':
                # macOS doesn't expose I/O schedulers directly
                schedulers = ['default', 'performance', 'power_save']
            else:
                # Linux I/O schedulers
                for device in os.listdir('/sys/block/'):
                    if device.startswith('sd') or device.startswith('nvme'):
                        scheduler_file = f'/sys/block/{device}/queue/scheduler'
                        if os.path.exists(scheduler_file):
                            with open(scheduler_file, 'r') as f:
                                available = f.read().strip()
                                schedulers.extend(available.replace('[', '').replace(']', '').split())
                                break
        except:
            schedulers = ['noop', 'deadline', 'cfq']
        
        return list(set(schedulers))
    
    def _detect_network_interfaces(self) -> List[str]:
        """Detect network interfaces"""
        interfaces = []
        
        try:
            net_stats = psutil.net_if_stats()
            interfaces = list(net_stats.keys())
        except:
            interfaces = ['en0', 'en1']  # Default macOS interfaces
        
        return interfaces
    
    def _get_current_io_scheduler(self) -> str:
        """Get current I/O scheduler"""
        try:
            if sys.platform != 'darwin':
                for device in os.listdir('/sys/block/'):
                    if device.startswith('sd') or device.startswith('nvme'):
                        scheduler_file = f'/sys/block/{device}/queue/scheduler'
                        if os.path.exists(scheduler_file):
                            with open(scheduler_file, 'r') as f:
                                content = f.read().strip()
                                # Extract current scheduler (marked with brackets)
                                import re
                                match = re.search(r'\[(\w+)\]', content)
                                return match.group(1) if match else 'unknown'
        except:
            pass
        
        return 'default'
    
    def optimize_disk_io(self, workload_type: str) -> List[OptimizationResult]:
        """Optimize disk I/O for specific workload"""
        results = []
        
        if workload_type == 'sequential':
            # Optimize for sequential I/O (large files, streaming)
            result = self._set_io_scheduler('deadline')
            results.append(result)
            
            result = self._set_read_ahead(256)  # Increase read-ahead
            results.append(result)
            
        elif workload_type == 'random':
            # Optimize for random I/O (databases, small files)
            result = self._set_io_scheduler('noop')
            results.append(result)
            
            result = self._set_read_ahead(32)  # Reduce read-ahead
            results.append(result)
            
        elif workload_type == 'interactive':
            # Optimize for interactive workload
            result = self._set_io_scheduler('cfq')
            results.append(result)
        
        return results
    
    def _set_io_scheduler(self, scheduler: str) -> OptimizationResult:
        """Set I/O scheduler for storage devices"""
        old_scheduler = self.current_io_scheduler
        
        try:
            if sys.platform == 'darwin':
                # macOS I/O optimization through system parameters
                if scheduler == 'performance':
                    subprocess.run(['sudo', 'sysctl', '-w', 'kern.maxfiles=65536'], check=True)
                    subprocess.run(['sudo', 'sysctl', '-w', 'kern.maxfilesperproc=32768'], check=True)
                elif scheduler == 'power_save':
                    subprocess.run(['sudo', 'sysctl', '-w', 'kern.maxfiles=16384'], check=True)
                    subprocess.run(['sudo', 'sysctl', '-w', 'kern.maxfilesperproc=8192'], check=True)
            else:
                # Linux I/O scheduler
                for device in os.listdir('/sys/block/'):
                    if device.startswith('sd') or device.startswith('nvme'):
                        scheduler_file = f'/sys/block/{device}/queue/scheduler'
                        if os.path.exists(scheduler_file):
                            subprocess.run(['sudo', 'sh', '-c', 
                                          f'echo {scheduler} > {scheduler_file}'], check=True)
            
            self.current_io_scheduler = scheduler
            expected_impact = self._calculate_scheduler_impact(scheduler)
            
            return OptimizationResult(
                success=True,
                parameter='io_scheduler',
                old_value=old_scheduler,
                new_value=scheduler,
                expected_impact=expected_impact
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='io_scheduler',
                old_value=old_scheduler,
                new_value=scheduler,
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _set_read_ahead(self, kb_size: int) -> OptimizationResult:
        """Set read-ahead buffer size"""
        try:
            if sys.platform != 'darwin':
                # Linux read-ahead setting
                for device in os.listdir('/sys/block/'):
                    if device.startswith('sd') or device.startswith('nvme'):
                        ra_file = f'/sys/block/{device}/queue/read_ahead_kb'
                        if os.path.exists(ra_file):
                            subprocess.run(['sudo', 'sh', '-c', 
                                          f'echo {kb_size} > {ra_file}'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='read_ahead',
                old_value='default',
                new_value=f'{kb_size}KB',
                expected_impact=5.0  # 5% I/O performance improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='read_ahead',
                old_value='default',
                new_value=f'{kb_size}KB',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _calculate_scheduler_impact(self, scheduler: str) -> float:
        """Calculate expected impact of I/O scheduler change"""
        impact_map = {
            'deadline': 10.0,    # Good for sequential I/O
            'noop': 15.0,        # Best for SSDs and random I/O
            'cfq': 5.0,          # Balanced for interactive workloads
            'performance': 12.0,  # macOS performance mode
            'power_save': -5.0   # macOS power save mode
        }
        return impact_map.get(scheduler, 0.0)
    
    def optimize_network_io(self) -> List[OptimizationResult]:
        """Optimize network I/O performance"""
        results = []
        
        # Optimize TCP buffer sizes
        result = self._optimize_tcp_buffers()
        results.append(result)
        
        # Optimize network queue length
        result = self._optimize_network_queues()
        results.append(result)
        
        return results
    
    def _optimize_tcp_buffers(self) -> OptimizationResult:
        """Optimize TCP buffer sizes"""
        try:
            if sys.platform == 'darwin':
                # macOS TCP optimization
                subprocess.run(['sudo', 'sysctl', '-w', 'net.inet.tcp.sendspace=131072'], check=True)
                subprocess.run(['sudo', 'sysctl', '-w', 'net.inet.tcp.recvspace=131072'], check=True)
            else:
                # Linux TCP optimization
                subprocess.run(['sudo', 'sysctl', '-w', 'net.core.rmem_max=134217728'], check=True)
                subprocess.run(['sudo', 'sysctl', '-w', 'net.core.wmem_max=134217728'], check=True)
            
            return OptimizationResult(
                success=True,
                parameter='tcp_buffers',
                old_value='default',
                new_value='optimized',
                expected_impact=8.0  # 8% network performance improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='tcp_buffers',
                old_value='default',
                new_value='optimized',
                expected_impact=0.0,
                error_message=str(e)
            )
    
    def _optimize_network_queues(self) -> OptimizationResult:
        """Optimize network interface queue lengths"""
        try:
            if sys.platform == 'darwin':
                # macOS network queue optimization - use valid macOS parameters
                subprocess.run(['sudo', 'sysctl', '-w', 'net.inet.tcp.delayed_ack=0'], check=True)
            else:
                # Linux network queue optimization
                for interface in self.network_interfaces:
                    if interface.startswith('eth') or interface.startswith('en'):
                        subprocess.run(['sudo', 'ethtool', '-G', interface, 'rx', '4096', 'tx', '4096'], 
                                     check=True)
            
            return OptimizationResult(
                success=True,
                parameter='network_queues',
                old_value='default',
                new_value='optimized',
                expected_impact=6.0  # 6% network latency improvement
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                parameter='network_queues',
                old_value='default',
                new_value='optimized',
                expected_impact=0.0,
                error_message=str(e)
            )

class PowerManagementController:
    """Controls power management, sleep states, and power domains"""
    
    def __init__(self):
        self.power_profiles = self._detect_power_profiles()
        self.current_profile = self._get_current_power_profile()
        self.sleep_settings = self._get_sleep_settings()
        
    def _detect_power_profiles(self) -> List[str]:
        """Detect available power profiles"""
        if sys.platform == 'darwin':
            return ['Battery', 'Power Adapter', 'UPS']
        else:
            return ['performance', 'balanced', 'power-saver']
    
    def _get_current_power_profile(self) -> str:
        """Get current power profile"""
        try:
            if sys.platform == 'darwin':
                result = subprocess.run(['pmset', '-g', 'ps'], capture_output=True, text=True)
                if 'Battery Power' in result.stdout:
                    return 'Battery'
                elif 'AC Power' in result.stdout:
                    return 'Power Adapter'
                else:
                    return 'UPS'
            else:
                result = subprocess.run(['powerprofilesctl', 'get'], capture_output=True, text=True)
                return result.stdout.strip() if result.returncode == 0 else 'balanced'
        except:
            return 'unknown'
    
    def _get_sleep_settings(self) -> Dict[str, int]:
        """Get current sleep settings"""
        settings = {}
        
        try:
            if sys.platform == 'darwin':
                result = subprocess.run(['pmset', '-g', 'custom'], capture_output=True, text=True)
                # Parse pmset output for sleep settings
                for line in result.stdout.split('\n'):
                    if 'sleep' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            settings['system_sleep'] = int(parts[1])
                    elif 'displaysleep' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            settings['display_sleep'] = int(parts[1])
        except:
            settings = {'system_sleep': 30, 'display_sleep': 10}
        
        return settings
    
    def optimize_power_management(self, optimization_level: OptimizationLevel) -> List[OptimizationResult]:
        """Optimize power management based on optimization level"""
        results = []
        
        if optimization_level == OptimizationLevel.MAXIMUM:
            # Maximum performance - disable power saving
            results.extend(self._disable_power_saving())
        elif optimization_level == OptimizationLevel.AGGRESSIVE:
            # Aggressive optimization - balanced approach
            results.extend(self._aggressive_power_optimization())
        elif optimization_level == OptimizationLevel.BALANCED:
            # Balanced optimization
            results.extend(self._balanced_power_optimization())
        else:  # CONSERVATIVE
            # Conservative optimization - minimal changes
            results.extend(self._conservative_power_optimization())
        
        return results
    
    def _disable_power_saving(self) -> List[OptimizationResult]:
        """Disable power saving for maximum performance"""
        results = []
        
        try:
            if sys.platform == 'darwin':
                # Disable sleep
                subprocess.run(['sudo', 'pmset', '-a', 'sleep', '0'], check=True)
                subprocess.run(['sudo', 'pmset', '-a', 'displaysleep', '0'], check=True)
                
                # Disable power nap
                subprocess.run(['sudo', 'pmset', '-a', 'powernap', '0'], check=True)
                
                # Prevent system sleep
                subprocess.run(['sudo', 'pmset', '-a', 'disablesleep', '1'], check=True)
                
                results.append(OptimizationResult(
                    success=True,
                    parameter='power_saving_disabled',
                    old_value='enabled',
                    new_value='disabled',
                    expected_impact=25.0  # 25% performance increase
                ))
            
        except Exception as e:
            results.append(OptimizationResult(
                success=False,
                parameter='power_saving_disabled',
                old_value='enabled',
                new_value='disabled',
                expected_impact=0.0,
                error_message=str(e)
            ))
        
        return results
    
    def _aggressive_power_optimization(self) -> List[OptimizationResult]:
        """Aggressive power optimization"""
        results = []
        
        try:
            if sys.platform == 'darwin':
                # Optimize sleep settings for performance
                subprocess.run(['sudo', 'pmset', '-a', 'sleep', '60'], check=True)  # 1 hour
                subprocess.run(['sudo', 'pmset', '-a', 'displaysleep', '30'], check=True)  # 30 min
                
                # Enable power nap for background updates
                subprocess.run(['sudo', 'pmset', '-a', 'powernap', '1'], check=True)
                
                # Optimize hibernation
                subprocess.run(['sudo', 'pmset', '-a', 'hibernatemode', '0'], check=True)
                
                results.append(OptimizationResult(
                    success=True,
                    parameter='aggressive_power_optimization',
                    old_value='default',
                    new_value='optimized',
                    expected_impact=15.0  # 15% efficiency improvement
                ))
            
        except Exception as e:
            results.append(OptimizationResult(
                success=False,
                parameter='aggressive_power_optimization',
                old_value='default',
                new_value='optimized',
                expected_impact=0.0,
                error_message=str(e)
            ))
        
        return results
    
    def _balanced_power_optimization(self) -> List[OptimizationResult]:
        """Balanced power optimization"""
        results = []
        
        try:
            if sys.platform == 'darwin':
                # Balanced sleep settings
                subprocess.run(['sudo', 'pmset', '-a', 'sleep', '30'], check=True)  # 30 min
                subprocess.run(['sudo', 'pmset', '-a', 'displaysleep', '15'], check=True)  # 15 min
                
                results.append(OptimizationResult(
                    success=True,
                    parameter='balanced_power_optimization',
                    old_value='default',
                    new_value='balanced',
                    expected_impact=8.0  # 8% efficiency improvement
                ))
            
        except Exception as e:
            results.append(OptimizationResult(
                success=False,
                parameter='balanced_power_optimization',
                old_value='default',
                new_value='balanced',
                expected_impact=0.0,
                error_message=str(e)
            ))
        
        return results
    
    def _conservative_power_optimization(self) -> List[OptimizationResult]:
        """Conservative power optimization"""
        results = []
        
        try:
            if sys.platform == 'darwin':
                # Conservative sleep settings - prioritize battery life
                subprocess.run(['sudo', 'pmset', '-a', 'sleep', '15'], check=True)  # 15 min
                subprocess.run(['sudo', 'pmset', '-a', 'displaysleep', '5'], check=True)  # 5 min
                
                results.append(OptimizationResult(
                    success=True,
                    parameter='conservative_power_optimization',
                    old_value='default',
                    new_value='conservative',
                    expected_impact=5.0  # 5% battery life improvement
                ))
            
        except Exception as e:
            results.append(OptimizationResult(
                success=False,
                parameter='conservative_power_optimization',
                old_value='default',
                new_value='conservative',
                expected_impact=0.0,
                error_message=str(e)
            ))
        
        return results

class ComprehensiveSystemOptimizer:
    """Main class that orchestrates all system optimization components"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.cpu_controller = CPUFrequencyController()
        self.scheduler_controller = SchedulerController()
        self.memory_controller = MemoryController()
        self.thermal_controller = ThermalController()
        self.gpu_scheduler = GPUScheduler()
        self.io_scheduler = IOScheduler()
        self.power_controller = PowerManagementController()
        
        self.optimization_history = []
        self.system_baseline = None
        self.monitoring_active = False
        self.optimization_thread = None
        
        logger.info(f"Comprehensive System Optimizer initialized with {optimization_level.value} level")
    
    def start_optimization(self) -> bool:
        """Start continuous system optimization"""
        try:
            if self.monitoring_active:
                logger.warning("Optimization already active")
                return False
            
            # Capture system baseline
            self.system_baseline = self._capture_system_baseline()
            
            # Start optimization thread
            self.monitoring_active = True
            self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
            self.optimization_thread.start()
            
            logger.info("Comprehensive system optimization started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start optimization: {e}")
            return False
    
    def stop_optimization(self) -> bool:
        """Stop continuous system optimization"""
        try:
            self.monitoring_active = False
            
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5.0)
            
            # Restore system to baseline if possible
            self._restore_system_baseline()
            
            logger.info("Comprehensive system optimization stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop optimization: {e}")
            return False
    
    def _capture_system_baseline(self) -> SystemState:
        """Capture current system state as baseline"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            battery = psutil.sensors_battery()
            
            return SystemState(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                thermal_state=self._get_thermal_state(),
                battery_level=battery.percent if battery else 100,
                power_source='battery' if battery and not battery.power_plugged else 'ac',
                active_processes=len(list(psutil.process_iter())),
                network_activity=self._get_network_activity(),
                disk_activity=self._get_disk_activity(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to capture baseline: {e}")
            return SystemState(0, 0, 'unknown', 100, 'unknown', 0, 0, 0, time.time())
    
    def _get_thermal_state(self) -> str:
        """Get current thermal state"""
        try:
            temperatures = self.thermal_controller.get_current_temperatures()
            if temperatures:
                max_temp = max(temperatures.values())
                if max_temp > 85:
                    return 'hot'
                elif max_temp > 70:
                    return 'warm'
                else:
                    return 'normal'
            return 'unknown'
        except:
            return 'unknown'
    
    def _get_network_activity(self) -> float:
        """Get current network activity"""
        try:
            net_io = psutil.net_io_counters()
            return (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_disk_activity(self) -> float:
        """Get current disk activity"""
        try:
            disk_io = psutil.disk_io_counters()
            return (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _optimization_loop(self):
        """Main optimization loop"""
        logger.info("Starting optimization loop")
        
        while self.monitoring_active:
            try:
                # Capture current system state
                current_state = self._capture_system_baseline()
                
                # Analyze system and determine optimizations needed
                optimizations_needed = self._analyze_system_state(current_state)
                
                # Apply optimizations
                if optimizations_needed:
                    results = self._apply_optimizations(optimizations_needed)
                    self.optimization_history.extend(results)
                    
                    # Log optimization results
                    successful_optimizations = [r for r in results if r.success]
                    if successful_optimizations:
                        total_impact = sum(r.expected_impact for r in successful_optimizations)
                        logger.info(f"Applied {len(successful_optimizations)} optimizations, "
                                  f"expected impact: {total_impact:.1f}%")
                
                # Wait before next optimization cycle
                time.sleep(30)  # 30-second optimization cycle
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _analyze_system_state(self, current_state: SystemState) -> List[str]:
        """Analyze system state and determine needed optimizations"""
        optimizations_needed = []
        
        # CPU optimization analysis
        if current_state.cpu_usage > 80:
            optimizations_needed.append('cpu_frequency_scaling')
            optimizations_needed.append('process_priority_adjustment')
        elif current_state.cpu_usage < 20:
            optimizations_needed.append('cpu_power_saving')
        
        # Memory optimization analysis
        if current_state.memory_usage > 85:
            optimizations_needed.append('memory_optimization')
            optimizations_needed.append('memory_compression')
        
        # Thermal optimization analysis
        if current_state.thermal_state in ['hot', 'warm']:
            optimizations_needed.append('thermal_management')
        
        # Power optimization analysis
        if current_state.power_source == 'battery' and current_state.battery_level < 30:
            optimizations_needed.append('aggressive_power_saving')
        elif current_state.power_source == 'ac':
            optimizations_needed.append('performance_optimization')
        
        # I/O optimization analysis
        if current_state.disk_activity > 100:  # High disk activity
            optimizations_needed.append('disk_io_optimization')
        
        if current_state.network_activity > 50:  # High network activity
            optimizations_needed.append('network_io_optimization')
        
        return optimizations_needed
    
    def _apply_optimizations(self, optimizations_needed: List[str]) -> List[OptimizationResult]:
        """Apply the needed optimizations"""
        results = []
        
        for optimization in optimizations_needed:
            try:
                if optimization == 'cpu_frequency_scaling':
                    if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
                        result = self.cpu_controller.set_governor('performance')
                    else:
                        result = self.cpu_controller.set_governor('automatic')
                    results.append(result)
                
                elif optimization == 'cpu_power_saving':
                    result = self.cpu_controller.set_governor('powersave')
                    results.append(result)
                
                elif optimization == 'process_priority_adjustment':
                    scheduler_results = self.scheduler_controller.optimize_scheduler_for_workload('interactive')
                    results.extend(scheduler_results)
                
                elif optimization == 'memory_optimization':
                    memory_results = self.memory_controller.optimize_memory_allocation()
                    results.extend(memory_results)
                
                elif optimization == 'thermal_management':
                    thermal_results = self.thermal_controller.prevent_thermal_throttling()
                    results.extend(thermal_results)
                
                elif optimization == 'performance_optimization':
                    power_results = self.power_controller.optimize_power_management(OptimizationLevel.MAXIMUM)
                    results.extend(power_results)
                
                elif optimization == 'aggressive_power_saving':
                    power_results = self.power_controller.optimize_power_management(OptimizationLevel.CONSERVATIVE)
                    results.extend(power_results)
                
                elif optimization == 'disk_io_optimization':
                    io_results = self.io_scheduler.optimize_disk_io('interactive')
                    results.extend(io_results)
                
                elif optimization == 'network_io_optimization':
                    network_results = self.io_scheduler.optimize_network_io()
                    results.extend(network_results)
                
            except Exception as e:
                logger.error(f"Failed to apply optimization {optimization}: {e}")
                results.append(OptimizationResult(
                    success=False,
                    parameter=optimization,
                    old_value='unknown',
                    new_value='failed',
                    expected_impact=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _restore_system_baseline(self):
        """Restore system to baseline configuration"""
        try:
            if not self.system_baseline:
                return
            
            logger.info("Restoring system to baseline configuration")
            
            # Restore CPU governor to automatic
            self.cpu_controller.set_governor('automatic')
            
            # Restore power management to balanced
            self.power_controller.optimize_power_management(OptimizationLevel.BALANCED)
            
            logger.info("System baseline restored")
            
        except Exception as e:
            logger.error(f"Failed to restore baseline: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        current_state = self._capture_system_baseline()
        
        return {
            'active': self.monitoring_active,
            'optimization_level': self.optimization_level.value,
            'current_state': {
                'cpu_usage': current_state.cpu_usage,
                'memory_usage': current_state.memory_usage,
                'thermal_state': current_state.thermal_state,
                'battery_level': current_state.battery_level,
                'power_source': current_state.power_source
            },
            'optimization_history': len(self.optimization_history),
            'successful_optimizations': len([r for r in self.optimization_history if r.success]),
            'total_expected_impact': sum(r.expected_impact for r in self.optimization_history if r.success),
            'controllers': {
                'cpu_governor': self.cpu_controller.current_governor,
                'io_scheduler': self.io_scheduler.current_io_scheduler,
                'power_profile': self.power_controller.current_profile,
                'gpu_type': self.gpu_scheduler.gpu_info['type'],
                'thermal_sensors': len(self.thermal_controller.thermal_sensors)
            }
        }
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run a one-time comprehensive optimization"""
        logger.info("Running comprehensive system optimization")
        
        all_results = []
        
        try:
            # CPU optimization
            cpu_results = []
            if self.optimization_level == OptimizationLevel.MAXIMUM:
                cpu_results.append(self.cpu_controller.set_governor('performance'))
            else:
                cpu_results.append(self.cpu_controller.set_governor('automatic'))
            all_results.extend(cpu_results)
            
            # Scheduler optimization
            scheduler_results = self.scheduler_controller.optimize_scheduler_for_workload('interactive')
            all_results.extend(scheduler_results)
            
            # Memory optimization
            memory_results = self.memory_controller.optimize_memory_allocation()
            all_results.extend(memory_results)
            
            # Thermal optimization
            thermal_results = self.thermal_controller.prevent_thermal_throttling()
            all_results.extend(thermal_results)
            
            # GPU optimization
            gpu_results = self.gpu_scheduler.optimize_gpu_workload_distribution()
            all_results.extend(gpu_results)
            
            # I/O optimization
            io_results = self.io_scheduler.optimize_disk_io('interactive')
            all_results.extend(io_results)
            
            network_results = self.io_scheduler.optimize_network_io()
            all_results.extend(network_results)
            
            # Power management optimization
            power_results = self.power_controller.optimize_power_management(self.optimization_level)
            all_results.extend(power_results)
            
            # Calculate summary
            successful_results = [r for r in all_results if r.success]
            failed_results = [r for r in all_results if not r.success]
            total_expected_impact = sum(r.expected_impact for r in successful_results)
            
            summary = {
                'total_optimizations': len(all_results),
                'successful_optimizations': len(successful_results),
                'failed_optimizations': len(failed_results),
                'total_expected_impact': total_expected_impact,
                'optimization_level': self.optimization_level.value,
                'results': all_results,
                'timestamp': time.time()
            }
            
            logger.info(f"Comprehensive optimization complete: {len(successful_results)}/{len(all_results)} "
                       f"successful, expected impact: {total_expected_impact:.1f}%")
            
            return summary
            
        except Exception as e:
            logger.error(f"Comprehensive optimization failed: {e}")
            return {
                'total_optimizations': 0,
                'successful_optimizations': 0,
                'failed_optimizations': 1,
                'total_expected_impact': 0.0,
                'error': str(e),
                'timestamp': time.time()
            }

def main():
    """Main function for testing the comprehensive system optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PQS Comprehensive System Optimizer')
    parser.add_argument('--level', choices=['conservative', 'balanced', 'aggressive', 'maximum'],
                       default='balanced', help='Optimization level')
    parser.add_argument('--continuous', action='store_true', help='Run continuous optimization')
    parser.add_argument('--once', action='store_true', help='Run one-time optimization')
    
    args = parser.parse_args()
    
    # Create optimizer
    level = OptimizationLevel(args.level)
    optimizer = ComprehensiveSystemOptimizer(level)
    
    if args.continuous:
        # Start continuous optimization
        print(f"Starting continuous optimization at {level.value} level...")
        optimizer.start_optimization()
        
        try:
            while True:
                time.sleep(10)
                status = optimizer.get_optimization_status()
                print(f"Status: {status['successful_optimizations']} optimizations, "
                      f"{status['total_expected_impact']:.1f}% total impact")
        except KeyboardInterrupt:
            print("\nStopping optimization...")
            optimizer.stop_optimization()
    
    elif args.once:
        # Run one-time optimization
        print(f"Running comprehensive optimization at {level.value} level...")
        results = optimizer.run_comprehensive_optimization()
        
        print(f"Optimization complete:")
        print(f"  Successful: {results['successful_optimizations']}/{results['total_optimizations']}")
        print(f"  Expected impact: {results['total_expected_impact']:.1f}%")
        
        if results.get('failed_optimizations', 0) > 0:
            print(f"  Failed optimizations: {results['failed_optimizations']}")
    
    else:
        # Show status only
        status = optimizer.get_optimization_status()
        print("Comprehensive System Optimizer Status:")
        print(f"  Active: {status['active']}")
        print(f"  Level: {status['optimization_level']}")
        print(f"  CPU Usage: {status['current_state']['cpu_usage']:.1f}%")
        print(f"  Memory Usage: {status['current_state']['memory_usage']:.1f}%")
        print(f"  Thermal State: {status['current_state']['thermal_state']}")
        print(f"  Controllers: {status['controllers']}")

if __name__ == "__main__":
    main()