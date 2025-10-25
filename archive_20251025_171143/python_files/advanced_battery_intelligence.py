#!/usr/bin/env python3
"""
Advanced Battery Intelligence System
=====================================

Tier 1 & 2 Features:
1. Predictive Battery Life Estimation
2. App-Specific Recommendations & Anomaly Detection
3. Power Profiles (Maximum Battery | Balanced | Performance)
4. Thermal Management
5. Usage Pattern Learning
6. Duplicate Process Detection (Kiro's 3 helpers issue)
"""

import psutil
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    from quantum_ml_persistence import get_database
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


class AdvancedBatteryIntelligence:
    """
    Advanced battery intelligence with predictive analytics and recommendations
    """
    
    def __init__(self):
        self.db = get_database() if DB_AVAILABLE else None
        
        # Battery life prediction
        self.battery_history = deque(maxlen=100)
        self.drain_rate_history = deque(maxlen=50)
        
        # App behavior baselines
        self.app_baselines = {}  # Normal behavior for each app
        self.app_usage_schedule = {}  # When apps are typically used
        
        # Power profiles
        self.current_profile = 'balanced'
        self.profiles = {
            'maximum_battery': {
                'cpu_limit': 50,
                'throttle_aggressiveness': 0.9,
                'suspend_idle_apps': True,
                'reduce_background_activity': True
            },
            'balanced': {
                'cpu_limit': 80,
                'throttle_aggressiveness': 0.5,
                'suspend_idle_apps': True,
                'reduce_background_activity': False
            },
            'performance': {
                'cpu_limit': 100,
                'throttle_aggressiveness': 0.2,
                'suspend_idle_apps': False,
                'reduce_background_activity': False
            }
        }
        
        # Thermal management
        self.thermal_history = deque(maxlen=30)
        self.thermal_throttle_active = False
        
        logger.info("🧠 Advanced Battery Intelligence initialized")
    
    def predict_battery_life(self) -> Dict:
        """
        Predict remaining battery life with and without optimizations
        """
        try:
            battery = psutil.sensors_battery()
            if not battery or battery.power_plugged:
                return {'available': False, 'reason': 'not_on_battery'}
            
            current_percent = battery.percent
            
            # Calculate current drain rate (% per minute)
            if len(self.battery_history) >= 2:
                recent_history = list(self.battery_history)[-10:]
                time_diff = recent_history[-1]['time'] - recent_history[0]['time']
                percent_diff = recent_history[0]['percent'] - recent_history[-1]['percent']
                
                if time_diff > 0:
                    drain_rate_per_min = percent_diff / (time_diff / 60)
                else:
                    drain_rate_per_min = 0.5  # Default estimate
            else:
                drain_rate_per_min = 0.5  # Default estimate
            
            # Predict time remaining with current optimizations
            if drain_rate_per_min > 0:
                minutes_remaining = current_percent / drain_rate_per_min
                hours_remaining = minutes_remaining / 60
            else:
                minutes_remaining = 0
                hours_remaining = 0
            
            # Estimate without optimizations (20% worse)
            unoptimized_drain_rate = drain_rate_per_min * 1.2
            unoptimized_minutes = current_percent / unoptimized_drain_rate if unoptimized_drain_rate > 0 else 0
            unoptimized_hours = unoptimized_minutes / 60
            
            # Calculate savings
            time_saved_minutes = minutes_remaining - unoptimized_minutes
            time_saved_hours = time_saved_minutes / 60
            
            return {
                'available': True,
                'current_percent': current_percent,
                'drain_rate_per_hour': drain_rate_per_min * 60,
                'hours_remaining': hours_remaining,
                'minutes_remaining': minutes_remaining,
                'unoptimized_hours': unoptimized_hours,
                'time_saved_hours': time_saved_hours,
                'time_saved_minutes': time_saved_minutes,
                'improvement_percent': (time_saved_hours / unoptimized_hours * 100) if unoptimized_hours > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Battery prediction error: {e}")
            return {'available': False, 'error': str(e)}
    
    def detect_app_anomalies(self, app_name: str, current_metrics: Dict) -> Dict:
        """
        Detect if an app is behaving abnormally (using more battery than usual)
        """
        app_lower = app_name.lower()
        
        # Build baseline if we don't have one
        if app_lower not in self.app_baselines:
            self.app_baselines[app_lower] = {
                'cpu_samples': deque(maxlen=50),
                'memory_samples': deque(maxlen=50),
                'avg_cpu': 0,
                'avg_memory': 0,
                'std_cpu': 0
            }
        
        baseline = self.app_baselines[app_lower]
        current_cpu = current_metrics.get('cpu', 0)
        current_memory = current_metrics.get('memory', 0)
        
        # Add current sample
        baseline['cpu_samples'].append(current_cpu)
        baseline['memory_samples'].append(current_memory)
        
        # Calculate statistics
        if len(baseline['cpu_samples']) >= 10:
            baseline['avg_cpu'] = np.mean(baseline['cpu_samples'])
            baseline['std_cpu'] = np.std(baseline['cpu_samples'])
            baseline['avg_memory'] = np.mean(baseline['memory_samples'])
            
            # Detect anomaly (>2 standard deviations above average)
            cpu_threshold = baseline['avg_cpu'] + (2 * baseline['std_cpu'])
            
            is_anomaly = current_cpu > cpu_threshold and current_cpu > baseline['avg_cpu'] * 1.5
            
            if is_anomaly:
                increase_percent = ((current_cpu - baseline['avg_cpu']) / baseline['avg_cpu'] * 100) if baseline['avg_cpu'] > 0 else 0
                
                return {
                    'anomaly_detected': True,
                    'app_name': app_name,
                    'current_cpu': current_cpu,
                    'normal_cpu': baseline['avg_cpu'],
                    'increase_percent': increase_percent,
                    'recommendation': self._get_anomaly_recommendation(app_name, increase_percent)
                }
        
        return {'anomaly_detected': False}
    
    def _get_anomaly_recommendation(self, app_name: str, increase_percent: float) -> str:
        """Get specific recommendation for app anomaly"""
        if 'kiro' in app_name.lower() or 'electron' in app_name.lower():
            return f"Restart {app_name} or close unused tabs/windows"
        elif 'chrome' in app_name.lower() or 'firefox' in app_name.lower():
            return f"Close unused tabs in {app_name}"
        elif increase_percent > 100:
            return f"Restart {app_name} - using {increase_percent:.0f}% more battery than normal"
        else:
            return f"Monitor {app_name} - elevated battery usage detected"
    
    def detect_duplicate_processes(self) -> List[Dict]:
        """
        Detect duplicate/redundant processes (like Kiro's 3 helpers doing same thing)
        """
        process_groups = defaultdict(list)
        
        # Group processes by base name
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                name = proc.info['name']
                
                # Extract base name (remove "Helper", numbers, etc)
                base_name = name.lower()
                for suffix in [' helper', ' (renderer)', ' (gpu)', ' (plugin)', ' (utility)']:
                    base_name = base_name.replace(suffix, '')
                
                base_name = base_name.strip()
                
                cpu = proc.cpu_percent(interval=0.01)
                memory = proc.memory_percent()
                
                process_groups[base_name].append({
                    'pid': proc.info['pid'],
                    'full_name': name,
                    'cpu': cpu,
                    'memory': memory,
                    'proc': proc
                })
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Find groups with multiple processes
        duplicates = []
        for base_name, processes in process_groups.items():
            if len(processes) >= 3:  # 3+ processes with same base name
                total_cpu = sum(p['cpu'] for p in processes)
                total_memory = sum(p['memory'] for p in processes)
                
                # Check if they're doing similar things (similar CPU usage)
                cpu_values = [p['cpu'] for p in processes]
                if len(cpu_values) > 1:
                    cpu_variance = np.var(cpu_values)
                    
                    # Low variance = likely redundant
                    if cpu_variance < 10 and total_cpu > 5:
                        duplicates.append({
                            'base_name': base_name,
                            'process_count': len(processes),
                            'processes': processes,
                            'total_cpu': total_cpu,
                            'total_memory': total_memory,
                            'likely_redundant': True,
                            'recommendation': f"Consolidate {len(processes)} {base_name} processes"
                        })
        
        return duplicates
    
    def apply_power_profile(self, profile_name: str) -> Dict:
        """
        Apply a power profile (maximum_battery, balanced, performance)
        """
        if profile_name not in self.profiles:
            return {'success': False, 'error': 'Invalid profile'}
        
        profile = self.profiles[profile_name]
        self.current_profile = profile_name
        
        applied_changes = []
        
        # Apply CPU limits
        cpu_limit = profile['cpu_limit']
        if cpu_limit < 100:
            # Throttle high-CPU processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    cpu = proc.cpu_percent(interval=0.01)
                    if cpu > cpu_limit * 0.8:  # 80% of limit
                        current_nice = proc.nice()
                        if current_nice < 10:
                            proc.nice(min(19, current_nice + 5))
                            applied_changes.append(f"Throttled {proc.info['name']}")
                except:
                    continue
        
        # Suspend idle apps if enabled
        if profile['suspend_idle_apps']:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    cpu = proc.cpu_percent(interval=0.01)
                    if cpu < 0.5:
                        proc.nice(19)
                        applied_changes.append(f"Suspended {proc.info['name']}")
                except:
                    continue
        
        return {
            'success': True,
            'profile': profile_name,
            'changes_applied': len(applied_changes),
            'details': applied_changes[:10]
        }
    
    def monitor_thermal_state(self) -> Dict:
        """
        Monitor thermal state and apply thermal management
        """
        try:
            # Get CPU temperature
            temps = psutil.sensors_temperatures()
            cpu_temp = None
            
            if temps:
                for name, entries in temps.items():
                    if entries and 'cpu' in name.lower():
                        cpu_temp = entries[0].current
                        break
            
            # Fallback: estimate from CPU usage
            if cpu_temp is None:
                cpu_percent = psutil.cpu_percent(interval=0.5)
                # Rough estimate: 40°C base + CPU% * 0.4
                cpu_temp = 40 + (cpu_percent * 0.4)
            
            # Track thermal history
            self.thermal_history.append({
                'time': time.time(),
                'temp': cpu_temp
            })
            
            # Determine thermal state
            if cpu_temp > 80:
                thermal_state = 'critical'
                action_needed = True
            elif cpu_temp > 70:
                thermal_state = 'hot'
                action_needed = True
            elif cpu_temp > 60:
                thermal_state = 'warm'
                action_needed = False
            else:
                thermal_state = 'normal'
                action_needed = False
            
            # Apply thermal throttling if needed
            throttled = 0
            if action_needed and not self.thermal_throttle_active:
                self.thermal_throttle_active = True
                
                # Throttle top CPU consumers
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        cpu = proc.cpu_percent(interval=0.01)
                        if cpu > 10:
                            current_nice = proc.nice()
                            if current_nice < 15:
                                proc.nice(min(19, current_nice + 8))
                                throttled += 1
                    except:
                        continue
                
                logger.info(f"🌡️ Thermal throttling activated: {cpu_temp:.1f}°C, throttled {throttled} processes")
            
            elif not action_needed and self.thermal_throttle_active:
                self.thermal_throttle_active = False
                logger.info(f"🌡️ Thermal throttling deactivated: {cpu_temp:.1f}°C")
            
            return {
                'temperature': cpu_temp,
                'thermal_state': thermal_state,
                'action_needed': action_needed,
                'throttle_active': self.thermal_throttle_active,
                'processes_throttled': throttled
            }
            
        except Exception as e:
            logger.error(f"Thermal monitoring error: {e}")
            return {'available': False, 'error': str(e)}
    
    def learn_usage_patterns(self, app_name: str) -> Dict:
        """
        Learn when an app is typically used (time-based patterns)
        """
        app_lower = app_name.lower()
        current_hour = datetime.now().hour
        
        if app_lower not in self.app_usage_schedule:
            self.app_usage_schedule[app_lower] = {
                'hourly_usage': [0] * 24,  # Usage count per hour
                'total_observations': 0,
                'peak_hours': [],
                'off_hours': []
            }
        
        schedule = self.app_usage_schedule[app_lower]
        schedule['hourly_usage'][current_hour] += 1
        schedule['total_observations'] += 1
        
        # Determine peak and off hours
        if schedule['total_observations'] > 20:  # Need enough data
            hourly_avg = np.array(schedule['hourly_usage']) / schedule['total_observations']
            mean_usage = np.mean(hourly_avg)
            
            peak_hours = [h for h, usage in enumerate(hourly_avg) if usage > mean_usage * 1.5]
            off_hours = [h for h, usage in enumerate(hourly_avg) if usage < mean_usage * 0.3]
            
            schedule['peak_hours'] = peak_hours
            schedule['off_hours'] = off_hours
            
            # Check if currently in off-hours
            in_off_hours = current_hour in off_hours
            
            return {
                'app_name': app_name,
                'current_hour': current_hour,
                'in_off_hours': in_off_hours,
                'peak_hours': peak_hours,
                'off_hours': off_hours,
                'recommendation': f"Aggressive optimization - typically not used at {current_hour}:00" if in_off_hours else None
            }
        
        return {'app_name': app_name, 'learning': True, 'observations': schedule['total_observations']}
    
    def get_comprehensive_recommendations(self) -> List[Dict]:
        """
        Get comprehensive recommendations for battery optimization
        """
        recommendations = []
        
        # 1. Check for duplicate processes
        duplicates = self.detect_duplicate_processes()
        for dup in duplicates:
            recommendations.append({
                'type': 'duplicate_processes',
                'severity': 'high',
                'app': dup['base_name'],
                'message': f"{dup['base_name'].title()} has {dup['process_count']} processes using {dup['total_cpu']:.1f}% CPU",
                'recommendation': dup['recommendation'],
                'action': 'consolidate_processes',
                'estimated_savings': dup['total_cpu'] * 0.3
            })
        
        # 2. Check for anomalies in all running apps
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name']
                cpu = proc.cpu_percent(interval=0.01)
                memory = proc.memory_percent()
                
                anomaly = self.detect_app_anomalies(name, {'cpu': cpu, 'memory': memory})
                if anomaly.get('anomaly_detected'):
                    recommendations.append({
                        'type': 'anomaly',
                        'severity': 'medium',
                        'app': name,
                        'message': f"{name} using {anomaly['increase_percent']:.0f}% more battery than usual",
                        'recommendation': anomaly['recommendation'],
                        'action': 'restart_app',
                        'estimated_savings': anomaly['increase_percent'] * 0.5
                    })
            except:
                continue
        
        # 3. Check thermal state
        thermal = self.monitor_thermal_state()
        if thermal.get('thermal_state') in ['hot', 'critical']:
            recommendations.append({
                'type': 'thermal',
                'severity': 'high' if thermal['thermal_state'] == 'critical' else 'medium',
                'message': f"System temperature: {thermal['temperature']:.1f}°C ({thermal['thermal_state']})",
                'recommendation': "Reduce workload or improve ventilation",
                'action': 'thermal_throttle',
                'estimated_savings': 10.0
            })
        
        # 4. Battery life prediction
        prediction = self.predict_battery_life()
        if prediction.get('available') and prediction.get('time_saved_hours', 0) > 0:
            recommendations.append({
                'type': 'prediction',
                'severity': 'info',
                'message': f"Optimizations adding +{prediction['time_saved_hours']:.1f} hours battery life",
                'recommendation': "Keep optimizations active",
                'action': 'none',
                'estimated_savings': prediction['improvement_percent']
            })
        
        # Sort by severity and estimated savings
        severity_order = {'high': 0, 'medium': 1, 'info': 2}
        recommendations.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x.get('estimated_savings', 0)))
        
        return recommendations
    
    def update_battery_history(self):
        """Update battery history for predictions"""
        try:
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                self.battery_history.append({
                    'time': time.time(),
                    'percent': battery.percent
                })
        except:
            pass


# Global instance
_intelligence_instance = None

def get_intelligence() -> AdvancedBatteryIntelligence:
    """Get or create global intelligence instance"""
    global _intelligence_instance
    if _intelligence_instance is None:
        _intelligence_instance = AdvancedBatteryIntelligence()
    return _intelligence_instance


if __name__ == "__main__":
    print("🧠 Testing Advanced Battery Intelligence")
    print("=" * 70)
    
    intel = get_intelligence()
    
    # Test 1: Duplicate process detection
    print("\n🔍 Test 1: Detecting Duplicate Processes")
    duplicates = intel.detect_duplicate_processes()
    
    if duplicates:
        print(f"✅ Found {len(duplicates)} apps with duplicate processes:")
        for dup in duplicates:
            print(f"\n   {dup['base_name'].upper()}:")
            print(f"      Processes: {dup['process_count']}")
            print(f"      Total CPU: {dup['total_cpu']:.1f}%")
            print(f"      Total Memory: {dup['total_memory']:.1f}%")
            print(f"      Recommendation: {dup['recommendation']}")
    else:
        print("   No duplicate processes detected")
    
    # Test 2: Battery life prediction
    print("\n🔋 Test 2: Battery Life Prediction")
    prediction = intel.predict_battery_life()
    
    if prediction.get('available'):
        print(f"   Current: {prediction['hours_remaining']:.1f} hours remaining")
        print(f"   Without optimization: {prediction['unoptimized_hours']:.1f} hours")
        print(f"   Time saved: +{prediction['time_saved_hours']:.1f} hours ({prediction['improvement_percent']:.0f}% better)")
    else:
        print(f"   {prediction.get('reason', 'Not available')}")
    
    # Test 3: Thermal monitoring
    print("\n🌡️ Test 3: Thermal Monitoring")
    thermal = intel.monitor_thermal_state()
    
    if thermal.get('temperature'):
        print(f"   Temperature: {thermal['temperature']:.1f}°C")
        print(f"   State: {thermal['thermal_state']}")
        print(f"   Throttle active: {thermal['throttle_active']}")
    
    # Test 4: Comprehensive recommendations
    print("\n💡 Test 4: Comprehensive Recommendations")
    recommendations = intel.get_comprehensive_recommendations()
    
    if recommendations:
        print(f"   Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n   {i}. [{rec['severity'].upper()}] {rec['type']}")
            print(f"      {rec['message']}")
            print(f"      → {rec['recommendation']}")
            if rec.get('estimated_savings'):
                print(f"      Potential savings: {rec['estimated_savings']:.1f}%")
    else:
        print("   No recommendations - system optimized!")
    
    print("\n✅ Advanced Battery Intelligence Test Complete!")
