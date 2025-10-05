import subprocess
import json
import time
from datetime import datetime, timedelta
import psutil
import threading
from collections import defaultdict, deque

class M3EnergyScheduler:
    """Advanced energy-aware scheduling for M3 MacBook Air"""
    
    def __init__(self):
        self.thermal_history = deque(maxlen=100)
        self.performance_cores = []
        self.efficiency_cores = []
        self.neural_engine_usage = 0
        self.gpu_usage = 0
        self.memory_pressure = 0
        
    def get_m3_chip_metrics(self):
        """Get M3-specific performance and thermal data"""
        try:
            # CPU cluster information (P-cores vs E-cores)
            cpu_info = subprocess.check_output([
                "system_profiler", "SPHardwareDataType", "-json"
            ], text=True)
            
            # Thermal state via powermetrics (requires sudo, fallback to estimates)
            thermal_state = self.get_thermal_state()
            
            # Memory pressure from kernel
            memory_pressure = self.get_memory_pressure()
            
            # GPU usage estimation
            gpu_usage = self.estimate_gpu_usage()
            
            return {
                "thermal_state": thermal_state,
                "memory_pressure": memory_pressure,
                "gpu_usage": gpu_usage,
                "cpu_clusters": self.analyze_cpu_clusters(),
                "neural_engine_load": self.estimate_neural_engine_usage()
            }
        except Exception as e:
            print(f"M3 metrics error: {e}")
            return {}
    
    def get_thermal_state(self):
        """Estimate thermal state from CPU frequency and usage"""
        try:
            # Use CPU frequency as thermal proxy
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # M3 base frequency is ~3.2GHz, max boost ~4.0GHz
            if cpu_freq and cpu_freq.current:
                thermal_ratio = cpu_freq.current / 4000  # Normalize to max freq
                
                # Estimate thermal state
                if thermal_ratio > 0.95 and cpu_percent > 80:
                    return "hot"  # Likely thermal throttling
                elif thermal_ratio > 0.85 and cpu_percent > 60:
                    return "warm"  # Elevated temperature
                else:
                    return "cool"  # Normal operation
            
            return "unknown"
        except:
            return "unknown"
    
    def get_memory_pressure(self):
        """Get memory pressure from system"""
        try:
            vm_stat = subprocess.check_output(["vm_stat"], text=True)
            
            # Parse memory pressure indicators
            lines = vm_stat.split('\n')
            page_size = 4096  # 4KB pages on Apple Silicon
            
            free_pages = 0
            inactive_pages = 0
            
            for line in lines:
                if "Pages free:" in line:
                    free_pages = int(line.split(':')[1].strip().rstrip('.'))
                elif "Pages inactive:" in line:
                    inactive_pages = int(line.split(':')[1].strip().rstrip('.'))
            
            # Calculate available memory
            available_mb = (free_pages + inactive_pages) * page_size / (1024 * 1024)
            total_memory = psutil.virtual_memory().total / (1024 * 1024)
            
            pressure_ratio = 1 - (available_mb / total_memory)
            
            if pressure_ratio > 0.9:
                return "high"
            elif pressure_ratio > 0.7:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            return "unknown"
    
    def estimate_gpu_usage(self):
        """Estimate GPU usage from process analysis"""
        gpu_intensive_apps = [
            "Final Cut Pro", "Motion", "Compressor", "Adobe Premiere",
            "Adobe After Effects", "Blender", "Unity", "Unreal Engine",
            "Chrome", "Safari", "Firefox"  # WebGL/video acceleration
        ]
        
        gpu_load = 0
        for proc in psutil.process_iter(['name', 'cpu_percent']):
            try:
                name = proc.info['name']
                cpu = proc.info['cpu_percent'] or 0
                
                for gpu_app in gpu_intensive_apps:
                    if gpu_app.lower() in name.lower():
                        # Estimate GPU load based on CPU usage of GPU-intensive apps
                        gpu_load += min(cpu * 1.5, 100)  # GPU often higher than CPU
                        break
            except:
                continue
        
        return min(gpu_load, 100)
    
    def analyze_cpu_clusters(self):
        """Analyze P-core vs E-core usage patterns"""
        try:
            # M3 has 4 P-cores + 4 E-cores
            cpu_count = psutil.cpu_count()
            per_cpu = psutil.cpu_percent(percpu=True, interval=1)
            
            if len(per_cpu) >= 8:
                # Assume first 4 are P-cores, next 4 are E-cores (typical Apple Silicon layout)
                p_cores = per_cpu[:4]
                e_cores = per_cpu[4:8]
                
                return {
                    "p_core_avg": sum(p_cores) / len(p_cores),
                    "e_core_avg": sum(e_cores) / len(e_cores),
                    "p_core_max": max(p_cores),
                    "e_core_max": max(e_cores),
                    "cluster_imbalance": abs(sum(p_cores) - sum(e_cores)) / 4
                }
            
            return {"error": "Cannot determine CPU clusters"}
        except:
            return {"error": "CPU analysis failed"}
    
    def estimate_neural_engine_usage(self):
        """Estimate Neural Engine usage from ML-intensive processes"""
        ml_apps = [
            "Photos", "Siri", "Spotlight", "CoreML", "CreateML",
            "Xcode", "Instruments", "Python", "TensorFlow", "PyTorch"
        ]
        
        neural_load = 0
        for proc in psutil.process_iter(['name', 'cpu_percent']):
            try:
                name = proc.info['name']
                cpu = proc.info['cpu_percent'] or 0
                
                for ml_app in ml_apps:
                    if ml_app.lower() in name.lower():
                        # Neural Engine usage correlates with ML workloads
                        neural_load += cpu * 0.8  # Estimate Neural Engine offload
                        break
            except:
                continue
        
        return min(neural_load, 100)

class AdaptiveScheduler:
    """Intelligent scheduling based on system state and user patterns"""
    
    def __init__(self):
        self.user_patterns = defaultdict(list)
        self.app_priorities = {}
        self.energy_scheduler = M3EnergyScheduler()
        
    def learn_usage_patterns(self):
        """Learn when user typically uses different apps"""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        active_apps = []
        for proc in psutil.process_iter(['name']):
            try:
                active_apps.append(proc.info['name'])
            except:
                continue
        
        # Store usage patterns by time
        pattern_key = f"{current_day}_{current_hour}"
        self.user_patterns[pattern_key] = active_apps
        
        return pattern_key, active_apps
    
    def predict_app_usage(self, lookahead_hours=2):
        """Predict which apps user will likely need soon"""
        current_time = datetime.now()
        predictions = defaultdict(float)
        
        for i in range(lookahead_hours):
            future_time = current_time + timedelta(hours=i+1)
            pattern_key = f"{future_time.weekday()}_{future_time.hour}"
            
            if pattern_key in self.user_patterns:
                for app in self.user_patterns[pattern_key]:
                    predictions[app] += 1.0 / (i + 1)  # Closer predictions weighted higher
        
        return dict(predictions)
    
    def calculate_app_priority(self, app_name, system_metrics):
        """Calculate dynamic app priority based on multiple factors"""
        base_priority = 50  # Default priority
        
        # Factor 1: User pattern prediction
        predictions = self.predict_app_usage()
        if app_name in predictions:
            base_priority += predictions[app_name] * 20
        
        # Factor 2: App category importance
        critical_apps = ["Terminal", "iTerm", "SSH", "VPN"]
        productivity_apps = ["Xcode", "VS Code", "Sublime", "Vim"]
        communication_apps = ["Slack", "Teams", "Zoom", "Messages"]
        
        if any(critical in app_name for critical in critical_apps):
            base_priority += 40
        elif any(prod in app_name for prod in productivity_apps):
            base_priority += 30
        elif any(comm in app_name for comm in communication_apps):
            base_priority += 20
        
        # Factor 3: System resource impact
        thermal_state = system_metrics.get("thermal_state", "unknown")
        memory_pressure = system_metrics.get("memory_pressure", "unknown")
        
        if thermal_state == "hot":
            base_priority -= 30  # Deprioritize during thermal stress
        elif memory_pressure == "high":
            base_priority -= 20  # Deprioritize during memory pressure
        
        return max(0, min(100, base_priority))
    
    def intelligent_suspension_order(self, candidate_apps, system_metrics):
        """Determine optimal order for suspending apps"""
        app_scores = []
        
        for app_name, pid in candidate_apps:
            priority = self.calculate_app_priority(app_name, system_metrics)
            
            try:
                proc = psutil.Process(pid)
                cpu_usage = proc.cpu_percent()
                memory_mb = proc.memory_info().rss / (1024 * 1024)
                
                # Suspension score: lower = suspend first
                # High resource usage + low priority = suspend first
                suspension_score = priority - (cpu_usage * 2) - (memory_mb / 100)
                
                app_scores.append((suspension_score, app_name, pid))
            except:
                continue
        
        # Sort by suspension score (lowest first)
        app_scores.sort(key=lambda x: x[0])
        return [(name, pid) for score, name, pid in app_scores]

class EnergyAwareOptimizer:
    """Main energy-aware optimization engine"""
    
    def __init__(self):
        self.scheduler = AdaptiveScheduler()
        self.energy_modes = {
            "maximum_performance": {"cpu_threshold": 15, "aggressive": False},
            "balanced": {"cpu_threshold": 8, "aggressive": False},
            "power_saver": {"cpu_threshold": 3, "aggressive": True},
            "thermal_protection": {"cpu_threshold": 1, "aggressive": True}
        }
        self.current_mode = "balanced"
        
    def determine_optimal_energy_mode(self, system_metrics, battery_level):
        """Dynamically select energy mode based on system state"""
        thermal_state = system_metrics.get("thermal_state", "unknown")
        memory_pressure = system_metrics.get("memory_pressure", "unknown")
        gpu_usage = system_metrics.get("gpu_usage", 0)
        
        # Thermal protection mode
        if thermal_state == "hot":
            return "thermal_protection"
        
        # Power saver mode
        if battery_level < 20:
            return "power_saver"
        
        # Maximum performance mode
        if battery_level > 80 and thermal_state == "cool" and gpu_usage > 50:
            return "maximum_performance"
        
        # Default balanced mode
        return "balanced"
    
    def adaptive_process_management(self, system_metrics, battery_level):
        """Advanced process management with energy awareness"""
        optimal_mode = self.determine_optimal_energy_mode(system_metrics, battery_level)
        
        if optimal_mode != self.current_mode:
            print(f"Energy mode changed: {self.current_mode} -> {optimal_mode}")
            self.current_mode = optimal_mode
        
        mode_config = self.energy_modes[optimal_mode]
        
        # Get candidate apps for suspension
        candidate_apps = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                name = proc.info['name']
                pid = proc.info['pid']
                cpu = proc.info['cpu_percent'] or 0
                
                if cpu > mode_config["cpu_threshold"]:
                    candidate_apps.append((name, pid))
            except:
                continue
        
        # Use intelligent ordering
        ordered_apps = self.scheduler.intelligent_suspension_order(
            candidate_apps, system_metrics
        )
        
        return ordered_apps, optimal_mode
    
    def proactive_optimization(self):
        """Proactive optimization based on predictions"""
        predictions = self.scheduler.predict_app_usage(lookahead_hours=1)
        
        # Pre-warm predicted apps (ensure they're not suspended)
        high_probability_apps = [
            app for app, prob in predictions.items() if prob > 0.7
        ]
        
        # Pre-suspend apps unlikely to be used
        low_probability_apps = []
        current_apps = set()
        
        for proc in psutil.process_iter(['name']):
            try:
                current_apps.add(proc.info['name'])
            except:
                continue
        
        for app in current_apps:
            if app not in predictions or predictions[app] < 0.2:
                low_probability_apps.append(app)
        
        return {
            "pre_warm": high_probability_apps,
            "pre_suspend": low_probability_apps
        }

# Integration with existing system
def enhanced_energy_aware_check():
    """Enhanced check with energy-aware scheduling"""
    optimizer = EnergyAwareOptimizer()
    
    # Get M3-specific metrics
    system_metrics = optimizer.scheduler.energy_scheduler.get_m3_chip_metrics()
    battery_level = get_battery_level()  # From existing code
    
    # Learn current usage patterns
    optimizer.scheduler.learn_usage_patterns()
    
    # Get adaptive process management recommendations
    ordered_apps, energy_mode = optimizer.adaptive_process_management(
        system_metrics, battery_level
    )
    
    # Get proactive optimization suggestions
    proactive_actions = optimizer.proactive_optimization()
    
    print(f"Energy Mode: {energy_mode}")
    print(f"System Metrics: {system_metrics}")
    print(f"Ordered suspension candidates: {len(ordered_apps)}")
    print(f"Proactive actions: {proactive_actions}")
    
    return {
        "energy_mode": energy_mode,
        "system_metrics": system_metrics,
        "suspension_order": ordered_apps,
        "proactive_actions": proactive_actions
    }

# Calendar integration for predictive scheduling
class CalendarAwareScheduler:
    """Schedule optimization based on calendar events"""
    
    def get_upcoming_meetings(self):
        """Get upcoming calendar events (requires Calendar.app integration)"""
        try:
            # AppleScript to get calendar events
            script = '''
            tell application "Calendar"
                set today to current date
                set tomorrow to today + (1 * days)
                set upcomingEvents to every event of every calendar whose start date is greater than today and start date is less than tomorrow
                set eventList to {}
                repeat with anEvent in upcomingEvents
                    set end of eventList to (summary of anEvent & "|" & (start date of anEvent as string))
                end repeat
                return eventList as string
            end tell
            '''
            
            result = subprocess.check_output([
                "osascript", "-e", script
            ], text=True).strip()
            
            events = []
            if result:
                for event_str in result.split(", "):
                    if "|" in event_str:
                        title, date_str = event_str.split("|", 1)
                        events.append({"title": title, "date": date_str})
            
            return events
        except:
            return []
    
    def optimize_for_meetings(self, events):
        """Optimize system for upcoming meetings"""
        meeting_keywords = ["meeting", "call", "zoom", "teams", "standup", "sync"]
        
        upcoming_meetings = []
        for event in events:
            title_lower = event["title"].lower()
            if any(keyword in title_lower for keyword in meeting_keywords):
                upcoming_meetings.append(event)
        
        if upcoming_meetings:
            # Prepare for meetings: keep communication apps active
            return {
                "action": "meeting_prep",
                "keep_active": ["Zoom", "Teams", "Slack", "Calendar", "Notes"],
                "suspend_aggressive": ["Chrome", "Safari", "Photoshop", "Xcode"]
            }
        
        return {"action": "normal_operation"}

if __name__ == "__main__":
    # Test the enhanced energy-aware system
    result = enhanced_energy_aware_check()
    print(json.dumps(result, indent=2, default=str))
