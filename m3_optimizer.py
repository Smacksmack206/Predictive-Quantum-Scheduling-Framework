#!/usr/bin/env python3
"""
M3 MacBook Air Advanced Energy Optimizer
Implements cutting-edge energy-aware scheduling and optimization
"""

import subprocess
import json
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import os

class M3ThermalManager:
    """Advanced thermal management for M3 chip"""
    
    def __init__(self):
        self.thermal_history = deque(maxlen=50)
        self.thermal_threshold_hot = 85  # Celsius
        self.thermal_threshold_warm = 70
        
    def get_cpu_temperature(self):
        """Get CPU temperature using powermetrics or estimation"""
        try:
            # Try powermetrics (requires sudo, so fallback to estimation)
            result = subprocess.run([
                "sudo", "powermetrics", "-n", "1", "-s", "cpu_power"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse temperature from powermetrics output
                for line in result.stdout.split('\n'):
                    if 'CPU die temperature' in line:
                        temp = float(line.split(':')[1].strip().split()[0])
                        return temp
        except:
            pass
        
        # Fallback: estimate temperature from CPU frequency and usage
        return self.estimate_temperature()
    
    def estimate_temperature(self):
        """Estimate CPU temperature from frequency and usage"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_freq and cpu_freq.current:
                # M3 base: ~3.2GHz, max: ~4.0GHz
                freq_ratio = cpu_freq.current / 4000.0
                usage_ratio = cpu_percent / 100.0
                
                # Thermal estimation model
                base_temp = 35  # Idle temperature
                freq_impact = freq_ratio * 30  # Frequency contribution
                usage_impact = usage_ratio * 25  # Usage contribution
                
                estimated_temp = base_temp + freq_impact + usage_impact
                return min(estimated_temp, 100)  # Cap at 100°C
            
            return 45  # Default safe estimate
        except:
            return 45
    
    def get_thermal_state(self):
        """Get current thermal state"""
        temp = self.get_cpu_temperature()
        self.thermal_history.append((time.time(), temp))
        
        if temp > self.thermal_threshold_hot:
            return "critical", temp
        elif temp > self.thermal_threshold_warm:
            return "warm", temp
        else:
            return "cool", temp
    
    def predict_thermal_throttling(self):
        """Predict if thermal throttling is imminent"""
        if len(self.thermal_history) < 5:
            return False, 0
        
        # Calculate temperature trend
        recent_temps = [temp for _, temp in list(self.thermal_history)[-5:]]
        temp_trend = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
        
        current_temp = recent_temps[-1]
        predicted_temp = current_temp + (temp_trend * 3)  # 3 intervals ahead
        
        throttling_risk = predicted_temp > self.thermal_threshold_hot
        return throttling_risk, predicted_temp

class M3PerformanceManager:
    """Performance cluster management for M3 chip"""
    
    def __init__(self):
        self.p_core_count = 4  # Performance cores
        self.e_core_count = 4  # Efficiency cores
        
    def get_core_usage(self):
        """Get per-core CPU usage"""
        try:
            per_cpu = psutil.cpu_percent(percpu=True, interval=1)
            
            if len(per_cpu) >= 8:
                # Assume first 4 are P-cores, next 4 are E-cores
                p_cores = per_cpu[:4]
                e_cores = per_cpu[4:8]
                
                return {
                    "p_cores": p_cores,
                    "e_cores": e_cores,
                    "p_core_avg": sum(p_cores) / len(p_cores),
                    "e_core_avg": sum(e_cores) / len(e_cores),
                    "total_usage": sum(per_cpu) / len(per_cpu)
                }
            
            return {"error": "Insufficient CPU data"}
        except Exception as e:
            return {"error": str(e)}
    
    def recommend_core_allocation(self, app_type):
        """Recommend optimal core allocation for app types"""
        core_usage = self.get_core_usage()
        
        if "error" in core_usage:
            return "balanced"
        
        p_core_load = core_usage["p_core_avg"]
        e_core_load = core_usage["e_core_avg"]
        
        # App type recommendations
        if app_type == "interactive":  # User-facing apps
            return "p_cores" if p_core_load < 70 else "e_cores"
        elif app_type == "background":  # Background tasks
            return "e_cores"
        elif app_type == "compute":  # CPU-intensive work
            return "p_cores" if p_core_load < 50 else "balanced"
        else:
            return "balanced"

class PredictiveScheduler:
    """AI-powered predictive scheduling"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.usage_patterns = defaultdict(list)
        self.init_prediction_db()
        
    def init_prediction_db(self):
        """Initialize prediction database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS usage_patterns (
                timestamp TEXT,
                hour INTEGER,
                day_of_week INTEGER,
                app_name TEXT,
                usage_duration INTEGER,
                context TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_app_usage(self, app_name, duration, context="normal"):
        """Log app usage for pattern learning"""
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO usage_patterns 
            (timestamp, hour, day_of_week, app_name, usage_duration, context)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now.isoformat(), now.hour, now.weekday(), app_name, duration, context))
        conn.commit()
        conn.close()
    
    def predict_app_usage(self, lookahead_minutes=60):
        """Predict app usage in the next period"""
        now = datetime.now()
        target_hour = now.hour
        target_day = now.weekday()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT app_name, AVG(usage_duration), COUNT(*) as frequency
            FROM usage_patterns 
            WHERE hour = ? AND day_of_week = ?
            GROUP BY app_name
            ORDER BY frequency DESC, AVG(usage_duration) DESC
        ''', (target_hour, target_day))
        
        predictions = []
        for row in cursor.fetchall():
            app_name, avg_duration, frequency = row
            confidence = min(frequency / 10.0, 1.0)  # Normalize confidence
            predictions.append({
                "app": app_name,
                "probability": confidence,
                "expected_duration": avg_duration
            })
        
        conn.close()
        return predictions[:10]  # Top 10 predictions
    
    def get_optimal_suspension_time(self, app_name):
        """Get optimal suspension time for an app"""
        now = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT AVG(usage_duration)
            FROM usage_patterns 
            WHERE app_name = ? AND hour = ? AND day_of_week = ?
        ''', (app_name, now.hour, now.weekday()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            # Suspend for half the typical usage gap
            return int(result[0] * 0.5)
        
        return 300  # Default 5 minutes

class CalendarIntegration:
    """Calendar-aware optimization"""
    
    def get_upcoming_events(self, hours_ahead=4):
        """Get upcoming calendar events"""
        try:
            script = f'''
            tell application "Calendar"
                set startDate to current date
                set endDate to startDate + ({hours_ahead} * hours)
                set upcomingEvents to every event of every calendar whose start date ≥ startDate and start date ≤ endDate
                
                set eventList to {{}}
                repeat with anEvent in upcomingEvents
                    set eventInfo to (summary of anEvent) & "|" & (start date of anEvent as string) & "|" & (end date of anEvent as string)
                    set end of eventList to eventInfo
                end repeat
                
                return my listToString(eventList, "\\n")
            end tell
            
            on listToString(lst, delim)
                set AppleScript's text item delimiters to delim
                set str to lst as string
                set AppleScript's text item delimiters to ""
                return str
            end listToString
            '''
            
            result = subprocess.run([
                "osascript", "-e", script
            ], capture_output=True, text=True, timeout=10)
            
            events = []
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            events.append({
                                "title": parts[0],
                                "start": parts[1],
                                "end": parts[2]
                            })
            
            return events
        except Exception as e:
            print(f"Calendar integration error: {e}")
            return []
    
    def analyze_meeting_requirements(self, events):
        """Analyze what apps are needed for upcoming meetings"""
        meeting_keywords = {
            "zoom": ["Zoom"],
            "teams": ["Microsoft Teams"],
            "meet": ["Google Chrome", "Safari"],
            "webex": ["Cisco Webex"],
            "call": ["FaceTime", "Phone"],
            "presentation": ["Keynote", "PowerPoint", "Google Chrome"]
        }
        
        required_apps = set()
        meeting_times = []
        
        for event in events:
            title_lower = event["title"].lower()
            meeting_times.append(event["start"])
            
            for keyword, apps in meeting_keywords.items():
                if keyword in title_lower:
                    required_apps.update(apps)
        
        return {
            "required_apps": list(required_apps),
            "meeting_times": meeting_times,
            "meeting_count": len(events)
        }

class AdvancedEnergyOptimizer:
    """Main advanced energy optimization engine"""
    
    def __init__(self, db_path):
        self.thermal_manager = M3ThermalManager()
        self.performance_manager = M3PerformanceManager()
        self.scheduler = PredictiveScheduler(db_path)
        self.calendar = CalendarIntegration()
        
        self.energy_modes = {
            "thermal_protection": {
                "cpu_threshold": 1,
                "aggressive_suspension": True,
                "thermal_priority": True
            },
            "meeting_mode": {
                "cpu_threshold": 15,
                "preserve_communication": True,
                "aggressive_suspension": False
            },
            "focus_mode": {
                "cpu_threshold": 5,
                "suspend_distractions": True,
                "preserve_productivity": True
            },
            "travel_mode": {
                "cpu_threshold": 2,
                "maximum_battery_saving": True,
                "offline_optimization": True
            },
            "performance_mode": {
                "cpu_threshold": 20,
                "aggressive_suspension": False,
                "thermal_priority": False
            }
        }
        
        self.current_mode = "balanced"
    
    def determine_optimal_mode(self, battery_level, on_battery):
        """Determine optimal energy mode based on context"""
        # Check thermal state
        thermal_state, temp = self.thermal_manager.get_thermal_state()
        throttling_risk, predicted_temp = self.thermal_manager.predict_thermal_throttling()
        
        if thermal_state == "critical" or throttling_risk:
            return "thermal_protection", f"Thermal protection (temp: {temp:.1f}°C)"
        
        # Check for upcoming meetings
        upcoming_events = self.calendar.get_upcoming_events(hours_ahead=1)
        if upcoming_events:
            return "meeting_mode", f"Meeting preparation ({len(upcoming_events)} events)"
        
        # Battery-based modes
        if on_battery:
            if battery_level < 15:
                return "travel_mode", "Critical battery conservation"
            elif battery_level < 30:
                return "focus_mode", "Power saving with productivity focus"
        
        # Default performance mode when on AC power
        if not on_battery and thermal_state == "cool":
            return "performance_mode", "Maximum performance (AC power)"
        
        return "balanced", "Balanced optimization"
    
    def get_intelligent_recommendations(self, current_apps, battery_level, on_battery):
        """Get AI-powered optimization recommendations"""
        # Get predictions
        predictions = self.scheduler.predict_app_usage()
        
        # Get calendar context
        upcoming_events = self.calendar.get_upcoming_events()
        meeting_requirements = self.calendar.analyze_meeting_requirements(upcoming_events)
        
        # Determine optimal mode
        optimal_mode, reason = self.determine_optimal_mode(battery_level, on_battery)
        mode_config = self.energy_modes.get(optimal_mode, {})
        
        recommendations = {
            "mode": optimal_mode,
            "reason": reason,
            "thermal_state": self.thermal_manager.get_thermal_state(),
            "predictions": predictions,
            "meeting_requirements": meeting_requirements,
            "actions": []
        }
        
        # Generate specific actions
        if optimal_mode == "thermal_protection":
            recommendations["actions"].append({
                "type": "suspend_aggressive",
                "apps": [app for app in current_apps if "browser" in app.lower()],
                "reason": "Thermal protection"
            })
        
        elif optimal_mode == "meeting_mode":
            keep_active = meeting_requirements["required_apps"] + ["Calendar", "Notes"]
            recommendations["actions"].append({
                "type": "preserve_apps",
                "apps": keep_active,
                "reason": "Meeting preparation"
            })
        
        elif optimal_mode == "focus_mode":
            distracting_apps = ["Safari", "Chrome", "Twitter", "Facebook", "Instagram", "TikTok"]
            recommendations["actions"].append({
                "type": "suspend_distractions",
                "apps": [app for app in current_apps if any(d in app for d in distracting_apps)],
                "reason": "Focus enhancement"
            })
        
        return recommendations
    
    def execute_optimization(self, recommendations):
        """Execute optimization recommendations"""
        executed_actions = []
        
        for action in recommendations["actions"]:
            if action["type"] == "suspend_aggressive":
                for app_name in action["apps"]:
                    # Find and suspend the app
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            if app_name.lower() in proc.info['name'].lower():
                                os.kill(proc.info['pid'], 19)  # SIGSTOP
                                executed_actions.append(f"Suspended {proc.info['name']}")
                                break
                        except:
                            continue
            
            elif action["type"] == "preserve_apps":
                # Ensure these apps are not suspended (resume if needed)
                for app_name in action["apps"]:
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            if app_name.lower() in proc.info['name'].lower():
                                os.kill(proc.info['pid'], 18)  # SIGCONT
                                executed_actions.append(f"Preserved {proc.info['name']}")
                                break
                        except:
                            continue
        
        return executed_actions

def main():
    """Main execution function"""
    db_path = os.path.expanduser("~/.battery_optimizer_advanced.db")
    optimizer = AdvancedEnergyOptimizer(db_path)
    
    # Get current system state
    battery = psutil.sensors_battery()
    battery_level = battery.percent if battery else 100
    on_battery = not battery.power_plugged if battery else False
    
    # Get current running apps
    current_apps = []
    for proc in psutil.process_iter(['name']):
        try:
            current_apps.append(proc.info['name'])
        except:
            continue
    
    # Get intelligent recommendations
    recommendations = optimizer.get_intelligent_recommendations(
        current_apps, battery_level, on_battery
    )
    
    print("🧠 Advanced M3 Energy Optimization")
    print("=" * 50)
    print(f"Current Mode: {recommendations['mode']}")
    print(f"Reason: {recommendations['reason']}")
    print(f"Thermal State: {recommendations['thermal_state']}")
    print(f"Battery: {battery_level}% ({'Battery' if on_battery else 'AC Power'})")
    
    if recommendations['predictions']:
        print("\n📊 Usage Predictions:")
        for pred in recommendations['predictions'][:5]:
            print(f"  • {pred['app']}: {pred['probability']:.1%} probability")
    
    if recommendations['meeting_requirements']['meeting_count'] > 0:
        print(f"\n📅 Upcoming Meetings: {recommendations['meeting_requirements']['meeting_count']}")
        print(f"Required Apps: {', '.join(recommendations['meeting_requirements']['required_apps'])}")
    
    print(f"\n🎯 Recommended Actions: {len(recommendations['actions'])}")
    for action in recommendations['actions']:
        print(f"  • {action['type']}: {action['reason']}")
    
    return recommendations

if __name__ == "__main__":
    main()
