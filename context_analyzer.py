#!/usr/bin/env python3
"""
Context-Aware Analysis for Advanced EAS
Analyzes system context including meetings, workflow phases, and user focus
"""

# Line 1-20: Context analysis imports and setup
import subprocess
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import os
from dataclasses import dataclass

@dataclass
class SystemContext:
    meeting_in_progress: bool
    upcoming_deadline: Optional[datetime]
    workflow_phase: str
    user_focus_level: float
    battery_level: float
    thermal_state: str
    time_of_day: str
    day_of_week: str

class ContextAnalyzer:
    def __init__(self):
        self.db_path = os.path.expanduser("~/.advanced_eas_context.db")
        self.init_database()
        
    def init_database(self):
        # Line 21-40: Database setup for context storage
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS context_history (
                timestamp TEXT,
                meeting_active BOOLEAN,
                workflow_phase TEXT,
                user_focus_level REAL,
                battery_level REAL,
                thermal_state TEXT,
                active_applications TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def detect_meeting_in_progress(self) -> bool:
        # Line 41-60: Meeting detection via active applications
        try:
            # Check for video conferencing apps
            meeting_apps = ['zoom', 'teams', 'webex', 'skype', 'facetime', 'meet', 'discord']
            
            for proc in psutil.process_iter(['name']):
                try:
                    proc_name = proc.info['name'].lower()
                    if any(app in proc_name for app in meeting_apps):
                        # Additional check: look for camera/microphone usage
                        if self._check_camera_microphone_usage():
                            return True
                except:
                    continue
                    
            return False
        except:
            return False
            
    def _check_camera_microphone_usage(self) -> bool:
        # Line 61-75: Check camera/microphone indicators
        try:
            # macOS specific: check for camera/microphone usage indicators
            result = subprocess.run([
                'lsof', '+c', '0'
            ], capture_output=True, text=True, timeout=2)
            
            output = result.stdout.lower()
            camera_indicators = ['videodecoderacceleration', 'avfoundation', 'coremedia']
            microphone_indicators = ['coreaudio', 'audiohardware']
            
            has_camera = any(indicator in output for indicator in camera_indicators)
            has_microphone = any(indicator in output for indicator in microphone_indicators)
            
            return has_camera or has_microphone
        except:
            return False
            
    def analyze_workflow_phase(self) -> str:
        # Line 76-100: Workflow phase detection
        try:
            # Get active applications
            active_apps = []
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name and not name.startswith('com.apple'):
                        active_apps.append(name.lower())
                except:
                    continue
                    
            # Classify workflow phase based on active applications
            development_apps = ['xcode', 'vscode', 'pycharm', 'intellij', 'terminal', 'iterm']
            design_apps = ['photoshop', 'illustrator', 'sketch', 'figma', 'blender']
            communication_apps = ['slack', 'teams', 'discord', 'messages', 'mail']
            browser_apps = ['safari', 'chrome', 'firefox', 'edge']
            
            if any(app in ' '.join(active_apps) for app in development_apps):
                return 'development'
            elif any(app in ' '.join(active_apps) for app in design_apps):
                return 'creative_work'
            elif any(app in ' '.join(active_apps) for app in communication_apps):
                return 'communication'
            elif any(app in ' '.join(active_apps) for app in browser_apps):
                return 'research_browsing'
            else:
                return 'general_computing'
                
        except:
            return 'unknown' 
           
    def calculate_user_focus_level(self) -> float:
        # Line 101-125: User focus level calculation
        try:
            # Factors that indicate high focus:
            # 1. Single application in foreground for extended time
            # 2. Low application switching frequency
            # 3. Consistent input patterns
            # 4. Time of day (focus hours)
            
            current_hour = datetime.now().hour
            
            # Focus score based on time of day
            if 9 <= current_hour <= 11 or 14 <= current_hour <= 16:
                time_focus_score = 0.8  # Peak focus hours
            elif 8 <= current_hour <= 12 or 13 <= current_hour <= 17:
                time_focus_score = 0.6  # Good focus hours
            else:
                time_focus_score = 0.3  # Low focus hours
                
            # Application switching frequency (simplified)
            app_switch_score = self._calculate_app_switch_score()
            
            # Meeting indicator (meetings often require high focus)
            meeting_score = 0.9 if self.detect_meeting_in_progress() else 0.5
            
            # Weighted average
            focus_level = (time_focus_score * 0.4 + app_switch_score * 0.3 + meeting_score * 0.3)
            return min(1.0, max(0.1, focus_level))
            
        except:
            return 0.5  # Default moderate focus
            
    def _calculate_app_switch_score(self) -> float:
        # Line 126-140: Application switching frequency analysis
        try:
            # Get number of running GUI applications
            gui_apps = 0
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name and not any(sys_prefix in name for sys_prefix in 
                                      ['com.apple', 'kernel', 'launchd', 'system']):
                        gui_apps += 1
                except:
                    continue
                    
            # Fewer apps = higher focus score
            if gui_apps <= 3:
                return 0.9
            elif gui_apps <= 6:
                return 0.7
            elif gui_apps <= 10:
                return 0.5
            else:
                return 0.3
                
        except:
            return 0.5

    def get_system_context(self) -> SystemContext:
        # Line 141-165: Comprehensive system context gathering
        try:
            # Battery level
            battery = psutil.sensors_battery()
            battery_level = battery.percent if battery else 100.0
            
            # Thermal state (simplified - would need more sophisticated monitoring)
            cpu_temp = self._get_cpu_temperature()
            if cpu_temp > 80:
                thermal_state = "hot"
            elif cpu_temp > 60:
                thermal_state = "warm"
            else:
                thermal_state = "cool"
                
            # Time context
            now = datetime.now()
            time_of_day = self._classify_time_of_day(now.hour)
            day_of_week = now.strftime("%A").lower()
            
            context = SystemContext(
                meeting_in_progress=self.detect_meeting_in_progress(),
                upcoming_deadline=None,  # Would integrate with calendar
                workflow_phase=self.analyze_workflow_phase(),
                user_focus_level=self.calculate_user_focus_level(),
                battery_level=battery_level,
                thermal_state=thermal_state,
                time_of_day=time_of_day,
                day_of_week=day_of_week
            )
            
            # Store context in database
            self._store_context(context)
            
            return context
            
        except Exception as e:
            # Return default context on error
            return SystemContext(
                meeting_in_progress=False,
                upcoming_deadline=None,
                workflow_phase="unknown",
                user_focus_level=0.5,
                battery_level=100.0,
                thermal_state="cool",
                time_of_day="day",
                day_of_week="weekday"
            )
            
    def _get_cpu_temperature(self) -> float:
        # Line 166-180: CPU temperature monitoring
        try:
            # macOS specific temperature monitoring
            from permission_manager import permission_manager
            result = permission_manager.execute_with_sudo([
                'powermetrics', '--samplers', 'smc', '-n', '1', '--show-initial-usage'
            ], timeout=5)
            
            # Parse temperature from output (simplified)
            for line in result.stdout.split('\n'):
                if 'CPU die temperature' in line:
                    temp_str = line.split(':')[1].strip().replace('C', '')
                    return float(temp_str)
                    
            return 50.0  # Default safe temperature
        except:
            return 50.0
            
    def _classify_time_of_day(self, hour: int) -> str:
        # Line 181-190: Time of day classification
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
            
    def _store_context(self, context: SystemContext):
        # Line 191-210: Store context in database for learning
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get active applications for context
            active_apps = []
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name:
                        active_apps.append(name)
                except:
                    continue
                    
            conn.execute('''
                INSERT INTO context_history 
                (timestamp, meeting_active, workflow_phase, user_focus_level, 
                 battery_level, thermal_state, active_applications)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                context.meeting_in_progress,
                context.workflow_phase,
                context.user_focus_level,
                context.battery_level,
                context.thermal_state,
                json.dumps(active_apps)
            ))
            
            conn.commit()
            conn.close()
        except:
            pass  # Fail silently for context storage

# Test function
def test_context_analyzer():
    """Test the context analyzer"""
    print("🎯 Testing Context Analyzer")
    print("=" * 50)
    
    analyzer = ContextAnalyzer()
    context = analyzer.get_system_context()
    
    print(f"📊 Current System Context:")
    print(f"  Meeting in Progress: {context.meeting_in_progress}")
    print(f"  Workflow Phase: {context.workflow_phase}")
    print(f"  User Focus Level: {context.user_focus_level:.2f}")
    print(f"  Battery Level: {context.battery_level:.1f}%")
    print(f"  Thermal State: {context.thermal_state}")
    print(f"  Time of Day: {context.time_of_day}")
    print(f"  Day of Week: {context.day_of_week}")

if __name__ == "__main__":
    test_context_analyzer()