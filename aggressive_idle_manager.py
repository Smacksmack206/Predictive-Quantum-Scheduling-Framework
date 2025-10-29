#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggressive Idle Power Manager
==============================
Detects true idle state and aggressively manages power:
- Deep sleep when truly idle
- Suspend battery-draining apps when lid closed
- Detect real workloads vs fake activity
- Override sleep-preventing apps (Amphetamine, Kiro, etc.)
"""

import psutil
import subprocess
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ActivityState:
    """System activity state"""
    cpu_active: bool
    user_input_recent: bool
    media_playing: bool
    network_active: bool
    disk_active: bool
    lid_open: bool
    power_plugged: bool
    battery_percent: float
    active_workload: bool
    timestamp: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0


class AggressiveIdleManager:
    """
    Aggressive power management when system is truly idle
    Enhanced with Quantum-ML intelligence
    """
    
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        
        # Quantum-ML optimizer
        self.quantum_ml_optimizer = None
        try:
            from quantum_ml_idle_optimizer import get_quantum_ml_optimizer
            self.quantum_ml_optimizer = get_quantum_ml_optimizer()
            logger.info("🧠 Quantum-ML optimizer integrated")
        except ImportError:
            logger.info("💻 Running without Quantum-ML optimization")
        
        # Activity thresholds
        self.cpu_idle_threshold = 5.0  # CPU < 5% = idle
        self.network_idle_threshold = 10000  # < 10KB/s = idle
        self.disk_idle_threshold = 50000  # < 50KB/s = idle
        
        # Timing thresholds
        self.idle_time_before_sleep = 120  # 2 minutes of idle
        self.lid_closed_suspend_delay = 30  # 30 seconds after lid close
        
        # State tracking
        self.last_activity_time = time.time()
        self.last_user_input_time = time.time()
        self.lid_closed_time = None
        self.suspended_apps = set()
        
        # Sleep-preventing apps to override
        self.sleep_preventers = [
            'Amphetamine',
            'Kiro',
            'caffeinate',
            'InsomniaX',
            'KeepingYouAwake'
        ]
        
        # Apps that indicate real work
        self.work_apps = [
            'Xcode',
            'Visual Studio Code',
            'PyCharm',
            'IntelliJ',
            'Terminal',
            'iTerm',
            'Docker',
            'VirtualBox',
            'Parallels',
            'VMware'
        ]
        
        # Media apps
        self.media_apps = [
            'Music',
            'Spotify',
            'VLC',
            'QuickTime Player',
            'Safari',
            'Chrome',
            'Firefox',
            'YouTube'
        ]
        
        # Track I/O baselines
        self._last_net_io = None
        self._last_disk_io = None
        self._last_io_time = None
        
        logger.info("🔋 Aggressive Idle Manager initialized")
    
    def get_activity_state(self) -> ActivityState:
        """Detect true system activity"""
        try:
            # CPU activity
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_active = cpu_percent > self.cpu_idle_threshold
            
            # Check for real workloads
            active_workload = self._detect_real_workload()
            
            # Media playback detection
            media_playing = self._detect_media_playback()
            
            # Network activity
            network_active = self._check_network_activity()
            
            # Disk activity
            disk_active = self._check_disk_activity()
            
            # User input detection (keyboard/mouse)
            user_input_recent = self._detect_user_input()
            
            # Lid state
            lid_open = self._check_lid_state()
            
            # Battery state
            battery = psutil.sensors_battery()
            power_plugged = battery.power_plugged if battery else True
            battery_percent = battery.percent if battery else 100.0
            
            return ActivityState(
                cpu_active=cpu_active,
                user_input_recent=user_input_recent,
                media_playing=media_playing,
                network_active=network_active,
                disk_active=disk_active,
                lid_open=lid_open,
                power_plugged=power_plugged,
                battery_percent=battery_percent,
                active_workload=active_workload,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error getting activity state: {e}")
            return None
    
    def _detect_real_workload(self) -> bool:
        """Detect if real work is being done"""
        try:
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    name = proc.info['name']
                    cpu = proc.info['cpu_percent'] or 0
                    
                    # Check if work app is active
                    if any(work_app.lower() in name.lower() for work_app in self.work_apps):
                        if cpu > 1.0:  # Actually doing something
                            return True
                    
                    # Check for compilation/build processes
                    if any(keyword in name.lower() for keyword in ['gcc', 'clang', 'swift', 'javac', 'npm', 'yarn', 'cargo', 'go']):
                        if cpu > 5.0:
                            return True
                            
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting workload: {e}")
            return False
    
    def _detect_media_playback(self) -> bool:
        """Detect if media is playing"""
        try:
            # Check for media apps with audio
            result = subprocess.run(
                ['pmset', '-g', 'assertions'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                output = result.stdout.lower()
                # Check for audio/video assertions
                if 'preventuseridledisplaysleep' in output or 'preventuseridlesystemsleep' in output:
                    # Verify it's actually media, not just Amphetamine
                    if any(media in output for media in ['audio', 'video', 'coreaudio', 'avfoundation']):
                        return True
            
            # Also check for media apps with CPU usage
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                try:
                    name = proc.info['name']
                    cpu = proc.info['cpu_percent'] or 0
                    
                    if any(media_app.lower() in name.lower() for media_app in self.media_apps):
                        if cpu > 2.0:  # Actually playing
                            return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting media: {e}")
            return False
    
    def _check_network_activity(self) -> bool:
        """Check for significant network activity"""
        try:
            net_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self._last_net_io is None:
                self._last_net_io = (net_io.bytes_sent + net_io.bytes_recv, current_time)
                return False
            
            bytes_total = net_io.bytes_sent + net_io.bytes_recv
            time_delta = current_time - self._last_net_io[1]
            
            if time_delta > 0:
                bytes_per_sec = (bytes_total - self._last_net_io[0]) / time_delta
                self._last_net_io = (bytes_total, current_time)
                
                return bytes_per_sec > self.network_idle_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking network: {e}")
            return False
    
    def _check_disk_activity(self) -> bool:
        """Check for significant disk activity"""
        try:
            disk_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self._last_disk_io is None:
                self._last_disk_io = (disk_io.read_bytes + disk_io.write_bytes, current_time)
                return False
            
            bytes_total = disk_io.read_bytes + disk_io.write_bytes
            time_delta = current_time - self._last_disk_io[1]
            
            if time_delta > 0:
                bytes_per_sec = (bytes_total - self._last_disk_io[0]) / time_delta
                self._last_disk_io = (bytes_total, current_time)
                
                return bytes_per_sec > self.disk_idle_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking disk: {e}")
            return False
    
    def _detect_user_input(self) -> bool:
        """Detect recent user input (keyboard/mouse)"""
        try:
            # Check system idle time
            result = subprocess.run(
                ['ioreg', '-c', 'IOHIDSystem'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'HIDIdleTime' in line:
                        # Extract idle time in nanoseconds
                        idle_ns = int(line.split('=')[1].strip())
                        idle_seconds = idle_ns / 1000000000
                        
                        # User input in last 60 seconds = active
                        return idle_seconds < 60
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting user input: {e}")
            return False
    
    def _check_lid_state(self) -> bool:
        """Check if laptop lid is open"""
        try:
            result = subprocess.run(
                ['ioreg', '-r', '-k', 'AppleClamshellState', '-d', '4'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # AppleClamshellState = Yes means lid is closed
                return 'AppleClamshellState" = Yes' not in result.stdout
            
            return True  # Assume open if can't detect
            
        except Exception as e:
            logger.error(f"Error checking lid state: {e}")
            return True

    def is_truly_idle(self, state: ActivityState) -> bool:
        """Determine if system is truly idle"""
        if state is None:
            return False
        
        # Not idle if any real activity
        if state.active_workload:
            return False
        
        if state.media_playing:
            return False
        
        if state.user_input_recent:
            return False
        
        if state.cpu_active:
            return False
        
        if state.network_active:
            return False
        
        if state.disk_active:
            return False
        
        # System is truly idle
        return True
    
    def suspend_battery_draining_apps(self):
        """Suspend apps that drain battery when idle"""
        try:
            suspended_count = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    name = proc.info['name']
                    pid = proc.info['pid']
                    cpu = proc.info['cpu_percent'] or 0
                    
                    # Check if it's a sleep preventer doing nothing useful
                    if any(preventer.lower() in name.lower() for preventer in self.sleep_preventers):
                        if cpu < 1.0:  # Not actually doing anything
                            # Send SIGSTOP to suspend
                            subprocess.run(['kill', '-STOP', str(pid)], timeout=1)
                            self.suspended_apps.add(pid)
                            suspended_count += 1
                            logger.info(f"⏸️  Suspended idle app: {name} (PID: {pid})")
                    
                    # Also suspend Electron apps (like Kiro) that are idle
                    if 'electron' in name.lower() or 'helper' in name.lower():
                        mem = proc.info['memory_percent'] or 0
                        if cpu < 0.5 and mem > 1.0:  # Idle but using memory
                            subprocess.run(['kill', '-STOP', str(pid)], timeout=1)
                            self.suspended_apps.add(pid)
                            suspended_count += 1
                            logger.info(f"⏸️  Suspended idle Electron app: {name} (PID: {pid})")
                            
                except:
                    continue
            
            if suspended_count > 0:
                logger.info(f"⏸️  Suspended {suspended_count} battery-draining apps")
            
        except Exception as e:
            logger.error(f"Error suspending apps: {e}")
    
    def resume_suspended_apps(self):
        """Resume previously suspended apps"""
        try:
            resumed_count = 0
            
            for pid in list(self.suspended_apps):
                try:
                    # Send SIGCONT to resume
                    subprocess.run(['kill', '-CONT', str(pid)], timeout=1)
                    resumed_count += 1
                    logger.info(f"▶️  Resumed app (PID: {pid})")
                except:
                    # Process might have exited
                    pass
                
                self.suspended_apps.discard(pid)
            
            if resumed_count > 0:
                logger.info(f"▶️  Resumed {resumed_count} apps")
                
        except Exception as e:
            logger.error(f"Error resuming apps: {e}")
    
    def force_deep_sleep(self):
        """Force Mac into deep sleep state"""
        try:
            logger.info("💤 Forcing deep sleep...")
            
            # Use pmset to sleep immediately
            subprocess.run(['pmset', 'sleepnow'], timeout=2)
            
        except Exception as e:
            logger.error(f"Error forcing sleep: {e}")
    
    def enable_aggressive_power_settings(self):
        """Enable aggressive power management settings"""
        try:
            logger.info("⚡ Enabling aggressive power settings...")
            
            # Set aggressive sleep timers (battery only)
            commands = [
                # Display sleep after 2 minutes
                ['pmset', '-b', 'displaysleep', '2'],
                # System sleep after 5 minutes
                ['pmset', '-b', 'sleep', '5'],
                # Disk sleep after 1 minute
                ['pmset', '-b', 'disksleep', '1'],
                # Enable Power Nap on battery
                ['pmset', '-b', 'powernap', '0'],
                # Reduce processor speed on battery
                ['pmset', '-b', 'reducebright', '1'],
                # Hibernate mode 25 (deep sleep)
                ['pmset', '-a', 'hibernatemode', '25'],
                # Standby delay 1 hour
                ['pmset', '-a', 'standbydelay', '3600']
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, timeout=2, check=False)
                except:
                    pass
            
            logger.info("✅ Aggressive power settings enabled")
            
        except Exception as e:
            logger.error(f"Error setting power settings: {e}")
    
    def start_monitoring(self):
        """Start aggressive idle monitoring"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Enable aggressive settings
        self.enable_aggressive_power_settings()
        
        logger.info("🔋 Aggressive idle monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Resume any suspended apps
        self.resume_suspended_apps()
        
        logger.info("⏹️  Aggressive idle monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop with Quantum-ML intelligence"""
        idle_start_time = None
        last_activity_type = 'active'
        
        while self.running:
            try:
                # Get current activity state
                state = self.get_activity_state()
                
                if state is None:
                    time.sleep(10)
                    continue
                
                # Check if truly idle
                is_idle = self.is_truly_idle(state)
                
                # Get Quantum-ML recommendation if available
                ml_recommendation = None
                if self.quantum_ml_optimizer and is_idle:
                    state_dict = {
                        'cpu_percent': state.cpu_percent,
                        'memory_percent': state.memory_percent,
                        'battery_percent': state.battery_percent,
                        'power_plugged': state.power_plugged,
                        'process_count': psutil.cpu_count() * 50  # Estimate
                    }
                    ml_recommendation = self.quantum_ml_optimizer.get_intelligent_recommendation(state_dict)
                
                if is_idle:
                    if idle_start_time is None:
                        idle_start_time = time.time()
                        if ml_recommendation:
                            logger.info(f"😴 System idle - {ml_recommendation['reasoning']}")
                        else:
                            logger.info("😴 System detected as idle")
                    
                    idle_duration = time.time() - idle_start_time
                    
                    # Use ML-optimized thresholds if available
                    suspend_threshold = 30
                    sleep_threshold = self.idle_time_before_sleep
                    
                    if ml_recommendation:
                        suspend_threshold = ml_recommendation['suspend_delay']
                        sleep_threshold = ml_recommendation['sleep_delay']
                        
                        # Log ML decision
                        if idle_duration > 60 and int(idle_duration) % 60 == 0:
                            logger.info(f"🧠 ML: {ml_recommendation['reasoning']}")
                    
                    # Suspend battery-draining apps
                    if idle_duration > suspend_threshold:
                        self.suspend_battery_draining_apps()
                    
                    # Handle lid closed scenario
                    if not state.lid_open:
                        if self.lid_closed_time is None:
                            self.lid_closed_time = time.time()
                            logger.info("🔒 Lid closed detected")
                        
                        lid_closed_duration = time.time() - self.lid_closed_time
                        
                        # Aggressive: suspend apps and sleep after 30 seconds
                        if lid_closed_duration > self.lid_closed_suspend_delay:
                            logger.info("🔒 Lid closed for 30s - suspending apps and sleeping")
                            self.suspend_battery_draining_apps()
                            time.sleep(2)
                            self.force_deep_sleep()
                    
                    # Force sleep after idle threshold (on battery only)
                    elif not state.power_plugged and idle_duration > sleep_threshold:
                        if ml_recommendation:
                            logger.info(f"💤 ML-optimized sleep after {idle_duration:.0f}s (predicted {ml_recommendation['predicted_return_seconds']/60:.0f}min away)")
                        else:
                            logger.info(f"💤 Idle for {idle_duration:.0f}s on battery - forcing sleep")
                        
                        # Learn from this session if ML available
                        if self.quantum_ml_optimizer:
                            self.quantum_ml_optimizer.learn_from_session(
                                idle_start_time, time.time(), last_activity_type, 'idle'
                            )
                        
                        self.force_deep_sleep()
                        idle_start_time = None
                    
                else:
                    # System is active
                    if idle_start_time is not None:
                        # Learn from this return
                        if self.quantum_ml_optimizer:
                            return_time = time.time()
                            self.quantum_ml_optimizer.learn_from_session(
                                idle_start_time, return_time, 'idle', 'active'
                            )
                        
                        logger.info("⚡ System active again")
                        idle_start_time = None
                    
                    # Track activity type for learning
                    if state.media_playing:
                        last_activity_type = 'media'
                    elif state.active_workload:
                        last_activity_type = 'work'
                    else:
                        last_activity_type = 'active'
                    
                    # Resume suspended apps
                    if self.suspended_apps:
                        self.resume_suspended_apps()
                    
                    # Reset lid closed time
                    if state.lid_open:
                        self.lid_closed_time = None
                
                # Check every 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def get_status(self) -> Dict:
        """Get current status"""
        state = self.get_activity_state()
        
        if state is None:
            return {'error': 'Could not get activity state'}
        
        return {
            'monitoring': self.running,
            'truly_idle': self.is_truly_idle(state),
            'suspended_apps': len(self.suspended_apps),
            'activity': {
                'cpu_active': state.cpu_active,
                'user_input_recent': state.user_input_recent,
                'media_playing': state.media_playing,
                'network_active': state.network_active,
                'disk_active': state.disk_active,
                'active_workload': state.active_workload
            },
            'system': {
                'lid_open': state.lid_open,
                'power_plugged': state.power_plugged,
                'battery_percent': state.battery_percent
            }
        }


# Global instance
_idle_manager = None

def get_idle_manager() -> AggressiveIdleManager:
    """Get or create global idle manager"""
    global _idle_manager
    
    if _idle_manager is None:
        _idle_manager = AggressiveIdleManager()
    
    return _idle_manager


if __name__ == "__main__":
    print("🔋 Aggressive Idle Manager Test")
    print("=" * 60)
    
    manager = AggressiveIdleManager()
    
    # Get current state
    state = manager.get_activity_state()
    
    if state:
        print(f"\n📊 Current Activity State:")
        print(f"   CPU Active: {state.cpu_active}")
        print(f"   User Input Recent: {state.user_input_recent}")
        print(f"   Media Playing: {state.media_playing}")
        print(f"   Network Active: {state.network_active}")
        print(f"   Disk Active: {state.disk_active}")
        print(f"   Active Workload: {state.active_workload}")
        print(f"   Lid Open: {state.lid_open}")
        print(f"   Power Plugged: {state.power_plugged}")
        print(f"   Battery: {state.battery_percent}%")
        
        print(f"\n💤 Truly Idle: {manager.is_truly_idle(state)}")
    
    print("\n✅ Test complete")
