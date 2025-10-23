#!/usr/bin/env python3
"""
Intelligent Universal Process Monitor
======================================

Learns normal behavior for ALL apps and detects:
- Abnormal multiple instances (should be killed)
- Normal multiple instances (expected behavior)
- Zombie/stuck processes
- Resource-hogging duplicates
- Memory leaks in multi-instance apps

Uses ML to learn what's "normal" for each app over time.
"""

import psutil
import time
import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

# Persistence file
LEARNING_DATA_FILE = os.path.expanduser("~/.pqs_process_learning.json")


class ProcessProfile:
    """Profile for a specific application's normal behavior"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
        self.observations = 0
        
        # Instance count learning
        self.instance_counts = deque(maxlen=200)  # Historical instance counts
        self.normal_instance_range = (1, 1)  # (min, max) normal instances
        
        # Resource usage per instance
        self.cpu_per_instance = deque(maxlen=100)
        self.memory_per_instance = deque(maxlen=100)
        
        # Behavior patterns
        self.typical_cpu_total = 0.0
        self.typical_memory_total = 0.0
        self.variance_tolerance = 2.0  # Standard deviations
        
        # Multi-instance classification
        self.multi_instance_type = 'unknown'  # 'normal', 'helper_based', 'single_only', 'variable'
        self.helper_process_names = set()  # Known helper process names
        
        # Anomaly tracking
        self.anomaly_count = 0
        self.last_anomaly_time = None
        
    def observe(self, instances: List[Dict]):
        """Record observation of app instances"""
        self.observations += 1
        instance_count = len(instances)
        
        # Track instance count
        self.instance_counts.append(instance_count)
        
        # Track per-instance resources
        if instance_count > 0:
            total_cpu = sum(inst['cpu'] for inst in instances)
            total_memory = sum(inst['memory'] for inst in instances)
            
            avg_cpu = total_cpu / instance_count
            avg_memory = total_memory / instance_count
            
            self.cpu_per_instance.append(avg_cpu)
            self.memory_per_instance.append(avg_memory)
            
            self.typical_cpu_total = np.mean([sum(inst['cpu'] for inst in instances)])
            self.typical_memory_total = np.mean([sum(inst['memory'] for inst in instances)])
        
        # Learn helper process names
        for inst in instances:
            name = inst['name'].lower()
            if 'helper' in name or 'renderer' in name or 'gpu' in name or 'plugin' in name:
                self.helper_process_names.add(inst['name'])
        
        # Classify multi-instance behavior after enough observations
        if self.observations >= 20:
            self._classify_multi_instance_behavior()
    
    def _classify_multi_instance_behavior(self):
        """Classify if multiple instances are normal for this app"""
        if len(self.instance_counts) < 10:
            return
        
        counts = list(self.instance_counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        
        # Determine normal range (mean ± 2 std dev)
        normal_min = max(1, int(mean_count - 2 * std_count))
        normal_max = int(mean_count + 2 * std_count)
        self.normal_instance_range = (normal_min, normal_max)
        
        # Classify behavior type
        if mean_count > 5 and std_count < 2:
            # Consistently many instances (like Electron apps with helpers)
            self.multi_instance_type = 'helper_based'
        elif max_count == 1 and min_count == 1:
            # Always single instance
            self.multi_instance_type = 'single_only'
        elif std_count > mean_count * 0.5:
            # Highly variable instance count
            self.multi_instance_type = 'variable'
        elif mean_count <= 3:
            # Typically few instances
            self.multi_instance_type = 'normal'
        else:
            self.multi_instance_type = 'normal'
    
    def is_anomalous(self, current_instances: List[Dict]) -> Tuple[bool, str, float]:
        """
        Determine if current state is anomalous
        Returns: (is_anomalous, reason, severity_score)
        """
        instance_count = len(current_instances)
        total_cpu = sum(inst['cpu'] for inst in current_instances)
        total_memory = sum(inst['memory'] for inst in current_instances)
        
        anomalies = []
        severity = 0.0
        
        # IMMEDIATE RED FLAGS (don't need learning)
        # Red Flag 1: Many instances with high total CPU
        if instance_count >= 10 and total_cpu > 50:
            anomalies.append(f"{instance_count} instances using {total_cpu:.1f}% CPU (excessive)")
            severity += instance_count * 2.0
        
        # Red Flag 2: Many idle instances (zombies)
        idle_count = sum(1 for inst in current_instances if inst['cpu'] < 0.1)
        if idle_count >= 5 and instance_count >= 8:
            anomalies.append(f"{idle_count} idle/zombie processes out of {instance_count}")
            severity += idle_count * 1.5
        
        # Red Flag 3: One process hogging resources while others idle
        if instance_count >= 5:
            cpu_values = [inst['cpu'] for inst in current_instances]
            max_cpu = max(cpu_values) if cpu_values else 0
            avg_cpu = np.mean(cpu_values) if cpu_values else 0
            
            if max_cpu > 50 and avg_cpu < 10:
                anomalies.append(f"One process using {max_cpu:.1f}% while others idle")
                severity += 5.0
        
        # If we have immediate red flags, return now
        if anomalies and severity >= 10.0:
            return True, "; ".join(anomalies), severity
        
        # LEARNED BEHAVIOR CHECKS (need observations)
        if self.observations < 10:
            return False, "Learning", 0.0
        
        # Check 1: Instance count anomaly
        min_normal, max_normal = self.normal_instance_range
        if instance_count > max_normal:
            excess = instance_count - max_normal
            anomalies.append(f"{excess} excess instances (normal: {min_normal}-{max_normal})")
            severity += excess * 2.0
        
        # Check 2: Total CPU anomaly
        if len(self.cpu_per_instance) > 5:
            expected_cpu = np.mean(self.cpu_per_instance) * instance_count
            cpu_threshold = expected_cpu + (2 * np.std(self.cpu_per_instance) * instance_count)
            
            if total_cpu > cpu_threshold and total_cpu > expected_cpu * 1.5:
                cpu_excess = total_cpu - expected_cpu
                anomalies.append(f"CPU {cpu_excess:.1f}% above normal")
                severity += cpu_excess * 0.5
        
        # Check 3: Zombie/stuck processes (high count, low CPU)
        if instance_count > max_normal and total_cpu < 1.0:
            anomalies.append(f"{instance_count} instances but only {total_cpu:.1f}% CPU (likely zombies)")
            severity += instance_count * 3.0
        
        # Check 4: Single-instance app with multiple instances
        if self.multi_instance_type == 'single_only' and instance_count > 1:
            anomalies.append(f"Should be single-instance but found {instance_count}")
            severity += (instance_count - 1) * 5.0
        
        # Check 5: Memory leak detection (increasing memory per instance)
        if len(self.memory_per_instance) > 20:
            recent_memory = list(self.memory_per_instance)[-10:]
            older_memory = list(self.memory_per_instance)[:10]
            
            if np.mean(recent_memory) > np.mean(older_memory) * 1.5:
                anomalies.append("Possible memory leak detected")
                severity += 3.0
        
        if anomalies:
            self.anomaly_count += 1
            self.last_anomaly_time = time.time()
            return True, "; ".join(anomalies), severity
        
        return False, "Normal", 0.0
    
    def _identify_main_process(self, instances: List[Dict]) -> Optional[Dict]:
        """
        Identify the main/parent process from a list of instances
        Heuristics:
        1. Oldest process (earliest create_time)
        2. Process without "Helper" in name
        3. Parent of other processes
        4. Highest memory usage (main app usually has more memory)
        """
        if not instances:
            return None
        
        # Score each process
        scores = []
        for inst in instances:
            score = 0
            name = inst['name'].lower()
            
            # Prefer non-helper processes
            if 'helper' not in name and 'renderer' not in name and 'gpu' not in name and 'plugin' not in name:
                score += 100
            
            # Prefer older processes (likely the parent)
            oldest_time = min(i['create_time'] for i in instances)
            if inst['create_time'] == oldest_time:
                score += 50
            
            # Prefer higher memory (main process usually has more state)
            score += inst['memory'] * 10
            
            # Check if it's a parent of other processes
            try:
                proc = inst['proc']
                children = proc.children()
                if children:
                    score += len(children) * 20
            except:
                pass
            
            scores.append((score, inst))
        
        # Return highest scoring process
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]
    
    def get_kill_recommendation(self, instances: List[Dict]) -> Dict:
        """
        Recommend which instances to kill (if any)
        SMART STRATEGY: Identify main thread/process and keep it, kill unnecessary helpers
        Returns: {should_kill: bool, kill_pids: [], reason: str, keep_pids: [], main_pid: int}
        """
        if not instances:
            return {'should_kill': False, 'kill_pids': [], 'keep_pids': [], 'reason': 'No instances'}
        
        is_anomalous, reason, severity = self.is_anomalous(instances)
        
        if not is_anomalous or severity < 5.0:
            return {
                'should_kill': False,
                'kill_pids': [],
                'keep_pids': [inst['pid'] for inst in instances],
                'reason': 'Normal behavior'
            }
        
        # SMART STRATEGY: Identify main process first
        main_process = self._identify_main_process(instances)
        
        if not main_process:
            return {
                'should_kill': False,
                'kill_pids': [],
                'keep_pids': [inst['pid'] for inst in instances],
                'reason': 'Could not identify main process'
            }
        
        main_pid = main_process['pid']
        main_name = main_process['name']
        
        # Separate main process from helpers
        helper_processes = [inst for inst in instances if inst['pid'] != main_pid]
        
        if not helper_processes:
            return {
                'should_kill': False,
                'kill_pids': [],
                'keep_pids': [main_pid],
                'reason': 'Only main process running',
                'main_pid': main_pid
            }
        
        # Determine which instances to kill
        kill_pids = []
        keep_pids = [main_pid]  # Always keep main process
        essential_kept = []  # Track what we're keeping
        
        # ALWAYS KEEP: Essential process types (but only one of each type)
        essential_types = {
            'gpu': [],
            'renderer': []
        }
        
        # Categorize helpers by type
        for inst in helper_processes:
            name_lower = inst['name'].lower()
            if 'gpu' in name_lower:
                essential_types['gpu'].append(inst)
            elif 'renderer' in name_lower:
                essential_types['renderer'].append(inst)
        
        # Keep ONE GPU process (the one with highest memory/most active)
        if essential_types['gpu']:
            if len(essential_types['gpu']) == 1:
                gpu_proc = essential_types['gpu'][0]
                keep_pids.append(gpu_proc['pid'])
                essential_kept.append(f"GPU ({gpu_proc['pid']})")
            else:
                # Multiple GPU processes - keep the most active one
                gpu_procs = sorted(essential_types['gpu'], key=lambda x: x['memory'] + x['cpu'], reverse=True)
                keep_pids.append(gpu_procs[0]['pid'])
                essential_kept.append(f"GPU ({gpu_procs[0]['pid']})")
                # Kill duplicate GPU processes
                for dup_gpu in gpu_procs[1:]:
                    kill_pids.append(dup_gpu['pid'])
        
        # Keep ONE main Renderer process (if exists)
        if essential_types['renderer']:
            if len(essential_types['renderer']) == 1:
                renderer_proc = essential_types['renderer'][0]
                keep_pids.append(renderer_proc['pid'])
                essential_kept.append(f"Renderer ({renderer_proc['pid']})")
            else:
                # Multiple renderers - keep the most active one
                renderer_procs = sorted(essential_types['renderer'], key=lambda x: x['memory'] + x['cpu'], reverse=True)
                keep_pids.append(renderer_procs[0]['pid'])
                essential_kept.append(f"Renderer ({renderer_procs[0]['pid']})")
        
        # Now handle remaining helpers (excluding already kept essential ones)
        remaining_helpers = [inst for inst in helper_processes if inst['pid'] not in keep_pids]
        
        # Strategy 1: Kill zombie/idle helper processes
        zombies = [inst for inst in remaining_helpers if inst['cpu'] < 0.1 and inst['memory'] < 0.5]
        active_helpers = [inst for inst in remaining_helpers if inst not in zombies]
        
        if zombies:
            kill_pids.extend([z['pid'] for z in zombies])
            
            # Keep some active helpers if they seem necessary (but not too many)
            if active_helpers and len(active_helpers) <= 2:
                # Keep a couple more active helpers
                keep_pids.extend([h['pid'] for h in active_helpers])
            elif active_helpers:
                # Too many active helpers, keep only the most important ones
                # Sort by memory (higher memory = more important state)
                active_helpers.sort(key=lambda x: x['memory'], reverse=True)
                essential_helpers = active_helpers[:1]  # Keep top 1
                keep_pids.extend([h['pid'] for h in essential_helpers])
                kill_pids.extend([h['pid'] for h in active_helpers[1:]])
            
            essential_str = f" + {', '.join(essential_kept)}" if essential_kept else ""
            return {
                'should_kill': True,
                'kill_pids': kill_pids,
                'keep_pids': keep_pids,
                'main_pid': main_pid,
                'main_name': main_name,
                'reason': f'Keeping main ({main_name} PID:{main_pid}){essential_str} + {len(keep_pids)-1-len(essential_kept)} helpers, killing {len(kill_pids)} unnecessary',
                'severity': severity
            }
        
        # Strategy 2: Too many active helpers (but respect essential processes already kept)
        remaining_helpers = [inst for inst in helper_processes if inst['pid'] not in keep_pids]
        
        if len(remaining_helpers) > 3:
            # Sort helpers by importance (memory + CPU)
            remaining_helpers.sort(key=lambda x: x['memory'] + x['cpu'] * 0.1, reverse=True)
            
            # Keep top 2 most important remaining helpers
            important_helpers = remaining_helpers[:2]
            excess_helpers = remaining_helpers[2:]
            
            keep_pids.extend([h['pid'] for h in important_helpers])
            kill_pids.extend([h['pid'] for h in excess_helpers])
            
            essential_str = f" + {', '.join(essential_kept)}" if essential_kept else ""
            return {
                'should_kill': True,
                'kill_pids': kill_pids,
                'keep_pids': keep_pids,
                'main_pid': main_pid,
                'main_name': main_name,
                'reason': f'Keeping main ({main_name} PID:{main_pid}){essential_str} + {len(important_helpers)} important helpers, killing {len(kill_pids)} excess',
                'severity': severity
            }
        
        # Strategy 3: Kill high-CPU outlier helpers
        if len(helper_processes) > 2:
            cpu_values = [inst['cpu'] for inst in helper_processes]
            mean_cpu = np.mean(cpu_values)
            std_cpu = np.std(cpu_values)
            
            outliers = [inst for inst in helper_processes if inst['cpu'] > mean_cpu + 2 * std_cpu and inst['cpu'] > 20]
            normal_helpers = [inst for inst in helper_processes if inst not in outliers]
            
            if outliers:
                kill_pids.extend([o['pid'] for o in outliers])
                keep_pids.extend([h['pid'] for h in normal_helpers])
                
                return {
                    'should_kill': True,
                    'kill_pids': kill_pids,
                    'keep_pids': keep_pids,
                    'main_pid': main_pid,
                    'main_name': main_name,
                    'reason': f'Keeping main process ({main_name} PID:{main_pid}) + normal helpers, killing {len(kill_pids)} high-CPU outliers',
                    'severity': severity
                }
        
        return {
            'should_kill': False,
            'kill_pids': [],
            'keep_pids': [inst['pid'] for inst in instances],
            'main_pid': main_pid,
            'main_name': main_name,
            'reason': 'Anomalous but helpers seem necessary'
        }
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'app_name': self.app_name,
            'observations': self.observations,
            'instance_counts': list(self.instance_counts),
            'cpu_per_instance': list(self.cpu_per_instance),
            'memory_per_instance': list(self.memory_per_instance),
            'normal_instance_range': self.normal_instance_range,
            'multi_instance_type': self.multi_instance_type,
            'helper_process_names': list(self.helper_process_names),
            'typical_cpu_total': self.typical_cpu_total,
            'typical_memory_total': self.typical_memory_total,
            'anomaly_count': self.anomaly_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessProfile':
        """Deserialize from dictionary"""
        profile = cls(data['app_name'])
        profile.observations = data.get('observations', 0)
        profile.instance_counts = deque(data.get('instance_counts', []), maxlen=200)
        profile.cpu_per_instance = deque(data.get('cpu_per_instance', []), maxlen=100)
        profile.memory_per_instance = deque(data.get('memory_per_instance', []), maxlen=100)
        profile.normal_instance_range = tuple(data.get('normal_instance_range', (1, 1)))
        profile.multi_instance_type = data.get('multi_instance_type', 'unknown')
        profile.helper_process_names = set(data.get('helper_process_names', []))
        profile.typical_cpu_total = data.get('typical_cpu_total', 0.0)
        profile.typical_memory_total = data.get('typical_memory_total', 0.0)
        profile.anomaly_count = data.get('anomaly_count', 0)
        return profile


class IntelligentProcessMonitor:
    """
    Universal process monitor that learns normal behavior for all apps
    """
    
    def __init__(self):
        self.profiles: Dict[str, ProcessProfile] = {}
        self.load_learning_data()
        self.scan_count = 0
        logger.info("🔍 Intelligent Process Monitor initialized")
    
    def load_learning_data(self):
        """Load learned profiles from disk"""
        try:
            if os.path.exists(LEARNING_DATA_FILE):
                with open(LEARNING_DATA_FILE, 'r') as f:
                    data = json.load(f)
                    for app_name, profile_data in data.items():
                        self.profiles[app_name] = ProcessProfile.from_dict(profile_data)
                logger.info(f"📚 Loaded {len(self.profiles)} learned app profiles")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
    
    def save_learning_data(self):
        """Save learned profiles to disk"""
        try:
            data = {name: profile.to_dict() for name, profile in self.profiles.items()}
            with open(LEARNING_DATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def normalize_app_name(self, process_name: str) -> str:
        """Normalize process name to app name"""
        name = process_name.lower()
        
        # Remove common suffixes
        for suffix in [' helper', ' (renderer)', ' (gpu)', ' (plugin)', ' (utility)', '.exe', '.app']:
            name = name.replace(suffix, '')
        
        # Remove version numbers
        import re
        name = re.sub(r'\d+', '', name)
        
        return name.strip()
    
    def scan_processes(self) -> Dict[str, List[Dict]]:
        """Scan all processes and group by app"""
        app_processes = defaultdict(list)
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                name = proc.info['name']
                app_name = self.normalize_app_name(name)
                
                if not app_name:
                    continue
                
                cpu = proc.cpu_percent(interval=0.01)
                memory = proc.memory_percent()
                
                app_processes[app_name].append({
                    'pid': proc.info['pid'],
                    'name': name,
                    'cpu': cpu,
                    'memory': memory,
                    'create_time': proc.info['create_time'],
                    'proc': proc
                })
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return app_processes
    
    def analyze_and_learn(self) -> List[Dict]:
        """
        Analyze all processes, learn patterns, and detect anomalies
        Returns list of anomalies with kill recommendations
        """
        self.scan_count += 1
        app_processes = self.scan_processes()
        anomalies = []
        
        for app_name, instances in app_processes.items():
            # Get or create profile
            if app_name not in self.profiles:
                self.profiles[app_name] = ProcessProfile(app_name)
            
            profile = self.profiles[app_name]
            
            # Learn from observation
            profile.observe(instances)
            
            # Check for anomalies
            is_anomalous, reason, severity = profile.is_anomalous(instances)
            
            if is_anomalous:
                # Get kill recommendation
                kill_rec = profile.get_kill_recommendation(instances)
                
                anomalies.append({
                    'app_name': app_name,
                    'instance_count': len(instances),
                    'total_cpu': sum(inst['cpu'] for inst in instances),
                    'total_memory': sum(inst['memory'] for inst in instances),
                    'reason': reason,
                    'severity': severity,
                    'multi_instance_type': profile.multi_instance_type,
                    'normal_range': profile.normal_instance_range,
                    'observations': profile.observations,
                    'kill_recommendation': kill_rec,
                    'instances': instances
                })
        
        # Save learning data periodically
        if self.scan_count % 10 == 0:
            self.save_learning_data()
        
        # Sort by severity
        anomalies.sort(key=lambda x: x['severity'], reverse=True)
        
        return anomalies
    
    def execute_kill_recommendation(self, anomaly: Dict, dry_run: bool = True) -> Dict:
        """
        Execute kill recommendation for an anomaly
        """
        kill_rec = anomaly['kill_recommendation']
        
        if not kill_rec['should_kill']:
            return {'executed': False, 'reason': 'No kill recommended'}
        
        killed = []
        failed = []
        
        for pid in kill_rec['kill_pids']:
            try:
                if not dry_run:
                    proc = psutil.Process(pid)
                    proc.terminate()  # Graceful termination
                    killed.append(pid)
                else:
                    killed.append(pid)  # Dry run
            except Exception as e:
                failed.append({'pid': pid, 'error': str(e)})
        
        return {
            'executed': True,
            'dry_run': dry_run,
            'killed_pids': killed,
            'failed': failed,
            'kept_pids': kill_rec['keep_pids'],
            'reason': kill_rec['reason']
        }
    
    def get_app_profile_summary(self, app_name: str) -> Optional[Dict]:
        """Get summary of learned profile for an app"""
        normalized = self.normalize_app_name(app_name)
        
        if normalized not in self.profiles:
            return None
        
        profile = self.profiles[normalized]
        
        return {
            'app_name': app_name,
            'observations': profile.observations,
            'normal_instance_range': profile.normal_instance_range,
            'multi_instance_type': profile.multi_instance_type,
            'typical_cpu': profile.typical_cpu_total,
            'typical_memory': profile.typical_memory_total,
            'anomaly_count': profile.anomaly_count,
            'helper_processes': list(profile.helper_process_names)
        }
    
    def get_all_profiles_summary(self) -> List[Dict]:
        """Get summary of all learned profiles"""
        summaries = []
        
        for app_name, profile in self.profiles.items():
            if profile.observations >= 5:  # Only show apps with enough data
                summaries.append({
                    'app_name': app_name,
                    'observations': profile.observations,
                    'normal_instances': f"{profile.normal_instance_range[0]}-{profile.normal_instance_range[1]}",
                    'type': profile.multi_instance_type,
                    'anomalies': profile.anomaly_count
                })
        
        summaries.sort(key=lambda x: x['observations'], reverse=True)
        return summaries


# Global instance
_monitor_instance = None

def get_monitor() -> IntelligentProcessMonitor:
    """Get or create global monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = IntelligentProcessMonitor()
    return _monitor_instance


if __name__ == "__main__":
    import sys
    
    # Check for auto-kill flag
    auto_kill = '--kill' in sys.argv or '--auto-kill' in sys.argv
    dry_run = '--dry-run' in sys.argv
    
    print("🔍 Intelligent Universal Process Monitor")
    print("=" * 70)
    if auto_kill:
        print("⚠️  AUTO-KILL MODE ENABLED" if not dry_run else "🧪 DRY-RUN MODE (no processes will be killed)")
    print("Learning normal behavior for all apps...")
    print()
    
    monitor = get_monitor()
    
    # Run analysis
    print("📊 Scanning processes...")
    
    # Debug: Show what we're seeing
    app_processes = monitor.scan_processes()
    print(f"\n🔎 Found {len(app_processes)} unique apps with {sum(len(procs) for procs in app_processes.values())} total processes")
    
    # Show apps with multiple instances
    multi_instance = {app: procs for app, procs in app_processes.items() if len(procs) > 1}
    if multi_instance:
        print(f"\n📦 Apps with multiple instances:")
        for app, procs in sorted(multi_instance.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            total_cpu = sum(p['cpu'] for p in procs)
            print(f"   {app}: {len(procs)} instances, {total_cpu:.1f}% CPU")
    
    anomalies = monitor.analyze_and_learn()
    
    if anomalies:
        print(f"\n⚠️  Found {len(anomalies)} anomalies:\n")
        
        for i, anomaly in enumerate(anomalies, 1):
            print(f"{i}. {anomaly['app_name'].upper()}")
            print(f"   Instances: {anomaly['instance_count']} (normal: {anomaly['normal_range'][0]}-{anomaly['normal_range'][1]})")
            print(f"   CPU: {anomaly['total_cpu']:.1f}% | Memory: {anomaly['total_memory']:.1f}%")
            print(f"   Type: {anomaly['multi_instance_type']}")
            print(f"   Issue: {anomaly['reason']}")
            print(f"   Severity: {anomaly['severity']:.1f}")
            
            kill_rec = anomaly['kill_recommendation']
            if kill_rec['should_kill']:
                print(f"   🎯 RECOMMENDATION: {kill_rec['reason']}")
                print(f"      Kill PIDs: {kill_rec['kill_pids']}")
                print(f"      Keep PIDs: {kill_rec['keep_pids']}")
                
                # Execute if auto-kill enabled
                if auto_kill and anomaly['severity'] >= 15.0:  # Only kill high severity
                    print(f"   ⚡ Executing kill recommendation...")
                    result = monitor.execute_kill_recommendation(anomaly, dry_run=dry_run)
                    if result['executed']:
                        if dry_run:
                            print(f"      [DRY RUN] Would kill {len(result['killed_pids'])} processes")
                        else:
                            print(f"      ✅ Killed {len(result['killed_pids'])} processes")
                            if result['failed']:
                                print(f"      ⚠️  Failed to kill {len(result['failed'])} processes")
            else:
                print(f"   ✓ No action needed: {kill_rec['reason']}")
            
            print()
    else:
        print("✅ No anomalies detected - all processes normal!")
    
    # Show learned profiles
    print("\n📚 Learned App Profiles:")
    profiles = monitor.get_all_profiles_summary()
    
    if profiles:
        print(f"\n{'App':<20} {'Observations':<15} {'Normal Instances':<20} {'Type':<15} {'Anomalies'}")
        print("-" * 90)
        for profile in profiles[:15]:  # Top 15
            print(f"{profile['app_name']:<20} {profile['observations']:<15} {profile['normal_instances']:<20} {profile['type']:<15} {profile['anomalies']}")
    else:
        print("   No profiles learned yet. Run multiple times to build learning data.")
    
    print(f"\n💾 Learning data saved to: {LEARNING_DATA_FILE}")
    
    if not auto_kill:
        print("\n💡 TIP: Run with --dry-run to see what would be killed")
        print("        Run with --kill to automatically kill problematic processes")
    
    print("\n✅ Scan complete!")
