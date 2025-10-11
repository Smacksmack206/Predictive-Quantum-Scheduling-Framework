#!/usr/bin/env python3
"""
Enhanced EAS Classifier - Dynamic Process Classification for Energy Aware Scheduling
Uses machine learning, behavioral analysis, and real-time metrics for intelligent workload classification
"""

import psutil
import time
import json
import sqlite3
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ProcessProfile:
    """Comprehensive process profile for classification"""
    name: str
    pid: int
    cpu_usage_history: List[float]
    memory_usage_history: List[float]
    io_read_history: List[int]
    io_write_history: List[int]
    network_activity: List[int]
    gpu_usage_estimate: float
    user_interaction_score: float
    energy_efficiency_score: float
    classification_confidence: float
    last_updated: datetime
    
class DynamicProcessClassifier:
    """Advanced process classifier using multiple intelligence sources"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.process_profiles = {}
        self.classification_history = deque(maxlen=1000)
        self.learning_data = defaultdict(list)
        self.init_database()
        
        # Dynamic classification thresholds (self-adjusting)
        self.thresholds = {
            'cpu_interactive': 15.0,
            'cpu_compute': 50.0,
            'memory_heavy': 500.0,  # MB
            'io_intensive': 10.0,   # MB/s
            'network_active': 1.0,  # MB/s
            'gpu_threshold': 20.0   # %
        }
        
        # Classification confidence weights
        self.confidence_weights = {
            'behavioral_analysis': 0.4,
            'resource_pattern': 0.3,
            'user_interaction': 0.2,
            'historical_data': 0.1
        }
        
        # Start background learning thread
        self.learning_thread = threading.Thread(target=self._continuous_learning, daemon=True)
        self.learning_thread.start()
    
    def init_database(self):
        """Initialize process learning database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS process_classifications (
                timestamp TEXT,
                process_name TEXT,
                pid INTEGER,
                classification TEXT,
                confidence REAL,
                cpu_usage REAL,
                memory_mb REAL,
                io_activity REAL,
                network_activity REAL,
                gpu_usage REAL,
                user_interaction_score REAL,
                energy_efficiency REAL,
                core_assignment TEXT,
                performance_impact REAL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS classification_feedback (
                timestamp TEXT,
                process_name TEXT,
                predicted_class TEXT,
                actual_performance TEXT,
                energy_impact REAL,
                user_satisfaction REAL,
                adjustment_needed BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_process_behavior(self, pid: int, name: str) -> Dict:
        """Comprehensive behavioral analysis of a process"""
        try:
            proc = psutil.Process(pid)
            
            # Get current metrics
            cpu_percent = proc.cpu_percent(interval=0.5)
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # I/O activity
            try:
                io_counters = proc.io_counters()
                io_read = io_counters.read_bytes
                io_write = io_counters.write_bytes
            except (psutil.AccessDenied, AttributeError):
                io_read = io_write = 0
            
            # Network activity estimation
            network_activity = self._estimate_network_activity(proc)
            
            # GPU usage estimation
            gpu_usage = self._estimate_gpu_usage(name, cpu_percent)
            
            # User interaction score
            user_interaction = self._calculate_user_interaction_score(name, proc)
            
            # Energy efficiency score
            energy_efficiency = self._calculate_energy_efficiency(cpu_percent, memory_mb, gpu_usage)
            
            return {
                'cpu_usage': cpu_percent,
                'memory_mb': memory_mb,
                'io_read': io_read,
                'io_write': io_write,
                'network_activity': network_activity,
                'gpu_usage': gpu_usage,
                'user_interaction_score': user_interaction,
                'energy_efficiency_score': energy_efficiency,
                'process_age': time.time() - proc.create_time(),
                'num_threads': proc.num_threads(),
                'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else 0
            }
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def _estimate_network_activity(self, proc: psutil.Process) -> float:
        """Estimate network activity for a process"""
        try:
            # Get process connections
            connections = proc.connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            # Estimate based on connection count and process type
            name_lower = proc.name().lower()
            
            # Network-heavy applications
            if any(app in name_lower for app in ['chrome', 'safari', 'firefox', 'zoom', 'teams', 'slack']):
                return active_connections * 2.0  # Higher weight for browsers/communication
            elif any(app in name_lower for app in ['dropbox', 'onedrive', 'googledrive', 'sync']):
                return active_connections * 3.0  # Even higher for sync apps
            else:
                return active_connections * 0.5  # Lower weight for other apps
                
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return 0.0
    
    def _estimate_gpu_usage(self, name: str, cpu_usage: float) -> float:
        """Estimate GPU usage based on process characteristics"""
        name_lower = name.lower()
        
        # GPU-intensive applications
        gpu_heavy_apps = {
            'final cut': 80, 'premiere': 75, 'after effects': 70, 'davinci': 85,
            'blender': 90, 'cinema4d': 85, 'maya': 80, 'unity': 70,
            'photoshop': 60, 'illustrator': 40, 'sketch': 30,
            'chrome': 20, 'safari': 15, 'firefox': 18,  # WebGL/video
            'zoom': 25, 'teams': 20, 'facetime': 30,    # Video calls
            'games': 95, 'steam': 80, 'epic': 75        # Gaming
        }
        
        base_gpu_usage = 0
        for app, usage in gpu_heavy_apps.items():
            if app in name_lower:
                base_gpu_usage = usage
                break
        
        # Adjust based on CPU usage (correlation)
        if base_gpu_usage > 0:
            cpu_factor = min(cpu_usage / 50.0, 2.0)  # Scale with CPU usage
            return min(base_gpu_usage * cpu_factor, 100)
        
        return 0
    
    def _calculate_user_interaction_score(self, name: str, proc: psutil.Process) -> float:
        """Calculate how much user interaction this process likely has"""
        name_lower = name.lower()
        
        # High interaction apps
        if any(app in name_lower for app in ['finder', 'safari', 'chrome', 'firefox', 'terminal', 'iterm']):
            return 0.9
        elif any(app in name_lower for app in ['xcode', 'vscode', 'sublime', 'photoshop', 'sketch']):
            return 0.8
        elif any(app in name_lower for app in ['zoom', 'teams', 'slack', 'messages', 'facetime']):
            return 0.7
        elif any(app in name_lower for app in ['music', 'spotify', 'vlc', 'quicktime']):
            return 0.6
        
        # System/background processes
        elif any(app in name_lower for app in ['kernel', 'launchd', 'system', 'daemon', 'helper']):
            return 0.1
        elif any(app in name_lower for app in ['backup', 'sync', 'cloud', 'update']):
            return 0.2
        
        # Check if process has GUI windows (macOS specific)
        try:
            # Use AppleScript to check if app has windows
            script = f'''
            tell application "System Events"
                try
                    set appProcess to first process whose name is "{name}"
                    set windowCount to count of windows of appProcess
                    return windowCount
                on error
                    return 0
                end try
            end tell
            '''
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                window_count = int(result.stdout.strip() or 0)
                if window_count > 0:
                    return 0.7  # Has GUI windows
                else:
                    return 0.3  # No GUI windows
            
        except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError):
            pass
        
        # Default based on process characteristics
        try:
            if proc.terminal():
                return 0.4  # Terminal process
            else:
                return 0.5  # Unknown GUI process
        except (psutil.AccessDenied, AttributeError):
            return 0.4  # Default
    
    def _calculate_energy_efficiency(self, cpu_usage: float, memory_mb: float, gpu_usage: float) -> float:
        """Calculate energy efficiency score (higher = more efficient)"""
        # Base efficiency (inverse of resource usage)
        cpu_efficiency = max(0, 100 - cpu_usage) / 100
        memory_efficiency = max(0, 100 - min(memory_mb / 1000, 100)) / 100  # Normalize to 1GB
        gpu_efficiency = max(0, 100 - gpu_usage) / 100
        
        # Weighted average
        efficiency = (cpu_efficiency * 0.5 + memory_efficiency * 0.3 + gpu_efficiency * 0.2)
        
        return efficiency
    
    def classify_process_intelligent(self, pid: int, name: str) -> Tuple[str, float]:
        """Intelligent process classification using multiple methods"""
        
        # Method 1: Behavioral Analysis
        behavior = self.analyze_process_behavior(pid, name)
        if not behavior:
            return self._fallback_classification(name), 0.3
        
        behavioral_class, behavioral_confidence = self._classify_by_behavior(behavior)
        
        # Method 2: Resource Pattern Analysis
        pattern_class, pattern_confidence = self._classify_by_resource_pattern(behavior)
        
        # Method 3: User Interaction Analysis
        interaction_class, interaction_confidence = self._classify_by_interaction(behavior)
        
        # Method 4: Historical Data
        historical_class, historical_confidence = self._classify_by_history(name)
        
        # Method 5: Machine Learning Classification
        ml_class, ml_confidence = self._classify_by_ml(name, behavior)
        
        # Combine classifications with weighted confidence
        classifications = [
            (behavioral_class, behavioral_confidence * self.confidence_weights['behavioral_analysis']),
            (pattern_class, pattern_confidence * self.confidence_weights['resource_pattern']),
            (interaction_class, interaction_confidence * self.confidence_weights['user_interaction']),
            (historical_class, historical_confidence * self.confidence_weights['historical_data']),
            (ml_class, ml_confidence * 0.3)  # ML gets additional weight
        ]
        
        # Weighted voting
        class_scores = defaultdict(float)
        total_confidence = 0
        
        for cls, confidence in classifications:
            class_scores[cls] += confidence
            total_confidence += confidence
        
        # Get best classification
        if class_scores:
            best_class = max(class_scores.items(), key=lambda x: x[1])
            final_confidence = best_class[1] / max(total_confidence, 1.0)
            
            # Store learning data
            self._store_classification_data(pid, name, best_class[0], final_confidence, behavior)
            
            return best_class[0], final_confidence
        
        # Fallback
        return self._fallback_classification(name), 0.2
    
    def _classify_by_behavior(self, behavior: Dict) -> Tuple[str, float]:
        """Classify based on current behavioral metrics"""
        cpu = behavior.get('cpu_usage', 0)
        memory = behavior.get('memory_mb', 0)
        gpu = behavior.get('gpu_usage', 0)
        interaction = behavior.get('user_interaction_score', 0)
        
        # High CPU + High GPU = Compute intensive
        if cpu > 50 and gpu > 30:
            return 'compute_intensive', 0.9
        
        # High interaction + Moderate resources = Interactive
        elif interaction > 0.7 and cpu > 5:
            return 'interactive_heavy', 0.8
        
        # High interaction + Low resources = Light interactive
        elif interaction > 0.6 and cpu < 15:
            return 'interactive_light', 0.8
        
        # Low interaction + Low resources = Background
        elif interaction < 0.3 and cpu < 10:
            return 'background', 0.7
        
        # High CPU + Low interaction = Background compute
        elif cpu > 30 and interaction < 0.4:
            return 'background_compute', 0.7
        
        # Medium resources = General purpose
        else:
            return 'general_purpose', 0.5
    
    def _classify_by_resource_pattern(self, behavior: Dict) -> Tuple[str, float]:
        """Classify based on resource usage patterns"""
        cpu = behavior.get('cpu_usage', 0)
        memory = behavior.get('memory_mb', 0)
        io_read = behavior.get('io_read', 0)
        io_write = behavior.get('io_write', 0)
        network = behavior.get('network_activity', 0)
        
        # I/O intensive
        if (io_read + io_write) > 50 * 1024 * 1024:  # 50MB/s
            return 'io_intensive', 0.8
        
        # Network intensive
        elif network > 5.0:
            return 'network_intensive', 0.8
        
        # Memory intensive
        elif memory > 1000:  # 1GB
            return 'memory_intensive', 0.7
        
        # CPU intensive
        elif cpu > 60:
            return 'cpu_intensive', 0.8
        
        # Balanced workload
        elif cpu > 20 and memory > 200:
            return 'balanced_workload', 0.6
        
        # Lightweight
        else:
            return 'lightweight', 0.6
    
    def _classify_by_interaction(self, behavior: Dict) -> Tuple[str, float]:
        """Classify based on user interaction patterns"""
        interaction_score = behavior.get('user_interaction_score', 0)
        
        if interaction_score > 0.8:
            return 'user_facing', 0.9
        elif interaction_score > 0.5:
            return 'semi_interactive', 0.7
        elif interaction_score > 0.2:
            return 'background_ui', 0.6
        else:
            return 'system_service', 0.8
    
    def _classify_by_history(self, name: str) -> Tuple[str, float]:
        """Classify based on historical performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT classification, AVG(confidence), COUNT(*) as frequency
            FROM process_classifications 
            WHERE process_name = ? AND timestamp > datetime('now', '-7 days')
            GROUP BY classification
            ORDER BY frequency DESC, AVG(confidence) DESC
            LIMIT 1
        ''', (name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[2] > 3:  # At least 3 historical records
            return result[0], min(result[1] * 1.2, 1.0)  # Boost historical confidence
        
        return 'unknown', 0.1
    
    def _classify_by_ml(self, name: str, behavior: Dict) -> Tuple[str, float]:
        """Simple ML-based classification using learned patterns"""
        # This is a simplified ML approach - could be enhanced with scikit-learn
        
        # Feature vector
        features = [
            behavior.get('cpu_usage', 0) / 100,
            behavior.get('memory_mb', 0) / 1000,
            behavior.get('gpu_usage', 0) / 100,
            behavior.get('user_interaction_score', 0),
            behavior.get('energy_efficiency_score', 0),
            behavior.get('network_activity', 0) / 10,
            len(name) / 50,  # Name length as feature
            1 if any(x in name.lower() for x in ['system', 'kernel', 'daemon']) else 0
        ]
        
        # Simple decision tree logic (could be replaced with trained model)
        score = sum(features) / len(features)
        
        if score > 0.7:
            return 'high_priority', 0.6
        elif score > 0.4:
            return 'medium_priority', 0.5
        else:
            return 'low_priority', 0.4
    
    def _fallback_classification(self, name: str) -> str:
        """Fallback classification for unknown processes"""
        name_lower = name.lower()
        
        # Enhanced fallback with more categories
        if any(x in name_lower for x in ['system', 'kernel', 'launchd', 'daemon']):
            return 'system_critical'
        elif any(x in name_lower for x in ['helper', 'agent', 'service']):
            return 'system_service'
        elif any(x in name_lower for x in ['backup', 'sync', 'cloud']):
            return 'background_sync'
        elif any(x in name_lower for x in ['update', 'installer']):
            return 'maintenance'
        elif name_lower.endswith('d'):  # Many daemons end with 'd'
            return 'daemon_process'
        else:
            return 'unknown_application'
    
    def _store_classification_data(self, pid: int, name: str, classification: str, 
                                 confidence: float, behavior: Dict):
        """Store classification data for learning"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO process_classifications 
            (timestamp, process_name, pid, classification, confidence, cpu_usage, 
             memory_mb, io_activity, network_activity, gpu_usage, user_interaction_score, 
             energy_efficiency, core_assignment, performance_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(), name, pid, classification, confidence,
            behavior.get('cpu_usage', 0), behavior.get('memory_mb', 0),
            behavior.get('io_read', 0) + behavior.get('io_write', 0),
            behavior.get('network_activity', 0), behavior.get('gpu_usage', 0),
            behavior.get('user_interaction_score', 0), behavior.get('energy_efficiency_score', 0),
            'pending', 0.0  # Will be updated after core assignment
        ))
        conn.commit()
        conn.close()
    
    def _continuous_learning(self):
        """Background thread for continuous learning and threshold adjustment"""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._adjust_thresholds()
                self._cleanup_old_data()
            except Exception as e:
                print(f"Learning thread error: {e}")
    
    def _adjust_thresholds(self):
        """Dynamically adjust classification thresholds based on performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Analyze recent classifications for threshold optimization
        cursor = conn.execute('''
            SELECT classification, AVG(cpu_usage), AVG(memory_mb), AVG(energy_efficiency)
            FROM process_classifications 
            WHERE timestamp > datetime('now', '-1 day')
            GROUP BY classification
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        # Adjust thresholds based on actual data
        for classification, avg_cpu, avg_memory, avg_efficiency in results:
            if classification == 'interactive_heavy' and avg_cpu < self.thresholds['cpu_interactive']:
                self.thresholds['cpu_interactive'] = max(5, avg_cpu * 0.8)
            elif classification == 'compute_intensive' and avg_cpu < self.thresholds['cpu_compute']:
                self.thresholds['cpu_compute'] = max(30, avg_cpu * 0.9)
    
    def _cleanup_old_data(self):
        """Clean up old classification data"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            DELETE FROM process_classifications 
            WHERE timestamp < datetime('now', '-30 days')
        ''')
        conn.commit()
        conn.close()
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics for monitoring"""
        conn = sqlite3.connect(self.db_path)
        
        # Recent classifications
        cursor = conn.execute('''
            SELECT classification, COUNT(*), AVG(confidence)
            FROM process_classifications 
            WHERE timestamp > datetime('now', '-1 day')
            GROUP BY classification
            ORDER BY COUNT(*) DESC
        ''')
        
        recent_stats = cursor.fetchall()
        
        # Overall accuracy (simplified)
        cursor = conn.execute('''
            SELECT AVG(confidence), COUNT(*)
            FROM process_classifications 
            WHERE timestamp > datetime('now', '-7 days')
        ''')
        
        accuracy_data = cursor.fetchone()
        conn.close()
        
        return {
            'recent_classifications': recent_stats,
            'average_confidence': accuracy_data[0] if accuracy_data[0] else 0,
            'total_classifications': accuracy_data[1] if accuracy_data[1] else 0,
            'current_thresholds': self.thresholds.copy()
        }

class EnhancedEASScheduler:
    """Enhanced EAS with dynamic classification"""
    
    def __init__(self, db_path: str):
        self.classifier = DynamicProcessClassifier(db_path)
        self.core_assignments = {}
        self.performance_history = deque(maxlen=100)
        
        # Enhanced core assignment strategies
        self.assignment_strategies = {
            'system_critical': 'p_core_reserved',
            'user_facing': 'p_core_priority',
            'interactive_heavy': 'p_core_balanced',
            'interactive_light': 'e_core_preferred',
            'compute_intensive': 'p_core_dedicated',
            'background_compute': 'e_core_compute',
            'background': 'e_core_only',
            'system_service': 'e_core_only',
            'io_intensive': 'balanced_io',
            'network_intensive': 'e_core_network',
            'memory_intensive': 'balanced_memory'
        }
    
    def classify_and_assign(self, pid: int, name: str) -> Dict:
        """Enhanced classification and core assignment"""
        
        # Get intelligent classification
        classification, confidence = self.classifier.classify_process_intelligent(pid, name)
        
        # Determine optimal core assignment strategy
        strategy = self.assignment_strategies.get(classification, 'balanced_default')
        
        # Calculate optimal assignment
        assignment = self._calculate_enhanced_assignment(pid, name, classification, strategy, confidence)
        
        return assignment
    
    def _calculate_enhanced_assignment(self, pid: int, name: str, classification: str, 
                                    strategy: str, confidence: float) -> Dict:
        """Calculate enhanced core assignment with multiple factors"""
        
        try:
            proc = psutil.Process(pid)
            cpu_usage = proc.cpu_percent(interval=0.1)
            
            # Get system load for dynamic adjustment
            system_load = psutil.cpu_percent(interval=0.1)
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
            
            if len(per_cpu) >= 8:
                p_core_load = sum(per_cpu[:4]) / 4
                e_core_load = sum(per_cpu[4:8]) / 4
            else:
                p_core_load = e_core_load = system_load
            
            # Strategy-based assignment
            if strategy == 'p_core_reserved':
                target_core = 'p_core'
                priority_adjustment = -5  # High priority
            elif strategy == 'p_core_priority':
                target_core = 'p_core' if p_core_load < 80 else 'e_core'
                priority_adjustment = -3
            elif strategy == 'p_core_balanced':
                target_core = 'p_core' if p_core_load < e_core_load + 20 else 'e_core'
                priority_adjustment = -1
            elif strategy == 'e_core_preferred':
                target_core = 'e_core' if e_core_load < 70 else 'p_core'
                priority_adjustment = 1
            elif strategy == 'p_core_dedicated':
                target_core = 'p_core'
                priority_adjustment = -4
            elif strategy == 'e_core_compute':
                target_core = 'e_core'
                priority_adjustment = 0
            elif strategy == 'e_core_only':
                target_core = 'e_core'
                priority_adjustment = 2
            elif strategy == 'balanced_io':
                target_core = 'e_core' if cpu_usage < 30 else 'p_core'
                priority_adjustment = 0
            elif strategy == 'e_core_network':
                target_core = 'e_core'
                priority_adjustment = 1
            elif strategy == 'balanced_memory':
                target_core = 'p_core' if system_load > 60 else 'e_core'
                priority_adjustment = 0
            else:  # balanced_default
                target_core = 'p_core' if cpu_usage > 25 else 'e_core'
                priority_adjustment = 0
            
            # Confidence-based adjustment
            if confidence < 0.5:
                priority_adjustment = min(priority_adjustment + 1, 5)  # Less aggressive if uncertain
            
            return {
                'pid': pid,
                'name': name,
                'classification': classification,
                'strategy': strategy,
                'target_core': target_core,
                'priority_adjustment': priority_adjustment,
                'confidence': confidence,
                'cpu_usage': cpu_usage,
                'system_load': system_load,
                'p_core_load': p_core_load,
                'e_core_load': e_core_load
            }
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                'pid': pid,
                'name': name,
                'classification': 'unknown',
                'strategy': 'balanced_default',
                'target_core': 'e_core',
                'priority_adjustment': 2,
                'confidence': 0.1,
                'error': 'process_access_denied'
            }

def main():
    """Test the enhanced EAS classifier"""
    db_path = os.path.expanduser("~/.battery_optimizer_enhanced_eas.db")
    scheduler = EnhancedEASScheduler(db_path)
    
    print("🧠 Enhanced EAS Classifier Test")
    print("=" * 50)
    
    # Test on current processes
    test_processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['pid'] > 100:  # Skip system processes
                test_processes.append((proc.info['pid'], proc.info['name']))
            if len(test_processes) >= 10:
                break
        except:
            continue
    
    print(f"Testing {len(test_processes)} processes:\n")
    
    for pid, name in test_processes:
        assignment = scheduler.classify_and_assign(pid, name)
        
        print(f"Process: {name[:25]:25}")
        print(f"  Classification: {assignment['classification']}")
        print(f"  Strategy: {assignment['strategy']}")
        print(f"  Target Core: {assignment['target_core']}")
        print(f"  Confidence: {assignment['confidence']:.2f}")
        print(f"  CPU Usage: {assignment.get('cpu_usage', 0):.1f}%")
        print()
    
    # Show classification stats
    stats = scheduler.classifier.get_classification_stats()
    print("📊 Classification Statistics:")
    print(f"  Total Classifications: {stats['total_classifications']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2f}")
    print(f"  Recent Classifications: {len(stats['recent_classifications'])}")
    
    return scheduler

if __name__ == "__main__":
    main()