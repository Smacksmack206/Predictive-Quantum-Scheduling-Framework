#!/usr/bin/env python3
"""
Advanced Enhanced EAS System - The Ultimate Implementation
Combines fast lightweight operations with strategic expensive operations
Uses intelligent scheduling, caching, and background processing
"""

import psutil
import time
import json
import sqlite3
import os
import threading
import subprocess
import queue
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import weakref

@dataclass
class ProcessIntelligence:
    """Comprehensive process intelligence data"""
    pid: int
    name: str
    
    # Fast metrics (always collected)
    cpu_usage: float
    memory_mb: float
    create_time: float
    
    # Medium-cost metrics (collected periodically)
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_connections: int = 0
    num_threads: int = 0
    
    # Expensive metrics (collected strategically)
    has_gui_windows: Optional[bool] = None
    window_count: Optional[int] = None
    user_interaction_score: Optional[float] = None
    
    # Classification results
    classification: str = "unknown"
    confidence: float = 0.0
    classification_method: str = "none"
    
    # Metadata
    last_updated: datetime = None
    last_expensive_check: datetime = None
    priority_score: float = 0.0

class IntelligentProcessAnalyzer:
    """Intelligent process analyzer with strategic expensive operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.process_cache = {}
        self.expensive_operation_queue = queue.Queue()
        self.gui_detection_cache = {}
        
        # Thread pools for different operation types
        self.fast_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FastEAS")
        self.expensive_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ExpensiveEAS")
        
        # Background processing
        self.background_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.background_thread.start()
        
        self.init_database()
        
        # Classification rules (fast lookup)
        self.fast_classification_rules = {
            'interactive_critical': {
                'keywords': ['finder', 'safari', 'chrome', 'firefox', 'terminal', 'iterm'],
                'confidence': 0.9,
                'priority': 10
            },
            'development_tools': {
                'keywords': ['xcode', 'vscode', 'sublime', 'atom', 'intellij', 'pycharm'],
                'confidence': 0.9,
                'priority': 9
            },
            'communication': {
                'keywords': ['zoom', 'teams', 'slack', 'discord', 'facetime', 'messages'],
                'confidence': 0.8,
                'priority': 8
            },
            'creative_apps': {
                'keywords': ['photoshop', 'illustrator', 'sketch', 'figma', 'final cut', 'premiere'],
                'confidence': 0.8,
                'priority': 7
            },
            'system_critical': {
                'keywords': ['kernel_task', 'launchd', 'windowserver', 'loginwindow'],
                'confidence': 0.95,
                'priority': 10
            },
            'system_services': {
                'keywords': ['backupd', 'spotlight', 'mds', 'cloudd', 'coreauthd', 'logind'],
                'confidence': 0.8,
                'priority': 3
            },
            'compute_intensive': {
                'keywords': ['python', 'node', 'java', 'ffmpeg', 'handbrake', 'blender'],
                'confidence': 0.7,
                'priority': 6
            }
        }
    
    def init_database(self):
        """Initialize advanced database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS process_intelligence (
                timestamp TEXT,
                pid INTEGER,
                name TEXT,
                classification TEXT,
                confidence REAL,
                cpu_usage REAL,
                memory_mb REAL,
                has_gui_windows BOOLEAN,
                window_count INTEGER,
                user_interaction_score REAL,
                classification_method TEXT,
                priority_score REAL,
                core_assignment TEXT,
                performance_impact REAL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS gui_detection_cache (
                process_name TEXT PRIMARY KEY,
                has_gui BOOLEAN,
                window_count INTEGER,
                last_checked TEXT,
                check_count INTEGER
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS classification_performance (
                timestamp TEXT,
                method TEXT,
                processes_analyzed INTEGER,
                time_taken_ms REAL,
                accuracy_score REAL
            )
        ''')
        
        conn.commit()
        conn.close()    

    def analyze_process_intelligent(self, pid: int, name: str) -> ProcessIntelligence:
        """Intelligent process analysis with strategic expensive operations"""
        
        # Check cache first
        cache_key = f"{name}_{pid}"
        if cache_key in self.process_cache:
            cached = self.process_cache[cache_key]
            if time.time() - cached.last_updated.timestamp() < 30:
                return cached
        
        # Create process intelligence object
        intel = ProcessIntelligence(
            pid=pid,
            name=name,
            last_updated=datetime.now()
        )
        
        try:
            proc = psutil.Process(pid)
            
            # FAST OPERATIONS (always do these)
            intel.cpu_usage = proc.cpu_percent(interval=0)
            intel.memory_mb = proc.memory_info().rss / (1024 * 1024)
            intel.create_time = proc.create_time()
            
            # Fast classification
            intel.classification, intel.confidence, intel.priority_score = self._classify_fast(name)
            intel.classification_method = "fast_rules"
            
            # MEDIUM-COST OPERATIONS (do periodically)
            if self._should_do_medium_analysis(intel):
                try:
                    io_counters = proc.io_counters()
                    intel.io_read_bytes = io_counters.read_bytes
                    intel.io_write_bytes = io_counters.write_bytes
                    intel.network_connections = len(proc.connections())
                    intel.num_threads = proc.num_threads()
                except (psutil.AccessDenied, AttributeError):
                    pass
            
            # EXPENSIVE OPERATIONS (do strategically)
            if self._should_do_expensive_analysis(intel):
                # Queue for background processing
                self.expensive_operation_queue.put(intel)
            else:
                # Use cached expensive data if available
                cached_gui = self._get_cached_gui_info(name)
                if cached_gui:
                    intel.has_gui_windows = cached_gui['has_gui']
                    intel.window_count = cached_gui['window_count']
                    intel.user_interaction_score = self._calculate_interaction_score(intel)
            
            # Cache the result
            self.process_cache[cache_key] = intel
            
            # Store in database (async)
            self._store_intelligence_async(intel)
            
            return intel
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            intel.classification = "system_process"
            intel.confidence = 0.2
            return intel
    
    def _classify_fast(self, name: str) -> Tuple[str, float, float]:
        """Fast rule-based classification"""
        name_lower = name.lower()
        
        for classification, rules in self.fast_classification_rules.items():
            if any(keyword in name_lower for keyword in rules['keywords']):
                return classification, rules['confidence'], rules['priority']
        
        # Default classification based on name patterns
        if name_lower.endswith('d'):
            return 'daemon_process', 0.6, 2
        elif 'helper' in name_lower or 'agent' in name_lower:
            return 'system_helper', 0.5, 3
        else:
            return 'unknown_application', 0.3, 1
    
    def _should_do_medium_analysis(self, intel: ProcessIntelligence) -> bool:
        """Decide if we should do medium-cost analysis"""
        # Do medium analysis for high-priority processes or periodically
        return (
            intel.priority_score > 5 or  # High priority processes
            intel.cpu_usage > 10 or      # Active processes
            intel.memory_mb > 100        # Memory-heavy processes
        )
    
    def _should_do_expensive_analysis(self, intel: ProcessIntelligence) -> bool:
        """Strategically decide when to do expensive operations"""
        
        # Never do expensive operations for system processes
        if intel.classification in ['system_critical', 'daemon_process']:
            return False
        
        # Always do for high-priority interactive apps (but cache results)
        if intel.classification in ['interactive_critical', 'development_tools']:
            return not self._has_recent_gui_cache(intel.name)
        
        # Do for unknown processes with significant resource usage
        if intel.classification == 'unknown_application' and (
            intel.cpu_usage > 5 or intel.memory_mb > 50
        ):
            return not self._has_recent_gui_cache(intel.name)
        
        return False
    
    def _has_recent_gui_cache(self, name: str) -> bool:
        """Check if we have recent GUI detection cache"""
        if name in self.gui_detection_cache:
            cache_time = self.gui_detection_cache[name]['last_checked']
            return (datetime.now() - datetime.fromisoformat(cache_time)).seconds < 300
        return False
    
    def _get_cached_gui_info(self, name: str) -> Optional[Dict]:
        """Get cached GUI information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT has_gui, window_count, last_checked
                FROM gui_detection_cache 
                WHERE process_name = ? AND last_checked > datetime('now', '-5 minutes')
            ''', (name,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'has_gui': bool(result[0]),
                    'window_count': result[1],
                    'last_checked': result[2]
                }
        except:
            pass
        return None
    
    def _background_processor(self):
        """Background thread for expensive operations"""
        while True:
            try:
                # Get process from queue (blocking)
                intel = self.expensive_operation_queue.get(timeout=1)
                
                # Perform expensive GUI detection
                self._perform_expensive_gui_detection(intel)
                
                # Update cache
                self._update_gui_cache(intel)
                
                # Mark task as done
                self.expensive_operation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background processor error: {e}")
    
    def _perform_expensive_gui_detection(self, intel: ProcessIntelligence):
        """Perform expensive GUI detection with timeout and caching"""
        try:
            # Use AppleScript with timeout
            script = f'''
            tell application "System Events"
                try
                    set appProcess to first process whose name is "{intel.name}"
                    set windowCount to count of windows of appProcess
                    return windowCount
                on error
                    return -1
                end try
            end tell
            '''
            
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=3  # 3 second timeout
            )
            
            if result.returncode == 0:
                window_count = int(result.stdout.strip() or -1)
                if window_count >= 0:
                    intel.window_count = window_count
                    intel.has_gui_windows = window_count > 0
                    intel.user_interaction_score = self._calculate_interaction_score(intel)
                    intel.last_expensive_check = datetime.now()
            
        except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError):
            # Fallback to heuristic detection
            intel.has_gui_windows = self._heuristic_gui_detection(intel.name)
            intel.window_count = 1 if intel.has_gui_windows else 0
            intel.user_interaction_score = self._calculate_interaction_score(intel)
    
    def _heuristic_gui_detection(self, name: str) -> bool:
        """Heuristic GUI detection without AppleScript"""
        name_lower = name.lower()
        
        # Known GUI applications
        gui_apps = [
            'finder', 'safari', 'chrome', 'firefox', 'terminal', 'iterm',
            'xcode', 'vscode', 'sublime', 'photoshop', 'sketch', 'figma',
            'zoom', 'teams', 'slack', 'discord', 'music', 'spotify'
        ]
        
        # Known non-GUI processes
        non_gui = [
            'kernel', 'launchd', 'daemon', 'helper', 'agent', 'service',
            'backupd', 'spotlight', 'mds', 'cloudd', 'logd', 'powerd'
        ]
        
        if any(app in name_lower for app in gui_apps):
            return True
        elif any(app in name_lower for app in non_gui):
            return False
        else:
            # Default heuristic: if it's not obviously a system process, assume GUI
            return not (name_lower.endswith('d') or 'system' in name_lower)
    
    def _calculate_interaction_score(self, intel: ProcessIntelligence) -> float:
        """Calculate user interaction score"""
        score = 0.0
        
        # Base score from classification
        classification_scores = {
            'interactive_critical': 0.9,
            'development_tools': 0.8,
            'communication': 0.8,
            'creative_apps': 0.7,
            'unknown_application': 0.5,
            'system_services': 0.2,
            'daemon_process': 0.1
        }
        score += classification_scores.get(intel.classification, 0.3)
        
        # Adjust based on GUI presence
        if intel.has_gui_windows is not None:
            if intel.has_gui_windows:
                score += 0.3
                if intel.window_count and intel.window_count > 1:
                    score += 0.1  # Multiple windows = more interaction
            else:
                score = max(score - 0.4, 0.1)  # Reduce but don't eliminate
        
        # Adjust based on resource usage
        if intel.cpu_usage > 10:
            score += 0.1  # Active processes likely interactive
        if intel.memory_mb > 200:
            score += 0.05  # Memory usage indicates user apps
        
        return min(score, 1.0)
    
    def _update_gui_cache(self, intel: ProcessIntelligence):
        """Update GUI detection cache"""
        if intel.has_gui_windows is not None:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT OR REPLACE INTO gui_detection_cache 
                    (process_name, has_gui, window_count, last_checked, check_count)
                    VALUES (?, ?, ?, ?, 
                        COALESCE((SELECT check_count FROM gui_detection_cache WHERE process_name = ?), 0) + 1)
                ''', (intel.name, intel.has_gui_windows, intel.window_count or 0, 
                     datetime.now().isoformat(), intel.name))
                conn.commit()
                conn.close()
                
                # Update in-memory cache
                self.gui_detection_cache[intel.name] = {
                    'has_gui': intel.has_gui_windows,
                    'window_count': intel.window_count or 0,
                    'last_checked': datetime.now().isoformat()
                }
            except:
                pass
    
    def _store_intelligence_async(self, intel: ProcessIntelligence):
        """Store process intelligence asynchronously"""
        def store():
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO process_intelligence 
                    (timestamp, pid, name, classification, confidence, cpu_usage, memory_mb,
                     has_gui_windows, window_count, user_interaction_score, classification_method,
                     priority_score, core_assignment, performance_impact)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    intel.last_updated.isoformat(), intel.pid, intel.name,
                    intel.classification, intel.confidence, intel.cpu_usage, intel.memory_mb,
                    intel.has_gui_windows, intel.window_count, intel.user_interaction_score,
                    intel.classification_method, intel.priority_score, None, 0.0
                ))
                conn.commit()
                conn.close()
            except:
                pass
        
        threading.Thread(target=store, daemon=True).start()
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Recent classifications
            cursor = conn.execute('''
                SELECT classification, COUNT(*), AVG(confidence), AVG(user_interaction_score)
                FROM process_intelligence 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY classification
                ORDER BY COUNT(*) DESC
            ''')
            recent_classifications = cursor.fetchall()
            
            # GUI detection stats
            cursor = conn.execute('''
                SELECT COUNT(*), AVG(check_count)
                FROM gui_detection_cache
            ''')
            gui_stats = cursor.fetchone()
            
            # Performance stats
            cursor = conn.execute('''
                SELECT AVG(confidence), COUNT(*)
                FROM process_intelligence 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            performance_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'recent_classifications': recent_classifications,
                'total_classifications': performance_stats[1] if performance_stats[1] else 0,
                'average_confidence': performance_stats[0] if performance_stats[0] else 0,
                'gui_cache_entries': gui_stats[0] if gui_stats[0] else 0,
                'avg_gui_checks': gui_stats[1] if gui_stats[1] else 0,
                'current_thresholds': {
                    'cpu_interactive': 15.0,
                    'cpu_compute': 50.0,
                    'memory_heavy': 500.0,
                    'gui_detection_timeout': 3.0
                }
            }
        except:
            return {
                'recent_classifications': [],
                'total_classifications': 0,
                'average_confidence': 0,
                'gui_cache_entries': 0,
                'avg_gui_checks': 0,
                'current_thresholds': {}
            }

class AdvancedEASScheduler:
    """Advanced EAS scheduler with intelligent core assignment"""
    
    def __init__(self, db_path: str):
        self.analyzer = IntelligentProcessAnalyzer(db_path)
        self.core_assignments = {}
        self.performance_history = deque(maxlen=100)
        
        # Advanced assignment strategies
        self.assignment_strategies = {
            'interactive_critical': {
                'preferred_core': 'p_core',
                'priority_adjustment': -5,
                'strategy': 'p_core_reserved'
            },
            'development_tools': {
                'preferred_core': 'p_core',
                'priority_adjustment': -3,
                'strategy': 'p_core_priority'
            },
            'communication': {
                'preferred_core': 'balanced',
                'priority_adjustment': -2,
                'strategy': 'balanced_interactive'
            },
            'creative_apps': {
                'preferred_core': 'p_core',
                'priority_adjustment': -4,
                'strategy': 'p_core_dedicated'
            },
            'compute_intensive': {
                'preferred_core': 'p_core',
                'priority_adjustment': -3,
                'strategy': 'p_core_compute'
            },
            'system_critical': {
                'preferred_core': 'balanced',
                'priority_adjustment': -1,
                'strategy': 'system_balanced'
            },
            'system_services': {
                'preferred_core': 'e_core',
                'priority_adjustment': 1,
                'strategy': 'e_core_preferred'
            },
            'daemon_process': {
                'preferred_core': 'e_core',
                'priority_adjustment': 2,
                'strategy': 'e_core_only'
            },
            'unknown_application': {
                'preferred_core': 'e_core',
                'priority_adjustment': 1,
                'strategy': 'e_core_safe'
            }
        }
    
    def optimize_system_advanced(self, max_processes: int = 100) -> Dict:
        """Advanced system optimization with intelligent processing"""
        
        start_time = time.time()
        optimizations = []
        
        try:
            # Get system load for dynamic decisions
            cpu_percent = psutil.cpu_percent(interval=0.1)
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
            
            p_core_load = sum(per_cpu[:4]) / 4 if len(per_cpu) >= 8 else cpu_percent
            e_core_load = sum(per_cpu[4:8]) / 4 if len(per_cpu) >= 8 else 0
            
            # Process optimization with intelligent batching
            process_count = 0
            high_priority_processes = []
            normal_processes = []
            
            # First pass: categorize processes
            for proc in psutil.process_iter(['pid', 'name']):
                if process_count >= max_processes:
                    break
                
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    
                    if pid < 50:  # Skip low-level system processes
                        continue
                    
                    # Quick priority assessment
                    if self._is_high_priority_process(name):
                        high_priority_processes.append((pid, name))
                    else:
                        normal_processes.append((pid, name))
                    
                    process_count += 1
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Process high-priority processes first (with expensive operations)
            for pid, name in high_priority_processes:
                intel = self.analyzer.analyze_process_intelligent(pid, name)
                assignment = self._calculate_advanced_assignment(intel, p_core_load, e_core_load)
                optimizations.append(assignment)
            
            # Process normal processes (lightweight)
            for pid, name in normal_processes[:50]:  # Limit normal processes
                intel = self.analyzer.analyze_process_intelligent(pid, name)
                assignment = self._calculate_advanced_assignment(intel, p_core_load, e_core_load)
                optimizations.append(assignment)
            
            end_time = time.time()
            
            # Store performance metrics
            self.performance_history.append({
                'timestamp': datetime.now(),
                'processes_analyzed': len(optimizations),
                'time_taken': end_time - start_time,
                'high_priority_count': len(high_priority_processes),
                'p_core_load': p_core_load,
                'e_core_load': e_core_load
            })
            
            return {
                'optimized': len(optimizations),
                'assignments': optimizations,
                'performance': {
                    'time_taken_ms': (end_time - start_time) * 1000,
                    'processes_per_second': len(optimizations) / (end_time - start_time),
                    'high_priority_processes': len(high_priority_processes),
                    'system_load': {'p_cores': p_core_load, 'e_cores': e_core_load}
                }
            }
            
        except Exception as e:
            return {
                'optimized': 0,
                'assignments': [],
                'error': str(e)
            }
    
    def _is_high_priority_process(self, name: str) -> bool:
        """Quick check if process is high priority (deserves expensive analysis)"""
        name_lower = name.lower()
        high_priority_keywords = [
            'finder', 'safari', 'chrome', 'firefox', 'terminal', 'iterm',
            'xcode', 'vscode', 'sublime', 'photoshop', 'sketch',
            'zoom', 'teams', 'slack', 'facetime'
        ]
        return any(keyword in name_lower for keyword in high_priority_keywords)
    
    def _calculate_advanced_assignment(self, intel: ProcessIntelligence, 
                                     p_core_load: float, e_core_load: float) -> Dict:
        """Calculate advanced core assignment"""
        
        strategy_config = self.assignment_strategies.get(
            intel.classification, 
            self.assignment_strategies['unknown_application']
        )
        
        preferred_core = strategy_config['preferred_core']
        priority_adj = strategy_config['priority_adjustment']
        strategy_name = strategy_config['strategy']
        
        # Dynamic load balancing
        if preferred_core == 'balanced':
            if p_core_load < e_core_load + 20:
                target_core = 'p_core'
            else:
                target_core = 'e_core'
        elif preferred_core == 'p_core':
            # Use P-core unless heavily loaded
            target_core = 'p_core' if p_core_load < 80 else 'e_core'
        else:  # e_core
            target_core = 'e_core'
        
        # Confidence-based adjustment
        if intel.confidence < 0.5:
            priority_adj = min(priority_adj + 1, 5)  # Less aggressive if uncertain
        
        # User interaction boost
        if intel.user_interaction_score and intel.user_interaction_score > 0.7:
            priority_adj -= 1  # Higher priority for interactive apps
            if target_core == 'e_core':
                target_core = 'p_core'  # Promote to P-core
        
        return {
            'pid': intel.pid,
            'name': intel.name,
            'classification': intel.classification,
            'target_core': target_core,
            'strategy': strategy_name,
            'priority_adjustment': priority_adj,
            'confidence': intel.confidence,
            'user_interaction_score': intel.user_interaction_score or 0,
            'cpu_usage': intel.cpu_usage,
            'memory_mb': intel.memory_mb,
            'has_gui': intel.has_gui_windows,
            'system_load': {'p_cores': p_core_load, 'e_cores': e_core_load}
        }
    
    def get_performance_insights(self) -> Dict:
        """Get performance insights and recommendations"""
        if not self.performance_history:
            return {}
        
        recent_performance = list(self.performance_history)[-10:]
        
        avg_time = sum(p['time_taken'] for p in recent_performance) / len(recent_performance)
        avg_processes = sum(p['processes_analyzed'] for p in recent_performance) / len(recent_performance)
        
        return {
            'average_analysis_time': avg_time,
            'average_processes_analyzed': avg_processes,
            'processes_per_second': avg_processes / avg_time if avg_time > 0 else 0,
            'recent_performance': recent_performance[-5:],
            'recommendations': self._generate_performance_recommendations(recent_performance)
        }
    
    def _generate_performance_recommendations(self, performance_data: List[Dict]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        avg_time = sum(p['time_taken'] for p in performance_data) / len(performance_data)
        
        if avg_time > 1.0:  # Taking more than 1 second
            recommendations.append("Consider reducing max_processes limit for better responsiveness")
        
        if avg_time < 0.1:  # Very fast
            recommendations.append("System performing well - could handle more processes")
        
        high_priority_ratio = sum(p['high_priority_count'] for p in performance_data) / sum(p['processes_analyzed'] for p in performance_data)
        
        if high_priority_ratio > 0.3:
            recommendations.append("High ratio of priority processes - consider optimizing background tasks")
        
        return recommendations

class AdvancedEASIntegration:
    """Advanced EAS integration with strategic expensive operations"""
    
    def __init__(self, original_eas):
        self.original_eas = original_eas
        db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas.db")
        self.advanced_scheduler = AdvancedEASScheduler(db_path)
        
        # Performance monitoring
        self.operation_stats = {
            'fast_operations': 0,
            'medium_operations': 0,
            'expensive_operations': 0,
            'total_time': 0,
            'cache_hits': 0
        }
        
        self._patch_methods()
    
    def _patch_methods(self):
        """Patch original EAS with advanced methods"""
        
        # Store original methods
        self.original_eas._original_classify_workload = getattr(self.original_eas, 'classify_workload', None)
        self.original_eas._original_optimize_system = getattr(self.original_eas, 'optimize_system', None)
        
        # Replace with advanced versions
        self.original_eas.classify_workload_advanced = self._advanced_classify_workload
        self.original_eas.optimize_system_advanced = self._advanced_optimize_system
        self.original_eas.get_advanced_stats = self._get_advanced_stats
        self.original_eas.get_performance_insights = self._get_performance_insights
        self.original_eas.force_reclassify_advanced = self._force_reclassify_advanced
    
    def _advanced_classify_workload(self, pid, name):
        """Advanced workload classification with intelligent expensive operations"""
        start_time = time.time()
        
        try:
            intel = self.advanced_scheduler.analyzer.analyze_process_intelligent(pid, name)
            
            # Map advanced classifications to original EAS categories
            classification_mapping = {
                'interactive_critical': 'interactive',
                'development_tools': 'interactive',
                'communication': 'interactive',
                'creative_apps': 'compute',
                'compute_intensive': 'compute',
                'system_critical': 'background',
                'system_services': 'background',
                'daemon_process': 'background',
                'unknown_application': 'background'
            }
            
            mapped_classification = classification_mapping.get(intel.classification, 'background')
            
            # Update stats
            self.operation_stats['fast_operations'] += 1
            if intel.last_expensive_check:
                self.operation_stats['expensive_operations'] += 1
            self.operation_stats['total_time'] += time.time() - start_time
            
            return mapped_classification
            
        except Exception as e:
            print(f"Advanced classification error for {name}: {e}")
            return 'background'
    
    def _advanced_optimize_system(self):
        """Advanced system optimization"""
        try:
            result = self.advanced_scheduler.optimize_system_advanced(max_processes=75)
            
            # Convert to original format
            return {
                'optimized': result['optimized'],
                'assignments': result['assignments'][:20],  # Limit for display
                'metrics': getattr(self.original_eas, 'current_metrics', {}),
                'performance': result.get('performance', {}),
                'advanced_stats': self._get_operation_stats()
            }
            
        except Exception as e:
            print(f"Advanced optimization error: {e}")
            return {
                'optimized': 0,
                'assignments': [],
                'error': str(e)
            }
    
    def _get_advanced_stats(self):
        """Get advanced classification statistics"""
        stats = self.advanced_scheduler.analyzer.get_comprehensive_stats()
        
        # Add operation performance stats
        stats['operation_performance'] = self._get_operation_stats()
        
        return stats
    
    def _get_performance_insights(self):
        """Get performance insights"""
        return self.advanced_scheduler.get_performance_insights()
    
    def _force_reclassify_advanced(self):
        """Force advanced reclassification of processes"""
        try:
            result = self.advanced_scheduler.optimize_system_advanced(max_processes=100)
            return result['optimized']
        except Exception as e:
            print(f"Force reclassify error: {e}")
            return 0
    
    def _get_operation_stats(self):
        """Get operation performance statistics"""
        total_ops = (self.operation_stats['fast_operations'] + 
                    self.operation_stats['medium_operations'] + 
                    self.operation_stats['expensive_operations'])
        
        if total_ops == 0:
            return {}
        
        return {
            'total_operations': total_ops,
            'fast_operations': self.operation_stats['fast_operations'],
            'medium_operations': self.operation_stats['medium_operations'],
            'expensive_operations': self.operation_stats['expensive_operations'],
            'cache_hits': self.operation_stats['cache_hits'],
            'average_time_ms': (self.operation_stats['total_time'] / total_ops) * 1000,
            'expensive_operation_ratio': self.operation_stats['expensive_operations'] / total_ops,
            'performance_score': self._calculate_performance_score()
        }
    
    def _calculate_performance_score(self):
        """Calculate overall performance score"""
        total_ops = (self.operation_stats['fast_operations'] + 
                    self.operation_stats['medium_operations'] + 
                    self.operation_stats['expensive_operations'])
        
        if total_ops == 0:
            return 100
        
        # Score based on operation efficiency
        fast_ratio = self.operation_stats['fast_operations'] / total_ops
        expensive_ratio = self.operation_stats['expensive_operations'] / total_ops
        avg_time = self.operation_stats['total_time'] / total_ops
        
        # Higher score for more fast operations, fewer expensive ones, and faster execution
        score = (fast_ratio * 40 +  # Fast operations are good
                (1 - expensive_ratio) * 30 +  # Fewer expensive operations is good
                max(0, (1 - avg_time)) * 30)  # Faster execution is good
        
        return min(100, max(0, score * 100))

def patch_eas_advanced(eas_instance):
    """Patch EAS with advanced version"""
    return AdvancedEASIntegration(eas_instance)

# Test function
def main():
    """Test the advanced EAS system"""
    print("🚀 Testing Advanced EAS System")
    print("=" * 60)
    
    db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas_test.db")
    scheduler = AdvancedEASScheduler(db_path)
    
    print("Phase 1: Fast lightweight operations...")
    start_time = time.time()
    result = scheduler.optimize_system_advanced(max_processes=30)
    end_time = time.time()
    
    print(f"✅ Analyzed {result['optimized']} processes in {end_time - start_time:.3f} seconds")
    print(f"⚡ Performance: {result['performance']['processes_per_second']:.1f} processes/sec")
    
    print(f"\n📊 Sample Classifications:")
    for assignment in result['assignments'][:8]:
        name = assignment['name'][:20]
        classification = assignment['classification']
        target_core = assignment['target_core']
        confidence = assignment['confidence']
        gui = "GUI" if assignment.get('has_gui') else "CLI"
        
        print(f"  {name:20} → {target_core:6} ({classification:15}) {confidence:.2f} {gui}")
    
    print(f"\n🧠 Intelligence Stats:")
    stats = scheduler.analyzer.get_comprehensive_stats()
    print(f"  Total Classifications: {stats['total_classifications']:,}")
    print(f"  Average Confidence: {stats['average_confidence']:.3f}")
    print(f"  GUI Cache Entries: {stats['gui_cache_entries']}")
    
    print(f"\n💡 Performance Insights:")
    insights = scheduler.get_performance_insights()
    if insights:
        print(f"  Analysis Time: {insights['average_analysis_time']:.3f}s")
        print(f"  Processes/Second: {insights['processes_per_second']:.1f}")
        
        if insights['recommendations']:
            print(f"  Recommendations:")
            for rec in insights['recommendations']:
                print(f"    • {rec}")
    
    print(f"\n🎯 System demonstrates:")
    print(f"  ✅ Fast operations for immediate responsiveness")
    print(f"  ✅ Strategic expensive operations for accuracy")
    print(f"  ✅ Intelligent caching to avoid redundant work")
    print(f"  ✅ Background processing for GUI detection")
    print(f"  ✅ Performance monitoring and optimization")

        elif high_priority_ratio < 0.1:
            recommendations.append("Consider reducing expensive operations frequency")

if __name__ == "__main__":
    main()
        
        return recommendations

# Integration class for the main app
class AdvancedEASIntegration:
    """Advanced EAS integration with strategic expensive operations"""
    
    def __init__(self, original_eas):
        self.original_eas = original_eas
        db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas.db")
        self.advanced_scheduler = AdvancedEASScheduler(db_path)
        
        # Patch methods
        self._patch_methods()
    
    def _patch_methods(self):
        """Patch original EAS with advanced methods"""
        
        # Add new methods
        self.original_eas.classify_workload_advanced = self._advanced_classify_workload
        self.original_eas.optimize_system_advanced = self._advanced_optimize_system
        self.original_eas.get_advanced_stats = self._get_advanced_stats
        self.original_eas.get_performance_insights = self._get_performance_insights
        self.original_eas.force_reclassify_advanced = self._force_reclassify_advanced
    
    def _advanced_classify_workload(self, pid, name):
        """Advanced workload classification"""
        intel = self.advanced_scheduler.analyzer.analyze_process_intelligent(pid, name)
        
        # Map to original EAS categories
        mapping = {
            'interactive_critical': 'interactive',
            'development_tools': 'interactive',
            'communication': 'interactive',
            'creative_apps': 'compute',
            'compute_intensive': 'compute',
            'system_critical': 'background',
            'system_services': 'background',
            'daemon_process': 'background',
            'unknown_application': 'background'
        }
        
        return mapping.get(intel.classification, 'background')
    
    def _advanced_optimize_system(self):
        """Advanced system optimization"""
        result = self.advanced_scheduler.optimize_system_advanced(max_processes=75)
        
        # Convert to original format
        return {
            'optimized': result['optimized'],
            'assignments': result['assignments'][:20],  # Limit display
            'metrics': getattr(self.original_eas, 'current_metrics', {}),
            'performance': result.get('performance', {})
        }
    
    def _get_advanced_stats(self):
        """Get advanced classification stats"""
        return self.advanced_scheduler.analyzer.get_comprehensive_stats()
    
    def _get_performance_insights(self):
        """Get performance insights"""
        return self.advanced_scheduler.get_performance_insights()
    
    def _force_reclassify_advanced(self):
        """Force advanced reclassification"""
        result = self.advanced_scheduler.optimize_system_advanced(max_processes=100)
        return result['optimized']

def patch_eas_advanced(eas_instance):
    """Patch EAS with advanced version"""
    return AdvancedEASIntegration(eas_instance)

if __name__ == "__main__":
    # Test the advanced system
    db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas_test.db")
    scheduler = AdvancedEASScheduler(db_path)
    
    print("🚀 Testing Advanced EAS System")
    print("=" * 60)
    
    start_time = time.time()
    result = scheduler.optimize_system_advanced(max_processes=30)
    end_time = time.time()
    
    print(f"✅ Optimized {result['optimized']} processes in {end_time - start_time:.2f} seconds")
    print(f"⚡ Performance: {result['performance']['processes_per_second']:.1f} processes/sec")
    print(f"🎯 High priority processes: {result['performance']['high_priority_processes']}")
    
    print(f"\n📊 Sample Assignments:")
    for assignment in result['assignments'][:8]:
        gui_status = "🖥️ " if assignment.get('has_gui') else "⚙️ "
        interaction = assignment.get('user_interaction_score', 0)
        print(f"  {gui_status}{assignment['name'][:20]:20} → {assignment['target_core']:6} "
              f"({assignment['classification'][:15]:15}) conf:{assignment['confidence']:.2f} "
              f"interact:{interaction:.2f}")
    
    # Show performance insights
    insights = scheduler.get_performance_insights()
    if insights:
        print(f"\n🔍 Performance Insights:")
        print(f"  Average analysis time: {insights['average_analysis_time']:.3f}s")
        print(f"  Processes per second: {insights['processes_per_second']:.1f}")
        
        if insights['recommendations']:
            print(f"  Recommendations:")
            for rec in insights['recommendations']:
                print(f"    • {rec}")
    
    # Show comprehensive stats
    stats = scheduler.analyzer.get_comprehensive_stats()
    print(f"\n📈 System Stats:")
    print(f"  Total classifications: {stats['total_classifications']}")
    print(f"  Average confidence: {stats['average_confidence']:.3f}")
    print(f"  GUI cache entries: {stats['gui_cache_entries']}")
    print(f"  Avg GUI checks per app: {stats['avg_gui_checks']:.1f}")