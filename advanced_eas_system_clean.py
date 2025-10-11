#!/usr/bin/env python3
"""
Advanced Enhanced EAS System - Clean Implementation
Strategic expensive operations with intelligent caching and background processing
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ProcessIntelligence:
    """Process intelligence data"""
    pid: int
    name: str
    cpu_usage: float = 0.0
    memory_mb: float = 0.0
    classification: str = "unknown"
    confidence: float = 0.0
    has_gui_windows: Optional[bool] = None
    user_interaction_score: Optional[float] = None
    last_updated: Optional[datetime] = None

class AdvancedProcessAnalyzer:
    """Advanced process analyzer with strategic expensive operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.process_cache = {}
        self.gui_cache = {}
        self.expensive_queue = queue.Queue()
        
        # Background thread for expensive operations
        self.background_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.background_thread.start()
        
        self.init_database()
        
        # Fast classification rules
        self.classification_rules = {
            'interactive_critical': {
                'keywords': ['finder', 'safari', 'chrome', 'firefox', 'terminal', 'iterm', 'xcode', 'vscode'],
                'confidence': 0.9
            },
            'communication': {
                'keywords': ['zoom', 'teams', 'slack', 'discord', 'facetime', 'messages'],
                'confidence': 0.8
            },
            'system_critical': {
                'keywords': ['kernel_task', 'launchd', 'windowserver', 'loginwindow'],
                'confidence': 0.95
            },
            'system_services': {
                'keywords': ['backupd', 'spotlight', 'mds', 'cloudd', 'coreauthd'],
                'confidence': 0.8
            },
            'compute_intensive': {
                'keywords': ['python', 'node', 'java', 'ffmpeg', 'handbrake', 'blender'],
                'confidence': 0.7
            }
        }
    
    def init_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS advanced_classifications (
                timestamp TEXT,
                pid INTEGER,
                name TEXT,
                classification TEXT,
                confidence REAL,
                cpu_usage REAL,
                memory_mb REAL,
                has_gui BOOLEAN,
                user_interaction_score REAL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS gui_cache (
                process_name TEXT PRIMARY KEY,
                has_gui BOOLEAN,
                last_checked TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_process_smart(self, pid: int, name: str) -> ProcessIntelligence:
        """Smart process analysis with strategic expensive operations"""
        
        # Check cache first
        cache_key = f"{name}_{pid}"
        if cache_key in self.process_cache:
            cached = self.process_cache[cache_key]
            if time.time() - cached.last_updated.timestamp() < 30:
                return cached
        
        intel = ProcessIntelligence(
            pid=pid,
            name=name,
            cpu_usage=0.0,
            memory_mb=0.0,
            last_updated=datetime.now()
        )
        
        try:
            proc = psutil.Process(pid)
            
            # FAST OPERATIONS (always do)
            intel.cpu_usage = proc.cpu_percent(interval=0)
            intel.memory_mb = proc.memory_info().rss / (1024 * 1024)
            
            # Fast classification
            intel.classification, intel.confidence = self._classify_fast(name)
            
            # EXPENSIVE OPERATIONS (strategic)
            if self._should_do_expensive_analysis(intel):
                # Queue for background processing
                self.expensive_queue.put(intel)
            else:
                # Use cached GUI info
                cached_gui = self._get_cached_gui_info(name)
                if cached_gui is not None:
                    intel.has_gui_windows = cached_gui
                    intel.user_interaction_score = self._calculate_interaction_score(intel)
            
            # Cache result
            self.process_cache[cache_key] = intel
            
            # Store in database (async)
            self._store_async(intel)
            
            return intel
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            intel.classification = "system_process"
            intel.confidence = 0.2
            
            # Cache and store exception cases too
            self.process_cache[cache_key] = intel
            self._store_async(intel)
            
            return intel
    
    def _classify_fast(self, name: str) -> Tuple[str, float]:
        """Fast rule-based classification"""
        name_lower = name.lower()
        
        for classification, rules in self.classification_rules.items():
            if any(keyword in name_lower for keyword in rules['keywords']):
                return classification, rules['confidence']
        
        # Default classification
        if name_lower.endswith('d'):
            return 'daemon_process', 0.6
        elif 'helper' in name_lower or 'agent' in name_lower:
            return 'system_helper', 0.5
        else:
            return 'unknown_application', 0.3
    
    def _should_do_expensive_analysis(self, intel: ProcessIntelligence) -> bool:
        """Decide when to do expensive GUI detection"""
        
        # Never for system processes
        if intel.classification in ['system_critical', 'daemon_process']:
            return False
        
        # Always for interactive apps (but cache results)
        if intel.classification == 'interactive_critical':
            return not self._has_recent_gui_cache(intel.name)
        
        # For unknown processes with significant usage
        if intel.classification == 'unknown_application' and (
            intel.cpu_usage > 5 or intel.memory_mb > 50
        ):
            return not self._has_recent_gui_cache(intel.name)
        
        return False
    
    def _has_recent_gui_cache(self, name: str) -> bool:
        """Check if we have recent GUI cache"""
        if name in self.gui_cache:
            cache_time = self.gui_cache[name]['last_checked']
            return (datetime.now() - datetime.fromisoformat(cache_time)).seconds < 300
        return False
    
    def _get_cached_gui_info(self, name: str) -> Optional[bool]:
        """Get cached GUI information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute('''
                SELECT has_gui FROM gui_cache 
                WHERE process_name = ? AND last_checked > datetime('now', '-5 minutes')
            ''', (name,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return bool(result[0])
        except:
            pass
        return None
    
    def _background_processor(self):
        """Background thread for expensive GUI detection"""
        while True:
            try:
                intel = self.expensive_queue.get(timeout=1)
                
                # Perform expensive GUI detection with timeout
                self._detect_gui_with_timeout(intel)
                
                # Update cache
                self._update_gui_cache(intel)
                
                self.expensive_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background processor error: {e}")
    
    def _detect_gui_with_timeout(self, intel: ProcessIntelligence):
        """Detect GUI with AppleScript timeout"""
        try:
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
                timeout=2  # 2 second timeout
            )
            
            if result.returncode == 0:
                window_count = int(result.stdout.strip() or -1)
                if window_count >= 0:
                    intel.has_gui_windows = window_count > 0
                    intel.user_interaction_score = self._calculate_interaction_score(intel)
            
        except (subprocess.TimeoutExpired, ValueError):
            # Fallback to heuristic
            intel.has_gui_windows = self._heuristic_gui_detection(intel.name)
            intel.user_interaction_score = self._calculate_interaction_score(intel)
    
    def _heuristic_gui_detection(self, name: str) -> bool:
        """Heuristic GUI detection"""
        name_lower = name.lower()
        
        gui_apps = ['finder', 'safari', 'chrome', 'firefox', 'terminal', 'xcode', 'vscode']
        non_gui = ['kernel', 'launchd', 'daemon', 'helper', 'backupd', 'spotlight']
        
        if any(app in name_lower for app in gui_apps):
            return True
        elif any(app in name_lower for app in non_gui):
            return False
        else:
            return not (name_lower.endswith('d') or 'system' in name_lower)
    
    def _calculate_interaction_score(self, intel: ProcessIntelligence) -> float:
        """Calculate user interaction score"""
        score = 0.0
        
        # Base score from classification
        classification_scores = {
            'interactive_critical': 0.9,
            'communication': 0.8,
            'unknown_application': 0.5,
            'system_services': 0.2,
            'daemon_process': 0.1
        }
        score += classification_scores.get(intel.classification, 0.3)
        
        # GUI bonus
        if intel.has_gui_windows:
            score += 0.3
        elif intel.has_gui_windows is False:
            score = max(score - 0.4, 0.1)
        
        # Resource usage bonus
        if intel.cpu_usage > 10:
            score += 0.1
        if intel.memory_mb > 200:
            score += 0.05
        
        return min(score, 1.0)
    
    def _update_gui_cache(self, intel: ProcessIntelligence):
        """Update GUI cache"""
        if intel.has_gui_windows is not None:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT OR REPLACE INTO gui_cache 
                    (process_name, has_gui, last_checked)
                    VALUES (?, ?, ?)
                ''', (intel.name, intel.has_gui_windows, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                
                self.gui_cache[intel.name] = {
                    'has_gui': intel.has_gui_windows,
                    'last_checked': datetime.now().isoformat()
                }
            except:
                pass
    
    def _store_async(self, intel: ProcessIntelligence):
        """Store intelligence data asynchronously"""
        def store():
            try:
                conn = sqlite3.connect(self.db_path)
                timestamp = intel.last_updated.isoformat() if intel.last_updated else datetime.now().isoformat()
                conn.execute('''
                    INSERT INTO advanced_classifications 
                    (timestamp, pid, name, classification, confidence, cpu_usage, memory_mb,
                     has_gui, user_interaction_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, intel.pid, intel.name,
                    intel.classification, intel.confidence, intel.cpu_usage, intel.memory_mb,
                    intel.has_gui_windows, intel.user_interaction_score
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Database storage error: {e}")
        
        threading.Thread(target=store, daemon=True).start()
    
    def get_stats(self) -> Dict:
        """Get classification statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute('''
                SELECT classification, COUNT(*), AVG(confidence)
                FROM advanced_classifications 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY classification
                ORDER BY COUNT(*) DESC
            ''')
            recent_classifications = cursor.fetchall()
            
            cursor = conn.execute('''
                SELECT AVG(confidence), COUNT(*)
                FROM advanced_classifications 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            performance_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'recent_classifications': recent_classifications,
                'total_classifications': performance_stats[1] if performance_stats[1] else 0,
                'average_confidence': performance_stats[0] if performance_stats[0] else 0,
                'current_thresholds': {
                    'cpu_interactive': 15.0,
                    'cpu_compute': 50.0,
                    'memory_heavy': 500.0
                }
            }
        except:
            return {
                'recent_classifications': [],
                'total_classifications': 0,
                'average_confidence': 0,
                'current_thresholds': {}
            }

class AdvancedEASScheduler:
    """Advanced EAS scheduler"""
    
    def __init__(self, db_path: str):
        self.analyzer = AdvancedProcessAnalyzer(db_path)
        
        # Core assignment strategies
        self.strategies = {
            'interactive_critical': {'core': 'p_core', 'priority': -5},
            'communication': {'core': 'balanced', 'priority': -2},
            'compute_intensive': {'core': 'p_core', 'priority': -3},
            'system_critical': {'core': 'balanced', 'priority': -1},
            'system_services': {'core': 'e_core', 'priority': 1},
            'daemon_process': {'core': 'e_core', 'priority': 2},
            'unknown_application': {'core': 'e_core', 'priority': 1}
        }
    
    def optimize_system_advanced(self, max_processes: int = 75) -> Dict:
        """Advanced system optimization"""
        
        start_time = time.time()
        optimizations = []
        
        try:
            # Get system load
            cpu_percent = psutil.cpu_percent(interval=0.1)
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.1)
            
            p_core_load = sum(per_cpu[:4]) / 4 if len(per_cpu) >= 8 else cpu_percent
            e_core_load = sum(per_cpu[4:8]) / 4 if len(per_cpu) >= 8 else 0
            
            # Process optimization
            process_count = 0
            for proc in psutil.process_iter(['pid', 'name']):
                if process_count >= max_processes:
                    break
                
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    
                    # Skip very low system PIDs and empty names
                    if pid < 10 or not name or name.strip() == '':
                        continue
                    
                    # Debug first few processes
                    if process_count < 3:
                        print(f"Processing: {pid} - {name}")
                    
                    try:
                        intel = self.analyzer.analyze_process_smart(pid, name)
                        assignment = self._calculate_assignment(intel, p_core_load, e_core_load)
                        optimizations.append(assignment)
                        process_count += 1
                    except Exception as analysis_error:
                        if process_count < 3:
                            print(f"  -> Analysis error: {analysis_error}")
                        continue
                    
                    if process_count < 3:
                        print(f"  -> Classification: {intel.classification}, Core: {assignment['target_core']}")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    if process_count < 3:
                        print(f"  -> Error: {e}")
                    continue
            
            end_time = time.time()
            
            return {
                'optimized': len(optimizations),
                'assignments': optimizations,
                'performance': {
                    'time_taken_ms': (end_time - start_time) * 1000,
                    'processes_per_second': len(optimizations) / (end_time - start_time),
                    'system_load': {'p_cores': p_core_load, 'e_cores': e_core_load}
                }
            }
            
        except Exception as e:
            return {
                'optimized': 0,
                'assignments': [],
                'error': str(e)
            }
    
    def _calculate_assignment(self, intel: ProcessIntelligence, p_core_load: float, e_core_load: float) -> Dict:
        """Calculate core assignment"""
        
        strategy = self.strategies.get(intel.classification, self.strategies['unknown_application'])
        
        preferred_core = strategy['core']
        priority_adj = strategy['priority']
        
        # Dynamic load balancing
        if preferred_core == 'balanced':
            target_core = 'p_core' if p_core_load < e_core_load + 20 else 'e_core'
        elif preferred_core == 'p_core':
            target_core = 'p_core' if p_core_load < 80 else 'e_core'
        else:
            target_core = 'e_core'
        
        # User interaction boost
        if intel.user_interaction_score and intel.user_interaction_score > 0.7:
            priority_adj -= 1
            if target_core == 'e_core':
                target_core = 'p_core'
        
        return {
            'pid': intel.pid,
            'name': intel.name,
            'classification': intel.classification,
            'target_core': target_core,
            'priority_adjustment': priority_adj,
            'confidence': intel.confidence,
            'user_interaction_score': intel.user_interaction_score or 0,
            'cpu_usage': intel.cpu_usage,
            'memory_mb': intel.memory_mb,
            'has_gui': intel.has_gui_windows
        }

class AdvancedEASIntegration:
    """Integration with existing EAS system"""
    
    def __init__(self, original_eas):
        self.original_eas = original_eas
        db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas.db")
        self.scheduler = AdvancedEASScheduler(db_path)
        
        self._patch_methods()
    
    def _patch_methods(self):
        """Patch original EAS methods"""
        self.original_eas.get_advanced_stats = self._get_advanced_stats
        self.original_eas.force_reclassify_advanced = self._force_reclassify_advanced
        self.original_eas.optimize_system_advanced = self._optimize_system_advanced
    
    def _get_advanced_stats(self):
        """Get advanced statistics"""
        return self.scheduler.analyzer.get_stats()
    
    def _force_reclassify_advanced(self):
        """Force reclassification"""
        result = self.scheduler.optimize_system_advanced(max_processes=100)
        return result['optimized']
    
    def _optimize_system_advanced(self):
        """Advanced optimization"""
        result = self.scheduler.optimize_system_advanced()
        return {
            'optimized': result['optimized'],
            'assignments': result['assignments'][:20],
            'metrics': getattr(self.original_eas, 'current_metrics', {}),
            'performance': result.get('performance', {})
        }

def patch_eas_advanced(eas_instance):
    """Patch EAS with advanced capabilities"""
    return AdvancedEASIntegration(eas_instance)

def main():
    """Test the advanced system"""
    print("🚀 Testing Advanced EAS System")
    print("=" * 60)
    
    db_path = os.path.expanduser("~/.battery_optimizer_advanced_eas_test.db")
    scheduler = AdvancedEASScheduler(db_path)
    
    print("Running advanced optimization...")
    
    # First, let's see what processes are available
    print("Available processes:")
    count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            pid = proc.info['pid']
            name = proc.info['name']
            if pid > 10 and name and count < 10:
                print(f"  {pid}: {name}")
                count += 1
        except:
            continue
    
    start_time = time.time()
    result = scheduler.optimize_system_advanced(max_processes=30)
    end_time = time.time()
    
    print(f"✅ Analyzed {result['optimized']} processes in {end_time - start_time:.3f} seconds")
    
    if 'performance' in result:
        print(f"⚡ Performance: {result['performance']['processes_per_second']:.1f} processes/sec")
    else:
        print(f"⚡ Performance: {result['optimized'] / (end_time - start_time):.1f} processes/sec")
    
    print(f"\n📊 Sample Classifications:")
    for assignment in result['assignments'][:8]:
        name = assignment['name'][:20]
        classification = assignment['classification']
        target_core = assignment['target_core']
        confidence = assignment['confidence']
        gui = "GUI" if assignment.get('has_gui') else "CLI"
        
        print(f"  {name:20} → {target_core:6} ({classification:15}) {confidence:.2f} {gui}")
    
    # Wait a moment for async database operations to complete
    print("\n⏳ Waiting for database operations to complete...")
    time.sleep(1)
    
    stats = scheduler.analyzer.get_stats()
    print(f"\n🧠 Intelligence Stats:")
    print(f"  Total Classifications: {stats['total_classifications']:,}")
    print(f"  Average Confidence: {stats['average_confidence']:.3f}")
    
    print(f"\n🎯 Advanced Features:")
    print(f"  ✅ Strategic expensive operations (GUI detection)")
    print(f"  ✅ Intelligent caching (5-minute GUI cache)")
    print(f"  ✅ Background processing (non-blocking)")
    print(f"  ✅ Dynamic load balancing")
    print(f"  ✅ User interaction scoring")

if __name__ == "__main__":
    main()