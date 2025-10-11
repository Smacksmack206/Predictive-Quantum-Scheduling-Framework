#!/usr/bin/env python3
"""
Lightweight EAS Classifier - Fast, non-blocking process classification
Optimized for performance to prevent app hanging
"""

import psutil
import time
import json
import sqlite3
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ProcessProfile:
    """Lightweight process profile"""
    name: str
    pid: int
    cpu_usage: float
    memory_mb: float
    classification: str
    confidence: float
    last_updated: datetime

class LightweightProcessClassifier:
    """Fast, lightweight process classifier"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.process_cache = {}
        self.classification_cache = {}
        self.last_cleanup = time.time()
        self.init_database()
        
        # Lightweight classification rules (no expensive operations)
        self.classification_rules = {
            # Interactive applications (user-facing)
            'interactive_heavy': [
                'safari', 'chrome', 'firefox', 'edge', 'brave',
                'finder', 'xcode', 'vscode', 'sublime', 'atom',
                'photoshop', 'illustrator', 'sketch', 'figma',
                'zoom', 'teams', 'slack', 'discord', 'facetime'
            ],
            
            # Background system processes
            'background': [
                'backupd', 'spotlight', 'mds', 'cloudd', 'bird',
                'syncdefaultsd', 'coreauthd', 'logind', 'launchd',
                'kernel_task', 'windowserver', 'dock', 'controlcenter'
            ],
            
            # Compute intensive
            'compute_intensive': [
                'python', 'node', 'java', 'ruby', 'go',
                'gcc', 'clang', 'swift', 'rustc',
                'ffmpeg', 'handbrake', 'compressor',
                'blender', 'cinema4d', 'maya'
            ],
            
            # System daemons
            'system_service': [
                'systemstats', 'powerd', 'configd', 'logd',
                'fseventsd', 'mediaremoted', 'remoted'
            ]
        }
    
    def init_database(self):
        """Initialize lightweight database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS lightweight_classifications (
                timestamp TEXT,
                process_name TEXT,
                classification TEXT,
                confidence REAL,
                cpu_usage REAL,
                memory_mb REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def classify_process_fast(self, pid: int, name: str) -> Tuple[str, float]:
        """Fast process classification without expensive operations"""
        
        # Check cache first
        cache_key = f"{name}_{pid}"
        if cache_key in self.classification_cache:
            cached = self.classification_cache[cache_key]
            if time.time() - cached['timestamp'] < 30:  # Cache for 30 seconds
                return cached['classification'], cached['confidence']
        
        try:
            # Get basic process info (fast)
            proc = psutil.Process(pid)
            cpu_percent = proc.cpu_percent(interval=0)  # Non-blocking
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            
            # Fast rule-based classification
            name_lower = name.lower()
            classification = 'unknown'
            confidence = 0.3
            
            # Check against classification rules
            for cls_type, keywords in self.classification_rules.items():
                if any(keyword in name_lower for keyword in keywords):
                    classification = cls_type
                    confidence = 0.8
                    break
            
            # Adjust based on resource usage (fast heuristics)
            if classification == 'unknown':
                if cpu_percent > 50:
                    classification = 'compute_intensive'
                    confidence = 0.6
                elif cpu_percent > 15:
                    classification = 'interactive_heavy'
                    confidence = 0.5
                elif memory_mb > 500:
                    classification = 'memory_intensive'
                    confidence = 0.5
                else:
                    classification = 'background'
                    confidence = 0.4
            
            # Cache the result
            self.classification_cache[cache_key] = {
                'classification': classification,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            # Store in database (async)
            self._store_classification_async(name, classification, confidence, cpu_percent, memory_mb)
            
            return classification, confidence
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 'system_process', 0.2
    
    def _store_classification_async(self, name: str, classification: str, confidence: float, cpu_usage: float, memory_mb: float):
        """Store classification asynchronously to avoid blocking"""
        def store():
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO lightweight_classifications 
                    (timestamp, process_name, classification, confidence, cpu_usage, memory_mb)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), name, classification, confidence, cpu_usage, memory_mb))
                conn.commit()
                conn.close()
            except:
                pass  # Ignore database errors to prevent blocking
        
        # Run in background thread
        threading.Thread(target=store, daemon=True).start()
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Recent classifications
            cursor = conn.execute('''
                SELECT classification, COUNT(*), AVG(confidence)
                FROM lightweight_classifications 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY classification
                ORDER BY COUNT(*) DESC
            ''')
            recent_stats = cursor.fetchall()
            
            # Overall stats
            cursor = conn.execute('''
                SELECT AVG(confidence), COUNT(*)
                FROM lightweight_classifications 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            accuracy_data = cursor.fetchone()
            conn.close()
            
            return {
                'recent_classifications': recent_stats,
                'average_confidence': accuracy_data[0] if accuracy_data[0] else 0,
                'total_classifications': accuracy_data[1] if accuracy_data[1] else 0,
                'current_thresholds': {
                    'cpu_interactive': 15.0,
                    'cpu_compute': 50.0,
                    'memory_heavy': 500.0
                }
            }
        except:
            return {
                'recent_classifications': [],
                'average_confidence': 0,
                'total_classifications': 0,
                'current_thresholds': {}
            }
    
    def cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # Every 5 minutes
            # Remove old cache entries
            expired_keys = [
                key for key, value in self.classification_cache.items()
                if current_time - value['timestamp'] > 300
            ]
            for key in expired_keys:
                del self.classification_cache[key]
            
            self.last_cleanup = current_time

class LightweightEASScheduler:
    """Lightweight EAS scheduler"""
    
    def __init__(self, db_path: str):
        self.classifier = LightweightProcessClassifier(db_path)
        self.core_assignments = {}
        
        # Simple assignment strategies
        self.assignment_strategies = {
            'interactive_heavy': 'p_core',
            'compute_intensive': 'p_core',
            'background': 'e_core',
            'system_service': 'e_core',
            'memory_intensive': 'balanced',
            'unknown': 'e_core'
        }
    
    def classify_and_assign_fast(self, pid: int, name: str) -> Dict:
        """Fast classification and assignment"""
        
        # Fast classification
        classification, confidence = self.classifier.classify_process_fast(pid, name)
        
        # Simple core assignment
        strategy = self.assignment_strategies.get(classification, 'e_core')
        
        if strategy == 'balanced':
            # Simple load balancing
            target_core = 'e_core'  # Default to efficiency
        else:
            target_core = strategy
        
        return {
            'pid': pid,
            'name': name,
            'classification': classification,
            'target_core': target_core,
            'confidence': confidence,
            'strategy': strategy
        }
    
    def optimize_system_lightweight(self, max_processes: int = 50) -> Dict:
        """Lightweight system optimization"""
        optimizations = []
        
        try:
            # Limit the number of processes to avoid hanging
            process_count = 0
            for proc in psutil.process_iter(['pid', 'name']):
                if process_count >= max_processes:
                    break
                
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    
                    if pid < 50:  # Skip low-level system processes
                        continue
                    
                    assignment = self.classify_and_assign_fast(pid, name)
                    optimizations.append(assignment)
                    process_count += 1
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Clean up cache periodically
            self.classifier.cleanup_cache()
            
            return {
                'optimized': len(optimizations),
                'assignments': optimizations,
                'limited_to': max_processes
            }
            
        except Exception as e:
            return {
                'optimized': 0,
                'assignments': [],
                'error': str(e)
            }

# Integration patch for the main app
class LightweightEASIntegration:
    """Lightweight EAS integration that won't hang the app"""
    
    def __init__(self, original_eas):
        self.original_eas = original_eas
        db_path = os.path.expanduser("~/.battery_optimizer_lightweight_eas.db")
        self.lightweight_scheduler = LightweightEASScheduler(db_path)
        
        # Replace heavy methods with lightweight versions
        self._patch_methods()
    
    def _patch_methods(self):
        """Patch original EAS with lightweight methods"""
        
        # Store original methods
        self.original_eas._original_classify_workload = getattr(self.original_eas, 'classify_workload', None)
        
        # Replace with lightweight versions
        self.original_eas.classify_workload_lightweight = self._lightweight_classify_workload
        self.original_eas.optimize_system_lightweight = self._lightweight_optimize_system
        self.original_eas.get_lightweight_stats = self._get_lightweight_stats
    
    def _lightweight_classify_workload(self, pid, name):
        """Lightweight workload classification"""
        classification, confidence = self.lightweight_scheduler.classifier.classify_process_fast(pid, name)
        
        # Map to original EAS categories
        mapping = {
            'interactive_heavy': 'interactive',
            'compute_intensive': 'compute',
            'background': 'background',
            'system_service': 'background',
            'memory_intensive': 'compute',
            'unknown': 'background'
        }
        
        return mapping.get(classification, 'background')
    
    def _lightweight_optimize_system(self):
        """Lightweight system optimization"""
        result = self.lightweight_scheduler.optimize_system_lightweight(max_processes=30)
        
        # Convert to original format
        return {
            'optimized': result['optimized'],
            'assignments': result['assignments'][:15],  # Limit display
            'metrics': getattr(self.original_eas, 'current_metrics', {})
        }
    
    def _get_lightweight_stats(self):
        """Get lightweight classification stats"""
        return self.lightweight_scheduler.classifier.get_classification_stats()

def patch_eas_lightweight(eas_instance):
    """Patch EAS with lightweight version"""
    return LightweightEASIntegration(eas_instance)

if __name__ == "__main__":
    # Test the lightweight classifier
    db_path = os.path.expanduser("~/.battery_optimizer_lightweight_eas_test.db")
    scheduler = LightweightEASScheduler(db_path)
    
    print("🚀 Testing Lightweight EAS Classifier")
    print("=" * 50)
    
    start_time = time.time()
    result = scheduler.optimize_system_lightweight(max_processes=20)
    end_time = time.time()
    
    print(f"✅ Optimized {result['optimized']} processes in {end_time - start_time:.2f} seconds")
    
    for assignment in result['assignments'][:10]:
        print(f"  {assignment['name'][:20]:20} → {assignment['target_core']:6} ({assignment['classification']})")
    
    stats = scheduler.classifier.get_classification_stats()
    print(f"\n📊 Stats: {stats['total_classifications']} classifications, {stats['average_confidence']:.3f} avg confidence")