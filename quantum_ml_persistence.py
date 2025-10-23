#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-ML Persistence Layer
=============================

Persistent storage for optimization history with:
- Local SQLite database
- Automatic save/load on startup
- Distributed sync capability
- Historical data analysis
- Real accumulated metrics
"""

import sqlite3
import json
import time
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class QuantumMLDatabase:
    """Persistent storage for quantum-ML optimization data"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database (default: ~/.pqs_quantum_ml.db)
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.pqs_quantum_ml.db")
        
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"📊 Quantum-ML database initialized: {db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Optimizations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    energy_saved REAL NOT NULL,
                    performance_gain REAL NOT NULL,
                    quantum_advantage REAL NOT NULL,
                    ml_confidence REAL NOT NULL,
                    optimization_strategy TEXT NOT NULL,
                    quantum_circuits_used INTEGER NOT NULL,
                    execution_time REAL NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    process_count INTEGER,
                    battery_level REAL,
                    power_plugged INTEGER,
                    thermal_state TEXT,
                    architecture TEXT
                )
            ''')
            
            # System stats table (cumulative)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    optimizations_run INTEGER NOT NULL,
                    total_energy_saved REAL NOT NULL,
                    ml_models_trained INTEGER NOT NULL,
                    quantum_operations INTEGER NOT NULL,
                    predictions_made INTEGER NOT NULL,
                    power_efficiency_score REAL NOT NULL,
                    current_savings_rate REAL NOT NULL,
                    ml_average_accuracy REAL NOT NULL,
                    architecture TEXT NOT NULL
                )
            ''')
            
            # ML accuracy history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_accuracy_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    confidence REAL NOT NULL,
                    architecture TEXT NOT NULL
                )
            ''')
            
            # Distributed sync table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS distributed_sync (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    sync_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    synced INTEGER DEFAULT 0
                )
            ''')
            
            # Process optimizations table - stores learned optimizations per process
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS process_optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    process_name TEXT NOT NULL,
                    optimization_strategy TEXT NOT NULL,
                    nice_adjustment INTEGER NOT NULL,
                    avg_energy_saved REAL NOT NULL,
                    times_applied INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 1.0,
                    last_applied REAL NOT NULL,
                    created_at REAL NOT NULL,
                    architecture TEXT NOT NULL,
                    UNIQUE(process_name, architecture)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimizations_timestamp ON optimizations(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_stats_timestamp ON system_stats(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_accuracy_timestamp ON ml_accuracy_history(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_process_optimizations_name ON process_optimizations(process_name, architecture)')
            
            self.conn.commit()
            logger.info("✅ Database tables initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_optimization(self, result: Dict, system_state: Dict, architecture: str):
        """Save optimization result to database"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO optimizations (
                        timestamp, energy_saved, performance_gain, quantum_advantage,
                        ml_confidence, optimization_strategy, quantum_circuits_used,
                        execution_time, cpu_percent, memory_percent, process_count,
                        battery_level, power_plugged, thermal_state, architecture
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(),
                    result.get('energy_saved', 0.0),
                    result.get('performance_gain', 0.0),
                    result.get('quantum_advantage', 1.0),
                    result.get('ml_confidence', 0.0),
                    result.get('optimization_strategy', 'Unknown'),
                    result.get('quantum_circuits_used', 0),
                    result.get('execution_time', 0.0),
                    system_state.get('cpu_percent', 0.0),
                    system_state.get('memory_percent', 0.0),
                    system_state.get('process_count', 0),
                    system_state.get('battery_level'),
                    1 if system_state.get('power_plugged') else 0,
                    system_state.get('thermal_state', 'normal'),
                    architecture
                ))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving optimization: {e}")
    
    def save_system_stats(self, stats: Dict, architecture: str):
        """Save current system stats to database"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO system_stats (
                        timestamp, optimizations_run, total_energy_saved,
                        ml_models_trained, quantum_operations, predictions_made,
                        power_efficiency_score, current_savings_rate,
                        ml_average_accuracy, architecture
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(),
                    stats.get('optimizations_run', 0),
                    stats.get('energy_saved', 0.0),
                    stats.get('ml_models_trained', 0),
                    stats.get('quantum_operations', 0),
                    stats.get('predictions_made', 0),
                    stats.get('power_efficiency_score', 85.0),
                    stats.get('current_savings_rate', 0.0),
                    stats.get('ml_average_accuracy', 0.0),
                    architecture
                ))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving system stats: {e}")
    
    def save_ml_accuracy(self, accuracy: float, confidence: float, architecture: str):
        """Save ML accuracy measurement"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO ml_accuracy_history (timestamp, accuracy, confidence, architecture)
                    VALUES (?, ?, ?, ?)
                ''', (time.time(), accuracy, confidence, architecture))
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving ML accuracy: {e}")
    
    def load_latest_stats(self, architecture: str) -> Optional[Dict]:
        """Load the most recent system stats from database"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT optimizations_run, total_energy_saved, ml_models_trained,
                           quantum_operations, predictions_made, power_efficiency_score,
                           current_savings_rate, ml_average_accuracy
                    FROM system_stats
                    WHERE architecture = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (architecture,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'optimizations_run': row[0],
                        'energy_saved': row[1],
                        'ml_models_trained': row[2],
                        'quantum_operations': row[3],
                        'predictions_made': row[4],
                        'power_efficiency_score': row[5],
                        'current_savings_rate': row[6],
                        'ml_average_accuracy': row[7]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error loading latest stats: {e}")
            return None
    
    def get_total_energy_saved(self, architecture: str = None, days: int = None) -> float:
        """Get total energy saved, optionally filtered by architecture and time"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                query = 'SELECT SUM(energy_saved) FROM optimizations WHERE 1=1'
                params = []
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                if days:
                    cutoff_time = time.time() - (days * 24 * 3600)
                    query += ' AND timestamp > ?'
                    params.append(cutoff_time)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
                
        except Exception as e:
            logger.error(f"Error getting total energy saved: {e}")
            return 0.0
    
    def get_optimization_count(self, architecture: str = None, days: int = None) -> int:
        """Get total number of optimizations"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                query = 'SELECT COUNT(*) FROM optimizations WHERE 1=1'
                params = []
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                if days:
                    cutoff_time = time.time() - (days * 24 * 3600)
                    query += ' AND timestamp > ?'
                    params.append(cutoff_time)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"Error getting optimization count: {e}")
            return 0
    
    def get_average_ml_accuracy(self, architecture: str = None, days: int = None) -> float:
        """Get average ML accuracy"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                query = 'SELECT AVG(accuracy) FROM ml_accuracy_history WHERE 1=1'
                params = []
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                if days:
                    cutoff_time = time.time() - (days * 24 * 3600)
                    query += ' AND timestamp > ?'
                    params.append(cutoff_time)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result[0] else 0.0
                
        except Exception as e:
            logger.error(f"Error getting average ML accuracy: {e}")
            return 0.0
    
    def get_recent_optimizations(self, limit: int = 100, architecture: str = None) -> List[Dict]:
        """Get recent optimization results"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                query = '''
                    SELECT timestamp, energy_saved, performance_gain, quantum_advantage,
                           ml_confidence, optimization_strategy, quantum_circuits_used,
                           cpu_percent, memory_percent, thermal_state
                    FROM optimizations
                    WHERE 1=1
                '''
                params = []
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [{
                    'timestamp': row[0],
                    'energy_saved': row[1],
                    'performance_gain': row[2],
                    'quantum_advantage': row[3],
                    'ml_confidence': row[4],
                    'optimization_strategy': row[5],
                    'quantum_circuits_used': row[6],
                    'cpu_percent': row[7],
                    'memory_percent': row[8],
                    'thermal_state': row[9]
                } for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting recent optimizations: {e}")
            return []
    
    def calculate_current_savings_rate(self, architecture: str = None, minutes: int = 5) -> float:
        """Calculate current savings rate based on recent optimizations"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                cutoff_time = time.time() - (minutes * 60)
                
                query = '''
                    SELECT SUM(energy_saved), MIN(timestamp), MAX(timestamp)
                    FROM optimizations
                    WHERE timestamp > ?
                '''
                params = [cutoff_time]
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result and result[0] and result[1] and result[2]:
                    total_saved = result[0]
                    time_span = (result[2] - result[1]) / 60.0  # Convert to minutes
                    
                    if time_span > 0:
                        return total_saved / time_span
                
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating savings rate: {e}")
            return 0.0
    
    def get_statistics(self, architecture: str = None) -> Dict:
        """Get comprehensive statistics"""
        try:
            return {
                'total_energy_saved': self.get_total_energy_saved(architecture),
                'total_optimizations': self.get_optimization_count(architecture),
                'average_ml_accuracy': self.get_average_ml_accuracy(architecture),
                'current_savings_rate': self.calculate_current_savings_rate(architecture),
                'last_7_days': {
                    'energy_saved': self.get_total_energy_saved(architecture, days=7),
                    'optimizations': self.get_optimization_count(architecture, days=7)
                },
                'last_24_hours': {
                    'energy_saved': self.get_total_energy_saved(architecture, days=1),
                    'optimizations': self.get_optimization_count(architecture, days=1)
                }
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            with self.lock:
                cutoff_time = time.time() - (days * 24 * 3600)
                cursor = self.conn.cursor()
                
                cursor.execute('DELETE FROM optimizations WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM ml_accuracy_history WHERE timestamp < ?', (cutoff_time,))
                
                self.conn.commit()
                logger.info(f"🧹 Cleaned up data older than {days} days")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def export_data(self, output_file: str, architecture: str = None):
        """Export data to JSON file"""
        try:
            data = {
                'export_timestamp': time.time(),
                'architecture': architecture,
                'statistics': self.get_statistics(architecture),
                'recent_optimizations': self.get_recent_optimizations(1000, architecture)
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"📤 Data exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def save_process_optimization(self, process_name: str, strategy: str, 
                                  nice_adjustment: int, energy_saved: float, 
                                  architecture: str, success: bool = True):
        """Save or update a learned process optimization"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                # Check if optimization already exists
                cursor.execute('''
                    SELECT id, times_applied, avg_energy_saved, success_rate
                    FROM process_optimizations
                    WHERE process_name = ? AND architecture = ?
                ''', (process_name, architecture))
                
                existing = cursor.fetchone()
                current_time = time.time()
                
                if existing:
                    # Update existing optimization with running average
                    opt_id, times_applied, avg_saved, success_rate = existing
                    new_times = times_applied + 1
                    new_avg = ((avg_saved * times_applied) + energy_saved) / new_times
                    new_success_rate = ((success_rate * times_applied) + (1.0 if success else 0.0)) / new_times
                    
                    cursor.execute('''
                        UPDATE process_optimizations
                        SET optimization_strategy = ?,
                            nice_adjustment = ?,
                            avg_energy_saved = ?,
                            times_applied = ?,
                            success_rate = ?,
                            last_applied = ?
                        WHERE id = ?
                    ''', (strategy, nice_adjustment, new_avg, new_times, new_success_rate, current_time, opt_id))
                else:
                    # Insert new optimization
                    cursor.execute('''
                        INSERT INTO process_optimizations (
                            process_name, optimization_strategy, nice_adjustment,
                            avg_energy_saved, times_applied, success_rate,
                            last_applied, created_at, architecture
                        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                    ''', (process_name, strategy, nice_adjustment, energy_saved, 
                          1.0 if success else 0.0, current_time, current_time, architecture))
                
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving process optimization: {e}")
    
    def load_process_optimizations(self, architecture: str, min_success_rate: float = 0.5) -> Dict[str, Dict]:
        """Load learned process optimizations"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT process_name, optimization_strategy, nice_adjustment,
                           avg_energy_saved, times_applied, success_rate, last_applied
                    FROM process_optimizations
                    WHERE architecture = ? AND success_rate >= ?
                    ORDER BY avg_energy_saved DESC
                ''', (architecture, min_success_rate))
                
                rows = cursor.fetchall()
                
                optimizations = {}
                for row in rows:
                    optimizations[row[0]] = {
                        'strategy': row[1],
                        'nice_adjustment': row[2],
                        'avg_energy_saved': row[3],
                        'times_applied': row[4],
                        'success_rate': row[5],
                        'last_applied': row[6]
                    }
                
                return optimizations
                
        except Exception as e:
            logger.error(f"Error loading process optimizations: {e}")
            return {}
    
    def get_process_optimization_stats(self, architecture: str = None) -> Dict:
        """Get statistics about learned process optimizations"""
        try:
            with self.lock:
                cursor = self.conn.cursor()
                
                query = '''
                    SELECT COUNT(*), AVG(avg_energy_saved), AVG(success_rate), SUM(times_applied)
                    FROM process_optimizations
                    WHERE 1=1
                '''
                params = []
                
                if architecture:
                    query += ' AND architecture = ?'
                    params.append(architecture)
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    return {
                        'total_learned_optimizations': result[0] or 0,
                        'avg_energy_saved_per_process': result[1] or 0.0,
                        'avg_success_rate': result[2] or 0.0,
                        'total_applications': result[3] or 0
                    }
                
                return {
                    'total_learned_optimizations': 0,
                    'avg_energy_saved_per_process': 0.0,
                    'avg_success_rate': 0.0,
                    'total_applications': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting process optimization stats: {e}")
            return {}
    
    def cleanup_stale_process_optimizations(self, days: int = 30):
        """Remove process optimizations that haven't been applied recently"""
        try:
            with self.lock:
                cutoff_time = time.time() - (days * 24 * 3600)
                cursor = self.conn.cursor()
                
                cursor.execute('''
                    DELETE FROM process_optimizations 
                    WHERE last_applied < ? OR success_rate < 0.3
                ''', (cutoff_time,))
                
                deleted = cursor.rowcount
                self.conn.commit()
                
                if deleted > 0:
                    logger.info(f"🧹 Cleaned up {deleted} stale process optimizations")
                
        except Exception as e:
            logger.error(f"Error cleaning up stale optimizations: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("📊 Database connection closed")

# Global database instance
_db_instance = None

def get_database() -> QuantumMLDatabase:
    """Get or create global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = QuantumMLDatabase()
    return _db_instance

if __name__ == "__main__":
    # Test the database
    print("🧪 Testing Quantum-ML Database")
    print("=" * 60)
    
    db = QuantumMLDatabase("test_quantum_ml.db")
    
    # Test saving optimization
    test_result = {
        'energy_saved': 5.5,
        'performance_gain': 4.5,
        'quantum_advantage': 2.3,
        'ml_confidence': 0.85,
        'optimization_strategy': 'Test Strategy',
        'quantum_circuits_used': 4,
        'execution_time': 0.05
    }
    
    test_state = {
        'cpu_percent': 45.0,
        'memory_percent': 60.0,
        'process_count': 150,
        'battery_level': 75.0,
        'power_plugged': False,
        'thermal_state': 'normal'
    }
    
    db.save_optimization(test_result, test_state, 'apple_silicon')
    print("✅ Saved test optimization")
    
    # Test loading stats
    stats = db.get_statistics('apple_silicon')
    print(f"\n📊 Statistics:")
    print(f"   Total energy saved: {stats['total_energy_saved']:.2f}%")
    print(f"   Total optimizations: {stats['total_optimizations']}")
    print(f"   Average ML accuracy: {stats['average_ml_accuracy']:.2f}")
    
    # Test export
    db.export_data("test_export.json", "apple_silicon")
    print("\n✅ Database test complete!")
    
    db.close()
