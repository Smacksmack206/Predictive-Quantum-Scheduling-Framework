#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-Lag Optimizer - Ensures Zero System Lag
=============================================

Implements async optimization, adaptive scheduling, and priority-based
process management to ensure PQS never causes system lag.

Priority: CRITICAL - Implements Phase 1 improvements
"""

import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SystemLoad:
    """Current system load metrics"""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    is_busy: bool
    timestamp: datetime


class AsyncOptimizer:
    """
    Async optimization engine that never blocks the UI.
    Runs optimizations in background threads.
    """
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.optimization_lock = threading.Lock()
        self.is_optimizing = False
        self.last_optimization = None
        self.optimization_queue = deque(maxlen=10)
        
        logger.info("🚀 Async Optimizer initialized")
    
    def run_optimization_async(
        self,
        optimization_func: Callable,
        callback: Optional[Callable] = None,
        priority: str = 'normal'
    ) -> Optional[Future]:
        """
        Run optimization asynchronously without blocking.
        
        Args:
            optimization_func: Function to run
            callback: Optional callback when complete
            priority: 'high', 'normal', or 'low'
        
        Returns:
            Future object or None if skipped
        """
        # Check if already optimizing
        if self.is_optimizing and priority != 'high':
            logger.debug("Optimization already running, skipping")
            return None
        
        with self.optimization_lock:
            self.is_optimizing = True
        
        def optimize_task():
            try:
                start_time = time.time()
                result = optimization_func()
                execution_time = (time.time() - start_time) * 1000
                
                self.last_optimization = {
                    'result': result,
                    'execution_time_ms': execution_time,
                    'timestamp': datetime.now()
                }
                
                if callback:
                    callback(result)
                
                logger.info(f"✅ Async optimization complete: {execution_time:.1f}ms")
                return result
                
            except Exception as e:
                logger.error(f"Async optimization error: {e}")
                return None
            finally:
                with self.optimization_lock:
                    self.is_optimizing = False
        
        # Submit to thread pool
        future = self.executor.submit(optimize_task)
        self.optimization_queue.append(future)
        
        return future
    
    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """Wait for current optimization to complete"""
        start = time.time()
        while self.is_optimizing and (time.time() - start) < timeout:
            time.sleep(0.1)
        return not self.is_optimizing
    
    def shutdown(self):
        """Shutdown executor gracefully"""
        self.executor.shutdown(wait=True)


class AdaptiveScheduler:
    """
    Adaptive scheduling that adjusts optimization frequency
    based on system load. Never optimizes when system is busy.
    """
    
    def __init__(self):
        self.base_interval = 30  # seconds
        self.current_interval = 30
        self.min_interval = 15
        self.max_interval = 120
        
        # Thresholds
        self.cpu_busy_threshold = 80
        self.cpu_idle_threshold = 30
        self.memory_pressure_threshold = 85
        
        # History
        self.load_history = deque(maxlen=60)  # Last 60 measurements
        
        logger.info("📊 Adaptive Scheduler initialized")
    
    def get_system_load(self) -> SystemLoad:
        """Get current system load"""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        
        # Estimate disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            disk_percent = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 100))
        except:
            disk_percent = 0
        
        is_busy = (
            cpu > self.cpu_busy_threshold or
            memory > self.memory_pressure_threshold
        )
        
        load = SystemLoad(
            cpu_percent=cpu,
            memory_percent=memory,
            disk_io_percent=disk_percent,
            is_busy=is_busy,
            timestamp=datetime.now()
        )
        
        self.load_history.append(load)
        return load
    
    def get_next_interval(self) -> float:
        """
        Calculate optimal next interval based on system load.
        Returns interval in seconds.
        """
        load = self.get_system_load()
        
        if load.is_busy:
            # System busy - wait longer
            self.current_interval = min(self.max_interval, self.current_interval * 1.5)
            logger.debug(f"System busy, increasing interval to {self.current_interval:.0f}s")
        
        elif load.cpu_percent < self.cpu_idle_threshold:
            # System idle - optimize more frequently
            self.current_interval = max(self.min_interval, self.current_interval * 0.8)
            logger.debug(f"System idle, decreasing interval to {self.current_interval:.0f}s")
        
        else:
            # Normal load - use base interval
            self.current_interval = self.base_interval
        
        return self.current_interval
    
    def should_optimize_now(self) -> bool:
        """Check if it's safe to optimize now"""
        load = self.get_system_load()
        
        if load.is_busy:
            logger.debug("System busy, skipping optimization")
            return False
        
        # Check recent history - don't optimize if system was recently busy
        if len(self.load_history) >= 3:
            recent = list(self.load_history)[-3:]
            if any(l.is_busy for l in recent):
                logger.debug("System recently busy, skipping optimization")
                return False
        
        return True
    
    def get_load_statistics(self) -> Dict[str, float]:
        """Get load statistics"""
        if not self.load_history:
            return {}
        
        recent = list(self.load_history)[-10:]
        
        return {
            'avg_cpu': sum(l.cpu_percent for l in recent) / len(recent),
            'avg_memory': sum(l.memory_percent for l in recent) / len(recent),
            'busy_rate': sum(1 for l in recent if l.is_busy) / len(recent),
            'current_interval': self.current_interval
        }


class PriorityProcessManager:
    """
    Priority-based process management.
    Never touches critical user-facing apps.
    """
    
    def __init__(self):
        # Apps that should never be throttled
        self.critical_apps = {
            'Terminal', 'iTerm', 'Code', 'VSCode', 'Xcode',
            'Chrome', 'Safari', 'Firefox', 'Arc',
            'Slack', 'Discord', 'Zoom', 'Teams',
            'Music', 'Spotify', 'VLC',
            'Finder', 'System Preferences', 'System Settings'
        }
        
        # Apps safe to optimize
        self.background_apps = {
            'Dropbox', 'Google Drive', 'OneDrive',
            'Time Machine', 'Backup',
            'Spotlight', 'mds', 'mdworker'
        }
        
        self.process_classifications = {}
        
        logger.info("🎯 Priority Process Manager initialized")
    
    def classify_process(self, proc: psutil.Process) -> str:
        """
        Classify process priority.
        Returns: 'critical', 'normal', or 'background'
        """
        try:
            name = proc.name()
            
            # Check if critical
            if any(critical in name for critical in self.critical_apps):
                return 'critical'
            
            # Check if background
            if any(bg in name for bg in self.background_apps):
                return 'background'
            
            # Check CPU usage
            cpu = proc.cpu_percent(interval=0.1)
            if cpu < 1.0:
                return 'background'
            
            # Default to normal
            return 'normal'
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 'unknown'
    
    def get_optimizable_processes(self) -> List[psutil.Process]:
        """Get list of processes safe to optimize"""
        optimizable = []
        
        for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
            try:
                classification = self.classify_process(proc)
                
                if classification == 'background':
                    optimizable.append(proc)
                elif classification == 'normal':
                    # Only optimize if low CPU
                    cpu_percent = proc.info.get('cpu_percent', 0)
                    if cpu_percent and cpu_percent < 5.0:
                        optimizable.append(proc)
                # Never optimize critical apps
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue
        
        return optimizable
    
    def get_process_statistics(self) -> Dict[str, int]:
        """Get process classification statistics"""
        stats = {'critical': 0, 'normal': 0, 'background': 0, 'unknown': 0}
        
        for proc in psutil.process_iter():
            try:
                classification = self.classify_process(proc)
                stats[classification] = stats.get(classification, 0) + 1
            except:
                continue
        
        return stats


class AntiLagSystem:
    """
    Complete anti-lag system combining all components.
    Ensures PQS never causes system lag.
    """
    
    def __init__(self):
        self.async_optimizer = AsyncOptimizer(max_workers=2)
        self.scheduler = AdaptiveScheduler()
        self.process_manager = PriorityProcessManager()
        
        self.optimization_count = 0
        self.skipped_count = 0
        
        logger.info("🛡️ Anti-Lag System initialized")
    
    def run_safe_optimization(
        self,
        optimization_func: Callable,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Run optimization safely without causing lag.
        
        Returns:
            True if optimization ran, False if skipped
        """
        # Check if safe to optimize
        if not self.scheduler.should_optimize_now():
            self.skipped_count += 1
            logger.debug("Optimization skipped - system busy")
            return False
        
        # Get optimizable processes
        optimizable = self.process_manager.get_optimizable_processes()
        logger.debug(f"Found {len(optimizable)} optimizable processes")
        
        # Run async
        future = self.async_optimizer.run_optimization_async(
            optimization_func,
            callback
        )
        
        if future:
            self.optimization_count += 1
            return True
        else:
            self.skipped_count += 1
            return False
    
    def get_next_optimization_time(self) -> float:
        """Get seconds until next optimization"""
        return self.scheduler.get_next_interval()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'optimizations_run': self.optimization_count,
            'optimizations_skipped': self.skipped_count,
            'skip_rate': self.skipped_count / max(1, self.optimization_count + self.skipped_count),
            'load_stats': self.scheduler.get_load_statistics(),
            'process_stats': self.process_manager.get_process_statistics(),
            'is_optimizing': self.async_optimizer.is_optimizing,
            'last_optimization': self.async_optimizer.last_optimization
        }
    
    def shutdown(self):
        """Shutdown gracefully"""
        self.async_optimizer.shutdown()


# Global instance
_anti_lag_system = None


def get_anti_lag_system() -> AntiLagSystem:
    """Get or create global anti-lag system"""
    global _anti_lag_system
    if _anti_lag_system is None:
        _anti_lag_system = AntiLagSystem()
    return _anti_lag_system


if __name__ == '__main__':
    print("🛡️ Testing Anti-Lag System...")
    
    system = get_anti_lag_system()
    
    # Test safe optimization
    print("\n⚡ Testing safe optimization...")
    
    def dummy_optimization():
        time.sleep(0.1)  # Simulate work
        return {'energy_saved': 25.0}
    
    def optimization_callback(result):
        print(f"   ✅ Optimization complete: {result}")
    
    # Run 5 optimizations
    for i in range(5):
        success = system.run_safe_optimization(
            dummy_optimization,
            optimization_callback
        )
        
        if success:
            print(f"   Cycle {i+1}: Started")
        else:
            print(f"   Cycle {i+1}: Skipped (system busy)")
        
        time.sleep(0.5)
    
    # Wait for completion
    system.async_optimizer.wait_for_completion()
    
    # Get statistics
    print("\n📊 Statistics:")
    stats = system.get_statistics()
    print(f"   Optimizations run: {stats['optimizations_run']}")
    print(f"   Optimizations skipped: {stats['optimizations_skipped']}")
    print(f"   Skip rate: {stats['skip_rate']:.1%}")
    
    if stats['load_stats']:
        print(f"\n   Load Statistics:")
        for key, value in stats['load_stats'].items():
            print(f"     {key}: {value:.1f}")
    
    print(f"\n   Process Statistics:")
    for key, value in stats['process_stats'].items():
        print(f"     {key}: {value}")
    
    # Shutdown
    system.shutdown()
    
    print("\n✅ Anti-lag system test complete!")
