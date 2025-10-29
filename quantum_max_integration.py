#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Max Scheduler Integration
==================================
Integrates the ultimate quantum scheduler with the PQS Framework
"""

import logging
from typing import Dict, Any, Optional
import threading
import time

logger = logging.getLogger(__name__)

class QuantumMaxIntegration:
    """Integration layer for Quantum Max Scheduler"""
    
    def __init__(self):
        self.quantum_scheduler = None
        self.is_active = False
        self.stats_lock = threading.Lock()
        
        try:
            from quantum_max_scheduler import get_quantum_max_scheduler
            self.quantum_scheduler = get_quantum_max_scheduler(max_qubits=48)
            logger.info("✅ Quantum Max Scheduler integration ready")
        except ImportError as e:
            logger.warning(f"⚠️ Quantum Max Scheduler not available: {e}")
    
    def activate_quantum_max_mode(self, interval: int = 10) -> bool:
        """Activate quantum maximum mode - NON-BLOCKING"""
        if not self.quantum_scheduler:
            logger.error("Quantum Max Scheduler not available")
            return False
        
        try:
            # Stop if already running (non-blocking)
            if self.quantum_scheduler.running:
                self.quantum_scheduler.stop_continuous_optimization()
                # Don't wait - let it stop asynchronously
            
            # Start with specified interval (non-blocking)
            self.quantum_scheduler.start_continuous_optimization(interval=interval)
            self.is_active = True
            
            logger.info(f"🚀 QUANTUM MAX MODE ACTIVATED")
            logger.info(f"   Max Qubits: {self.quantum_scheduler.max_qubits}")
            logger.info(f"   Optimization Interval: {interval}s")
            logger.info(f"   Strategies: Performance, Battery, Thermal, RAM, Balanced")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate Quantum Max Mode: {e}")
            return False
    
    def deactivate_quantum_max_mode(self) -> bool:
        """Deactivate quantum maximum mode"""
        if not self.quantum_scheduler:
            return False
        
        try:
            self.quantum_scheduler.stop_continuous_optimization()
            self.is_active = False
            logger.info("⏹️ Quantum Max Mode deactivated")
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate Quantum Max Mode: {e}")
            return False
    
    def get_quantum_max_status(self) -> Dict[str, Any]:
        """Get quantum max scheduler status"""
        if not self.quantum_scheduler:
            return {
                'available': False,
                'active': False,
                'error': 'Quantum Max Scheduler not available'
            }
        
        try:
            stats = self.quantum_scheduler.get_statistics()
            metrics = self.quantum_scheduler.get_system_metrics()
            
            return {
                'available': True,
                'active': self.is_active,
                'running': self.quantum_scheduler.running,
                'max_qubits': self.quantum_scheduler.max_qubits,
                'active_qubits': self.quantum_scheduler.active_qubits,
                'total_optimizations': stats['total_optimizations'],
                'total_energy_saved': stats['total_energy_saved'],
                'total_lag_prevented': stats['total_lag_prevented'],
                'total_ram_freed': stats['total_ram_freed'],
                'recent_performance': stats['recent_performance'],
                'current_metrics': {
                    'cpu_percent': metrics.cpu_percent if metrics else 0,
                    'memory_percent': metrics.memory_percent if metrics else 0,
                    'thermal_state': metrics.thermal_state if metrics else 'unknown',
                    'battery_percent': metrics.battery_percent if metrics else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting Quantum Max status: {e}")
            return {
                'available': True,
                'active': self.is_active,
                'error': str(e)
            }
    
    def run_single_optimization(self) -> Optional[Dict[str, Any]]:
        """Run a single quantum max optimization"""
        if not self.quantum_scheduler:
            return None
        
        try:
            metrics = self.quantum_scheduler.get_system_metrics()
            if not metrics:
                return None
            
            result = self.quantum_scheduler.optimize_system(metrics)
            
            return {
                'strategy': result.strategy,
                'energy_saved': result.energy_saved,
                'performance_boost': result.performance_boost,
                'lag_reduction': result.lag_reduction,
                'ram_freed_mb': result.ram_freed_mb,
                'thermal_reduction': result.thermal_reduction,
                'quantum_advantage': result.quantum_advantage,
                'qubits_used': result.qubits_used,
                'circuit_depth': result.circuit_depth,
                'execution_time_ms': result.execution_time_ms
            }
        except Exception as e:
            logger.error(f"Single optimization error: {e}")
            return None


# Global instance
_quantum_max_integration = None

def get_quantum_max_integration() -> QuantumMaxIntegration:
    """Get or create global quantum max integration instance"""
    global _quantum_max_integration
    
    if _quantum_max_integration is None:
        _quantum_max_integration = QuantumMaxIntegration()
    
    return _quantum_max_integration
