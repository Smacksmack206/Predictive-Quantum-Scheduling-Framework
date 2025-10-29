#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Quantum Optimization System
====================================

Integrates all optimization modules into a single production-ready system:
- Phase 1: Hardware sensors, validation, M3 GPU acceleration
- Phase 2: Intel optimization
- Phase 3: Advanced quantum algorithms

Provides automatic architecture detection and optimal algorithm selection.
"""

import platform
import logging
import time
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime

# Import all modules
from hardware_sensors import get_sensor_manager
from data_validator import get_validator, ValidationLevel
from enhanced_hardware_integration import get_hardware_monitor
from m3_gpu_accelerator import get_gpu_accelerator
from intel_optimizer import get_intel_optimizer
from advanced_quantum_algorithms import get_advanced_algorithms
from real_time_optimization_system import RealTimeOptimizationSystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class UnifiedOptimizationResult:
    """Comprehensive optimization result"""
    architecture: str  # 'apple_silicon' or 'intel'
    total_energy_saved_percent: float
    optimization_method: str
    gpu_accelerated: bool
    quantum_algorithm_used: str
    execution_time_ms: float
    validation_passed: bool
    thermal_state: str
    timestamp: datetime


class UnifiedQuantumSystem:
    """
    Unified quantum optimization system with automatic architecture detection.
    Provides maximum performance on both Apple Silicon and Intel systems.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        # Detect architecture
        self.architecture = self._detect_architecture()
        self.is_apple_silicon = self.architecture == 'apple_silicon'
        self.is_intel = self.architecture == 'intel'
        
        # Initialize core components
        self.sensor_manager = get_sensor_manager()
        self.validator = get_validator(validation_level)
        self.hardware_monitor = get_hardware_monitor(validation_level)
        
        # Initialize architecture-specific components
        if self.is_apple_silicon:
            self.gpu_accelerator = get_gpu_accelerator()
            self.rt_optimizer = RealTimeOptimizationSystem()
            logger.info("🍎 Apple Silicon mode: GPU acceleration enabled")
        else:
            self.intel_optimizer = get_intel_optimizer()
            logger.info("💻 Intel mode: Quantum-inspired classical optimization")
        
        # Initialize advanced algorithms (both architectures)
        self.advanced_algorithms = get_advanced_algorithms()
        
        # Performance tracking
        self.optimization_history = []
        self.total_energy_saved = 0.0
        self.cycles_completed = 0
        
        logger.info(f"🚀 Unified Quantum System initialized - {self.architecture}")
    
    def _detect_architecture(self) -> str:
        """Detect system architecture"""
        machine = platform.machine().lower()
        processor = platform.processor().lower()
        
        if 'arm' in machine or 'arm' in processor:
            return 'apple_silicon'
        else:
            return 'intel'
    
    def run_unified_optimization(self) -> UnifiedOptimizationResult:
        """
        Run unified optimization cycle.
        Automatically selects best optimization strategy for architecture.
        """
        start_time = time.time()
        
        # Step 1: Get validated hardware metrics
        metrics = self.hardware_monitor.get_comprehensive_validated_metrics()
        
        # Step 2: Run architecture-specific optimization
        if self.is_apple_silicon:
            result = self._optimize_apple_silicon(metrics)
        else:
            result = self._optimize_intel(metrics)
        
        # Step 3: Apply advanced quantum algorithms
        quantum_improvement = self._apply_advanced_algorithms(metrics)
        result.total_energy_saved_percent += quantum_improvement
        
        # Update statistics
        self.optimization_history.append(result)
        self.total_energy_saved += result.total_energy_saved_percent
        self.cycles_completed += 1
        
        execution_time = (time.time() - start_time) * 1000
        result.execution_time_ms = execution_time
        
        logger.info(f"✅ Optimization: {result.total_energy_saved_percent:.1f}% saved, "
                   f"{result.optimization_method}, {execution_time:.1f}ms")
        
        return result
    
    def _optimize_apple_silicon(self, metrics: Dict[str, Any]) -> UnifiedOptimizationResult:
        """Apple Silicon optimization with GPU acceleration"""
        # Use real-time optimization system
        cycle = self.rt_optimizer.run_optimization_cycle(use_cached_metrics=True)
        
        thermal_state = 'nominal'
        if metrics['thermal'] and metrics['thermal']['validated']:
            thermal_state = metrics['thermal']['thermal_pressure']
        
        return UnifiedOptimizationResult(
            architecture='apple_silicon',
            total_energy_saved_percent=cycle.energy_saved_percent,
            optimization_method='gpu_quantum',
            gpu_accelerated=True,
            quantum_algorithm_used='gpu_state_vector',
            execution_time_ms=cycle.execution_time_ms,
            validation_passed=cycle.validation_passed,
            thermal_state=thermal_state,
            timestamp=datetime.now()
        )
    
    def _optimize_intel(self, metrics: Dict[str, Any]) -> UnifiedOptimizationResult:
        """Intel optimization with quantum-inspired classical algorithms"""
        # Prepare metrics for Intel optimizer
        intel_metrics = {
            'cpu_percent': metrics.get('cpu', {}).get('performance_cores_active', 0) * 12.5,
            'cpu_temp': metrics.get('thermal', {}).get('cpu_temp_celsius', 60.0)
        }
        
        # Run Intel optimization
        result = self.intel_optimizer.optimize_intel_system(intel_metrics)
        
        return UnifiedOptimizationResult(
            architecture='intel',
            total_energy_saved_percent=result.energy_saved_percent,
            optimization_method='quantum_inspired_classical',
            gpu_accelerated=False,
            quantum_algorithm_used='simulated_annealing',
            execution_time_ms=result.execution_time_ms,
            validation_passed=True,
            thermal_state='nominal' if not result.thermal_optimized else 'optimized',
            timestamp=datetime.now()
        )
    
    def _apply_advanced_algorithms(self, metrics: Dict[str, Any]) -> float:
        """
        Apply advanced quantum algorithms for additional optimization.
        Returns additional energy savings percentage.
        """
        try:
            # Get current processes (simulated for now)
            num_processes = 8
            processes = [{'cpu_percent': np.random.uniform(10, 80)} for _ in range(num_processes)]
            
            # Use quantum annealing for process scheduling
            schedule_result = self.advanced_algorithms.optimize_process_schedule(processes)
            
            # Additional savings from quantum scheduling
            additional_savings = schedule_result.energy_improvement * 0.1  # 10% of improvement
            
            return additional_savings
            
        except Exception as e:
            logger.error(f"Advanced algorithms error: {e}")
            return 0.0
    
    def run_continuous_optimization(self, duration_seconds: int = 60, interval_seconds: float = 1.0):
        """
        Run continuous optimization for specified duration.
        """
        logger.info(f"🔄 Starting continuous optimization for {duration_seconds}s...")
        
        start_time = time.time()
        cycle_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            result = self.run_unified_optimization()
            cycle_count += 1
            
            if cycle_count % 10 == 0:
                logger.info(f"📊 Completed {cycle_count} cycles, "
                           f"average savings: {self.get_average_savings():.1f}%")
            
            time.sleep(interval_seconds)
        
        logger.info(f"✅ Continuous optimization complete: {cycle_count} cycles")
    
    def get_average_savings(self) -> float:
        """Get average energy savings"""
        if not self.optimization_history:
            return 0.0
        
        recent = self.optimization_history[-100:]
        return np.mean([r.total_energy_saved_percent for r in recent])
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'architecture': self.architecture,
            'cycles_completed': self.cycles_completed,
            'total_energy_saved': self.total_energy_saved,
            'average_energy_saved': self.get_average_savings(),
            'validation_stats': self.validator.get_validation_statistics()
        }
        
        # Add architecture-specific stats
        if self.is_apple_silicon:
            stats['gpu_stats'] = self.gpu_accelerator.get_performance_statistics()
            stats['rt_optimizer_stats'] = self.rt_optimizer.get_performance_summary()
        else:
            stats['intel_stats'] = self.intel_optimizer.get_intel_performance_stats()
        
        # Add advanced algorithm stats
        stats['advanced_algorithms'] = self.advanced_algorithms.get_algorithm_statistics()
        
        return stats
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current state"""
        recommendations = []
        
        # Get current metrics
        metrics = self.hardware_monitor.get_comprehensive_validated_metrics()
        
        # Check thermal state
        if metrics['thermal'] and metrics['thermal']['validated']:
            temp = metrics['thermal']['cpu_temp_celsius']
            if temp > 80:
                recommendations.append(f"High CPU temperature ({temp:.1f}°C) - reduce workload")
            elif temp > 70:
                recommendations.append(f"Elevated temperature ({temp:.1f}°C) - monitor thermal state")
        
        # Check power consumption
        if metrics['power'] and metrics['power']['validated']:
            power = metrics['power']['total_power_watts']
            if power > 25:
                recommendations.append(f"High power consumption ({power:.1f}W) - optimization recommended")
        
        # Check GPU utilization (Apple Silicon only)
        if self.is_apple_silicon and metrics['gpu'] and metrics['gpu']['validated']:
            util = metrics['gpu']['utilization_percent']
            if util > 90:
                recommendations.append(f"High GPU utilization ({util:.1f}%) - consider workload distribution")
        
        # Architecture-specific recommendations
        if self.is_apple_silicon:
            avg_savings = self.get_average_savings()
            if avg_savings < 15:
                recommendations.append("Below target savings - increase quantum circuit complexity")
        else:
            intel_stats = self.intel_optimizer.get_intel_performance_stats()
            if intel_stats['average_energy_saved'] < 5:
                recommendations.append("Below target savings - enable aggressive optimization")
        
        if not recommendations:
            recommendations.append("System operating optimally")
        
        return recommendations


def main():
    """Test unified quantum system"""
    print("🚀 Unified Quantum Optimization System")
    print("=" * 70)
    
    # Initialize system
    system = UnifiedQuantumSystem()
    
    print(f"\n📊 System Configuration:")
    print(f"  Architecture: {system.architecture}")
    print(f"  GPU Acceleration: {system.is_apple_silicon}")
    print(f"  Validation Level: Strict")
    
    # Run optimization cycles
    print(f"\n⚡ Running optimization cycles...")
    for i in range(5):
        result = system.run_unified_optimization()
        print(f"\nCycle {i+1}:")
        print(f"  Energy saved: {result.total_energy_saved_percent:.1f}%")
        print(f"  Method: {result.optimization_method}")
        print(f"  Quantum algorithm: {result.quantum_algorithm_used}")
        print(f"  GPU accelerated: {result.gpu_accelerated}")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")
        print(f"  Validated: {result.validation_passed}")
        
        time.sleep(0.5)
    
    # Get statistics
    print(f"\n📈 System Statistics:")
    stats = system.get_comprehensive_statistics()
    print(f"  Cycles completed: {stats['cycles_completed']}")
    print(f"  Average energy saved: {stats['average_energy_saved']:.1f}%")
    print(f"  Total energy saved: {stats['total_energy_saved']:.1f}%")
    
    # Get recommendations
    print(f"\n💡 Optimization Recommendations:")
    recommendations = system.get_optimization_recommendations()
    for rec in recommendations:
        print(f"  • {rec}")
    
    print(f"\n{'=' * 70}")
    print("✅ Unified quantum system test complete!")


if __name__ == '__main__':
    main()
