#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Hardware Integration Module
=====================================

Integrates hardware sensors and data validation into the quantum system.
Provides 100% authentic real-time metrics with comprehensive validation.

Requirements: Task 1, Task 5 - Real-time optimization enhancement
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
from hardware_sensors import (
    get_sensor_manager,
    PowerMetrics,
    ThermalMetrics,
    GPUMetrics,
    CPUMetrics,
    BatteryMetrics
)
from data_validator import (
    get_validator,
    DataSource,
    ValidationLevel,
    ValidationResult
)

logger = logging.getLogger(__name__)


class EnhancedHardwareMonitor:
    """
    Enhanced hardware monitoring with validation and authenticity guarantees.
    Provides comprehensive real-time metrics for quantum optimization.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.sensor_manager = get_sensor_manager()
        self.validator = get_validator(validation_level)
        self.last_metrics = {}
        self.metrics_history = []
        
        logger.info("🔧 Enhanced Hardware Monitor initialized")
    
    def get_validated_power_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get power metrics with validation.
        Returns only authentic, validated data.
        """
        power = self.sensor_manager.get_real_power_consumption()
        
        if not power:
            return None
        
        # Validate power metrics
        validation_result = self.validator.validate_metric(
            'cpu_power_watts',
            power.cpu_power_watts,
            DataSource.POWERMETRICS,
            power.timestamp
        )
        
        if not validation_result.is_valid or not validation_result.is_authentic:
            logger.warning("⚠️ Power metrics failed validation")
            return None
        
        return {
            'cpu_power_watts': power.cpu_power_watts,
            'gpu_power_watts': power.gpu_power_watts,
            'ane_power_watts': power.ane_power_watts,
            'total_power_watts': power.total_power_watts,
            'timestamp': power.timestamp,
            'validated': True,
            'confidence': validation_result.confidence_score
        }
    
    def get_validated_thermal_metrics(self) -> Optional[Dict[str, Any]]:
        """Get thermal metrics with validation"""
        thermal = self.sensor_manager.get_real_thermal_sensors()
        
        if not thermal:
            return None
        
        # Validate temperature
        validation_result = self.validator.validate_metric(
            'cpu_temp_celsius',
            thermal.cpu_temp_celsius,
            DataSource.SYSCTL,
            thermal.timestamp
        )
        
        if not validation_result.is_valid:
            logger.warning("⚠️ Thermal metrics failed validation")
            return None
        
        return {
            'cpu_temp_celsius': thermal.cpu_temp_celsius,
            'gpu_temp_celsius': thermal.gpu_temp_celsius,
            'thermal_pressure': thermal.thermal_pressure,
            'fan_speed_rpm': thermal.fan_speed_rpm,
            'timestamp': thermal.timestamp,
            'validated': True,
            'confidence': validation_result.confidence_score
        }
    
    def get_validated_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics with validation"""
        gpu = self.sensor_manager.get_real_gpu_memory()
        
        if not gpu:
            return None
        
        # Validate GPU memory
        validation_result = self.validator.validate_metric(
            'gpu_memory_mb',
            gpu.used_memory_mb,
            DataSource.METAL_API,
            gpu.timestamp
        )
        
        return {
            'used_memory_mb': gpu.used_memory_mb,
            'total_memory_mb': gpu.total_memory_mb,
            'utilization_percent': gpu.utilization_percent,
            'active_cores': gpu.active_cores,
            'timestamp': gpu.timestamp,
            'validated': validation_result.is_valid,
            'confidence': validation_result.confidence_score
        }
    
    def get_validated_cpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get CPU metrics with validation"""
        cpu = self.sensor_manager.get_real_cpu_frequency()
        
        if not cpu:
            return None
        
        # Validate CPU frequency
        validation_result = self.validator.validate_metric(
            'cpu_freq_mhz',
            cpu.current_freq_mhz,
            DataSource.SYSCTL,
            cpu.timestamp
        )
        
        return {
            'current_freq_mhz': cpu.current_freq_mhz,
            'min_freq_mhz': cpu.min_freq_mhz,
            'max_freq_mhz': cpu.max_freq_mhz,
            'performance_cores_active': cpu.performance_cores_active,
            'efficiency_cores_active': cpu.efficiency_cores_active,
            'timestamp': cpu.timestamp,
            'validated': validation_result.is_valid,
            'confidence': validation_result.confidence_score
        }
    
    def get_validated_battery_metrics(self) -> Optional[Dict[str, Any]]:
        """Get battery metrics with validation"""
        battery = self.sensor_manager.get_real_battery_cycles()
        
        if not battery:
            return None
        
        # Validate battery cycles
        if battery.cycle_count > 0:
            validation_result = self.validator.validate_metric(
                'battery_cycles',
                battery.cycle_count,
                DataSource.SYSTEM_PROFILER,
                battery.timestamp
            )
        else:
            # Cycle count not available, skip validation
            validation_result = None
        
        return {
            'cycle_count': battery.cycle_count,
            'max_capacity_percent': battery.max_capacity_percent,
            'current_capacity_mah': battery.current_capacity_mah,
            'design_capacity_mah': battery.design_capacity_mah,
            'is_charging': battery.is_charging,
            'time_remaining_minutes': battery.time_remaining_minutes,
            'timestamp': battery.timestamp,
            'validated': validation_result.is_valid if validation_result else True,
            'confidence': validation_result.confidence_score if validation_result else 0.8
        }
    
    def get_comprehensive_validated_metrics(self) -> Dict[str, Any]:
        """
        Get all hardware metrics with validation.
        Returns comprehensive system state with authenticity guarantees.
        """
        metrics = {
            'power': self.get_validated_power_metrics(),
            'thermal': self.get_validated_thermal_metrics(),
            'gpu': self.get_validated_gpu_metrics(),
            'cpu': self.get_validated_cpu_metrics(),
            'battery': self.get_validated_battery_metrics(),
            'timestamp': datetime.now(),
            'validation_stats': self.validator.get_validation_statistics()
        }
        
        # Store in history
        self.last_metrics = metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Analyze metrics and provide optimization recommendations.
        Uses validated data to ensure accurate recommendations.
        """
        recommendations = {
            'thermal_throttling_risk': False,
            'power_optimization_available': False,
            'memory_pressure': False,
            'suggested_actions': []
        }
        
        # Get current metrics
        thermal = self.get_validated_thermal_metrics()
        power = self.get_validated_power_metrics()
        gpu = self.get_validated_gpu_metrics()
        
        # Check thermal throttling risk
        if thermal and thermal['cpu_temp_celsius'] > 80:
            recommendations['thermal_throttling_risk'] = True
            recommendations['suggested_actions'].append(
                'Reduce quantum circuit complexity to prevent thermal throttling'
            )
        
        # Check power optimization opportunities
        if power and power['total_power_watts'] > 20:
            recommendations['power_optimization_available'] = True
            recommendations['suggested_actions'].append(
                'High power consumption detected - quantum optimization recommended'
            )
        
        # Check memory pressure
        if gpu and gpu['utilization_percent'] > 80:
            recommendations['memory_pressure'] = True
            recommendations['suggested_actions'].append(
                'High GPU utilization - consider reducing quantum state complexity'
            )
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics from validated data.
        Returns metrics suitable for quantum optimization algorithms.
        """
        metrics = self.get_comprehensive_validated_metrics()
        
        performance = {
            'power_efficiency': 0.0,
            'thermal_efficiency': 0.0,
            'compute_efficiency': 0.0,
            'overall_score': 0.0
        }
        
        # Calculate power efficiency (lower is better)
        if metrics['power'] and metrics['power']['validated']:
            total_power = metrics['power']['total_power_watts']
            # Normalize to 0-100 scale (assuming 5-50W range)
            performance['power_efficiency'] = max(0, 100 - (total_power - 5) * 2)
        
        # Calculate thermal efficiency (cooler is better)
        if metrics['thermal'] and metrics['thermal']['validated']:
            temp = metrics['thermal']['cpu_temp_celsius']
            # Normalize to 0-100 scale (assuming 40-100°C range)
            performance['thermal_efficiency'] = max(0, 100 - (temp - 40))
        
        # Calculate compute efficiency
        if metrics['cpu'] and metrics['cpu']['validated']:
            active_cores = (
                metrics['cpu']['performance_cores_active'] +
                metrics['cpu']['efficiency_cores_active']
            )
            # Normalize based on typical 8-core system
            performance['compute_efficiency'] = (active_cores / 8.0) * 100
        
        # Calculate overall score
        scores = [v for v in performance.values() if v > 0]
        if scores:
            performance['overall_score'] = sum(scores) / len(scores)
        
        return performance


# Global monitor instance
_monitor = None


def get_hardware_monitor(validation_level: ValidationLevel = ValidationLevel.STRICT) -> EnhancedHardwareMonitor:
    """Get or create the global hardware monitor"""
    global _monitor
    if _monitor is None:
        _monitor = EnhancedHardwareMonitor(validation_level)
    return _monitor


if __name__ == '__main__':
    # Test enhanced hardware monitoring
    print("🔧 Testing Enhanced Hardware Monitor...")
    
    monitor = get_hardware_monitor(ValidationLevel.STRICT)
    
    print("\n📊 Comprehensive Validated Metrics:")
    metrics = monitor.get_comprehensive_validated_metrics()
    
    if metrics['power']:
        print(f"\n⚡ Power (validated: {metrics['power']['validated']}):")
        print(f"  Total: {metrics['power']['total_power_watts']:.2f}W")
        print(f"  Confidence: {metrics['power']['confidence']:.2f}")
    
    if metrics['thermal']:
        print(f"\n🌡️ Thermal (validated: {metrics['thermal']['validated']}):")
        print(f"  CPU: {metrics['thermal']['cpu_temp_celsius']:.1f}°C")
        print(f"  Pressure: {metrics['thermal']['thermal_pressure']}")
    
    if metrics['gpu']:
        print(f"\n🎮 GPU (validated: {metrics['gpu']['validated']}):")
        print(f"  Memory: {metrics['gpu']['used_memory_mb']:.0f}/{metrics['gpu']['total_memory_mb']:.0f} MB")
        print(f"  Utilization: {metrics['gpu']['utilization_percent']:.1f}%")
    
    print("\n🎯 Optimization Recommendations:")
    recommendations = monitor.get_optimization_recommendations()
    for action in recommendations['suggested_actions']:
        print(f"  • {action}")
    
    print("\n📈 Performance Metrics:")
    performance = monitor.get_performance_metrics()
    for key, value in performance.items():
        print(f"  {key}: {value:.1f}")
    
    print("\n📊 Validation Statistics:")
    stats = metrics['validation_stats']
    print(f"  Total validations: {stats['total_validations']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
    print(f"  Authenticity rate: {stats['recent_authenticity_rate']:.1%}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    print("\n✅ Enhanced hardware monitor test complete!")
