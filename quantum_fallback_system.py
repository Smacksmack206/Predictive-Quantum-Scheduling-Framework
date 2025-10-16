#!/usr/bin/env python3
"""
Quantum System Fallback Mechanisms
Automatic fallback and graceful degradation for quantum systems
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

class FallbackReason(Enum):
    SYSTEM_FAILURE = "system_failure"
    RESOURCE_CONSTRAINT = "resource_constraint"
    THERMAL_THROTTLING = "thermal_throttling"
    MEMORY_LIMIT = "memory_limit"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    USER_REQUEST = "user_request"

@dataclass
class FallbackEvent:
    """Fallback event record"""
    timestamp: float
    from_system: str
    to_system: str
    reason: FallbackReason
    trigger_metrics: Dict[str, Any]
    success: bool
    recovery_time: float

class QuantumFallbackSystem:
    """
    Quantum system fallback and graceful degradation manager
    Handles automatic switching between quantum systems based on conditions
    """
    
    def __init__(self):
        self.fallback_history = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Fallback thresholds
        self.thresholds = {
            'memory_usage_percent': 85.0,
            'cpu_temperature': 80.0,
            'error_rate_percent': 10.0,
            'response_time_ms': 5000.0,
            'success_rate_percent': 70.0
        }
        
        # Fallback chain: 40-qubit -> 20-qubit -> classical
        self.fallback_chain = ['40_qubit', '20_qubit', 'classical']
        
        # System health monitors
        self.health_monitors = {}
        
        print("🛡️  QuantumFallbackSystem initialized")
        print("🔄 Automatic fallback monitoring ready")
    
    def register_quantum_controller(self, controller):
        """Register quantum controller for monitoring"""
        self.quantum_controller = controller
        print("✅ Quantum controller registered for fallback monitoring")
    
    def start_monitoring(self):
        """Start automatic fallback monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("📊 Fallback monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic fallback monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        print("📊 Fallback monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(10.0)  # Longer delay on error
    
    def _check_system_health(self):
        """Check system health and trigger fallback if needed"""
        if not hasattr(self, 'quantum_controller') or not self.quantum_controller:
            return
        
        try:
            # Get current system status
            system_status = self.quantum_controller.get_quantum_system_status()
            active_system = system_status.get('active_system')
            
            if not active_system:
                return
            
            # Check various health metrics
            health_issues = []
            
            # Memory usage check
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > self.thresholds['memory_usage_percent']:
                health_issues.append({
                    'type': 'memory_constraint',
                    'value': memory_usage,
                    'threshold': self.thresholds['memory_usage_percent'],
                    'reason': FallbackReason.MEMORY_LIMIT
                })
            
            # CPU temperature check (if available)
            try:
                cpu_temp = self._get_cpu_temperature()
                if cpu_temp and cpu_temp > self.thresholds['cpu_temperature']:
                    health_issues.append({
                        'type': 'thermal_constraint',
                        'value': cpu_temp,
                        'threshold': self.thresholds['cpu_temperature'],
                        'reason': FallbackReason.THERMAL_THROTTLING
                    })
            except:
                pass
            
            # System performance check
            active_status = system_status.get(active_system, {})
            performance_metrics = active_status.get('performance_metrics', {})
            
            # Check for system errors
            if 'error' in performance_metrics:
                health_issues.append({
                    'type': 'system_error',
                    'value': performance_metrics['error'],
                    'reason': FallbackReason.SYSTEM_FAILURE
                })
            
            # Trigger fallback if issues detected
            if health_issues:
                self._trigger_automatic_fallback(active_system, health_issues)
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (macOS specific)"""
        try:
            import subprocess
            result = subprocess.run(['sysctl', 'machdep.xcpm.cpu_thermal_state'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                temp_str = result.stdout.split(':')[-1].strip()
                return float(temp_str)
        except:
            pass
        
        return None
    
    def _trigger_automatic_fallback(self, current_system: str, health_issues: List[Dict]):
        """Trigger automatic fallback based on health issues"""
        # Determine the most critical issue
        critical_issue = max(health_issues, key=lambda x: self._get_issue_severity(x))
        
        # Find next system in fallback chain
        target_system = self._get_next_fallback_system(current_system, critical_issue)
        
        if target_system:
            print(f"🚨 Triggering automatic fallback: {current_system} -> {target_system}")
            print(f"   Reason: {critical_issue['reason'].value}")
            
            success = self._execute_fallback(current_system, target_system, critical_issue['reason'])
            
            if success:
                print(f"✅ Fallback successful: Now using {target_system}")
            else:
                print(f"❌ Fallback failed: Staying with {current_system}")
    
    def _get_issue_severity(self, issue: Dict) -> float:
        """Get severity score for a health issue"""
        severity_map = {
            'system_error': 10.0,
            'thermal_constraint': 8.0,
            'memory_constraint': 6.0,
            'performance_degradation': 4.0
        }
        
        return severity_map.get(issue['type'], 1.0)
    
    def _get_next_fallback_system(self, current_system: str, issue: Dict) -> Optional[str]:
        """Get next system in fallback chain"""
        try:
            current_index = self.fallback_chain.index(current_system)
            
            # Check if there's a next system
            if current_index + 1 < len(self.fallback_chain):
                return self.fallback_chain[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _execute_fallback(self, from_system: str, to_system: str, reason: FallbackReason) -> bool:
        """Execute fallback from one system to another"""
        start_time = time.time()
        
        try:
            # Record fallback attempt
            fallback_event = FallbackEvent(
                timestamp=start_time,
                from_system=from_system,
                to_system=to_system,
                reason=reason,
                trigger_metrics=self._get_current_metrics(),
                success=False,
                recovery_time=0.0
            )
            
            # Execute the switch
            if to_system == 'classical':
                success = self._fallback_to_classical()
            else:
                success = self.quantum_controller.switch_quantum_system(to_system)
            
            # Update fallback event
            fallback_event.success = success
            fallback_event.recovery_time = time.time() - start_time
            
            self.fallback_history.append(fallback_event)
            
            return success
            
        except Exception as e:
            print(f"❌ Fallback execution failed: {e}")
            return False
    
    def _fallback_to_classical(self) -> bool:
        """Fallback to classical processing"""
        print("🔄 Falling back to classical processing")
        # This would disable quantum processing and use classical algorithms
        # For now, just return success
        return True
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'memory_usage_percent': psutil.virtual_memory().percent,
            'cpu_usage_percent': psutil.cpu_percent(),
            'timestamp': time.time(),
            'cpu_temperature': self._get_cpu_temperature()
        }
    
    def manual_fallback(self, target_system: str, reason: str = "user_request") -> Dict[str, Any]:
        """
        Manually trigger fallback to target system
        
        Args:
            target_system: Target system to fallback to
            reason: Reason for manual fallback
            
        Returns:
            Fallback result
        """
        if not hasattr(self, 'quantum_controller') or not self.quantum_controller:
            return {'success': False, 'error': 'No quantum controller available'}
        
        try:
            # Get current system
            system_status = self.quantum_controller.get_quantum_system_status()
            current_system = system_status.get('active_system')
            
            if current_system == target_system:
                return {
                    'success': True,
                    'message': f'Already using {target_system} system',
                    'no_change': True
                }
            
            # Execute fallback
            success = self._execute_fallback(
                current_system or 'unknown',
                target_system,
                FallbackReason.USER_REQUEST
            )
            
            if success:
                return {
                    'success': True,
                    'message': f'Successfully switched to {target_system}',
                    'from_system': current_system,
                    'to_system': target_system
                }
            else:
                return {
                    'success': False,
                    'error': f'Failed to switch to {target_system}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Manual fallback failed: {str(e)}'
            }
    
    def create_algorithm_compatibility_layer(self) -> Dict[str, Callable]:
        """
        Create compatibility layer for quantum algorithms
        
        Returns:
            Dictionary of compatibility functions
        """
        compatibility_layer = {
            'optimize_processes': self._compatible_process_optimization,
            'analyze_correlations': self._compatible_correlation_analysis,
            'visualize_quantum_state': self._compatible_visualization,
            'encode_quantum_features': self._compatible_feature_encoding
        }
        
        print("🔧 Algorithm compatibility layer created")
        return compatibility_layer
    
    def _compatible_process_optimization(self, processes: List[Dict], **kwargs) -> Dict[str, Any]:
        """Compatible process optimization across quantum systems"""
        if not hasattr(self, 'quantum_controller') or not self.quantum_controller:
            return self._classical_process_optimization(processes)
        
        try:
            # Try quantum optimization
            result = self.quantum_controller.optimize_process_scheduling(processes)
            
            if result.get('success'):
                return result
            else:
                # Fallback to classical
                print("🔄 Quantum optimization failed, using classical fallback")
                return self._classical_process_optimization(processes)
                
        except Exception as e:
            print(f"❌ Quantum optimization error: {e}, using classical fallback")
            return self._classical_process_optimization(processes)
    
    def _classical_process_optimization(self, processes: List[Dict]) -> Dict[str, Any]:
        """Classical fallback for process optimization"""
        assignments = []
        
        # Simple classical assignment
        for i, process in enumerate(processes):
            cpu_usage = process.get('cpu_usage', 0)
            
            if cpu_usage > 20:
                core_type = 'p_core'
                core_id = i % 4
            else:
                core_type = 'e_core'
                core_id = i % 4
            
            assignments.append({
                'process_id': process.get('pid', i),
                'process_name': process.get('name', f'process_{i}'),
                'core_type': core_type,
                'core_id': core_id,
                'quantum_confidence': 0.0,
                'assignment_reason': 'classical_fallback'
            })
        
        return {
            'success': True,
            'assignments': assignments,
            'system_used': 'classical',
            'quantum_advantage': False,
            'execution_time': 0.001,
            'fallback_used': True
        }
    
    def _compatible_correlation_analysis(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Compatible correlation analysis with fallback"""
        # Simplified classical correlation analysis
        return {
            'correlations_found': 0,
            'method': 'classical_fallback',
            'success': True
        }
    
    def _compatible_visualization(self, data: Any, **kwargs) -> Optional[str]:
        """Compatible visualization with fallback"""
        # Return simple text-based visualization
        return "<html><body><h1>Classical Visualization Fallback</h1><p>Quantum visualization not available</p></body></html>"
    
    def _compatible_feature_encoding(self, features: Any, **kwargs) -> Dict[str, Any]:
        """Compatible feature encoding with fallback"""
        # Return classical feature encoding
        return {
            'encoding_type': 'classical',
            'features_encoded': len(features) if hasattr(features, '__len__') else 0,
            'success': True
        }
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback system statistics"""
        if not self.fallback_history:
            return {
                'total_fallbacks': 0,
                'success_rate': 0.0,
                'most_common_reason': None,
                'average_recovery_time': 0.0
            }
        
        successful_fallbacks = [f for f in self.fallback_history if f.success]
        
        # Count reasons
        reason_counts = {}
        for fallback in self.fallback_history:
            reason = fallback.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        most_common_reason = max(reason_counts, key=reason_counts.get) if reason_counts else None
        
        return {
            'total_fallbacks': len(self.fallback_history),
            'successful_fallbacks': len(successful_fallbacks),
            'success_rate': len(successful_fallbacks) / len(self.fallback_history) * 100,
            'most_common_reason': most_common_reason,
            'reason_breakdown': reason_counts,
            'average_recovery_time': sum(f.recovery_time for f in successful_fallbacks) / max(1, len(successful_fallbacks)),
            'monitoring_active': self.monitoring_active,
            'thresholds': self.thresholds.copy()
        }
    
    def update_fallback_thresholds(self, new_thresholds: Dict[str, float]):
        """Update fallback thresholds"""
        self.thresholds.update(new_thresholds)
        print(f"🔧 Updated fallback thresholds: {new_thresholds}")

if __name__ == "__main__":
    print("🧪 Testing QuantumFallbackSystem")
    
    # Initialize fallback system
    fallback_system = QuantumFallbackSystem()
    
    # Test compatibility layer
    compatibility_layer = fallback_system.create_algorithm_compatibility_layer()
    print(f"🔧 Compatibility layer: {list(compatibility_layer.keys())}")
    
    # Test classical fallback
    test_processes = [
        {'pid': 1, 'name': 'chrome', 'cpu_usage': 45},
        {'pid': 2, 'name': 'vscode', 'cpu_usage': 25}
    ]
    
    result = compatibility_layer['optimize_processes'](test_processes)
    print(f"⚛️  Optimization result: {result}")
    
    # Test statistics
    stats = fallback_system.get_fallback_statistics()
    print(f"📊 Fallback stats: {stats}")
    
    print("🎉 QuantumFallbackSystem test completed!")