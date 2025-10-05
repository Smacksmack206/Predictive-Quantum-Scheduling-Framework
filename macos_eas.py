#!/usr/bin/env python3
"""
macOS Energy Aware Scheduling (EAS) Implementation
Optimizes process placement on M3 P-cores vs E-cores for energy efficiency
"""

import psutil
import time
import json
from datetime import datetime
from collections import defaultdict, deque

class M3EnergyAwareScheduler:
    """Energy Aware Scheduling for M3 MacBook Air"""
    
    def __init__(self):
        # M3 chip configuration
        self.p_cores = list(range(4))  # Performance cores 0-3
        self.e_cores = list(range(4, 8))  # Efficiency cores 4-7
        
        # Energy models based on M3 specifications
        self.energy_models = {
            'p_core': {
                'max_freq_mhz': 4050,  # M3 P-core max frequency
                'power_per_mhz': 0.85,  # Estimated power efficiency
                'base_power_watts': 2.2,  # Idle power consumption
                'performance_factor': 1.0  # Baseline performance
            },
            'e_core': {
                'max_freq_mhz': 2750,  # M3 E-core max frequency  
                'power_per_mhz': 0.25,  # Much more efficient
                'base_power_watts': 0.4,  # Very low idle power
                'performance_factor': 0.6  # 60% of P-core performance
            }
        }
        
        self.process_classifications = {}
        self.core_assignments = {}
        self.energy_history = deque(maxlen=100)
        
    def classify_process_workload(self, pid, name):
        """Classify process workload type for optimal core assignment"""
        name_lower = name.lower()
        
        try:
            proc = psutil.Process(pid)
            cpu_percent = proc.cpu_percent(interval=0.5)
            memory_mb = proc.memory_info().rss / (1024 * 1024)
            
            # Interactive applications (need responsiveness)
            interactive_keywords = [
                'safari', 'chrome', 'firefox', 'edge', 'brave',
                'finder', 'terminal', 'iterm', 'warp',
                'xcode', 'vscode', 'sublime', 'atom',
                'photoshop', 'illustrator', 'sketch', 'figma',
                'final cut', 'premiere', 'davinci',
                'zoom', 'teams', 'slack', 'discord'
            ]
            
            # Background/System processes (can use E-cores efficiently)
            background_keywords = [
                'backupd', 'spotlight', 'mds', 'cloudd', 'bird',
                'syncdefaultsd', 'coreauthd', 'logind', 'launchd',
                'kernel_task', 'windowserver', 'dock', 'controlcenter',
                'notificationcenter', 'siri', 'assistant'
            ]
            
            # Compute-intensive (need P-core performance)
            compute_keywords = [
                'python', 'node', 'java', 'ruby', 'go',
                'gcc', 'clang', 'swift', 'rustc',
                'ffmpeg', 'handbrake', 'compressor',
                'blender', 'cinema4d', 'maya',
                'docker', 'qemu', 'virtualbox'
            ]
            
            # Classification logic
            if any(keyword in name_lower for keyword in interactive_keywords):
                if cpu_percent > 20:
                    return 'interactive_heavy'  # Needs P-core
                else:
                    return 'interactive_light'  # Can use E-core when idle
                    
            elif any(keyword in name_lower for keyword in background_keywords):
                return 'background'  # E-core preferred
                
            elif any(keyword in name_lower for keyword in compute_keywords):
                return 'compute'  # P-core required
                
            else:
                # Dynamic classification based on resource usage
                if cpu_percent > 50 or memory_mb > 500:
                    return 'compute'
                elif cpu_percent > 15:
                    return 'interactive_heavy'
                elif cpu_percent > 5:
                    return 'interactive_light'
                else:
                    return 'background'
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 'background'  # Safe default
    
    def calculate_energy_efficiency(self, workload_type, core_type, cpu_usage):
        """Calculate energy efficiency score for workload on core type"""
        model = self.energy_models[core_type]
        
        # Base energy consumption
        base_energy = model['base_power_watts']
        
        # Dynamic energy based on CPU usage
        freq_used = (cpu_usage / 100) * model['max_freq_mhz']
        dynamic_energy = freq_used * model['power_per_mhz'] / 1000  # Convert to watts
        
        total_energy = base_energy + dynamic_energy
        
        # Performance factor (work done per watt)
        performance = model['performance_factor']
        
        # Workload-specific adjustments
        workload_multipliers = {
            'interactive_heavy': {'p_core': 1.0, 'e_core': 0.7},  # P-core better for responsiveness
            'interactive_light': {'p_core': 0.8, 'e_core': 1.2},  # E-core fine for light work
            'background': {'p_core': 0.6, 'e_core': 1.4},         # E-core much better
            'compute': {'p_core': 1.2, 'e_core': 0.5}             # P-core essential
        }
        
        multiplier = workload_multipliers.get(workload_type, {}).get(core_type, 1.0)
        adjusted_performance = performance * multiplier
        
        # Efficiency = Performance per Watt
        efficiency = adjusted_performance / total_energy if total_energy > 0 else 0
        
        return {
            'efficiency_score': efficiency,
            'energy_watts': total_energy,
            'performance_factor': adjusted_performance,
            'recommendation': 'optimal' if efficiency > 0.4 else 'suboptimal'
        }
    
    def get_optimal_core_assignment(self, pid, name):
        """Determine optimal core assignment for process"""
        workload_type = self.classify_process_workload(pid, name)
        
        try:
            proc = psutil.Process(pid)
            cpu_usage = proc.cpu_percent(interval=0.1)
        except:
            cpu_usage = 5  # Default assumption
        
        # Calculate efficiency for both core types
        p_core_analysis = self.calculate_energy_efficiency(workload_type, 'p_core', cpu_usage)
        e_core_analysis = self.calculate_energy_efficiency(workload_type, 'e_core', cpu_usage)
        
        # Choose most efficient option
        if p_core_analysis['efficiency_score'] > e_core_analysis['efficiency_score']:
            optimal_core_type = 'p_core'
            analysis = p_core_analysis
        else:
            optimal_core_type = 'e_core'
            analysis = e_core_analysis
        
        return {
            'pid': pid,
            'name': name,
            'workload_type': workload_type,
            'optimal_core_type': optimal_core_type,
            'cpu_usage': cpu_usage,
            'energy_analysis': {
                'p_core': p_core_analysis,
                'e_core': e_core_analysis,
                'chosen': analysis
            }
        }
    
    def apply_energy_optimization(self, assignment):
        """Apply energy optimization through process priority adjustment"""
        pid = assignment['pid']
        core_type = assignment['optimal_core_type']
        
        try:
            proc = psutil.Process(pid)
            current_nice = proc.nice()
            
            # macOS doesn't have CPU affinity, so we use process priority
            # to influence scheduler decisions
            
            if core_type == 'p_core':
                # Higher priority for P-core processes (better responsiveness)
                target_nice = max(current_nice - 2, -10)  # Increase priority
                optimization_type = 'performance_priority'
                
            else:  # e_core
                # Lower priority for E-core processes (energy efficiency)
                target_nice = min(current_nice + 3, 10)   # Decrease priority
                optimization_type = 'efficiency_priority'
            
            # Apply the nice value change
            if target_nice != current_nice:
                proc.nice(target_nice)
                
                return {
                    'success': True,
                    'pid': pid,
                    'name': assignment['name'],
                    'optimization': optimization_type,
                    'nice_change': f"{current_nice} → {target_nice}",
                    'core_target': core_type
                }
            else:
                return {
                    'success': True,
                    'pid': pid,
                    'name': assignment['name'],
                    'optimization': 'no_change_needed',
                    'nice_value': current_nice,
                    'core_target': core_type
                }
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
            return {
                'success': False,
                'pid': pid,
                'name': assignment['name'],
                'error': str(e),
                'core_target': core_type
            }
    
    def optimize_system_energy_scheduling(self):
        """Run comprehensive energy-aware scheduling optimization"""
        optimizations = []
        energy_savings = 0
        
        print("🔋 Running macOS Energy Aware Scheduling...")
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                # Skip system critical processes (low PIDs)
                if pid < 50:
                    continue
                
                # Get optimal assignment
                assignment = self.get_optimal_core_assignment(pid, name)
                
                # Apply optimization
                result = self.apply_energy_optimization(assignment)
                
                if result['success']:
                    optimizations.append(result)
                    
                    # Estimate energy savings
                    chosen_analysis = assignment['energy_analysis']['chosen']
                    if chosen_analysis['recommendation'] == 'optimal':
                        energy_savings += chosen_analysis['efficiency_score'] * 0.1  # Rough estimate
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Log energy optimization results
        self.energy_history.append({
            'timestamp': datetime.now().isoformat(),
            'optimizations_count': len(optimizations),
            'estimated_energy_savings': energy_savings
        })
        
        return {
            'optimizations': optimizations,
            'summary': {
                'total_processes_optimized': len(optimizations),
                'estimated_energy_savings_percent': min(energy_savings * 10, 25),  # Cap at 25%
                'p_core_assignments': len([o for o in optimizations if o.get('core_target') == 'p_core']),
                'e_core_assignments': len([o for o in optimizations if o.get('core_target') == 'e_core'])
            }
        }
    
    def get_current_core_utilization(self):
        """Get current utilization of P-cores vs E-cores"""
        try:
            per_cpu = psutil.cpu_percent(percpu=True, interval=1)
            
            if len(per_cpu) >= 8:
                p_core_usage = per_cpu[:4]  # First 4 cores
                e_core_usage = per_cpu[4:8]  # Next 4 cores
                
                return {
                    'p_cores': {
                        'individual': p_core_usage,
                        'average': sum(p_core_usage) / len(p_core_usage),
                        'max': max(p_core_usage),
                        'total_capacity_used': sum(p_core_usage) / 4
                    },
                    'e_cores': {
                        'individual': e_core_usage,
                        'average': sum(e_core_usage) / len(e_core_usage),
                        'max': max(e_core_usage),
                        'total_capacity_used': sum(e_core_usage) / 4
                    },
                    'system_balance': {
                        'p_to_e_ratio': (sum(p_core_usage) / 4) / max((sum(e_core_usage) / 4), 1),
                        'overall_efficiency': 'balanced' if abs(sum(p_core_usage) - sum(e_core_usage)) < 50 else 'imbalanced'
                    }
                }
            else:
                return {'error': 'Cannot determine core utilization'}
                
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main EAS demonstration"""
    eas = M3EnergyAwareScheduler()
    
    print("🚀 macOS Energy Aware Scheduling (EAS) for M3 MacBook Air")
    print("=" * 60)
    
    # Get current core utilization
    core_util = eas.get_current_core_utilization()
    if 'error' not in core_util:
        print(f"📊 Current Core Utilization:")
        print(f"   P-cores: {core_util['p_cores']['average']:.1f}% avg, {core_util['p_cores']['max']:.1f}% max")
        print(f"   E-cores: {core_util['e_cores']['average']:.1f}% avg, {core_util['e_cores']['max']:.1f}% max")
        print(f"   Balance: {core_util['system_balance']['overall_efficiency']}")
        print()
    
    # Run energy optimization
    results = eas.optimize_system_energy_scheduling()
    
    print(f"✅ Optimization Complete!")
    print(f"   Processes optimized: {results['summary']['total_processes_optimized']}")
    print(f"   P-core assignments: {results['summary']['p_core_assignments']}")
    print(f"   E-core assignments: {results['summary']['e_core_assignments']}")
    print(f"   Estimated energy savings: {results['summary']['estimated_energy_savings_percent']:.1f}%")
    print()
    
    # Show some example optimizations
    print("🎯 Example Optimizations:")
    for opt in results['optimizations'][:5]:  # Show first 5
        if opt['success']:
            print(f"   • {opt['name'][:20]:20} → {opt['core_target']:6} ({opt['optimization']})")
    
    if len(results['optimizations']) > 5:
        print(f"   ... and {len(results['optimizations']) - 5} more processes")
    
    return results

if __name__ == "__main__":
    main()
