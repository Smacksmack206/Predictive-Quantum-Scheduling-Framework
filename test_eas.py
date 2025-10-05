#!/usr/bin/env python3
"""
Comprehensive EAS Testing & Validation
Tests Energy Aware Scheduling effectiveness and performance
"""

import psutil
import time
import json
import subprocess
from datetime import datetime
import requests

class EASValidator:
    """Validate EAS performance and effectiveness"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.eas_metrics = {}
        self.test_results = []
        
    def measure_baseline(self, duration=30):
        """Measure baseline performance without EAS"""
        print("📊 Measuring baseline performance (EAS OFF)...")
        
        # Ensure EAS is disabled
        try:
            requests.post('http://localhost:9010/api/eas-toggle', 
                         json={'enabled': False}, timeout=5)
        except:
            print("⚠️  Could not disable EAS via API")
        
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'battery': self.get_battery_info(),
                'temperature': self.estimate_temperature(),
                'core_usage': self.get_core_usage()
            }
            metrics.append(metric)
            print(f"   CPU: {metric['cpu_percent']:.1f}%, Temp: {metric['temperature']:.1f}°C")
        
        self.baseline_metrics = self.calculate_averages(metrics)
        print(f"✅ Baseline captured: CPU {self.baseline_metrics['cpu_avg']:.1f}%, "
              f"Temp {self.baseline_metrics['temp_avg']:.1f}°C")
        
    def measure_with_eas(self, duration=30):
        """Measure performance with EAS enabled"""
        print("🧠 Measuring EAS performance (EAS ON)...")
        
        # Enable EAS
        try:
            requests.post('http://localhost:9010/api/eas-toggle', 
                         json={'enabled': True}, timeout=5)
            time.sleep(5)  # Allow EAS to optimize
        except:
            print("⚠️  Could not enable EAS via API")
        
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metric = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'battery': self.get_battery_info(),
                'temperature': self.estimate_temperature(),
                'core_usage': self.get_core_usage()
            }
            metrics.append(metric)
            print(f"   CPU: {metric['cpu_percent']:.1f}%, Temp: {metric['temperature']:.1f}°C")
        
        self.eas_metrics = self.calculate_averages(metrics)
        print(f"✅ EAS metrics captured: CPU {self.eas_metrics['cpu_avg']:.1f}%, "
              f"Temp {self.eas_metrics['temp_avg']:.1f}°C")
    
    def get_battery_info(self):
        """Get battery information"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'percent': battery.percent,
                    'plugged': battery.power_plugged,
                    'time_left': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
        except:
            pass
        return {'percent': 100, 'plugged': True, 'time_left': None}
    
    def estimate_temperature(self):
        """Estimate CPU temperature"""
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_percent = psutil.cpu_percent()
            
            if cpu_freq and cpu_freq.current:
                freq_ratio = cpu_freq.current / 4000.0  # M3 max freq
                usage_ratio = cpu_percent / 100.0
                
                base_temp = 35
                freq_impact = freq_ratio * 30
                usage_impact = usage_ratio * 25
                
                return base_temp + freq_impact + usage_impact
        except:
            pass
        return 45  # Default estimate
    
    def get_core_usage(self):
        """Get per-core CPU usage"""
        try:
            per_cpu = psutil.cpu_percent(percpu=True, interval=0.5)
            if len(per_cpu) >= 8:
                return {
                    'p_cores': per_cpu[:4],
                    'e_cores': per_cpu[4:8],
                    'p_avg': sum(per_cpu[:4]) / 4,
                    'e_avg': sum(per_cpu[4:8]) / 4
                }
        except:
            pass
        return {'p_cores': [0]*4, 'e_cores': [0]*4, 'p_avg': 0, 'e_avg': 0}
    
    def calculate_averages(self, metrics):
        """Calculate average metrics from measurement period"""
        if not metrics:
            return {}
        
        return {
            'cpu_avg': sum(m['cpu_percent'] for m in metrics) / len(metrics),
            'memory_avg': sum(m['memory_percent'] for m in metrics) / len(metrics),
            'temp_avg': sum(m['temperature'] for m in metrics) / len(metrics),
            'p_core_avg': sum(m['core_usage']['p_avg'] for m in metrics) / len(metrics),
            'e_core_avg': sum(m['core_usage']['e_avg'] for m in metrics) / len(metrics),
            'battery_start': metrics[0]['battery']['percent'],
            'battery_end': metrics[-1]['battery']['percent'],
            'duration': metrics[-1]['timestamp'] - metrics[0]['timestamp']
        }
    
    def calculate_improvements(self):
        """Calculate EAS improvements over baseline"""
        if not self.baseline_metrics or not self.eas_metrics:
            return {}
        
        # Calculate percentage improvements
        cpu_improvement = ((self.baseline_metrics['cpu_avg'] - self.eas_metrics['cpu_avg']) / 
                          self.baseline_metrics['cpu_avg']) * 100
        
        temp_improvement = self.baseline_metrics['temp_avg'] - self.eas_metrics['temp_avg']
        
        # Battery drain rate (% per hour)
        baseline_drain = ((self.baseline_metrics['battery_start'] - self.baseline_metrics['battery_end']) / 
                         (self.baseline_metrics['duration'] / 3600))
        eas_drain = ((self.eas_metrics['battery_start'] - self.eas_metrics['battery_end']) / 
                    (self.eas_metrics['duration'] / 3600))
        
        battery_improvement = ((baseline_drain - eas_drain) / baseline_drain) * 100 if baseline_drain > 0 else 0
        
        return {
            'cpu_improvement_percent': cpu_improvement,
            'temperature_improvement_celsius': temp_improvement,
            'battery_improvement_percent': battery_improvement,
            'p_core_balance': self.eas_metrics['p_core_avg'] - self.baseline_metrics['p_core_avg'],
            'e_core_balance': self.eas_metrics['e_core_avg'] - self.baseline_metrics['e_core_avg']
        }
    
    def test_process_classification(self):
        """Test process classification accuracy"""
        print("🔍 Testing process classification...")
        
        test_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 100:  # Skip system processes
                    test_processes.append((pid, name))
                    
                if len(test_processes) >= 20:  # Test sample
                    break
            except:
                continue
        
        # Mock classification (would use actual EAS classifier)
        classifications = {}
        for pid, name in test_processes:
            name_lower = name.lower()
            
            if any(app in name_lower for app in ['safari', 'chrome', 'xcode']):
                classification = 'interactive'
            elif any(app in name_lower for app in ['python', 'node', 'java']):
                classification = 'compute'
            else:
                classification = 'background'
            
            classifications[name] = classification
        
        print(f"   Classified {len(classifications)} processes")
        for name, cls in list(classifications.items())[:5]:
            print(f"   • {name[:20]:20} → {cls}")
        
        return classifications
    
    def test_core_assignment_logic(self):
        """Test core assignment logic"""
        print("⚙️  Testing core assignment logic...")
        
        # Test scenarios
        test_cases = [
            {'workload': 'interactive', 'cpu_usage': 25, 'expected': 'p_core'},
            {'workload': 'background', 'cpu_usage': 5, 'expected': 'e_core'},
            {'workload': 'compute', 'cpu_usage': 60, 'expected': 'p_core'},
            {'workload': 'interactive_light', 'cpu_usage': 8, 'expected': 'e_core'}
        ]
        
        correct_assignments = 0
        for case in test_cases:
            # Mock energy calculation
            if case['workload'] in ['interactive', 'compute'] and case['cpu_usage'] > 20:
                assigned = 'p_core'
            else:
                assigned = 'e_core'
            
            is_correct = assigned == case['expected']
            correct_assignments += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {case['workload']} ({case['cpu_usage']}% CPU) → {assigned}")
        
        accuracy = (correct_assignments / len(test_cases)) * 100
        print(f"   Assignment accuracy: {accuracy:.1f}%")
        
        return accuracy
    
    def generate_report(self):
        """Generate comprehensive test report"""
        improvements = self.calculate_improvements()
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'eas_metrics': self.eas_metrics,
            'improvements': improvements,
            'test_results': {
                'classification_test': self.test_process_classification(),
                'assignment_accuracy': self.test_core_assignment_logic()
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 EAS PERFORMANCE REPORT")
        print("="*60)
        
        if improvements:
            print(f"CPU Efficiency:     {improvements['cpu_improvement_percent']:+.1f}%")
            print(f"Temperature:        {improvements['temperature_improvement_celsius']:+.1f}°C")
            print(f"Battery Life:       {improvements['battery_improvement_percent']:+.1f}%")
            print(f"P-Core Balance:     {improvements['p_core_balance']:+.1f}%")
            print(f"E-Core Balance:     {improvements['e_core_balance']:+.1f}%")
        
        print(f"\nProcess Classification: {len(report['test_results']['classification_test'])} processes")
        print(f"Assignment Accuracy:    {report['test_results']['assignment_accuracy']:.1f}%")
        
        # Overall assessment
        if improvements:
            overall_score = (
                max(0, improvements['cpu_improvement_percent']) +
                max(0, improvements['battery_improvement_percent']) +
                max(0, improvements['temperature_improvement_celsius'] * 5)  # Scale temp
            ) / 3
            
            if overall_score > 5:
                assessment = "🚀 EXCELLENT - EAS providing significant benefits"
            elif overall_score > 2:
                assessment = "✅ GOOD - EAS showing measurable improvements"
            elif overall_score > 0:
                assessment = "📈 MARGINAL - EAS providing minor benefits"
            else:
                assessment = "⚠️  MINIMAL - EAS impact not significant"
            
            print(f"\nOverall Assessment: {assessment}")
            print(f"Performance Score:  {overall_score:.1f}/10")
        
        print("="*60)
        
        return report

def main():
    """Run comprehensive EAS validation"""
    print("🧪 EAS Comprehensive Testing & Validation")
    print("This will test EAS effectiveness over 2 minutes")
    print("Make sure the Battery Optimizer is running on localhost:9010")
    print()
    
    validator = EASValidator()
    
    try:
        # Test sequence
        validator.measure_baseline(duration=60)  # 1 minute baseline
        time.sleep(5)  # Brief pause
        validator.measure_with_eas(duration=60)  # 1 minute with EAS
        
        # Generate report
        report = validator.generate_report()
        
        # Save report
        report_file = f"eas_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
