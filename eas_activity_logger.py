#!/usr/bin/env python3
"""
Enhanced EAS Activity Logger
Logs detailed activity to show exactly what the Enhanced EAS is doing
"""

import time
import requests
import json
import os
from datetime import datetime
from collections import defaultdict

class EASActivityLogger:
    """Logs detailed EAS activity for analysis"""
    
    def __init__(self, log_file="eas_activity.log"):
        self.log_file = log_file
        self.previous_state = {}
        self.activity_counts = defaultdict(int)
        
    def log_activity(self, message, level="INFO"):
        """Log activity with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def get_current_state(self):
        """Get current EAS state"""
        try:
            # Get all relevant data
            learning_response = requests.get('http://localhost:9010/api/eas-learning-stats', timeout=10)
            eas_response = requests.get('http://localhost:9010/api/eas-status', timeout=10)
            insights_response = requests.get('http://localhost:9010/api/eas-insights', timeout=10)
            
            return {
                'learning': learning_response.json() if learning_response.status_code == 200 else {},
                'eas': eas_response.json() if eas_response.status_code == 200 else {},
                'insights': insights_response.json() if insights_response.status_code == 200 else {},
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.log_activity(f"Error getting current state: {e}", "ERROR")
            return None
    
    def analyze_changes(self, current_state):
        """Analyze what changed since last check"""
        if not self.previous_state:
            self.log_activity("Starting EAS activity monitoring", "INFO")
            self.previous_state = current_state
            return
        
        changes = []
        
        # Check learning changes
        if 'learning' in current_state and 'learning' in self.previous_state:
            current_classifications = current_state['learning'].get('total_classifications', 0)
            previous_classifications = self.previous_state['learning'].get('total_classifications', 0)
            
            if current_classifications > previous_classifications:
                new_classifications = current_classifications - previous_classifications
                changes.append(f"LEARNING: +{new_classifications:,} new classifications (total: {current_classifications:,})")
                self.activity_counts['new_classifications'] += new_classifications
            
            # Check confidence changes
            current_confidence = current_state['learning'].get('average_confidence', 0)
            previous_confidence = self.previous_state['learning'].get('average_confidence', 0)
            
            if abs(current_confidence - previous_confidence) > 0.001:
                confidence_change = current_confidence - previous_confidence
                changes.append(f"CONFIDENCE: {confidence_change:+.4f} (now: {current_confidence:.4f})")
        
        # Check EAS optimization changes
        if 'eas' in current_state and 'eas' in self.previous_state:
            current_optimized = current_state['eas'].get('processes_optimized', 0)
            previous_optimized = self.previous_state['eas'].get('processes_optimized', 0)
            
            if current_optimized != previous_optimized:
                diff = current_optimized - previous_optimized
                if diff > 0:
                    changes.append(f"OPTIMIZATION: +{diff} processes added (total: {current_optimized})")
                    self.activity_counts['processes_added'] += diff
                else:
                    changes.append(f"OPTIMIZATION: {abs(diff)} processes removed (total: {current_optimized})")
                    self.activity_counts['processes_removed'] += abs(diff)
            
            # Check performance score changes
            current_performance = current_state['eas'].get('performance_score', 0)
            previous_performance = self.previous_state['eas'].get('performance_score', 0)
            
            if abs(current_performance - previous_performance) > 0.5:
                perf_change = current_performance - previous_performance
                changes.append(f"PERFORMANCE: {perf_change:+.1f} (now: {current_performance:.1f}/100)")
            
            # Check core utilization changes
            current_core = current_state['eas'].get('core_utilization', {})
            previous_core = self.previous_state['eas'].get('core_utilization', {})
            
            if current_core and previous_core:
                p_core_change = current_core.get('p_avg', 0) - previous_core.get('p_avg', 0)
                e_core_change = current_core.get('e_avg', 0) - previous_core.get('e_avg', 0)
                
                if abs(p_core_change) > 5 or abs(e_core_change) > 5:
                    changes.append(f"CORES: P-cores {p_core_change:+.1f}%, E-cores {e_core_change:+.1f}%")
        
        # Check process assignment changes
        if 'eas' in current_state and 'eas' in self.previous_state:
            current_assignments = current_state['eas'].get('process_assignments', [])
            previous_assignments = self.previous_state['eas'].get('process_assignments', [])
            
            # Count assignments by core type
            current_p_cores = len([a for a in current_assignments if a.get('core_type') == 'p_core'])
            current_e_cores = len([a for a in current_assignments if a.get('core_type') == 'e_core'])
            
            previous_p_cores = len([a for a in previous_assignments if a.get('core_type') == 'p_core'])
            previous_e_cores = len([a for a in previous_assignments if a.get('core_type') == 'e_core'])
            
            if current_p_cores != previous_p_cores or current_e_cores != previous_e_cores:
                p_diff = current_p_cores - previous_p_cores
                e_diff = current_e_cores - previous_e_cores
                changes.append(f"ASSIGNMENTS: P-cores {p_diff:+d}, E-cores {e_diff:+d}")
        
        # Log all changes
        for change in changes:
            self.log_activity(change, "ACTIVITY")
        
        if not changes:
            self.log_activity("No significant changes detected", "DEBUG")
        
        self.previous_state = current_state
    
    def log_detailed_state(self, current_state):
        """Log detailed current state"""
        if not current_state:
            return
        
        self.log_activity("=== DETAILED STATE SNAPSHOT ===", "INFO")
        
        # Learning state
        if 'learning' in current_state:
            learning = current_state['learning']
            self.log_activity(f"Learning Engine:", "INFO")
            self.log_activity(f"  Total Classifications: {learning.get('total_classifications', 0):,}", "INFO")
            self.log_activity(f"  Average Confidence: {learning.get('average_confidence', 0):.4f}", "INFO")
            
            recent_classifications = learning.get('recent_classifications', [])
            if recent_classifications:
                self.log_activity(f"  Classification Types:", "INFO")
                for cls_name, count, confidence in recent_classifications[:5]:
                    self.log_activity(f"    • {cls_name}: {count:,} times (conf: {confidence:.3f})", "INFO")
        
        # EAS state
        if 'eas' in current_state:
            eas = current_state['eas']
            self.log_activity(f"EAS Engine:", "INFO")
            self.log_activity(f"  Enabled: {eas.get('enabled', False)}", "INFO")
            self.log_activity(f"  Processes Optimized: {eas.get('processes_optimized', 0)}", "INFO")
            self.log_activity(f"  Performance Score: {eas.get('performance_score', 0):.1f}/100", "INFO")
            
            core_util = eas.get('core_utilization', {})
            if core_util:
                self.log_activity(f"  P-Core Usage: {core_util.get('p_avg', 0):.1f}%", "INFO")
                self.log_activity(f"  E-Core Usage: {core_util.get('e_avg', 0):.1f}%", "INFO")
            
            # Recent process assignments
            assignments = eas.get('process_assignments', [])
            if assignments:
                self.log_activity(f"  Recent Process Assignments:", "INFO")
                for assignment in assignments[:10]:
                    name = assignment.get('name', 'Unknown')[:20]
                    core_type = assignment.get('core_type', 'unknown')
                    workload = assignment.get('workload_type', 'unknown')
                    cpu_usage = assignment.get('cpu_usage', 0)
                    self.log_activity(f"    • {name:20} → {core_type:6} ({workload}) {cpu_usage:5.1f}%", "INFO")
        
        self.log_activity("=== END STATE SNAPSHOT ===", "INFO")
    
    def log_activity_summary(self):
        """Log summary of activity since start"""
        self.log_activity("=== ACTIVITY SUMMARY ===", "INFO")
        self.log_activity(f"New Classifications: {self.activity_counts['new_classifications']:,}", "INFO")
        self.log_activity(f"Processes Added: {self.activity_counts['processes_added']}", "INFO")
        self.log_activity(f"Processes Removed: {self.activity_counts['processes_removed']}", "INFO")
        self.log_activity("=== END SUMMARY ===", "INFO")
    
    def run_continuous_monitoring(self, interval=5, detailed_interval=60):
        """Run continuous monitoring"""
        self.log_activity("Starting Enhanced EAS continuous monitoring", "INFO")
        self.log_activity(f"Update interval: {interval}s, Detailed logging: {detailed_interval}s", "INFO")
        
        last_detailed_log = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Get current state
                current_state = self.get_current_state()
                
                if current_state:
                    # Analyze changes
                    self.analyze_changes(current_state)
                    
                    # Detailed logging at intervals
                    if current_time - last_detailed_log >= detailed_interval:
                        self.log_detailed_state(current_state)
                        last_detailed_log = current_time
                
                # Wait for next update
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.log_activity("Monitoring stopped by user", "INFO")
            self.log_activity_summary()
        except Exception as e:
            self.log_activity(f"Monitoring error: {e}", "ERROR")

def main():
    """Main function"""
    print("🔍 Enhanced EAS Activity Logger")
    print("=" * 50)
    print("This will log detailed EAS activity to show exactly what's happening")
    print("Log file: eas_activity.log")
    print("Press Ctrl+C to stop")
    print()
    
    logger = EASActivityLogger()
    logger.run_continuous_monitoring(interval=3, detailed_interval=30)

if __name__ == "__main__":
    main()