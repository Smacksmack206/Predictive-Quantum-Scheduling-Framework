#!/usr/bin/env python3
"""
Real-time Enhanced EAS Activity Monitor
Shows exactly what the Enhanced EAS is doing moment by moment
"""

import time
import requests
import json
import os
import sys
from datetime import datetime
from collections import defaultdict, deque

class RealTimeEASMonitor:
    """Real-time monitoring of Enhanced EAS activity"""
    
    def __init__(self):
        self.previous_classifications = 0
        self.previous_optimized = 0
        self.activity_log = deque(maxlen=50)
        self.classification_changes = deque(maxlen=20)
        self.process_activity = defaultdict(list)
        self.performance_history = deque(maxlen=30)
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_current_activity(self):
        """Get current EAS activity"""
        try:
            # Get learning stats
            response = requests.get('http://localhost:9010/api/eas-learning-stats', timeout=5)
            learning_data = response.json() if response.status_code == 200 else {}
            
            # Get EAS status
            response = requests.get('http://localhost:9010/api/eas-status', timeout=5)
            eas_data = response.json() if response.status_code == 200 else {}
            
            # Get insights
            response = requests.get('http://localhost:9010/api/eas-insights', timeout=5)
            insights_data = response.json() if response.status_code == 200 else {}
            
            # Get basic status
            response = requests.get('http://localhost:9010/api/status', timeout=5)
            status_data = response.json() if response.status_code == 200 else {}
            
            return {
                'learning': learning_data,
                'eas': eas_data,
                'insights': insights_data,
                'status': status_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def analyze_activity_changes(self, current_data):
        """Analyze what changed since last check"""
        changes = []
        
        if 'learning' in current_data:
            total_classifications = current_data['learning'].get('total_classifications', 0)
            
            if total_classifications > self.previous_classifications:
                new_classifications = total_classifications - self.previous_classifications
                changes.append(f"📚 +{new_classifications:,} new classifications")
                self.previous_classifications = total_classifications
        
        if 'eas' in current_data:
            processes_optimized = current_data['eas'].get('processes_optimized', 0)
            
            if processes_optimized != self.previous_optimized:
                diff = processes_optimized - self.previous_optimized
                if diff > 0:
                    changes.append(f"⚡ +{diff} processes optimized")
                elif diff < 0:
                    changes.append(f"📉 {abs(diff)} processes removed from optimization")
                self.previous_optimized = processes_optimized
        
        return changes
    
    def log_activity(self, changes, current_data):
        """Log activity for history"""
        timestamp = current_data.get('timestamp', datetime.now())
        
        for change in changes:
            self.activity_log.append({
                'time': timestamp.strftime('%H:%M:%S'),
                'activity': change
            })
    
    def display_real_time_dashboard(self, current_data):
        """Display real-time dashboard"""
        self.clear_screen()
        
        print("🔍 REAL-TIME ENHANCED EAS MONITOR")
        print("=" * 80)
        print(f"⏰ Last Update: {current_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Status
        if 'status' in current_data:
            status = current_data['status']
            battery = status.get('battery_level', 0)
            enabled = status.get('enabled', False)
            on_battery = status.get('on_battery', False)
            
            print("🔋 SYSTEM STATUS")
            print("-" * 40)
            print(f"Battery Level:     {battery}% {'🔋' if on_battery else '🔌'}")
            print(f"Service Enabled:   {'✅ YES' if enabled else '❌ NO'}")
            print(f"Power Source:      {'Battery' if on_battery else 'AC Power'}")
            print()
        
        # Enhanced EAS Status
        if 'learning' in current_data:
            learning = current_data['learning']
            total_classifications = learning.get('total_classifications', 0)
            avg_confidence = learning.get('average_confidence', 0)
            recent_classifications = learning.get('recent_classifications', [])
            
            print("🧠 ENHANCED EAS LEARNING ENGINE")
            print("-" * 40)
            print(f"Total Classifications: {total_classifications:,}")
            print(f"Average Confidence:    {avg_confidence:.3f}")
            print(f"Classification Types:  {len(recent_classifications)}")
            
            if recent_classifications:
                print("\n📊 Active Classification Types:")
                for i, (cls_name, count, confidence) in enumerate(recent_classifications[:5]):
                    bar_length = min(int(count / max(1, recent_classifications[0][1]) * 20), 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"   {cls_name[:20]:20} {bar} {count:,} ({confidence:.3f})")
            print()
        
        # Current Process Optimization
        if 'eas' in current_data:
            eas = current_data['eas']
            enabled = eas.get('enabled', False)
            processes_optimized = eas.get('processes_optimized', 0)
            performance_score = eas.get('performance_score', 0)
            process_assignments = eas.get('process_assignments', [])
            
            print("⚡ PROCESS OPTIMIZATION ENGINE")
            print("-" * 40)
            print(f"EAS Enabled:           {'✅ YES' if enabled else '❌ NO'}")
            print(f"Processes Optimized:   {processes_optimized}")
            print(f"Performance Score:     {performance_score:.1f}/100")
            
            # Core utilization
            core_util = eas.get('core_utilization', {})
            if core_util and 'p_avg' in core_util:
                p_avg = core_util.get('p_avg', 0)
                e_avg = core_util.get('e_avg', 0)
                print(f"P-Core Utilization:    {p_avg:.1f}%")
                print(f"E-Core Utilization:    {e_avg:.1f}%")
            
            # Recent process assignments
            if process_assignments:
                print(f"\n🎯 Recent Process Assignments:")
                for i, assignment in enumerate(process_assignments[:8]):
                    name = assignment.get('name', 'Unknown')[:15]
                    core_type = assignment.get('core_type', 'unknown')
                    workload = assignment.get('workload_type', 'unknown')[:12]
                    cpu_usage = assignment.get('cpu_usage', 0)
                    
                    core_icon = "🔥" if core_type == 'p_core' else "💚"
                    print(f"   {core_icon} {name:15} → {core_type:6} ({workload:12}) {cpu_usage:5.1f}%")
            print()
        
        # Learning Insights
        if 'insights' in current_data:
            insights = current_data['insights']
            learning_effectiveness = insights.get('learning_effectiveness', 0)
            total_processes = insights.get('total_processes_classified', 0)
            
            print("📈 LEARNING INSIGHTS")
            print("-" * 40)
            print(f"Learning Effectiveness: {learning_effectiveness:.3f}")
            print(f"Currently Classified:   {total_processes}")
            
            # Confidence distribution
            confidence_dist = insights.get('confidence_distribution', {})
            if confidence_dist:
                high_conf = confidence_dist.get('high_confidence', 0)
                medium_conf = confidence_dist.get('medium_confidence', 0)
                low_conf = confidence_dist.get('low_confidence', 0)
                total_conf = high_conf + medium_conf + low_conf
                
                if total_conf > 0:
                    print(f"\n🎯 Confidence Distribution:")
                    print(f"   High (>0.8):   {high_conf:3d} ({high_conf/total_conf*100:5.1f}%)")
                    print(f"   Medium (0.5-0.8): {medium_conf:3d} ({medium_conf/total_conf*100:5.1f}%)")
                    print(f"   Low (<0.5):    {low_conf:3d} ({low_conf/total_conf*100:5.1f}%)")
            print()
        
        # Recent Activity Log
        print("📝 RECENT ACTIVITY LOG")
        print("-" * 40)
        if self.activity_log:
            for activity in list(self.activity_log)[-10:]:
                print(f"   {activity['time']} - {activity['activity']}")
        else:
            print("   No recent activity detected...")
        print()
        
        # Performance History
        if len(self.performance_history) > 1:
            print("📊 PERFORMANCE TREND")
            print("-" * 40)
            recent_perf = list(self.performance_history)[-10:]
            for i, perf in enumerate(recent_perf):
                timestamp = perf['time']
                classifications = perf.get('classifications', 0)
                optimized = perf.get('optimized', 0)
                
                trend_icon = "📈" if i > 0 and classifications > recent_perf[i-1].get('classifications', 0) else "📊"
                print(f"   {trend_icon} {timestamp} - {classifications:,} classifications, {optimized} optimized")
            print()
        
        # Instructions
        print("🎮 CONTROLS")
        print("-" * 40)
        print("Press Ctrl+C to stop monitoring")
        print("Refresh rate: 3 seconds")
        print()
    
    def run_monitor(self):
        """Run the real-time monitor"""
        print("🚀 Starting Real-Time Enhanced EAS Monitor...")
        print("Connecting to Battery Optimizer Pro...")
        
        try:
            while True:
                # Get current activity
                current_data = self.get_current_activity()
                
                if 'error' in current_data:
                    self.clear_screen()
                    print("❌ CONNECTION ERROR")
                    print("=" * 50)
                    print(f"Error: {current_data['error']}")
                    print("\nMake sure Battery Optimizer Pro is running:")
                    print("python3 enhanced_app.py")
                    print("\nRetrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Analyze changes
                changes = self.analyze_activity_changes(current_data)
                
                # Log activity
                if changes:
                    self.log_activity(changes, current_data)
                
                # Store performance data
                if 'learning' in current_data and 'eas' in current_data:
                    self.performance_history.append({
                        'time': current_data['timestamp'].strftime('%H:%M:%S'),
                        'classifications': current_data['learning'].get('total_classifications', 0),
                        'optimized': current_data['eas'].get('processes_optimized', 0),
                        'confidence': current_data['learning'].get('average_confidence', 0)
                    })
                
                # Display dashboard
                self.display_real_time_dashboard(current_data)
                
                # Wait before next update
                time.sleep(3)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print("Enhanced EAS continues running in the background")
        except Exception as e:
            print(f"\n\n❌ Monitor error: {e}")

def main():
    """Main function"""
    monitor = RealTimeEASMonitor()
    monitor.run_monitor()

if __name__ == "__main__":
    main()