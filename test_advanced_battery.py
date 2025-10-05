#!/usr/bin/env python3
"""
Test script for advanced battery analytics
Shows comprehensive power consumption breakdown and intelligent predictions
"""

import time
import requests
import json
from datetime import datetime

def test_advanced_battery():
    print("🔋 Advanced Battery Analytics Test")
    print("=" * 60)
    
    try:
        # Test battery debug API
        print("📊 Fetching comprehensive battery analytics...")
        response = requests.get('http://localhost:9010/api/battery-debug', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n🔋 Battery Status:")
            battery = data['battery_info']
            print(f"  Level: {battery['percent']}%")
            print(f"  Power Source: {'AC Power' if battery['power_plugged'] else 'Battery'}")
            
            print(f"\n⚡ Current Metrics:")
            metrics = data['current_metrics']
            print(f"  Current Draw: {metrics.get('current_ma_drain', 'N/A')} mA")
            print(f"  Current Charge: {metrics.get('current_ma_charge', 'N/A')} mA")
            print(f"  Predicted Runtime: {metrics.get('predicted_battery_hours', 'N/A')} hours")
            print(f"  Time on Battery: {metrics.get('time_on_battery_hours', 'N/A')} hours")
            print(f"  Temperature: {metrics.get('temperature', 'N/A')}°C")
            
            print(f"\n📈 Analytics Data:")
            analytics = data.get('analytics', {})
            print(f"  History Records: {analytics.get('recent_history_count', 0)}")
            print(f"  Power Samples: {len(analytics.get('recent_power_consumption', []))}")
            print(f"  Drain Samples: {len(analytics.get('recent_drain_samples', []))}")
            
            if analytics.get('recent_drain_samples'):
                recent_drains = analytics['recent_drain_samples']
                avg_drain = sum(recent_drains) / len(recent_drains)
                print(f"  Average Measured Drain: {avg_drain:.0f} mA")
            
            if analytics.get('recent_power_consumption'):
                recent_power = analytics['recent_power_consumption']
                avg_power = sum(recent_power) / len(recent_power)
                print(f"  Average Calculated Draw: {avg_power:.0f} mA")
        
        # Test power breakdown API
        print(f"\n🔧 Fetching power consumption breakdown...")
        response = requests.get('http://localhost:9010/api/power-breakdown', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n⚡ Power Breakdown:")
            print(f"  Total Power: {data.get('total_power_mw', 0):.0f} mW")
            print(f"  Total Current: {data.get('total_current_ma', 0):.0f} mA")
            
            breakdown = data.get('breakdown', {})
            print(f"\n📊 Component Breakdown:")
            
            # Sort by power consumption
            components = []
            for comp, details in breakdown.items():
                if comp == 'cpu':
                    power = details.get('total_mw', 0)
                else:
                    power = details.get('power_mw', 0)
                percentage = details.get('percentage', 0)
                components.append((comp, power, percentage))
            
            components.sort(key=lambda x: x[1], reverse=True)
            
            for comp, power, percentage in components:
                print(f"  {comp.upper():12}: {power:6.0f} mW ({percentage:5.1f}%)")
            
            system_state = data.get('system_state', {})
            print(f"\n🖥️  System State:")
            print(f"  CPU Usage: {system_state.get('cpu_usage', 0):.1f}%")
            print(f"  EAS Enabled: {system_state.get('eas_enabled', False)}")
        
        # Test EAS status for additional metrics
        print(f"\n🧠 Fetching EAS performance data...")
        response = requests.get('http://localhost:9010/api/eas-status', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            advanced_battery = data.get('advanced_battery', {})
            print(f"\n🔋 Advanced Battery Metrics:")
            print(f"  Battery Level: {advanced_battery.get('battery_level', 0)}%")
            print(f"  Plugged In: {advanced_battery.get('plugged', False)}")
            print(f"  Time on Battery: {advanced_battery.get('time_on_battery_hours', 0):.2f}h")
            print(f"  Current Drain: {advanced_battery.get('current_ma_drain', 0):.0f} mA")
            print(f"  Predicted Runtime: {advanced_battery.get('predicted_battery_hours', 0):.1f}h")
            print(f"  Temperature: {advanced_battery.get('temperature_celsius', 0):.1f}°C")
            
            print(f"\n🎯 EAS Performance:")
            print(f"  Battery Improvement: {data.get('battery_improvement', 0):.1f}%")
            print(f"  Performance Score: {data.get('performance_score', 0):.0f}")
            print(f"  Thermal Improvement: {data.get('thermal_improvement', 0):.1f}°C")
            print(f"  Processes Optimized: {data.get('processes_optimized', 0)}")
        
        print(f"\n" + "=" * 60)
        print("✅ Advanced battery analytics test complete!")
        print("\n💡 Key Features:")
        print("• Dynamic power consumption modeling")
        print("• Multi-source drain rate measurement")
        print("• Intelligent runtime prediction with trend analysis")
        print("• Component-level power breakdown")
        print("• Machine learning calibration")
        print("• Real-time system state analysis")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Battery Optimizer")
        print("Make sure the app is running on localhost:9010")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def monitor_real_time():
    """Monitor battery metrics in real-time"""
    print("\n🔄 Real-time monitoring (Press Ctrl+C to stop)...")
    print("Time     | Battery | Current | Predicted | CPU  | Temp")
    print("-" * 55)
    
    try:
        while True:
            response = requests.get('http://localhost:9010/api/eas-status', timeout=2)
            if response.status_code == 200:
                data = response.json()
                battery = data.get('advanced_battery', {})
                
                current_time = datetime.now().strftime("%H:%M:%S")
                battery_level = battery.get('battery_level', 0)
                current_draw = battery.get('current_ma_drain', 0)
                predicted_hours = battery.get('predicted_battery_hours', 0)
                temp = battery.get('temperature_celsius', 0)
                
                # Get CPU usage
                cpu_response = requests.get('http://localhost:9010/api/battery-debug', timeout=2)
                cpu_usage = 0
                if cpu_response.status_code == 200:
                    cpu_data = cpu_response.json()
                    system_state = cpu_data.get('system_state', {})
                    cpu_usage = system_state.get('cpu_usage', 0)
                
                print(f"{current_time} | {battery_level:6.1f}% | {current_draw:6.0f}mA | {predicted_hours:8.1f}h | {cpu_usage:4.1f}% | {temp:4.1f}°C")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    test_advanced_battery()
    
    # Ask if user wants real-time monitoring
    try:
        choice = input("\n🔄 Start real-time monitoring? (y/N): ").lower()
        if choice == 'y':
            monitor_real_time()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")