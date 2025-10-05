#!/usr/bin/env python3
"""
Test script to verify battery metrics are working correctly
"""

import time
import psutil
from enhanced_app import EnergyAwareScheduler

def test_battery_metrics():
    print("🔋 Testing Battery Metrics")
    print("=" * 40)
    
    # Create EAS instance
    eas = EnergyAwareScheduler()
    
    # Get initial battery info
    battery = psutil.sensors_battery()
    if not battery:
        print("❌ No battery found - are you on a desktop?")
        return
    
    print(f"Initial Battery Level: {battery.percent}%")
    print(f"Power Source: {'AC Power' if battery.power_plugged else 'Battery'}")
    print()
    
    # Test metrics updates
    print("Testing metrics updates...")
    for i in range(5):
        print(f"\n--- Update {i+1} ---")
        eas.update_performance_metrics()
        
        # Print current metrics
        metrics = eas.current_metrics
        print(f"Time on Battery: {metrics.get('time_on_battery_hours', 0):.2f} hours")
        print(f"Current Drain: {metrics.get('current_ma_drain', 0):.0f} mA")
        print(f"Current Charge: {metrics.get('current_ma_charge', 0):.0f} mA")
        print(f"Predicted Runtime: {metrics.get('predicted_battery_hours', 0):.1f} hours")
        print(f"Battery Level: {metrics.get('battery_level', 0)}%")
        print(f"Plugged: {metrics.get('plugged', False)}")
        
        if i < 4:  # Don't sleep on last iteration
            time.sleep(2)
    
    print("\n✅ Battery metrics test complete!")
    print("\nIf you see zeros, try:")
    print("1. Wait a few minutes for battery level changes")
    print("2. Unplug/plug power adapter to test transitions")
    print("3. Check the debug logs in the main app")

if __name__ == "__main__":
    test_battery_metrics()