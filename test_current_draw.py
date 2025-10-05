#!/usr/bin/env python3
"""
Test script specifically for Current Draw and Predicted Runtime
"""

import time
import psutil
import requests

def test_current_draw():
    print("🔋 Testing Current Draw & Predicted Runtime")
    print("=" * 50)
    
    # Get battery info
    battery = psutil.sensors_battery()
    if not battery:
        print("❌ No battery found")
        return
    
    print(f"Battery Level: {battery.percent}%")
    print(f"Power Source: {'AC Power' if battery.power_plugged else 'Battery'}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    print()
    
    # Test the API endpoint
    try:
        print("Testing battery debug API...")
        response = requests.get('http://localhost:9010/api/battery-debug', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response:")
            print(f"  Battery: {data['battery_info']['percent']}% ({'Plugged' if data['battery_info']['power_plugged'] else 'Battery'})")
            
            metrics = data['eas_metrics']
            print(f"  Current Drain: {metrics.get('current_ma_drain', 'N/A')} mA")
            print(f"  Current Charge: {metrics.get('current_ma_charge', 'N/A')} mA")
            print(f"  Predicted Runtime: {metrics.get('predicted_battery_hours', 'N/A')} hours")
            print(f"  Time on Battery: {metrics.get('time_on_battery_hours', 'N/A')} hours")
            
            if data['last_battery_reading']:
                last_level, last_time = data['last_battery_reading']
                time_diff = data['current_time'] - last_time
                print(f"  Last Reading: {last_level}% ({time_diff:.0f}s ago)")
            
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        print("Make sure the Battery Optimizer is running on localhost:9010")
    
    print("\n" + "=" * 50)
    print("💡 Tips to get accurate readings:")
    print("1. Wait 2-3 minutes for initial measurements")
    print("2. Use the device normally to generate CPU activity")
    print("3. On battery power, readings are more accurate")
    print("4. Fallback estimation is used when no battery level changes detected")

if __name__ == "__main__":
    test_current_draw()