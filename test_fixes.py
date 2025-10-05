#!/usr/bin/env python3
"""
Test script to verify all the fixes are working
"""

import requests
import json
import time

def test_status_api():
    print("🔍 Testing Status API (Dashboard)")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/api/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Status API responded")
            
            # Check for current metrics
            current_metrics = data.get('current_metrics', {})
            battery_info = data.get('battery_info', {})
            analytics = data.get('analytics', {})
            
            print(f"\n📊 Analytics:")
            print(f"   Status: {analytics.get('status', 'N/A')}")
            print(f"   Hours Saved: {analytics.get('estimated_hours_saved', 0)}h")
            print(f"   Power Savings: {analytics.get('savings_percentage', 0)}%")
            print(f"   Data Points: {analytics.get('data_points', 0)}")
            
            print(f"\n🔋 Battery Info:")
            print(f"   Level: {battery_info.get('percent', 0)}%")
            print(f"   Plugged: {battery_info.get('power_plugged', False)}")
            
            print(f"\n⚡ Current Metrics:")
            print(f"   Current Draw: {current_metrics.get('current_ma_drain', 'N/A')}mA")
            print(f"   Current Charge: {current_metrics.get('current_ma_charge', 'N/A')}mA")
            print(f"   Plugged: {current_metrics.get('plugged', 'N/A')}")
            
            # Check if analytics are working
            if analytics.get('status') != 'Collecting data - check back in a few hours':
                print("✅ Analytics are active!")
            else:
                print("⚠️  Analytics still collecting data")
                
        else:
            print(f"❌ Status API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status API error: {e}")

def test_eas_api():
    print("\n🔍 Testing EAS API (EAS Monitor)")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/api/eas-status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            print("✅ EAS API responded")
            
            advanced_battery = data.get('advanced_battery', {})
            
            print(f"\n⚡ Advanced Battery:")
            print(f"   Current Draw: {advanced_battery.get('current_ma_drain', 'N/A')}mA")
            print(f"   Current Charge: {advanced_battery.get('current_ma_charge', 'N/A')}mA")
            print(f"   Plugged: {advanced_battery.get('plugged', 'N/A')}")
            print(f"   Battery Level: {advanced_battery.get('battery_level', 'N/A')}%")
            print(f"   Time on Battery: {advanced_battery.get('time_on_battery_hours', 'N/A')}h")
            
            # Check charging detection
            if advanced_battery.get('plugged') and advanced_battery.get('current_ma_charge', 0) > 0:
                print("✅ Charging current detected!")
            elif advanced_battery.get('plugged'):
                print("⚠️  Plugged in but no charging current")
            else:
                print("🔋 On battery power")
                
        else:
            print(f"❌ EAS API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ EAS API error: {e}")

def test_battery_history_api():
    print("\n🔍 Testing Battery History API")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            history = data.get('history', [])
            statistics = data.get('statistics', {})
            
            print("✅ Battery History API responded")
            print(f"   History Points: {len(history)}")
            print(f"   Statistics: {statistics}")
            
            if history:
                sample = history[0]
                print(f"\n📊 Sample Data Point:")
                print(f"   Timestamp: {sample.get('timestamp', 'N/A')}")
                print(f"   Battery Level: {sample.get('battery_level', 'N/A')}%")
                print(f"   Current Draw: {sample.get('current_draw', 'N/A')}mA")
                print(f"   EAS Active: {sample.get('eas_active', 'N/A')}")
                print(f"   Power Source: {sample.get('power_source', 'N/A')}")
                
                print("✅ Battery history has data!")
            else:
                print("⚠️  No battery history data")
                
        else:
            print(f"❌ Battery History API failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Battery History API error: {e}")

def test_real_time_updates():
    print("\n🔄 Testing Real-time Updates")
    print("=" * 50)
    
    print("Monitoring APIs for 30 seconds...")
    print("Time     | Status | EAS    | History | Charging")
    print("-" * 55)
    
    try:
        for i in range(6):  # 6 iterations, 5 seconds apart
            current_time = time.strftime("%H:%M:%S")
            
            # Test status API
            try:
                status_response = requests.get('http://localhost:9010/api/status', timeout=2)
                status_ok = "✅" if status_response.status_code == 200 else "❌"
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_metrics = status_data.get('current_metrics', {})
                    charge_rate = current_metrics.get('current_ma_charge', 0)
                    charging = f"+{charge_rate:.0f}mA" if charge_rate > 0 else "No charge"
                else:
                    charging = "Error"
            except:
                status_ok = "❌"
                charging = "Error"
            
            # Test EAS API
            try:
                eas_response = requests.get('http://localhost:9010/api/eas-status', timeout=2)
                eas_ok = "✅" if eas_response.status_code == 200 else "❌"
            except:
                eas_ok = "❌"
            
            # Test History API
            try:
                history_response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=2)
                history_ok = "✅" if history_response.status_code == 200 else "❌"
            except:
                history_ok = "❌"
            
            print(f"{current_time} | {status_ok:6} | {eas_ok:6} | {history_ok:7} | {charging}")
            
            if i < 5:  # Don't sleep on last iteration
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    test_status_api()
    test_eas_api()
    test_battery_history_api()
    
    # Ask if user wants real-time monitoring
    try:
        choice = input("\n🔄 Start real-time monitoring? (y/N): ").lower()
        if choice == 'y':
            test_real_time_updates()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")