#!/usr/bin/env python3
"""
Test script to verify battery history page is working correctly
"""

import requests
import time
import json

def test_battery_history_api():
    print("🔋 Testing Battery History API")
    print("=" * 50)
    
    ranges = ['today', 'week', 'month', 'all']
    
    for range_param in ranges:
        print(f"\n📊 Testing range: {range_param}")
        try:
            response = requests.get(f'http://localhost:9010/api/battery-history?range={range_param}', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                history = data.get('history', [])
                cycles = data.get('cycles', [])
                app_changes = data.get('app_changes', [])
                statistics = data.get('statistics', {})
                
                print(f"✅ Range '{range_param}' responded successfully")
                print(f"   History points: {len(history)}")
                print(f"   Battery cycles: {len(cycles)}")
                print(f"   App changes: {len(app_changes)}")
                print(f"   Statistics: {statistics}")
                
                if history:
                    sample = history[0]
                    print(f"   Sample data:")
                    print(f"     Timestamp: {sample.get('timestamp', 'N/A')}")
                    print(f"     Battery: {sample.get('battery_level', 'N/A')}%")
                    print(f"     Current Draw: {sample.get('current_draw', 'N/A')}mA")
                    print(f"     EAS Active: {sample.get('eas_active', 'N/A')}")
                    print(f"     Power Source: {sample.get('power_source', 'N/A')}")
                
                # Verify data structure
                required_fields = ['timestamp', 'battery_level', 'current_draw', 'eas_active', 'power_source']
                if history:
                    missing_fields = [field for field in required_fields if field not in history[0]]
                    if missing_fields:
                        print(f"   ⚠️  Missing fields: {missing_fields}")
                    else:
                        print(f"   ✅ All required fields present")
                
            else:
                print(f"❌ Range '{range_param}' failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Range '{range_param}' error: {e}")

def test_battery_history_page():
    print("\n🌐 Testing Battery History Web Page")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/history', timeout=10)
        
        if response.status_code == 200:
            html_content = response.text
            print("✅ Battery history page loaded successfully")
            
            # Check for key elements
            checks = [
                ('Chart.js library', 'chart.js' in html_content.lower()),
                ('Battery chart canvas', 'batteryChart' in html_content),
                ('Theme selector', 'themeSelect' in html_content),
                ('Statistics panel', 'avgBatteryLife' in html_content),
                ('Time range buttons', 'range-btn' in html_content),
                ('EAS toggle', 'showEAS' in html_content)
            ]
            
            for check_name, check_result in checks:
                status = "✅" if check_result else "❌"
                print(f"   {status} {check_name}")
                
        else:
            print(f"❌ Battery history page failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Battery history page error: {e}")

def test_real_time_updates():
    print("\n🔄 Testing Real-time Data Updates")
    print("=" * 50)
    
    print("Monitoring battery history API for changes...")
    print("Time     | Points | Battery | Current | EAS | Status")
    print("-" * 60)
    
    try:
        previous_count = 0
        for i in range(5):  # Monitor for 5 iterations
            response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                history = data.get('history', [])
                stats = data.get('statistics', {})
                
                current_time = time.strftime("%H:%M:%S")
                point_count = len(history)
                
                if history:
                    latest = history[-1]
                    battery_level = latest.get('battery_level', 0)
                    current_draw = latest.get('current_draw', 0)
                    eas_active = "Yes" if latest.get('eas_active') else "No"
                    
                    change_indicator = ""
                    if point_count > previous_count:
                        change_indicator = f"(+{point_count - previous_count})"
                    
                    print(f"{current_time} | {point_count:6}{change_indicator:8} | {battery_level:6.1f}% | {current_draw:6.0f}mA | {eas_active:3} | OK")
                    previous_count = point_count
                else:
                    print(f"{current_time} | {point_count:6} | No data available")
            else:
                print(f"{time.strftime('%H:%M:%S')} | API Error: {response.status_code}")
            
            if i < 4:  # Don't sleep on last iteration
                time.sleep(10)  # Wait 10 seconds between checks
                
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    test_battery_history_api()
    test_battery_history_page()
    
    # Ask if user wants real-time monitoring
    try:
        choice = input("\n🔄 Start real-time monitoring? (y/N): ").lower()
        if choice == 'y':
            test_real_time_updates()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")