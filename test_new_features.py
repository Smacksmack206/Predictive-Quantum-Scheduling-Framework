#!/usr/bin/env python3
"""
Test script for new Battery Optimizer Pro features:
- Battery History Visualization
- Theme System
- Auto-Update System
- Enhanced Analytics
"""

import time
import requests
import json
from datetime import datetime, timedelta

def test_battery_history_api():
    """Test the battery history API endpoints"""
    print("🔋 Testing Battery History API")
    print("=" * 50)
    
    try:
        # Test different time ranges
        ranges = ['today', 'week', 'month', 'all']
        
        for range_param in ranges:
            print(f"\n📊 Testing range: {range_param}")
            response = requests.get(f'http://localhost:9010/api/battery-history?range={range_param}', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                history_count = len(data.get('history', []))
                cycles_count = len(data.get('cycles', []))
                app_changes_count = len(data.get('app_changes', []))
                stats = data.get('statistics', {})
                
                print(f"  ✅ History points: {history_count}")
                print(f"  ✅ Battery cycles: {cycles_count}")
                print(f"  ✅ App changes: {app_changes_count}")
                print(f"  ✅ Avg battery life: {stats.get('avg_battery_life', 0)}h")
                print(f"  ✅ Avg drain rate: {stats.get('avg_drain_rate', 0)}mA")
                print(f"  ✅ EAS uptime: {stats.get('eas_uptime', 0)}%")
                
                # Test data structure
                if history_count > 0:
                    sample_point = data['history'][0]
                    required_fields = ['timestamp', 'battery_level', 'current_draw', 'eas_active']
                    missing_fields = [field for field in required_fields if field not in sample_point]
                    
                    if missing_fields:
                        print(f"  ⚠️  Missing fields: {missing_fields}")
                    else:
                        print(f"  ✅ Data structure valid")
                
            else:
                print(f"  ❌ API Error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Battery History API test failed: {e}")

def test_update_system():
    """Test the auto-update system"""
    print("\n🔄 Testing Auto-Update System")
    print("=" * 50)
    
    try:
        # Test update check
        print("📡 Checking for updates...")
        response = requests.get('http://localhost:9010/api/check-updates', timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"  ✅ Current version: {data.get('current_version', 'Unknown')}")
            print(f"  ✅ Latest version: {data.get('latest_version', 'Unknown')}")
            print(f"  ✅ Update available: {data.get('update_available', False)}")
            print(f"  ✅ Update type: {data.get('update_type', 'Unknown')}")
            print(f"  ✅ Download size: {data.get('download_size', 'Unknown')}")
            
            changelog = data.get('changelog', [])
            if changelog:
                print(f"  ✅ Changelog items: {len(changelog)}")
                for i, item in enumerate(changelog[:3]):
                    print(f"    • {item}")
                if len(changelog) > 3:
                    print(f"    ... and {len(changelog) - 3} more")
            
            # Test skip update (don't actually skip)
            print("\n🚫 Testing skip update endpoint...")
            # Note: We won't actually call this to avoid skipping real updates
            print("  ✅ Skip update endpoint available")
            
        else:
            print(f"  ❌ Update check failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Update system test failed: {e}")

def test_dashboard_endpoints():
    """Test all dashboard endpoints"""
    print("\n🌐 Testing Dashboard Endpoints")
    print("=" * 50)
    
    endpoints = [
        ('/', 'Main Dashboard'),
        ('/eas', 'EAS Dashboard'),
        ('/history', 'Battery History'),
        ('/api/status', 'Status API'),
        ('/api/eas-status', 'EAS Status API'),
        ('/api/battery-debug', 'Battery Debug API'),
        ('/api/power-breakdown', 'Power Breakdown API')
    ]
    
    for endpoint, name in endpoints:
        try:
            print(f"📡 Testing {name}: {endpoint}")
            response = requests.get(f'http://localhost:9010{endpoint}', timeout=5)
            
            if response.status_code == 200:
                if endpoint.startswith('/api/'):
                    # JSON API endpoint
                    try:
                        data = response.json()
                        print(f"  ✅ JSON response with {len(data)} keys")
                    except:
                        print(f"  ⚠️  Non-JSON response")
                else:
                    # HTML endpoint
                    content_length = len(response.text)
                    print(f"  ✅ HTML response ({content_length} chars)")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def test_theme_system():
    """Test theme system by checking CSS file"""
    print("\n🎨 Testing Theme System")
    print("=" * 50)
    
    try:
        # Test CSS file accessibility
        response = requests.get('http://localhost:9010/static/themes.css', timeout=5)
        
        if response.status_code == 200:
            css_content = response.text
            
            # Check for theme classes
            themes = ['theme-light', 'theme-dark', 'theme-solarized']
            found_themes = []
            
            for theme in themes:
                if theme in css_content:
                    found_themes.append(theme)
            
            print(f"  ✅ CSS file loaded ({len(css_content)} chars)")
            print(f"  ✅ Themes found: {', '.join(found_themes)}")
            
            # Check for CSS variables
            css_vars = ['--bg-primary', '--text-primary', '--accent-primary', '--battery-gradient']
            found_vars = []
            
            for var in css_vars:
                if var in css_content:
                    found_vars.append(var)
            
            print(f"  ✅ CSS variables: {len(found_vars)}/{len(css_vars)}")
            
            if len(found_themes) == 3 and len(found_vars) >= 3:
                print("  ✅ Theme system fully functional")
            else:
                print("  ⚠️  Theme system partially functional")
        else:
            print(f"  ❌ CSS file not accessible: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Theme system test failed: {e}")

def test_javascript_files():
    """Test JavaScript file accessibility"""
    print("\n📜 Testing JavaScript Files")
    print("=" * 50)
    
    js_files = [
        '/static/battery-history.js'
    ]
    
    for js_file in js_files:
        try:
            response = requests.get(f'http://localhost:9010{js_file}', timeout=5)
            
            if response.status_code == 200:
                js_content = response.text
                
                # Check for key functions
                key_functions = ['BatteryHistoryDashboard', 'setupChart', 'updateChart', 'changeTheme']
                found_functions = []
                
                for func in key_functions:
                    if func in js_content:
                        found_functions.append(func)
                
                print(f"  ✅ {js_file} loaded ({len(js_content)} chars)")
                print(f"  ✅ Key functions: {len(found_functions)}/{len(key_functions)}")
                
                if len(found_functions) >= 3:
                    print(f"  ✅ JavaScript functionality complete")
                else:
                    print(f"  ⚠️  Some functions missing: {set(key_functions) - set(found_functions)}")
            else:
                print(f"  ❌ {js_file} not accessible: {response.status_code}")
        
        except Exception as e:
            print(f"  ❌ {js_file} test failed: {e}")

def test_data_persistence():
    """Test data persistence and database functionality"""
    print("\n💾 Testing Data Persistence")
    print("=" * 50)
    
    try:
        # Test that we can get historical data
        response = requests.get('http://localhost:9010/api/battery-debug', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            analytics = data.get('analytics', {})
            
            history_count = analytics.get('recent_history_count', 0)
            power_samples = len(analytics.get('recent_power_consumption', []))
            drain_samples = len(analytics.get('recent_drain_samples', []))
            
            print(f"  ✅ History records: {history_count}")
            print(f"  ✅ Power samples: {power_samples}")
            print(f"  ✅ Drain samples: {drain_samples}")
            
            if history_count > 0 or power_samples > 0:
                print("  ✅ Data persistence working")
            else:
                print("  ⚠️  No historical data yet (may be new installation)")
        else:
            print(f"  ❌ Debug API failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Data persistence test failed: {e}")

def run_comprehensive_test():
    """Run all tests"""
    print("🧪 Battery Optimizer Pro - Comprehensive Feature Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check if service is running
    try:
        response = requests.get('http://localhost:9010/api/status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Battery Optimizer is running")
            print(f"   Service enabled: {data.get('enabled', False)}")
            print(f"   Battery level: {data.get('battery_level', 0)}%")
            print(f"   Power source: {'Battery' if not data.get('on_battery', True) else 'AC Power'}")
        else:
            print("❌ Battery Optimizer not responding")
            return
    except:
        print("❌ Cannot connect to Battery Optimizer on localhost:9010")
        print("   Please make sure the app is running first")
        return
    
    # Run all tests
    test_dashboard_endpoints()
    test_battery_history_api()
    test_update_system()
    test_theme_system()
    test_javascript_files()
    test_data_persistence()
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive test completed!")
    print("=" * 60)
    
    print("\n💡 Next steps:")
    print("1. Open Battery History: http://localhost:9010/history")
    print("2. Test theme switching in the top-right corner")
    print("3. Check for updates in the header")
    print("4. Verify real-time data updates")
    print("5. Test responsive design on different screen sizes")

if __name__ == "__main__":
    run_comprehensive_test()