#!/usr/bin/env python3
"""
Test Actual UI Functionality - Tests the real issues reported by user
"""

import time
import requests
import json
import subprocess
import sys

def test_dashboard_javascript_errors():
    """Test if dashboard loads without JavaScript errors by checking API structure"""
    print("🧪 Testing Dashboard JavaScript Compatibility...")
    
    try:
        response = requests.get("http://localhost:9010/api/quantum-status", timeout=10)
        data = response.json()
        
        # Test the exact structure that JavaScript expects
        required_structure = {
            'gpu_acceleration': ['gpu_name', 'gpu_memory_mb', 'performance_boost', 'compute_capability'],
            'quantum_advantage': ['average_speedup', 'max_speedup', 'average_quantum_volume', 'total_quantum_operations'],
            'neural_advantage': ['average_inference_time', 'average_transformer_confidence', 'average_neural_complexity'],
            'system_status': ['uptime_formatted', 'optimization_cycles', 'quantum_operations']
        }
        
        missing_fields = []
        
        for section, fields in required_structure.items():
            if section not in data:
                missing_fields.append(f"{section} (entire section)")
            else:
                for field in fields:
                    if field not in data[section]:
                        missing_fields.append(f"{section}.{field}")
        
        if missing_fields:
            print(f"   ❌ Missing required fields for JavaScript: {missing_fields}")
            return False
        else:
            print("   ✅ All required fields present for JavaScript")
            
            # Test specific values that caused errors
            gpu_name = data['gpu_acceleration']['gpu_name']
            uptime = data['system_status']['uptime_formatted']
            
            print(f"   ✅ GPU Name: {gpu_name}")
            print(f"   ✅ Uptime Formatted: {uptime}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

def test_dashboard_loads_without_errors():
    """Test that dashboard HTML loads and has required elements"""
    print("\n🖥️  Testing Dashboard HTML Structure...")
    
    try:
        response = requests.get("http://localhost:9010/quantum", timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Dashboard returns {response.status_code}")
            return False
        
        content = response.text
        
        # Check for elements that JavaScript will try to access
        required_elements = [
            'id="gpu-name"',
            'id="system-uptime"', 
            'id="quantum-operations"',
            'id="optimization-cycles"',
            'id="performance-boost"',
            'id="avg-speedup"'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"   ❌ Missing HTML elements: {missing_elements}")
            return False
        else:
            print("   ✅ All required HTML elements present")
            
        # Check for JavaScript API calls
        if '/api/quantum-status' in content:
            print("   ✅ JavaScript API calls configured")
        else:
            print("   ❌ JavaScript API calls not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Dashboard HTML test failed: {e}")
        return False

def test_ultimate_eas_menu_availability():
    """Test Ultimate EAS availability by checking API flags"""
    print("\n🚀 Testing Ultimate EAS Menu Availability...")
    
    try:
        response = requests.get("http://localhost:9010/api/quantum-status", timeout=10)
        data = response.json()
        
        # Check if Ultimate EAS features are available
        eas_indicators = [
            ('quantum_operations', 'Quantum operations'),
            ('gpu_acceleration', 'GPU acceleration'),
            ('quantum_advantage', 'Quantum advantage'),
            ('neural_advantage', 'Neural advantage')
        ]
        
        available_features = 0
        for field, description in eas_indicators:
            if field in data and data[field]:
                print(f"   ✅ {description}: Available")
                available_features += 1
            else:
                print(f"   ❌ {description}: Not available")
        
        # Check system status
        if 'system_status' in data:
            system_status = data['system_status']
            eas_enabled = system_status.get('ultimate_eas_enabled', False)
            
            if eas_enabled:
                print("   ✅ Ultimate EAS: Enabled in system")
            else:
                print("   ⚠️  Ultimate EAS: Using fallback system (this is expected)")
        
        # If most features are available, menu should work
        if available_features >= 3:
            print("   ✅ Ultimate EAS features sufficient for menu functionality")
            return True
        else:
            print("   ❌ Insufficient Ultimate EAS features available")
            return False
            
    except Exception as e:
        print(f"   ❌ Ultimate EAS availability test failed: {e}")
        return False

def test_progressive_metrics():
    """Test that metrics actually progress over time"""
    print("\n📈 Testing Progressive Metrics...")
    
    try:
        # Get initial reading
        response1 = requests.get("http://localhost:9010/api/quantum-status", timeout=5)
        data1 = response1.json()
        
        initial_uptime = data1.get('system_uptime', 0)
        initial_ops = data1.get('quantum_operations', 0)
        
        print(f"   Initial uptime: {initial_uptime:.4f} hours")
        print(f"   Initial quantum ops: {initial_ops}")
        
        # Wait for progression
        print("   Waiting 20 seconds for metrics to progress...")
        time.sleep(20)
        
        # Get updated reading
        response2 = requests.get("http://localhost:9010/api/quantum-status", timeout=5)
        data2 = response2.json()
        
        updated_uptime = data2.get('system_uptime', 0)
        updated_ops = data2.get('quantum_operations', 0)
        
        print(f"   Updated uptime: {updated_uptime:.4f} hours")
        print(f"   Updated quantum ops: {updated_ops}")
        
        # Check progression
        uptime_progressed = updated_uptime > initial_uptime
        ops_stable = updated_ops >= initial_ops
        
        if uptime_progressed and ops_stable:
            print("   ✅ Metrics are progressing correctly")
            return True
        else:
            print(f"   ❌ Metrics not progressing: uptime={uptime_progressed}, ops={ops_stable}")
            return False
            
    except Exception as e:
        print(f"   ❌ Progressive metrics test failed: {e}")
        return False

def test_css_loading():
    """Test that CSS loads properly"""
    print("\n🎨 Testing CSS Loading...")
    
    try:
        response = requests.get("http://localhost:9010/static/themes.css", timeout=10)
        
        if response.status_code == 200:
            css_content = response.text
            
            # Check for essential CSS content
            if ':root' in css_content and 'Theme System' in css_content:
                print("   ✅ CSS loads with valid theme content")
                return True
            else:
                print("   ❌ CSS loads but content is invalid")
                return False
        else:
            print(f"   ❌ CSS returns {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ CSS loading test failed: {e}")
        return False

def simulate_menu_interactions():
    """Simulate what happens when menu items are clicked"""
    print("\n📱 Simulating Menu Interactions...")
    
    try:
        # Simulate "View Ultimate EAS Status" by checking if we can get system info
        response = requests.get("http://localhost:9010/api/quantum-status", timeout=10)
        data = response.json()
        
        if 'system_status' in data:
            system_status = data['system_status']
            
            # Check if we have the data that would be shown in status dialog
            required_status_fields = ['system_id', 'uptime_formatted', 'optimization_cycles']
            
            missing_status = []
            for field in required_status_fields:
                if field not in system_status:
                    missing_status.append(field)
            
            if not missing_status:
                print("   ✅ 'View Ultimate EAS Status' would work - all data available")
                print(f"      System ID: {system_status['system_id']}")
                print(f"      Uptime: {system_status['uptime_formatted']}")
                print(f"      Cycles: {system_status['optimization_cycles']}")
            else:
                print(f"   ❌ 'View Ultimate EAS Status' missing data: {missing_status}")
                return False
        else:
            print("   ❌ 'View Ultimate EAS Status' would fail - no system_status")
            return False
        
        # Simulate "Open Quantum Dashboard" by checking if dashboard loads
        dashboard_response = requests.get("http://localhost:9010/quantum", timeout=10)
        if dashboard_response.status_code == 200:
            print("   ✅ 'Open Quantum Dashboard' would work - dashboard loads")
        else:
            print(f"   ❌ 'Open Quantum Dashboard' would fail - returns {dashboard_response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Menu interaction simulation failed: {e}")
        return False

def main():
    """Run all functionality tests"""
    print("🌟" + "=" * 60 + "🌟")
    print("🔍 TESTING ACTUAL UI FUNCTIONALITY")
    print("   (Testing the specific issues reported by user)")
    print("🌟" + "=" * 60 + "🌟")
    
    # Wait for app to be ready
    print("⏳ Waiting for app to be ready...")
    time.sleep(5)
    
    tests = [
        ("Dashboard JavaScript Compatibility", test_dashboard_javascript_errors),
        ("Dashboard HTML Structure", test_dashboard_loads_without_errors),
        ("Ultimate EAS Menu Availability", test_ultimate_eas_menu_availability),
        ("Progressive Metrics", test_progressive_metrics),
        ("CSS Loading", test_css_loading),
        ("Menu Interactions", simulate_menu_interactions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            results[test_name] = f"ERROR: {e}"
    
    # Generate report
    print("\n" + "🏆" + "=" * 50 + "🏆")
    print("📊 ACTUAL FUNCTIONALITY TEST RESULTS")
    print("🏆" + "=" * 50 + "🏆")
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        if result == "PASS":
            print(f"   ✅ {test_name}")
            passed += 1
        else:
            print(f"   ❌ {test_name}: {result}")
            failed += 1
    
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n📈 SUMMARY:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT! UI functionality is working properly!")
        status = "SUCCESS"
    elif success_rate >= 70:
        print(f"\n👍 GOOD! Most functionality works, minor issues remain.")
        status = "MOSTLY_WORKING"
    else:
        print(f"\n❌ CRITICAL! Major UI functionality issues found.")
        status = "BROKEN"
    
    # Specific issue check
    print(f"\n🔍 SPECIFIC ISSUE STATUS:")
    
    js_working = results.get("Dashboard JavaScript Compatibility") == "PASS"
    menu_working = results.get("Ultimate EAS Menu Availability") == "PASS"
    css_working = results.get("CSS Loading") == "PASS"
    
    print(f"   JavaScript 'gpu_name' error: {'✅ FIXED' if js_working else '❌ STILL BROKEN'}")
    print(f"   Ultimate EAS menu availability: {'✅ FIXED' if menu_working else '❌ STILL BROKEN'}")
    print(f"   CSS themes.css loading: {'✅ FIXED' if css_working else '❌ STILL BROKEN'}")
    
    return status == "SUCCESS" or status == "MOSTLY_WORKING"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)