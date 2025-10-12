#!/usr/bin/env python3
"""
Test Ultimate EAS default enabled functionality
"""

import requests
import time
import json

def test_ultimate_eas_default():
    """Test that Ultimate EAS is enabled by default"""
    print("🧪 Testing Ultimate EAS Default Enabled...")
    
    try:
        # Test the quantum status API
        response = requests.get("http://localhost:9010/api/quantum-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Check if ultimate_eas_enabled is True by default
            ultimate_enabled = data.get('ultimate_eas_enabled', False)
            print(f"   Ultimate EAS Enabled: {ultimate_enabled}")
            
            if ultimate_enabled:
                print("   ✅ Ultimate EAS is enabled by default!")
                
                # Check system status
                if 'system_status' in data:
                    system_status = data['system_status']
                    system_enabled = system_status.get('ultimate_eas_enabled', False)
                    print(f"   System Status Enabled: {system_enabled}")
                    
                    if system_enabled:
                        print("   ✅ System status also shows enabled!")
                    else:
                        print("   ❌ System status shows disabled")
                        return False
                
                # Check for quantum metrics
                if 'quantum_metrics' in data:
                    quantum_metrics = data['quantum_metrics']
                    print(f"   Quantum Operations: {quantum_metrics.get('quantum_operations', 0)}")
                    print(f"   Neural Classifications: {quantum_metrics.get('neural_classifications', 0)}")
                    print("   ✅ Quantum metrics are available!")
                
                return True
            else:
                print("   ❌ Ultimate EAS is not enabled by default")
                return False
        else:
            print(f"   ❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_view_ultimate_eas():
    """Test that View Ultimate EAS Status works"""
    print("\n🧪 Testing View Ultimate EAS Status functionality...")
    
    # This would normally be tested through the menu, but we can check
    # if the required data is available via API
    try:
        response = requests.get("http://localhost:9010/api/quantum-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have the required fields for the status view
            required_fields = ['quantum_operations', 'neural_classifications', 'energy_predictions']
            
            if 'quantum_metrics' in data:
                quantum_metrics = data['quantum_metrics']
                missing_fields = []
                
                for field in required_fields:
                    if field not in quantum_metrics:
                        missing_fields.append(field)
                
                if not missing_fields:
                    print("   ✅ All required fields for View Ultimate EAS Status are available!")
                    return True
                else:
                    print(f"   ❌ Missing fields: {missing_fields}")
                    return False
            else:
                print("   ❌ No quantum_metrics in response")
                return False
        else:
            print(f"   ❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🌟" + "=" * 50 + "🌟")
    print("🧪 ULTIMATE EAS DEFAULT ENABLED TEST")
    print("🌟" + "=" * 50 + "🌟")
    
    print("\n1. Make sure the enhanced_app.py is running")
    print("2. This test will check if Ultimate EAS is enabled by default")
    print("3. And verify that View Ultimate EAS Status has all required data")
    
    input("\nPress Enter to start the test...")
    
    # Test 1: Default enabled
    test1_result = test_ultimate_eas_default()
    
    # Test 2: View functionality
    test2_result = test_view_ultimate_eas()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Ultimate EAS Default Enabled: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   View Ultimate EAS Status: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED! Ultimate EAS is working correctly!")
    else:
        print("\n❌ Some tests failed. Check the issues above.")
    
    print("=" * 60)