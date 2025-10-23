#!/usr/bin/env python3
"""
Test if the methods exist in UniversalQuantumSystem
"""

import sys
import os
import psutil

# Mock the missing modules
class MockRumps:
    class App:
        def __init__(self, name):
            self.name = name
        def run(self):
            pass
    def clicked(self, name):
        def decorator(func):
            return func
        return decorator

sys.modules['rumps'] = MockRumps()

try:
    from universal_pqs_app import UniversalQuantumSystem, UniversalSystemDetector
    
    print("🧪 Testing UniversalQuantumSystem methods...")
    
    # Create a system detector
    detector = UniversalSystemDetector()
    
    # Create a quantum system
    quantum_system = UniversalQuantumSystem(detector)
    
    # Check if methods exist
    methods_to_check = [
        '_get_system_processes',
        '_calculate_quantum_operations', 
        '_calculate_real_energy_savings'
    ]
    
    for method_name in methods_to_check:
        if hasattr(quantum_system, method_name):
            print(f"✅ {method_name} exists")
            try:
                method = getattr(quantum_system, method_name)
                result = method()
                print(f"   Result: {result}")
            except Exception as e:
                print(f"   Error calling: {e}")
        else:
            print(f"❌ {method_name} missing")
    
    print("🎉 Method check complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()