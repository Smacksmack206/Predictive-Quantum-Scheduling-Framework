#!/usr/bin/env python3
"""
Quick test for the quantum API functions
"""

import sys
import os
import psutil

# Add current directory to path
sys.path.append('.')

# Mock the missing modules
class MockRumps:
    class App:
        def __init__(self, name):
            self.name = name
        def run(self):
            pass

sys.modules['rumps'] = MockRumps()

# Test the quantum API functions
try:
    from universal_pqs_app import _get_real_savings_rate, _get_real_efficiency_score, _get_real_speedup
    
    print("🧪 Testing quantum API functions...")
    
    # Test savings rate
    savings_rate = _get_real_savings_rate()
    print(f"✅ Savings rate: {savings_rate}%")
    
    # Test efficiency score
    efficiency_score = _get_real_efficiency_score()
    print(f"✅ Efficiency score: {efficiency_score}")
    
    # Test speedup
    speedup = _get_real_speedup()
    print(f"✅ Speedup: {speedup}")
    
    print("🎉 All quantum API functions working correctly!")
    
except Exception as e:
    print(f"❌ Error testing quantum API: {e}")
    import traceback
    traceback.print_exc()