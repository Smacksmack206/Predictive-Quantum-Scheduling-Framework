#!/usr/bin/env python3
"""
Test Energy Savings Fix
========================

Validates that energy savings shows rolling average, not cumulative total
"""

import time

print("🧪 Testing Energy Savings Fix")
print("=" * 60)

# Test 1: Check quantum-ML system
print("\n1️⃣ Testing Quantum-ML System...")
try:
    from real_quantum_ml_system import get_quantum_ml_system
    
    qml = get_quantum_ml_system()
    if qml and qml.available:
        print(f"✅ Quantum-ML system available")
        
        # Check initial stats
        initial_energy = qml.stats.get('energy_saved', 0.0)
        print(f"   Initial energy saved: {initial_energy:.1f}%")
        
        # Run multiple optimizations
        print(f"\n   Running 5 optimization cycles...")
        savings_list = []
        
        for i in range(5):
            state = qml._get_system_state()
            result = qml.run_comprehensive_optimization(state)
            
            current_saved = qml.stats.get('energy_saved', 0.0)
            current_instant = qml.stats.get('current_energy_savings', 0.0)
            
            savings_list.append(result.energy_saved)
            
            print(f"   Cycle {i+1}:")
            print(f"      This cycle: {result.energy_saved:.1f}%")
            print(f"      Average: {current_saved:.1f}%")
            print(f"      Current: {current_instant:.1f}%")
            
            time.sleep(0.5)
        
        # Verify it's not accumulating to 100%
        final_energy = qml.stats.get('energy_saved', 0.0)
        print(f"\n   Final average energy saved: {final_energy:.1f}%")
        
        # Calculate expected average
        expected_avg = sum(savings_list) / len(savings_list)
        print(f"   Expected average: {expected_avg:.1f}%")
        
        if abs(final_energy - expected_avg) < 1.0:
            print(f"✅ Energy savings using rolling average (correct!)")
        else:
            print(f"⚠️  Energy savings mismatch")
            print(f"      Got: {final_energy:.1f}%")
            print(f"      Expected: {expected_avg:.1f}%")
        
        # Check it's not at 100%
        if final_energy < 50.0:
            print(f"✅ Energy savings is reasonable ({final_energy:.1f}% < 50%)")
        else:
            print(f"⚠️  Energy savings seems too high: {final_energy:.1f}%")
            
    else:
        print("❌ Quantum-ML system not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check energy savings history
print("\n2️⃣ Testing Energy Savings History...")
try:
    if hasattr(qml, 'energy_savings_history'):
        print(f"✅ Energy savings history exists")
        print(f"   History length: {len(qml.energy_savings_history)}")
        print(f"   Max length: {qml.energy_savings_history.maxlen}")
        
        if len(qml.energy_savings_history) > 0:
            print(f"   Recent savings: {list(qml.energy_savings_history)[-5:]}")
            avg = sum(qml.energy_savings_history) / len(qml.energy_savings_history)
            print(f"   Average: {avg:.1f}%")
    else:
        print(f"⚠️  Energy savings history not found")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Simulate many cycles to ensure it doesn't reach 100%
print("\n3️⃣ Testing Long-Term Accumulation...")
try:
    print(f"   Running 20 more optimization cycles...")
    
    for i in range(20):
        state = qml._get_system_state()
        result = qml.run_comprehensive_optimization(state)
        
        if (i + 1) % 5 == 0:
            current_saved = qml.stats.get('energy_saved', 0.0)
            print(f"   After {i+1} cycles: {current_saved:.1f}%")
    
    final_energy = qml.stats.get('energy_saved', 0.0)
    print(f"\n   Final average after 25 total cycles: {final_energy:.1f}%")
    
    if final_energy < 50.0:
        print(f"✅ Energy savings stable and reasonable")
    elif final_energy < 80.0:
        print(f"⚠️  Energy savings high but not at cap: {final_energy:.1f}%")
    else:
        print(f"❌ Energy savings too high: {final_energy:.1f}%")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("📊 Test Summary")
print("=" * 60)

try:
    print(f"Total Optimizations: {qml.stats['optimizations_run']}")
    print(f"Average Energy Saved: {qml.stats['energy_saved']:.1f}%")
    print(f"Current Energy Savings: {qml.stats.get('current_energy_savings', 0):.1f}%")
    print(f"History Length: {len(qml.energy_savings_history)}")
    
    if qml.stats['energy_saved'] < 50.0:
        print(f"\n✅ Energy savings calculation is working correctly!")
        print(f"   Using rolling average instead of cumulative total")
        print(f"   Average: {qml.stats['energy_saved']:.1f}%")
    else:
        print(f"\n⚠️  Energy savings may still be accumulating")
        print(f"   Current value: {qml.stats['energy_saved']:.1f}%")
        
except Exception as e:
    print(f"⚠️  Could not generate summary: {e}")

print("\n💡 Expected Behavior:")
print("   • Energy saved should be 15-25% (rolling average)")
print("   • Should NOT accumulate to 100%")
print("   • Should stabilize after ~10 cycles")
print("   • Current savings may vary per cycle")
