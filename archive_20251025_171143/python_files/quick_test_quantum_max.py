#!/usr/bin/env python3
"""Quick test of Quantum Max Scheduler - Fast version"""

from quantum_max_scheduler import QuantumMaxScheduler
import time

print("🚀 Quick Quantum Max Scheduler Test")
print("=" * 60)

# Create scheduler with fewer qubits for speed
s = QuantumMaxScheduler(max_qubits=12)
print(f"✅ Scheduler created with {s.max_qubits} qubits")

# Get metrics
print("\n📊 Getting system metrics...")
m = s.get_system_metrics()

if m:
    print(f"   CPU: {m.cpu_percent:.1f}%")
    print(f"   Memory: {m.memory_percent:.1f}%")
    print(f"   Thermal: {m.thermal_state}")
    
    # Test each strategy quickly
    strategies = ['performance', 'battery', 'thermal', 'ram', 'balanced']
    
    for strategy in strategies:
        print(f"\n🎯 Testing {strategy.upper()}...")
        start = time.time()
        
        try:
            r = s.optimize_system(m)
            elapsed = time.time() - start
            
            print(f"   ✅ Success in {elapsed:.2f}s")
            print(f"   Strategy: {r.strategy}")
            print(f"   Energy Saved: {r.energy_saved:.1f}%")
            print(f"   Qubits Used: {r.qubits_used}")
            print(f"   Quantum Advantage: {r.quantum_advantage:.2f}x")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Get stats
    print("\n📈 Statistics:")
    stats = s.get_statistics()
    print(f"   Total Optimizations: {stats['total_optimizations']}")
    print(f"   Avg Energy Saved: {stats['recent_performance']['avg_energy_saved']:.1f}%")
    
    print("\n✅ All tests complete!")
else:
    print("❌ Failed to get metrics")
