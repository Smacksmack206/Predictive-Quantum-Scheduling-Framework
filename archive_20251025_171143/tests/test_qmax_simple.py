#!/usr/bin/env python3
"""Simple test of Quantum Max Scheduler"""

from quantum_max_scheduler import QuantumMaxScheduler

print("🚀 Testing Quantum Max Scheduler")
print("=" * 60)

# Create scheduler
s = QuantumMaxScheduler(max_qubits=20)
print(f"✅ Scheduler created with {s.max_qubits} qubits")

# Get metrics
m = s.get_system_metrics()

if m:
    print(f"\n📊 System Metrics:")
    print(f"   CPU: {m.cpu_percent:.1f}%")
    print(f"   Memory: {m.memory_percent:.1f}%")
    print(f"   Thermal: {m.thermal_state}")
    
    # Run one optimization
    print(f"\n⚛️ Running optimization...")
    r = s.optimize_system(m)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Strategy: {r.strategy}")
    print(f"   Energy Saved: {r.energy_saved:.1f}%")
    print(f"   Performance Boost: {r.performance_boost:.1f}%")
    print(f"   Lag Reduction: {r.lag_reduction:.1f}%")
    print(f"   RAM Freed: {r.ram_freed_mb:.1f} MB")
    print(f"   Thermal Reduction: {r.thermal_reduction:.1f}%")
    print(f"   Quantum Advantage: {r.quantum_advantage:.2f}x")
    print(f"   Qubits Used: {r.qubits_used}")
    print(f"   Execution Time: {r.execution_time_ms:.2f} ms")
else:
    print("❌ Failed to get metrics")
