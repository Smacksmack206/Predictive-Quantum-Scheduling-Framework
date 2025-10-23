#!/usr/bin/env python3
"""
Test Dual Quantum Engine Support
=================================

Tests both Cirq and Qiskit engines to ensure they work correctly.
"""

import sys
import time

print("🧪 Testing Dual Quantum Engine Support")
print("=" * 70)

# Test Cirq
print("\n1️⃣ Testing Cirq Engine...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, CIRQ_AVAILABLE
    
    if CIRQ_AVAILABLE:
        print("✅ Cirq available")
        
        # Initialize with Cirq
        system_cirq = RealQuantumMLSystem(quantum_engine='cirq')
        print(f"✅ Cirq system initialized")
        print(f"   Engine: {system_cirq.quantum_engine}")
        print(f"   Architecture: {system_cirq.architecture}")
        
        # Test optimization
        test_processes = [
            {'pid': 1, 'name': 'Chrome', 'cpu': 45.2, 'memory': 15.3},
            {'pid': 2, 'name': 'VSCode', 'cpu': 32.1, 'memory': 12.1},
            {'pid': 3, 'name': 'Slack', 'cpu': 18.5, 'memory': 8.2},
        ]
        
        from real_quantum_ml_system import SystemState
        state = system_cirq._get_system_state()
        result = system_cirq.run_comprehensive_optimization(state)
        
        print(f"✅ Cirq optimization successful")
        print(f"   Energy saved: {result.energy_saved:.1f}%")
        print(f"   Quantum advantage: {result.quantum_advantage:.2f}x")
        
    else:
        print("⚠️ Cirq not available - install: pip install cirq")
        
except Exception as e:
    print(f"❌ Cirq test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Qiskit
print("\n2️⃣ Testing Qiskit Engine...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, QISKIT_AVAILABLE
    
    if QISKIT_AVAILABLE:
        print("✅ Qiskit available")
        
        # Initialize with Qiskit
        system_qiskit = RealQuantumMLSystem(quantum_engine='qiskit')
        print(f"✅ Qiskit system initialized")
        print(f"   Engine: {system_qiskit.quantum_engine}")
        print(f"   Architecture: {system_qiskit.architecture}")
        
        # Test optimization
        state = system_qiskit._get_system_state()
        result = system_qiskit.run_comprehensive_optimization(state)
        
        print(f"✅ Qiskit optimization successful")
        print(f"   Energy saved: {result.energy_saved:.1f}%")
        print(f"   Quantum advantage: {result.quantum_advantage:.2f}x")
        
        # Test Qiskit-specific features
        if hasattr(system_qiskit, 'qiskit_engine'):
            print("\n🔬 Testing Qiskit-specific features...")
            qiskit_engine = system_qiskit.qiskit_engine
            
            test_processes = [
                {'pid': 1, 'name': 'Chrome', 'cpu': 45.2, 'memory': 15.3},
                {'pid': 2, 'name': 'VSCode', 'cpu': 32.1, 'memory': 12.1},
                {'pid': 3, 'name': 'Slack', 'cpu': 18.5, 'memory': 8.2},
                {'pid': 4, 'name': 'Terminal', 'cpu': 5.2, 'memory': 2.1},
                {'pid': 5, 'name': 'Finder', 'cpu': 3.1, 'memory': 1.5},
            ]
            
            qiskit_result = qiskit_engine.optimize_processes(test_processes)
            
            print(f"✅ Qiskit engine test successful")
            print(f"   Algorithm: {qiskit_result.algorithm}")
            print(f"   Energy saved: {qiskit_result.energy_savings:.1f}%")
            print(f"   Quantum advantage: {qiskit_result.quantum_advantage:.2f}x")
            print(f"   Circuit depth: {qiskit_result.circuit_depth}")
            print(f"   Qubits used: {qiskit_result.qubits_used}")
            print(f"   Confidence: {qiskit_result.confidence:.1%}")
            
            # Test quantum advantage demonstration
            print("\n🎯 Testing quantum advantage demonstration...")
            advantage = qiskit_engine.demonstrate_quantum_advantage(test_processes)
            
            if advantage.get('advantage_demonstrated'):
                print(f"✅ QUANTUM ADVANTAGE DEMONSTRATED!")
                print(f"   Speedup: {advantage['speedup']:.2f}x")
                print(f"   Energy improvement: {advantage['energy_improvement_percent']:.1f}%")
            else:
                print(f"⚠️ Quantum advantage: {advantage.get('speedup', 1.0):.2f}x (needs larger problem)")
        
    else:
        print("⚠️ Qiskit not available - install: pip install qiskit qiskit-algorithms qiskit-aer")
        
except Exception as e:
    print(f"❌ Qiskit test failed: {e}")
    import traceback
    traceback.print_exc()

# Test engine switching
print("\n3️⃣ Testing Engine Switching...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, CIRQ_AVAILABLE, QISKIT_AVAILABLE
    
    if CIRQ_AVAILABLE and QISKIT_AVAILABLE:
        print("✅ Both engines available - testing switch")
        
        # Start with Cirq
        system1 = RealQuantumMLSystem(quantum_engine='cirq')
        assert system1.quantum_engine == 'cirq', "Should be Cirq"
        print(f"✅ System 1: {system1.quantum_engine}")
        
        # Switch to Qiskit
        system2 = RealQuantumMLSystem(quantum_engine='qiskit')
        assert system2.quantum_engine == 'qiskit', "Should be Qiskit"
        print(f"✅ System 2: {system2.quantum_engine}")
        
        print("✅ Engine switching works correctly")
        
    elif CIRQ_AVAILABLE:
        print("⚠️ Only Cirq available - testing fallback")
        system = RealQuantumMLSystem(quantum_engine='qiskit')
        assert system.quantum_engine == 'cirq', "Should fallback to Cirq"
        print(f"✅ Correctly fell back to: {system.quantum_engine}")
        
    elif QISKIT_AVAILABLE:
        print("⚠️ Only Qiskit available - testing fallback")
        system = RealQuantumMLSystem(quantum_engine='cirq')
        assert system.quantum_engine == 'qiskit', "Should fallback to Qiskit"
        print(f"✅ Correctly fell back to: {system.quantum_engine}")
        
    else:
        print("⚠️ No quantum engines available - testing classical fallback")
        system = RealQuantumMLSystem(quantum_engine='cirq')
        assert system.quantum_engine == 'classical', "Should fallback to classical"
        print(f"✅ Correctly fell back to: {system.quantum_engine}")
        
except Exception as e:
    print(f"❌ Engine switching test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("📊 Test Summary")
print("=" * 70)

from real_quantum_ml_system import CIRQ_AVAILABLE, QISKIT_AVAILABLE

print(f"Cirq Available: {'✅' if CIRQ_AVAILABLE else '❌'}")
print(f"Qiskit Available: {'✅' if QISKIT_AVAILABLE else '❌'}")

if CIRQ_AVAILABLE and QISKIT_AVAILABLE:
    print("\n🎉 FULL DUAL ENGINE SUPPORT ACTIVE!")
    print("   Users can choose between Cirq (optimized) and Qiskit (experimental)")
elif CIRQ_AVAILABLE:
    print("\n✅ Cirq engine active (Qiskit optional)")
elif QISKIT_AVAILABLE:
    print("\n✅ Qiskit engine active (Cirq optional)")
else:
    print("\n⚠️ No quantum engines available - classical fallback only")

print("\n✅ All tests complete!")
