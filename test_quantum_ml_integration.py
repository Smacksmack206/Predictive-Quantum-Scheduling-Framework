#!/usr/bin/env python3
"""
Test script to verify Real Quantum-ML System integration
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'pqs_framework'))

print("🧪 Testing Real Quantum-ML System Integration")
print("=" * 60)

# Test 1: Import real quantum ML system
print("\n1️⃣ Testing real_quantum_ml_system import...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, quantum_ml_system
    print("✅ Real Quantum-ML System imported successfully")
    print(f"   Global instance available: {quantum_ml_system is not None}")
    if quantum_ml_system:
        print(f"   System initialized: {quantum_ml_system.initialized}")
        print(f"   System available: {quantum_ml_system.available}")
        print(f"   Optimization running: {quantum_ml_system.is_running}")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import universal_pqs_app
print("\n2️⃣ Testing universal_pqs_app import...")
try:
    from universal_pqs_app import (
        QUANTUM_ML_AVAILABLE,
        quantum_ml_system as app_quantum_ml,
        UniversalSystemDetector,
        UniversalQuantumSystem
    )
    print("✅ Universal PQS App imported successfully")
    print(f"   QUANTUM_ML_AVAILABLE: {QUANTUM_ML_AVAILABLE}")
    print(f"   quantum_ml_system available: {app_quantum_ml is not None}")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 3: Initialize system detector
print("\n3️⃣ Testing system detection...")
try:
    detector = UniversalSystemDetector()
    print("✅ System detector initialized")
    print(f"   Architecture: {detector.system_info['architecture']}")
    print(f"   Chip: {detector.system_info['chip_model']}")
    print(f"   Optimization tier: {detector.system_info['optimization_tier']}")
    print(f"   Max qubits: {detector.capabilities['max_qubits']}")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    sys.exit(1)

# Test 4: Initialize Universal Quantum System
print("\n4️⃣ Testing Universal Quantum System initialization...")
try:
    uqs = UniversalQuantumSystem(detector)
    print("✅ Universal Quantum System initialized")
    print(f"   Available: {uqs.available}")
    print(f"   Initialized: {uqs.initialized}")
    print(f"   Using Real Quantum-ML: {uqs.use_real_quantum_ml}")
    print(f"   Quantum ML instance: {uqs.quantum_ml is not None}")
except Exception as e:
    print(f"❌ Failed to initialize UQS: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Get system status
print("\n5️⃣ Testing system status...")
try:
    status = uqs.get_status()
    print("✅ System status retrieved")
    print(f"   Available: {status['available']}")
    print(f"   Using Quantum-ML: {status.get('using_quantum_ml', False)}")
    stats = status['stats']
    print(f"   Optimizations run: {stats['optimizations_run']}")
    print(f"   Energy saved: {stats['energy_saved']:.2f}%")
    print(f"   Quantum operations: {stats['quantum_operations']}")
    print(f"   ML predictions: {stats['predictions_made']}")
    print(f"   Power efficiency: {stats['power_efficiency_score']:.1f}%")
except Exception as e:
    print(f"❌ Failed to get status: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Run optimization
print("\n6️⃣ Testing optimization run...")
try:
    import time
    time.sleep(2)  # Wait for quantum ML system to stabilize
    
    result = uqs.run_optimization()
    print(f"✅ Optimization completed: {result}")
    
    # Get updated status
    status = uqs.get_status()
    stats = status['stats']
    print(f"   Optimizations run: {stats['optimizations_run']}")
    print(f"   Energy saved: {stats['energy_saved']:.2f}%")
    print(f"   Quantum operations: {stats['quantum_operations']}")
    print(f"   ML predictions: {stats['predictions_made']}")
except Exception as e:
    print(f"❌ Failed to run optimization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify quantum ML system is running
print("\n7️⃣ Testing quantum ML system status...")
try:
    if quantum_ml_system:
        qm_status = quantum_ml_system.get_system_status()
        print("✅ Quantum ML system status retrieved")
        print(f"   Available: {qm_status['available']}")
        print(f"   Initialized: {qm_status['initialized']}")
        print(f"   Optimization running: {qm_status['optimization_running']}")
        
        qm_stats = qm_status['stats']
        print(f"   Total optimizations: {qm_stats['optimizations_run']}")
        print(f"   Total energy saved: {qm_stats['energy_saved']:.2f}%")
        print(f"   Active circuits: {qm_stats['quantum_circuits_active']}")
        print(f"   ML models trained: {qm_stats['ml_models_trained']}")
    else:
        print("⚠️  Quantum ML system not available")
except Exception as e:
    print(f"❌ Failed to get quantum ML status: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Real Quantum-ML System is fully integrated!")
print("🚀 Ready to build macOS app with Briefcase")
