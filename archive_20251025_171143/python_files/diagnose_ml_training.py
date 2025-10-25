#!/usr/bin/env python3
"""
Diagnose ML Training Issue
===========================

Checks why ML Models Trained shows as 0
"""

import sys
import time

print("🔍 Diagnosing ML Training Issue")
print("=" * 70)

# Check 1: PyTorch availability
print("\n1️⃣ Checking PyTorch...")
try:
    import torch
    import torch.nn as nn
    print(f"✅ PyTorch {torch.__version__} available")
    PYTORCH_OK = True
except ImportError as e:
    print(f"❌ PyTorch NOT available: {e}")
    print("   This is the problem! Install: pip install torch")
    PYTORCH_OK = False

# Check 2: Import real_quantum_ml_system
print("\n2️⃣ Checking real_quantum_ml_system...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, PYTORCH_AVAILABLE
    print(f"✅ Module imported")
    print(f"   PYTORCH_AVAILABLE flag: {PYTORCH_AVAILABLE}")
    
    if not PYTORCH_AVAILABLE:
        print(f"   ⚠️ Module says PyTorch not available!")
        print(f"   This means ML training will use fallback mode")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Check 3: Initialize system
print("\n3️⃣ Initializing system...")
try:
    system = RealQuantumMLSystem(quantum_engine='cirq')
    print(f"✅ System initialized")
    
    # Check ML model
    if hasattr(system, 'ml_model'):
        if system.ml_model is not None:
            print(f"✅ ML model exists: {type(system.ml_model).__name__}")
        else:
            print(f"⚠️ ML model is None (PyTorch not available)")
    else:
        print(f"❌ No ml_model attribute")
    
    # Check stats
    print(f"\n📊 Initial stats:")
    print(f"   ml_models_trained: {system.stats.get('ml_models_trained', 'KEY MISSING')}")
    print(f"   optimizations_run: {system.stats.get('optimizations_run', 'KEY MISSING')}")
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check 4: Run optimization and training
print("\n4️⃣ Running optimization cycle...")
try:
    initial_count = system.stats.get('ml_models_trained', 0)
    print(f"   Initial ml_models_trained: {initial_count}")
    
    # Get state
    state = system._get_system_state()
    print(f"   System state: CPU {state.cpu_percent:.1f}%, Memory {state.memory_percent:.1f}%")
    
    # Run optimization
    result = system.run_comprehensive_optimization(state)
    print(f"   Optimization: {result.energy_saved:.1f}% energy saved")
    
    # Check if training was called
    if PYTORCH_AVAILABLE and hasattr(system, 'ml_model') and system.ml_model is not None:
        print(f"   ✅ Conditions met for ML training")
        system._train_ml_model(state, result)
        print(f"   ✅ _train_ml_model() called")
    else:
        print(f"   ⚠️ ML training conditions NOT met:")
        print(f"      PYTORCH_AVAILABLE: {PYTORCH_AVAILABLE}")
        print(f"      has ml_model: {hasattr(system, 'ml_model')}")
        print(f"      ml_model not None: {system.ml_model is not None if hasattr(system, 'ml_model') else False}")
    
    # Check count after
    after_count = system.stats.get('ml_models_trained', 0)
    print(f"   After ml_models_trained: {after_count}")
    
    if after_count > initial_count:
        print(f"   ✅ SUCCESS! Counter incremented by {after_count - initial_count}")
    else:
        print(f"   ❌ PROBLEM! Counter did not increment")
        print(f"   This is why dashboard shows 0!")
    
except Exception as e:
    print(f"❌ Optimization failed: {e}")
    import traceback
    traceback.print_exc()

# Check 5: Check get_system_status
print("\n5️⃣ Checking get_system_status()...")
try:
    status = system.get_system_status()
    
    if 'stats' in status:
        stats = status['stats']
        print(f"✅ Stats in status:")
        print(f"   ml_models_trained: {stats.get('ml_models_trained', 'MISSING')}")
        print(f"   optimizations_run: {stats.get('optimizations_run', 'MISSING')}")
    else:
        print(f"❌ No 'stats' key in status")
    
    if 'ml_status' in status:
        ml_status = status['ml_status']
        print(f"✅ ML status:")
        print(f"   models_trained: {ml_status.get('models_trained', 'MISSING')}")
        print(f"   is_learning: {ml_status.get('is_learning', 'MISSING')}")
        print(f"   training_active: {ml_status.get('training_active', 'MISSING')}")
    else:
        print(f"⚠️ No 'ml_status' key in status")
    
except Exception as e:
    print(f"❌ Status check failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("📊 Diagnosis Summary")
print("=" * 70)

if not PYTORCH_OK:
    print("❌ ROOT CAUSE: PyTorch is not installed!")
    print("   Solution: pip install torch")
    print("   Without PyTorch, ML training uses fallback mode")
elif not PYTORCH_AVAILABLE:
    print("⚠️ PyTorch installed but module doesn't detect it")
    print("   Check import errors in real_quantum_ml_system.py")
elif not hasattr(system, 'ml_model') or system.ml_model is None:
    print("❌ ML model not initialized")
    print("   Check _initialize_ml_components() method")
elif system.stats.get('ml_models_trained', 0) == 0:
    print("❌ ML training not incrementing counter")
    print("   Check _train_ml_model() method")
else:
    print("✅ Everything looks good!")
    print(f"   ML models trained: {system.stats.get('ml_models_trained', 0)}")

print("\n🔍 Diagnosis complete!")
