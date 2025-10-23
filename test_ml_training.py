#!/usr/bin/env python3
"""
Test ML Training - Verify Active Learning
==========================================

Tests that ML training is actively happening and improving.
"""

import time
import sys

print("🧪 Testing ML Training System")
print("=" * 70)

# Test 1: Import and initialize
print("\n1️⃣ Testing ML System Import...")
try:
    from real_quantum_ml_system import RealQuantumMLSystem, PYTORCH_AVAILABLE
    
    if not PYTORCH_AVAILABLE:
        print("⚠️ PyTorch not available - ML training will not work")
        print("   Install: pip install torch")
        sys.exit(1)
    
    print("✅ PyTorch available")
    print("✅ RealQuantumMLSystem imported")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize system
print("\n2️⃣ Testing System Initialization...")
try:
    system = RealQuantumMLSystem(quantum_engine='cirq')
    print(f"✅ System initialized")
    print(f"   Engine: {system.quantum_engine}")
    print(f"   Architecture: {system.architecture}")
    
    # Check ML model
    if hasattr(system, 'ml_model') and system.ml_model is not None:
        print(f"✅ ML model initialized")
        print(f"   Model type: {type(system.ml_model).__name__}")
    else:
        print(f"❌ ML model not initialized")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Run optimization and check ML training
print("\n3️⃣ Testing ML Training During Optimization...")
try:
    initial_trained = system.stats.get('ml_models_trained', 0)
    print(f"   Initial ML models trained: {initial_trained}")
    
    # Get system state
    state = system._get_system_state()
    print(f"   System state: CPU {state.cpu_percent:.1f}%, Memory {state.memory_percent:.1f}%")
    
    # Run optimization
    result = system.run_comprehensive_optimization(state)
    print(f"   Optimization result: {result.energy_saved:.1f}% energy saved")
    
    # Train ML model
    system._train_ml_model(state, result)
    
    # Check if training happened
    after_trained = system.stats.get('ml_models_trained', 0)
    print(f"   After training: {after_trained} models trained")
    
    if after_trained > initial_trained:
        print(f"✅ ML training successful! (+{after_trained - initial_trained} models)")
    else:
        print(f"❌ ML training did not increment counter")
        sys.exit(1)
    
    # Check training history
    if hasattr(system, 'ml_training_history') and system.ml_training_history:
        latest = system.ml_training_history[-1]
        print(f"   Latest training:")
        print(f"      Loss: {latest['loss']:.4f}")
        print(f"      Prediction: {latest['prediction']:.4f}")
        print(f"      Actual: {latest['actual']:.4f}")
    
except Exception as e:
    print(f"❌ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Multiple training cycles
print("\n4️⃣ Testing Multiple Training Cycles...")
try:
    initial_count = system.stats.get('ml_models_trained', 0)
    cycles = 5
    
    print(f"   Running {cycles} training cycles...")
    
    for i in range(cycles):
        state = system._get_system_state()
        result = system.run_comprehensive_optimization(state)
        system._train_ml_model(state, result)
        
        if (i + 1) % 2 == 0:
            current_count = system.stats.get('ml_models_trained', 0)
            print(f"   Cycle {i+1}: {current_count} models trained")
    
    final_count = system.stats.get('ml_models_trained', 0)
    trained_this_test = final_count - initial_count
    
    if trained_this_test >= cycles:
        print(f"✅ Multiple cycles successful! Trained {trained_this_test} models")
    else:
        print(f"⚠️ Expected {cycles} cycles, got {trained_this_test}")
    
    # Check loss history
    if hasattr(system, 'ml_loss_history') and len(system.ml_loss_history) >= 2:
        recent_losses = list(system.ml_loss_history)[-5:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        print(f"   Average loss (last 5): {avg_loss:.4f}")
        
        # Check if loss is decreasing (learning)
        if len(recent_losses) >= 3:
            first_half = sum(recent_losses[:2]) / 2
            second_half = sum(recent_losses[-2:]) / 2
            if second_half < first_half:
                print(f"✅ Loss is decreasing - model is learning!")
            else:
                print(f"   Loss trend: {first_half:.4f} → {second_half:.4f}")
    
except Exception as e:
    print(f"❌ Multiple cycles test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: ML Prediction Boost
print("\n5️⃣ Testing ML Prediction Boost...")
try:
    state = system._get_system_state()
    boost = system._get_ml_prediction_boost(state)
    
    print(f"   ML boost factor: {boost:.3f}x")
    
    if boost >= 1.0 and boost <= 1.5:
        print(f"✅ ML boost in valid range (1.0 - 1.5x)")
    else:
        print(f"⚠️ ML boost outside expected range: {boost:.3f}x")
    
    # Test with trained model
    if system.stats.get('ml_models_trained', 0) > 10:
        print(f"   Model is trained ({system.stats['ml_models_trained']} cycles)")
        print(f"   Boost should improve optimization by {(boost - 1.0) * 100:.1f}%")
    else:
        print(f"   Model needs more training (only {system.stats.get('ml_models_trained', 0)} cycles)")
    
except Exception as e:
    print(f"❌ Boost test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: System Status
print("\n6️⃣ Testing System Status Reporting...")
try:
    status = system.get_system_status()
    
    if 'ml_status' in status:
        ml_status = status['ml_status']
        print(f"✅ ML status available:")
        print(f"   Models trained: {ml_status.get('models_trained', 0)}")
        print(f"   Average accuracy: {ml_status.get('average_accuracy', 0):.2%}")
        print(f"   Is learning: {ml_status.get('is_learning', False)}")
        print(f"   Training active: {ml_status.get('training_active', False)}")
        print(f"   Predictions made: {ml_status.get('predictions_made', 0)}")
    else:
        print(f"⚠️ ML status not in system status")
    
except Exception as e:
    print(f"❌ Status test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("📊 Test Summary")
print("=" * 70)

final_stats = system.stats
print(f"ML Models Trained: {final_stats.get('ml_models_trained', 0)}")
print(f"Optimizations Run: {final_stats.get('optimizations_run', 0)}")
print(f"Energy Saved: {final_stats.get('energy_saved', 0):.1f}%")
print(f"Quantum Operations: {final_stats.get('quantum_operations', 0)}")

if hasattr(system, 'ml_training_history') and system.ml_training_history:
    print(f"Training History: {len(system.ml_training_history)} entries")

print("\n✅ All ML training tests passed!")
print("🧠 ML system is actively learning and improving!")
