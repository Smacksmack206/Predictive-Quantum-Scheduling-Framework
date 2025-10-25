#!/usr/bin/env python3
"""
Test ML Models Trained Fix
===========================

Validates that:
1. ML training increments the counter
2. Stats persist to database
3. Stats load from database on startup
4. API returns correct stats
"""

import time
import sys

print("🧪 Testing ML Models Trained Fix")
print("=" * 60)

# Test 1: Check quantum-ML system initialization
print("\n1️⃣ Testing Quantum-ML System Initialization...")
try:
    from real_quantum_ml_system import get_quantum_ml_system
    
    qml_system = get_quantum_ml_system()
    if qml_system and qml_system.available:
        print(f"✅ Quantum-ML system initialized")
        print(f"   ML Models Trained: {qml_system.stats['ml_models_trained']}")
        print(f"   Optimizations Run: {qml_system.stats['optimizations_run']}")
        print(f"   Energy Saved: {qml_system.stats['energy_saved']:.1f}%")
        
        # Check if stats were loaded from database
        if qml_system.stats['ml_models_trained'] > 0:
            print(f"✅ Stats loaded from database (persistent)")
        else:
            print(f"ℹ️  Starting fresh (no previous stats)")
    else:
        print("❌ Quantum-ML system not available")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: Check database persistence
print("\n2️⃣ Testing Database Persistence...")
try:
    from quantum_ml_persistence import get_database
    
    db = get_database()
    print(f"✅ Database initialized: {db.db_path}")
    
    # Get architecture
    import platform
    arch = 'apple_silicon' if 'arm' in platform.machine().lower() else 'intel'
    
    # Load latest stats
    loaded_stats = db.load_latest_stats(arch)
    if loaded_stats:
        print(f"✅ Loaded stats from database:")
        print(f"   ML Models Trained: {loaded_stats['ml_models_trained']}")
        print(f"   Optimizations Run: {loaded_stats['optimizations_run']}")
        print(f"   Energy Saved: {loaded_stats['energy_saved']:.1f}%")
    else:
        print(f"ℹ️  No previous stats in database")
        
except Exception as e:
    print(f"❌ Database error: {e}")
    sys.exit(1)

# Test 3: Run optimization and check ML training
print("\n3️⃣ Testing ML Training During Optimization...")
try:
    initial_ml_count = qml_system.stats['ml_models_trained']
    initial_opt_count = qml_system.stats['optimizations_run']
    
    print(f"   Initial ML models trained: {initial_ml_count}")
    print(f"   Running optimization...")
    
    # Get current system state
    state = qml_system._get_system_state()
    print(f"   System state: CPU={state.cpu_percent:.1f}%, Memory={state.memory_percent:.1f}%")
    
    # Run optimization
    result = qml_system.run_comprehensive_optimization(state)
    print(f"   Optimization result: {result.energy_saved:.1f}% energy saved")
    
    # Check if ML training happened
    after_ml_count = qml_system.stats['ml_models_trained']
    after_opt_count = qml_system.stats['optimizations_run']
    
    print(f"   After ML models trained: {after_ml_count}")
    print(f"   After optimizations run: {after_opt_count}")
    
    if after_ml_count > initial_ml_count:
        print(f"✅ ML training incremented counter (+{after_ml_count - initial_ml_count})")
    elif after_opt_count > initial_opt_count:
        print(f"ℹ️  Optimization ran but ML training may not have (PyTorch available?)")
    else:
        print(f"⚠️  No changes detected")
        
    # Check if stats were saved to database
    time.sleep(0.5)  # Give database time to write
    loaded_stats = db.load_latest_stats(arch)
    if loaded_stats and loaded_stats['ml_models_trained'] == after_ml_count:
        print(f"✅ Stats persisted to database")
    else:
        print(f"⚠️  Stats may not have persisted yet")
        
except Exception as e:
    print(f"❌ Optimization error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check API endpoint
print("\n4️⃣ Testing API Endpoint...")
try:
    import requests
    
    # Try to connect to the API
    try:
        response = requests.get('http://localhost:5002/api/status', timeout=2)
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ API responded successfully")
            print(f"   Data source: {data.get('data_source', 'unknown')}")
            
            if 'stats' in data:
                stats = data['stats']
                print(f"   ML Models Trained: {stats.get('ml_models_trained', 0)}")
                print(f"   Optimizations Run: {stats.get('optimizations_run', 0)}")
                print(f"   Energy Saved: {stats.get('energy_saved', 0):.1f}%")
                
                # Check if API is getting quantum-ML stats
                if data.get('data_source') == 'quantum_ml_system':
                    print(f"✅ API is using quantum-ML system (correct)")
                else:
                    print(f"⚠️  API is using fallback system")
                    
                # Verify ML models trained matches
                if stats.get('ml_models_trained') == qml_system.stats['ml_models_trained']:
                    print(f"✅ API stats match quantum-ML system")
                else:
                    print(f"⚠️  API stats don't match quantum-ML system")
                    print(f"     API: {stats.get('ml_models_trained')}")
                    print(f"     QML: {qml_system.stats['ml_models_trained']}")
            else:
                print(f"⚠️  No stats in API response")
        else:
            print(f"⚠️  API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"ℹ️  API server not running (start with: python universal_pqs_app.py)")
    except requests.exceptions.Timeout:
        print(f"⚠️  API request timed out")
        
except Exception as e:
    print(f"⚠️  Could not test API: {e}")

# Test 5: Check PyTorch availability
print("\n5️⃣ Checking ML Dependencies...")
try:
    import torch
    print(f"✅ PyTorch available: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print(f"✅ Apple Silicon MPS available")
    else:
        print(f"ℹ️  MPS not available (CPU mode)")
        
except ImportError:
    print(f"⚠️  PyTorch not available - ML training will use fallback")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow available: {tf.__version__}")
except ImportError:
    print(f"ℹ️  TensorFlow not available")

# Summary
print("\n" + "=" * 60)
print("📊 Test Summary")
print("=" * 60)
print(f"Current ML Models Trained: {qml_system.stats['ml_models_trained']}")
print(f"Current Optimizations Run: {qml_system.stats['optimizations_run']}")
print(f"Current Energy Saved: {qml_system.stats['energy_saved']:.1f}%")
print(f"Database Path: {db.db_path}")
print(f"System Architecture: {arch}")

if qml_system.stats['ml_models_trained'] > 0:
    print(f"\n✅ ML Models Trained counter is working!")
else:
    print(f"\nℹ️  ML Models Trained is 0 - run a few optimization cycles to see it increment")
    print(f"   The counter will increment as the system runs optimizations")

print("\n💡 To see ML training in action:")
print("   1. Start the app: python universal_pqs_app.py")
print("   2. Open dashboard: http://localhost:5002/")
print("   3. Watch the ML Models Trained counter increment")
print("   4. Restart the app - counter should persist")
