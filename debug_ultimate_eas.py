#!/usr/bin/env python3
"""
Debug Ultimate EAS availability in the enhanced app
"""

print("🔍 Debugging Ultimate EAS availability...")

# Test the exact import sequence from enhanced_app.py
try:
    print("Testing imports...")
    from ultimate_eas_system import UltimateEASSystem
    print("✅ UltimateEASSystem imported")
    
    from permission_manager import permission_manager
    print("✅ permission_manager imported")
    
    from gpu_acceleration import gpu_engine
    print("✅ gpu_engine imported")
    
    from pure_cirq_quantum_system import PureCirqQuantumSystem
    print("✅ PureCirqQuantumSystem imported")
    
    ULTIMATE_EAS_AVAILABLE = True
    print("🚀 Ultimate EAS System with Quantum Supremacy available")
    print("   Features: M3 GPU acceleration, Quantum circuits, Advanced AI")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    ULTIMATE_EAS_AVAILABLE = False

print(f"\n🎯 ULTIMATE_EAS_AVAILABLE = {ULTIMATE_EAS_AVAILABLE}")

# Test creating the system
if ULTIMATE_EAS_AVAILABLE:
    try:
        print("\n🧪 Testing Ultimate EAS System creation...")
        ultimate_eas = UltimateEASSystem(enable_distributed=False)
        print("✅ Ultimate EAS System created successfully")
    except Exception as e:
        print(f"❌ Ultimate EAS System creation failed: {e}")
        import traceback
        traceback.print_exc()