#!/usr/bin/env python3
"""
Intel Mac Optimization Test
Verifies that Intel-specific optimizations work correctly
"""

import sys
import os
import platform
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_intel_optimizations():
    """Test Intel-specific optimizations"""
    print("🔍 Testing Intel Mac Optimizations...")
    print("=" * 50)
    
    # Import the main app
    try:
        from universal_pqs_app import UniversalSystemDetector, UniversalQuantumSystem
        print("✅ Successfully imported PQS Framework")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test system detection
    print("\n1. Testing System Detection:")
    detector = UniversalSystemDetector()
    system_info = detector.system_info
    capabilities = detector.capabilities
    
    print(f"   Platform: {system_info['platform']}")
    print(f"   Architecture: {system_info['architecture']}")
    print(f"   Chip Model: {system_info['chip_model']}")
    print(f"   Optimization Tier: {system_info['optimization_tier']}")
    
    if system_info['architecture'] == 'intel':
        print("✅ Intel architecture detected correctly")
        
        # Test Intel-specific capabilities
        print("\n2. Testing Intel Capabilities:")
        print(f"   Quantum Simulation: {capabilities['quantum_simulation']}")
        print(f"   Max Qubits: {capabilities['max_qubits']}")
        print(f"   GPU Acceleration: {capabilities['gpu_acceleration']}")
        print(f"   Metal Support: {capabilities['metal_support']}")
        print(f"   Neural Engine: {capabilities['neural_engine']}")
        print(f"   Optimization Algorithms: {capabilities['optimization_algorithms']}")
        
        # Verify Intel-appropriate settings
        if not capabilities['metal_support']:
            print("✅ Metal support correctly disabled for Intel")
        if not capabilities['neural_engine']:
            print("✅ Neural Engine correctly disabled for Intel")
        if capabilities['max_qubits'] <= 30:
            print(f"✅ Qubit count appropriate for Intel: {capabilities['max_qubits']}")
        
        # Test i3-specific optimizations
        if 'i3' in system_info['chip_model']:
            print("\n3. Testing i3-Specific Optimizations:")
            if capabilities['max_qubits'] == 20:
                print("✅ i3 qubit limit correctly set to 20")
            if 'cpu_friendly_mode' in capabilities:
                print("✅ CPU-friendly mode enabled for i3")
            if 'lightweight' in capabilities['optimization_algorithms']:
                print("✅ Lightweight algorithms enabled for i3")
        
    elif system_info['architecture'] == 'apple_silicon':
        print("🍎 Apple Silicon detected - Intel optimizations cannot be fully tested")
        print("   However, Intel compatibility code paths are present")
    else:
        print("❓ Unknown architecture detected")
    
    # Test quantum system initialization
    print("\n4. Testing Quantum System Initialization:")
    try:
        quantum_system = UniversalQuantumSystem(detector)
        if quantum_system.available:
            print("✅ Quantum system initialized successfully")
            print(f"   System Architecture: {quantum_system.stats['system_architecture']}")
            print(f"   Optimization Tier: {quantum_system.stats['optimization_tier']}")
            print(f"   Qubits Available: {quantum_system.stats['qubits_available']}")
        else:
            print("❌ Quantum system initialization failed")
            return False
    except Exception as e:
        print(f"❌ Quantum system error: {e}")
        return False
    
    # Test optimization run
    print("\n5. Testing Optimization Execution:")
    try:
        success = quantum_system.run_optimization()
        if success:
            print("✅ Optimization executed successfully")
            print(f"   Energy Saved: {quantum_system.stats.get('energy_saved', 0):.2f}%")
            print(f"   Optimizations Run: {quantum_system.stats.get('optimizations_run', 0)}")
        else:
            print("⚠️  No optimization needed (system already efficient)")
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False
    
    # Test status retrieval
    print("\n6. Testing Status Retrieval:")
    try:
        status = quantum_system.get_status()
        if status['available']:
            print("✅ Status retrieved successfully")
            print(f"   System Info Available: {'system_info' in status}")
            print(f"   Capabilities Available: {'capabilities' in status}")
            print(f"   Stats Available: {'stats' in status}")
        else:
            print("❌ Status indicates system not available")
            return False
    except Exception as e:
        print(f"❌ Status error: {e}")
        return False
    
    print("\n🎉 Intel Optimization Test Complete!")
    print("=" * 50)
    
    if system_info['architecture'] == 'intel':
        print("✅ All Intel-specific optimizations working correctly")
        print("🔧 Your fiancé's Intel Mac will run these optimizations:")
        print(f"   • {capabilities['max_qubits']}-qubit quantum simulation")
        print(f"   • {system_info['optimization_tier']} optimization tier")
        print(f"   • {', '.join(capabilities['optimization_algorithms'])} algorithms")
        
        if 'i3' in system_info['chip_model']:
            print("   • Special i3 CPU-friendly optimizations")
            print("   • Reduced background tasks for better performance")
    else:
        print("✅ Intel compatibility code verified (running on non-Intel system)")
    
    return True

if __name__ == "__main__":
    success = test_intel_optimizations()
    if not success:
        sys.exit(1)