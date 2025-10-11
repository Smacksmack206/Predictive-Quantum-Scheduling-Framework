#!/usr/bin/env python3
"""
Simple test of the Ultimate EAS System
"""

print("🚀 Testing Ultimate EAS System Components")

try:
    print("1. Testing GPU acceleration...")
    from gpu_acceleration import gpu_engine
    gpu_status = gpu_engine.get_acceleration_status()
    print(f"   GPU Available: {gpu_status['gpu_available']}")
    print(f"   GPU Name: {gpu_status['gpu_name']}")
    print(f"   Performance Boost: {gpu_status['performance_boost']}x")
except Exception as e:
    print(f"   ❌ GPU acceleration failed: {e}")

try:
    print("2. Testing Pure Cirq Quantum System...")
    from pure_cirq_quantum_system import PureCirqQuantumSystem
    quantum_system = PureCirqQuantumSystem(num_qubits=8)
    print("   ✅ Pure Cirq Quantum System initialized")
except Exception as e:
    print(f"   ❌ Pure Cirq Quantum System failed: {e}")

try:
    print("3. Testing Permission Manager...")
    from permission_manager import permission_manager
    permissions = permission_manager.get_permission_status()
    print(f"   Permissions: {permissions}")
except Exception as e:
    print(f"   ❌ Permission Manager failed: {e}")

try:
    print("4. Testing Quantum Neural EAS...")
    from quantum_neural_eas import QuantumNeuralEAS
    quantum_neural = QuantumNeuralEAS()
    print("   ✅ Quantum Neural EAS initialized")
except Exception as e:
    print(f"   ❌ Quantum Neural EAS failed: {e}")

print("🎯 Component testing complete!")