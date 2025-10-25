#!/usr/bin/env python3
"""
Quick test of the quantum engine selection prompt
"""

def select_quantum_engine():
    """
    Prompt user to select quantum engine at startup
    
    Returns:
        str: 'cirq' or 'qiskit'
    """
    print("\n" + "="*70)
    print("⚛️  QUANTUM ENGINE SELECTION")
    print("="*70)
    print("\nChoose your quantum computing engine:\n")
    print("1. 🚀 OPTIMIZED (Cirq)")
    print("   - Lightweight and fast")
    print("   - Best for real-time optimization")
    print("   - Proven performance on macOS")
    print("   - Recommended for daily use")
    print()
    print("2. 🔬 EXPERIMENTAL (Qiskit)")
    print("   - IBM's quantum framework")
    print("   - Advanced algorithms (VQE, QAOA, QPE)")
    print("   - Academic-grade quantum advantage")
    print("   - Groundbreaking research features")
    print("   - May be slower but more powerful")
    print()
    print("="*70)
    
    while True:
        try:
            choice = input("\nSelect engine [1 for Cirq, 2 for Qiskit] (default: 1): ").strip()
            
            if choice == '' or choice == '1':
                print("\n✅ Selected: Cirq (Optimized)")
                print("   Fast, lightweight, perfect for real-time optimization")
                return 'cirq'
            elif choice == '2':
                print("\n✅ Selected: Qiskit (Experimental)")
                print("   🔬 Activating groundbreaking quantum algorithms...")
                print("   ⚛️ VQE, QAOA, and advanced features enabled")
                print("   🎯 Academic-grade quantum advantage mode")
                return 'qiskit'
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n⚠️ Defaulting to Cirq (Optimized)")
            return 'cirq'
        except Exception as e:
            print(f"❌ Error: {e}. Defaulting to Cirq.")
            return 'cirq'


if __name__ == "__main__":
    print("🧪 Testing Quantum Engine Selection Prompt")
    print("=" * 70)
    
    engine = select_quantum_engine()
    
    print(f"\n🎯 You selected: {engine.upper()}")
    print(f"✅ Prompt test complete!")
