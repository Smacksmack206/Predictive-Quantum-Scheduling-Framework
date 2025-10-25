
import sys
try:
    import cirq
    print("✅ Cirq quantum library loaded")
except ImportError as e:
    print(f"⚠️ Cirq not available: {e}")

try:
    import tensorflow as tf
    print("✅ TensorFlow loaded")
    print(f"   Version: {tf.__version__}")
    
    # Check for GPU devices (Metal on Apple Silicon)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🍎 TensorFlow Metal GPU acceleration: {len(gpus)} GPU(s)")
        else:
            print("💻 TensorFlow CPU only")
    except AttributeError:
        # Older TensorFlow version
        print("💻 TensorFlow loaded (version may not support GPU detection)")
        
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")

try:
    from real_quantum_ml_system import RealQuantumMLSystem
    print("✅ Real Quantum-ML System loaded")
except ImportError as e:
    print(f"⚠️ Real Quantum-ML System not available: {e}")
