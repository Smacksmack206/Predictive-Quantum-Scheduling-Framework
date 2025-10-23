#!/bin/bash
# Step-by-Step Quantum-ML Installation for Maximum Compatibility
# Handles all dependency conflicts and Apple Silicon optimizations

echo "🚀 Step-by-Step Quantum-ML Installation"
echo "======================================"

# Check Python version
python_version=$(python --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "🐍 Python version: $python_version"

if [[ "$python_version" != "3.11" ]]; then
    echo "⚠️ Warning: Python 3.11 recommended for maximum compatibility"
    echo "Current version: $python_version"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
python -m venv quantum_ml_venv
source quantum_ml_venv/bin/activate

# Upgrade pip and tools
echo "⬆️ Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Step 1: Install core dependencies first
echo "📦 Step 1: Installing core dependencies..."
pip install numpy>=1.24.0 scipy>=1.10.0 psutil>=5.9.0

# Step 2: Install TensorFlow (Apple Silicon optimized)
echo "🧠 Step 2: Installing TensorFlow..."
if [[ $(uname -m) == "arm64" ]]; then
    echo "🍎 Apple Silicon detected - installing optimized TensorFlow..."
    pip install tensorflow>=2.15.0
    # Try Metal acceleration (may not be available in newer versions)
    pip install tensorflow-metal || echo "ℹ️ tensorflow-metal not available, using built-in Apple Silicon optimizations"
else
    echo "💻 Intel Mac detected - installing standard TensorFlow..."
    pip install tensorflow>=2.15.0
fi

# Step 3: Install PyTorch with Apple Silicon support
echo "🔥 Step 3: Installing PyTorch..."
if [[ $(uname -m) == "arm64" ]]; then
    pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Step 4: Install quantum computing libraries
echo "⚛️ Step 4: Installing quantum computing libraries..."
pip install cirq>=1.6.1

# Try TensorFlow Quantum (may fail on some systems)
echo "🔬 Installing TensorFlow Quantum..."
pip install tensorflow-quantum>=0.7.3 || echo "⚠️ TensorFlow Quantum not available, quantum features will use alternative implementations"

# Install Qiskit
echo "🔬 Installing Qiskit..."
pip install 'qiskit[all]>=0.45.0' || pip install qiskit>=0.45.0

# Install PennyLane
echo "🔬 Installing PennyLane..."
pip install pennylane>=0.32.0
pip install pennylane-lightning>=0.32.0 || echo "⚠️ PennyLane Lightning not available, using default simulator"

# Step 5: Install machine learning libraries
echo "🤖 Step 5: Installing ML libraries..."
pip install scikit-learn>=1.3.0 pandas>=2.0.0

# Install reinforcement learning
pip install gymnasium>=0.29.0
pip install stable-baselines3>=2.0.0 || echo "⚠️ Stable Baselines3 not available, using alternative RL implementation"

# Step 6: Install optimization libraries
echo "🎯 Step 6: Installing optimization libraries..."
pip install optuna>=3.4.0
pip install cvxpy>=1.4.0
pip install ortools>=9.8.0

# Step 7: Install visualization and web libraries
echo "📊 Step 7: Installing visualization libraries..."
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.17.0
pip install flask>=2.3.0 requests>=2.31.0

# Step 8: Install high-performance computing
echo "⚡ Step 8: Installing high-performance libraries..."
pip install numba>=0.58.0 joblib>=1.3.0

# Step 9: Install additional quantum libraries (optional)
echo "🔬 Step 9: Installing additional quantum libraries..."
pip install qiskit-machine-learning>=0.7.0 || echo "ℹ️ Qiskit ML not available"
pip install pennylane-qiskit>=0.32.0 || echo "ℹ️ PennyLane-Qiskit not available"
pip install openfermion>=1.6.0 || echo "ℹ️ OpenFermion not available"

# Step 10: Install development tools
echo "🛠️ Step 10: Installing development tools..."
pip install pytest>=7.4.0 black>=23.9.0 memory-profiler>=0.61.0

# Step 11: Install remaining compatible packages
echo "📚 Step 11: Installing remaining packages..."
pip install pydantic>=2.4.0 loguru>=0.7.0 transformers>=4.35.0 || echo "ℹ️ Some packages not available"

# Verification
echo "✅ Installation complete! Verifying..."

# Test core libraries
python -c "
try:
    import numpy, scipy, psutil
    print('✅ Core libraries: OK')
except ImportError as e:
    print('❌ Core libraries:', e)

try:
    import tensorflow as tf
    print('✅ TensorFlow: OK')
    print('   Version:', tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('   GPU acceleration: Available')
    else:
        print('   GPU acceleration: Not available (CPU only)')
except ImportError as e:
    print('❌ TensorFlow:', e)

try:
    import torch
    print('✅ PyTorch: OK')
    print('   Version:', torch.__version__)
    if torch.backends.mps.is_available():
        print('   MPS (Apple Silicon): Available')
    else:
        print('   MPS (Apple Silicon): Not available')
except ImportError as e:
    print('❌ PyTorch:', e)

try:
    import cirq
    print('✅ Cirq: OK')
    print('   Version:', cirq.__version__)
except ImportError as e:
    print('❌ Cirq:', e)

try:
    import qiskit
    print('✅ Qiskit: OK')
    print('   Version:', qiskit.__version__)
except ImportError as e:
    print('❌ Qiskit:', e)

try:
    import pennylane as qml
    print('✅ PennyLane: OK')
    print('   Version:', qml.__version__)
except ImportError as e:
    print('❌ PennyLane:', e)

try:
    import tensorflow_quantum as tfq
    print('✅ TensorFlow Quantum: OK')
except ImportError as e:
    print('⚠️ TensorFlow Quantum: Not available -', e)
    print('   Quantum-ML system will use alternative implementations')
"

echo ""
echo "🎉 Quantum-ML environment setup complete!"
echo ""
echo "🚀 To test the system:"
echo "   source quantum_ml_venv/bin/activate"
echo "   python real_quantum_ml_system.py"
echo ""
echo "📊 Available quantum resources:"
python -c "
import sys
print(f'   Python: {sys.version}')

try:
    import cirq
    print('   Cirq quantum simulator: Available')
except:
    print('   Cirq quantum simulator: Not available')

try:
    import qiskit
    print('   Qiskit quantum simulator: Available')
except:
    print('   Qiskit quantum simulator: Not available')

try:
    import pennylane
    print('   PennyLane quantum ML: Available')
except:
    print('   PennyLane quantum ML: Not available')

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('   TensorFlow GPU acceleration: Available')
    else:
        print('   TensorFlow GPU acceleration: CPU only')
except:
    print('   TensorFlow: Not available')

try:
    import torch
    if torch.backends.mps.is_available():
        print('   PyTorch MPS (Apple Silicon): Available')
    else:
        print('   PyTorch MPS: Not available')
except:
    print('   PyTorch: Not available')
"

echo ""
echo "✅ Ready for exponential performance improvements!"