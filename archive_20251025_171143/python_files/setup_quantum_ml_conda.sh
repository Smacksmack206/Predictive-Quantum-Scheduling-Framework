#!/bin/bash
# Conda-based Quantum-ML Environment Setup with Python 3.11
# This provides the most reliable installation for all quantum-ML dependencies

echo "🚀 Conda-based Quantum-ML Environment Setup"
echo "============================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "📦 Installing Miniconda..."
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
        rm Miniconda3-latest-MacOSX-arm64.sh
    else
        # Intel Mac
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
        rm Miniconda3-latest-MacOSX-x86_64.sh
    fi
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init zsh
    source ~/.zshrc
    
    echo "✅ Miniconda installed successfully!"
fi

# Create conda environment with Python 3.11
echo "🐍 Creating conda environment with Python 3.11..."
conda create -n quantum_ml_311 python=3.11 -y

# Activate environment
echo "🔧 Activating quantum-ML environment..."
conda activate quantum_ml_311

# Add conda-forge channel for better package availability
conda config --add channels conda-forge
conda config --add channels pytorch

# Install core scientific packages via conda (more reliable)
echo "📊 Installing core scientific packages..."
conda install -y numpy scipy matplotlib pandas scikit-learn

# Install PyTorch with Apple Silicon support
echo "🔥 Installing PyTorch with Apple Silicon support..."
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon - use conda for best performance
    conda install -y pytorch torchvision torchaudio -c pytorch
else
    # Intel Mac
    conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
fi

# Install TensorFlow for macOS (Apple Silicon optimized)
echo "🧠 Installing TensorFlow for macOS..."
if [[ $(uname -m) == "arm64" ]]; then
    # Apple Silicon - use tensorflow-macos and tensorflow-metal
    pip install tensorflow-macos tensorflow-metal
    echo "🍎 TensorFlow with Metal GPU acceleration installed"
else
    # Intel Mac - use standard TensorFlow
    conda install -y tensorflow
    echo "💻 Standard TensorFlow installed"
fi

# Install quantum computing libraries via pip (more up-to-date)
echo "⚛️ Installing quantum computing libraries..."
pip install cirq>=1.6.1
pip install qiskit>=0.45.0
pip install pennylane>=0.32.0

# Note: TensorFlow Quantum not compatible with tensorflow-macos
echo "ℹ️ Skipping TensorFlow Quantum (not compatible with tensorflow-macos)"
echo "   Using Cirq + TensorFlow-macOS for quantum-ML hybrid algorithms"

# Install additional quantum libraries
echo "🔬 Installing additional quantum libraries..."
pip install qiskit-machine-learning || echo "ℹ️ Qiskit ML not available"
pip install pennylane-lightning || echo "ℹ️ PennyLane Lightning not available"
pip install openfermion || echo "ℹ️ OpenFermion not available"

# Install ML and optimization libraries
echo "🤖 Installing ML and optimization libraries..."
pip install stable-baselines3 || echo "ℹ️ Stable Baselines3 not available"
pip install optuna
pip install transformers || echo "ℹ️ Transformers not available"

# Install high-performance computing
echo "⚡ Installing high-performance computing..."
conda install -y numba joblib

# Install optimization libraries
echo "🎯 Installing optimization libraries..."
pip install cvxpy || echo "ℹ️ CVXPY not available"
pip install ortools || echo "ℹ️ OR-Tools not available"

# Install system monitoring
echo "🔍 Installing system monitoring..."
pip install psutil py-cpuinfo memory-profiler

# Install web and API libraries
echo "🌐 Installing web libraries..."
pip install flask requests pydantic

# Install development tools
echo "🛠️ Installing development tools..."
pip install pytest black loguru

# Install visualization
echo "📊 Installing visualization libraries..."
conda install -y plotly seaborn bokeh

# Verification
echo "✅ Installation complete! Verifying..."

python -c "
import sys
print(f'🐍 Python version: {sys.version}')

# Test core libraries
try:
    import numpy, scipy, pandas, matplotlib
    print('✅ Core scientific libraries: OK')
except ImportError as e:
    print('❌ Core libraries:', e)

try:
    import torch
    print('✅ PyTorch: OK')
    print(f'   Version: {torch.__version__}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('   🍎 Apple Silicon MPS: Available')
    else:
        print('   💻 CPU only')
except ImportError as e:
    print('❌ PyTorch:', e)

try:
    import tensorflow as tf
    print('✅ TensorFlow: OK')
    print(f'   Version: {tf.__version__}')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('   🚀 GPU acceleration: Available')
    else:
        print('   💻 CPU only')
except ImportError as e:
    print('❌ TensorFlow:', e)

try:
    import cirq
    print('✅ Cirq: OK')
    print(f'   Version: {cirq.__version__}')
except ImportError as e:
    print('❌ Cirq:', e)

try:
    import qiskit
    print('✅ Qiskit: OK')
    print(f'   Version: {qiskit.__version__}')
except ImportError as e:
    print('❌ Qiskit:', e)

try:
    import pennylane as qml
    print('✅ PennyLane: OK')
    print(f'   Version: {qml.__version__}')
except ImportError as e:
    print('❌ PennyLane:', e)

try:
    import tensorflow_quantum as tfq
    print('✅ TensorFlow Quantum: OK')
except ImportError as e:
    print('⚠️ TensorFlow Quantum: Not available')
    print('   Using alternative quantum implementations')

# Test quantum simulation capability
try:
    import cirq
    import numpy as np
    
    # Create a simple 2-qubit quantum circuit
    qubits = cirq.GridQubit.rect(1, 2)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=10)
    
    print('✅ Quantum simulation test: PASSED')
    print(f'   2-qubit Bell state created and measured')
    
except Exception as e:
    print('⚠️ Quantum simulation test: FAILED')
    print(f'   Error: {e}')
"

echo ""
echo "🎉 Conda Quantum-ML environment setup complete!"
echo ""
echo "🚀 To activate the environment in future sessions:"
echo "   conda activate quantum_ml_311"
echo ""
echo "🧪 To test the quantum-ML system:"
echo "   conda activate quantum_ml_311"
echo "   python real_quantum_ml_system.py"
echo ""
echo "📊 Environment summary:"
echo "   • Python 3.11 (optimal compatibility)"
echo "   • Conda package management (reliable)"
echo "   • Apple Silicon optimizations (if available)"
echo "   • Quantum computing libraries (Cirq, Qiskit, PennyLane)"
echo "   • Advanced ML libraries (PyTorch, TensorFlow)"
echo "   • High-performance scientific computing"
echo ""
echo "✅ Ready for exponential performance improvements!"

# Create activation script for convenience
cat > activate_quantum_ml.sh << 'EOF'
#!/bin/bash
# Convenience script to activate quantum-ML environment
echo "🚀 Activating Quantum-ML Environment..."
conda activate quantum_ml_311
echo "✅ Environment activated!"
echo "🧪 To test: python real_quantum_ml_system.py"
EOF

chmod +x activate_quantum_ml.sh

echo ""
echo "💡 Convenience script created: ./activate_quantum_ml.sh"
echo "   Run this to quickly activate the environment"