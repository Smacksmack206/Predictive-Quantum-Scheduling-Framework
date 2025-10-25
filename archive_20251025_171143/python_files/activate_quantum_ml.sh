#!/bin/bash
# Convenience script to activate quantum-ML environment
echo "🚀 Activating Quantum-ML 311 Environment..."
eval "$(conda shell.bash hook)"
conda activate /opt/homebrew/Caskroom/miniconda/base/envs/quantum_ml_311
echo "✅ Environment activated!"
echo "🧪 To test: python real_quantum_ml_system.py"
