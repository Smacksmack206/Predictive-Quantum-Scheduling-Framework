#!/bin/bash
# Install Quantum-ML Dependencies into Briefcase App Bundle

set -e

echo "🚀 Installing Quantum-ML Dependencies into App Bundle"
echo "============================================================"

# App bundle paths
APP_DIR="build/pqs-framework/macos/app/PQS Framework 40-Qubit.app"
APP_PACKAGES="$APP_DIR/Contents/Resources/app_packages"

# Check if app exists
if [ ! -d "$APP_DIR" ]; then
    echo "❌ App bundle not found: $APP_DIR"
    echo "   Run 'briefcase build macOS' first"
    exit 1
fi

echo "📦 App bundle found: $APP_DIR"
echo "📁 Installing to: $APP_PACKAGES"

# Create packages directory if it doesn't exist
mkdir -p "$APP_PACKAGES"

# Activate Python 3.11 environment with quantum-ML packages
echo "🔄 Activating Python 3.11 environment..."
source quantum_ml_311/bin/activate

# Install quantum-ML packages directly into app bundle
echo "⚛️  Installing Qiskit packages..."
pip install --target="$APP_PACKAGES" --upgrade qiskit qiskit-aer qiskit-algorithms

echo "🧠 Installing TensorFlow packages..."
pip install --target="$APP_PACKAGES" --upgrade tensorflow-macos tensorflow-metal

echo "🔧 Installing additional dependencies..."
pip install --target="$APP_PACKAGES" --upgrade scipy matplotlib networkx pandas

echo "✅ All quantum-ML dependencies installed!"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
echo "Checking app_packages directory:"
ls -la "$APP_PACKAGES" | head -20

echo ""
echo "📊 Package sizes:"
du -sh "$APP_PACKAGES"/* 2>/dev/null | head -10

echo ""
echo "✅ Installation complete!"
echo "🚀 App is ready to run with full quantum-ML capabilities"
echo ""
echo "📱 To test:"
echo "   open '$APP_DIR'"
