#!/bin/bash
# Fixed version of your original build command
# Key fix: Removed --no-deps so all dependencies are installed

set -e

echo "🚀 Building PQS Framework (Fixed Version)"
echo "=========================================="

# Ensure modules are in pqs_framework
echo "📋 Preparing pqs_framework..."
for f in quantum_ml_persistence.py macos_power_metrics.py qiskit_quantum_engine.py \
         quantum_process_optimizer.py quantum_ml_idle_optimizer.py \
         quantum_max_scheduler.py quantum_max_integration.py; do
    [ -f "$f" ] && [ ! -f "pqs_framework/$f" ] && cp "$f" "pqs_framework/"
done

[ -d "templates" ] && [ ! -d "pqs_framework/templates" ] && cp -r templates pqs_framework/
[ -d "static" ] && [ ! -d "pqs_framework/static" ] && cp -r static pqs_framework/
[ -f "config.json" ] && [ ! -f "pqs_framework/config.json" ] && cp config.json pqs_framework/

echo "✅ pqs_framework prepared"
echo ""

# Your original command, but FIXED (removed --no-deps)
echo "🔨 Running briefcase build..."
briefcase create macOS && \
briefcase build macOS && \
pip install rumps cirq qiskit tensorflow-macos tensorflow-metal torch numpy \
    -t "build/pqs_framework/macos/app/PQS Framework.app/Contents/Resources/app_packages/" && \
briefcase package macOS --adhoc-sign

echo ""
echo "✅ Build complete!"
echo "📦 DMG: dist/PQS Framework-1.0.0.dmg"
