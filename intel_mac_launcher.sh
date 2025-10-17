#!/bin/bash
# Intel Mac Launcher for PQS Framework 40-Qubit
# This script ensures the app runs correctly on Intel Macs

echo "🚀 PQS Framework Intel Mac Launcher"
echo "=================================="

# Detect architecture
ARCH=$(uname -m)
echo "🔍 Detected architecture: $ARCH"

# Check if we're on an Intel Mac
if [[ "$ARCH" == "x86_64" ]]; then
    echo "💻 Intel Mac detected - launching with x86_64 architecture"
    arch -x86_64 open "dist/PQS Framework 40-Qubit.app"
elif [[ "$ARCH" == "arm64" ]]; then
    echo "🍎 Apple Silicon detected - launching normally"
    open "dist/PQS Framework 40-Qubit.app"
else
    echo "❓ Unknown architecture - attempting normal launch"
    open "dist/PQS Framework 40-Qubit.app"
fi

echo "✅ Launch command executed"
echo "💡 If the app doesn't start, try running:"
echo "   arch -x86_64 open 'dist/PQS Framework 40-Qubit.app'"