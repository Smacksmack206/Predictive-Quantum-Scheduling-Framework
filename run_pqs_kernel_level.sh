#!/bin/bash
#
# Run PQS with Kernel-Level Optimization
# This script runs the PQS app with root privileges for full kernel access
#

echo "🚀 Starting PQS with Kernel-Level Optimization..."
echo ""
echo "⚠️  This requires root privileges for:"
echo "   - Process scheduler optimization"
echo "   - Memory management tuning"
echo "   - I/O subsystem optimization"
echo "   - Power management control"
echo "   - Thermal management"
echo ""
echo "You will be prompted for your password."
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python path
PYTHON_PATH="/Users/home/Projects/system-tools/m3.macbook.air/quantum_ml_311/bin/python3.11"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Python not found at: $PYTHON_PATH"
    exit 1
fi

# Check if universal_pqs_app.py exists
if [ ! -f "$SCRIPT_DIR/universal_pqs_app.py" ]; then
    echo "❌ universal_pqs_app.py not found in: $SCRIPT_DIR"
    exit 1
fi

echo "✅ Found Python at: $PYTHON_PATH"
echo "✅ Found PQS app at: $SCRIPT_DIR/universal_pqs_app.py"
echo ""

# Run with sudo
echo "🔐 Requesting root privileges..."
sudo "$PYTHON_PATH" "$SCRIPT_DIR/universal_pqs_app.py"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PQS stopped successfully"
else
    echo ""
    echo "❌ PQS exited with error"
    exit 1
fi
