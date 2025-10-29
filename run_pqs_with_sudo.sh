#!/bin/bash
# Run PQS Framework with elevated privileges
# This ensures all optimizations work correctly

set -e

echo "🚀 Starting PQS Framework with elevated privileges"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "✅ Already running as root"
else
    echo "🔐 Requesting sudo privileges..."
    echo "   This allows PQS Framework to:"
    echo "   - Optimize system power settings"
    echo "   - Manage process priorities"
    echo "   - Control system services"
    echo "   - Apply advanced battery optimizations"
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "quantum_ml_311" ]; then
    echo "📦 Activating virtual environment..."
    source quantum_ml_311/bin/activate
fi

# Run with sudo
echo ""
echo "🎯 Launching PQS Framework..."
echo ""

sudo -E python3 -m pqs_framework

echo ""
echo "✅ PQS Framework stopped"
