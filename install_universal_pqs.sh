#!/bin/bash

# Universal PQS Framework Installation Script
# Compatible with all Mac architectures

echo "🌍 Universal PQS Framework Installation"
echo "========================================"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This installer requires macOS"
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "🍎 Apple Silicon Mac detected"
    OPTIMIZATION_LEVEL="Maximum Quantum Acceleration"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "💻 Intel Mac detected"
    OPTIMIZATION_LEVEL="Classical Optimization"
else
    echo "❓ Unknown architecture: $ARCH"
    OPTIMIZATION_LEVEL="Basic Compatibility"
fi

echo "🎯 Optimization Level: $OPTIMIZATION_LEVEL"
echo

# Check Python installation
echo "🐍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "📦 Install Python 3 from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"

# Check pip
echo "📦 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed"
    echo "📦 Install pip3 with: python3 -m ensurepip --upgrade"
    exit 1
fi
echo "✅ pip3 found"

# Install required packages
echo
echo "📦 Installing required packages..."
REQUIRED_PACKAGES="rumps psutil flask numpy"

for package in $REQUIRED_PACKAGES; do
    echo "   Installing $package..."
    if pip3 install "$package" --quiet; then
        echo "   ✅ $package installed"
    else
        echo "   ⚠️ Failed to install $package (may already be installed)"
    fi
done

# Create templates directory if it doesn't exist
echo
echo "📁 Setting up directories..."
mkdir -p templates
echo "✅ Templates directory ready"

# Check if files exist
echo
echo "📄 Checking application files..."
if [[ -f "universal_pqs_app.py" ]]; then
    echo "✅ universal_pqs_app.py found"
else
    echo "❌ universal_pqs_app.py not found"
    echo "   Make sure all files are in the same directory"
    exit 1
fi

if [[ -f "launch_universal_pqs.py" ]]; then
    echo "✅ launch_universal_pqs.py found"
else
    echo "❌ launch_universal_pqs.py not found"
    exit 1
fi

if [[ -f "templates/universal_dashboard.html" ]]; then
    echo "✅ universal_dashboard.html found"
else
    echo "❌ templates/universal_dashboard.html not found"
    exit 1
fi

# Make scripts executable
echo
echo "🔧 Setting permissions..."
chmod +x launch_universal_pqs.py
chmod +x install_universal_pqs.sh
echo "✅ Permissions set"

# Create launch alias (optional)
echo
echo "🚀 Installation complete!"
echo
echo "To launch Universal PQS Framework:"
echo "   python3 launch_universal_pqs.py"
echo
echo "Or make it executable and run directly:"
echo "   ./launch_universal_pqs.py"
echo
echo "🌐 Dashboard will be available at: http://localhost:5003"
echo "📱 Menu bar app will appear in your menu bar"
echo

# Architecture-specific notes
if [[ "$ARCH" == "arm64" ]]; then
    echo "🔥 Apple Silicon Optimizations:"
    echo "   • Full 40-qubit quantum acceleration"
    echo "   • Metal GPU support"
    echo "   • Neural Engine integration"
    echo "   • Maximum energy optimization"
elif [[ "$ARCH" == "x86_64" ]]; then
    echo "💻 Intel Mac Optimizations:"
    echo "   • CPU-optimized quantum simulation"
    echo "   • Classical optimization algorithms"
    echo "   • Power-efficient processing"
    
    # Check for i3 specifically
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    if [[ "$CPU_INFO" == *"i3"* ]]; then
        echo "   • Special i3 MacBook Air optimizations active"
        echo "   • Reduced qubit count for better performance"
        echo "   • CPU-friendly algorithms"
    fi
fi

echo
echo "✅ Ready to launch Universal PQS Framework!"
echo "   Run: python3 launch_universal_pqs.py"