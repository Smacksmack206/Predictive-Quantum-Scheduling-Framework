#!/bin/bash

# Test and Build Script for 40-Qubit PQS Framework
# Intel Mac Compatible Version

echo "🔬 PQS Framework 40-Qubit - Test and Build"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install py2app setuptools wheel

# Run comprehensive tests
echo "🧪 Running comprehensive tests..."
python test_40_qubit_system.py

# Ask user if they want to proceed with build
echo ""
read -p "🚀 Proceed with app bundle build? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 Building app bundle..."
    python build_40_qubit_app.py
    
    echo ""
    echo "✅ Build process complete!"
    echo "📦 Check the dist/ folder for your app bundle"
else
    echo "⏹️  Build cancelled by user"
fi

echo ""
echo "👋 Done!"