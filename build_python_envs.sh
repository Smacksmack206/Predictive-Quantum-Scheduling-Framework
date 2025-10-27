#!/bin/bash
# Build Python environments for both architectures

set -e

echo "🏗️  Building Universal Python Environments"

# Create virtual environments
echo "📦 Creating ARM64 environment..."
arch -arm64 python3 -m venv python_arm64
arch -arm64 python_arm64/bin/pip install --upgrade pip
arch -arm64 python_arm64/bin/pip install -r app_requirements.txt

echo "📦 Creating x86_64 environment..."
arch -x86_64 python3 -m venv python_x86_64
arch -x86_64 python_x86_64/bin/pip install --upgrade pip
arch -x86_64 python_x86_64/bin/pip install -r app_requirements.txt

echo "✅ Python environments built"
