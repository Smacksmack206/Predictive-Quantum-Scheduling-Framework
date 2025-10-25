#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQS Framework Quantum-ML App Builder
===================================

Comprehensive build script for creating a production-ready macOS app
with the new quantum-ML system integrated.

This script:
1. Sets up the quantum-ML environment
2. Installs all dependencies
3. Builds the universal app bundle
4. Creates a distributable .dmg file
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command with proper error handling"""
    print(f"\n🔧 {description}")
    print(f"💻 Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"📝 Details: {e.stderr.strip()}")
        if check:
            sys.exit(1)
        return e

def check_dependencies():
    """Check if required build tools are available"""
    print("🔍 Checking build dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} is too old. Need Python 3.8+")
        sys.exit(1)
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for py2app
    try:
        import py2app
        print(f"✅ py2app {py2app.__version__}")
    except ImportError:
        print("❌ py2app not found. Installing...")
        run_command("pip install py2app", "Installing py2app")
    
    # Check for conda (recommended for quantum ML)
    conda_result = run_command("conda --version", "Checking conda", check=False)
    if conda_result.returncode == 0:
        print("✅ Conda available for quantum-ML environment")
        return True
    else:
        print("⚠️ Conda not available. Will use pip for quantum-ML dependencies")
        return False

def setup_quantum_ml_environment():
    """Set up the quantum-ML environment"""
    print("\n🚀 Setting up Quantum-ML Environment...")
    
    # Check if conda is available
    has_conda = check_dependencies()
    
    # Use current environment for building (simpler and more reliable)
    print("📦 Installing dependencies in current environment")
    
    # Install app bundle dependencies first
    run_command(
        "pip install -r requirements-app-bundle.txt",
        "Installing app bundle dependencies"
    )
    
    # Install TensorFlow for macOS (Apple Silicon optimized)
    import platform
    if platform.machine() == 'arm64':
        print("🍎 Detected Apple Silicon - installing tensorflow-macos")
        run_command(
            "pip install tensorflow-macos tensorflow-metal",
            "Installing TensorFlow for Apple Silicon",
            check=False
        )
    else:
        print("💻 Detected Intel Mac - using standard TensorFlow")
        run_command(
            "pip install tensorflow",
            "Installing standard TensorFlow",
            check=False
        )
    
    # Install quantum dependencies (optional)
    run_command(
        "pip install -r requirements_quantum_ml.txt",
        "Installing quantum-ML dependencies (optional)",
        check=False  # Don't fail if quantum deps can't be installed
    )
    
    # Quick verification without external script
    print("\n🔬 Verifying quantum libraries...")
    
    try:
        import cirq
        print("✅ Cirq quantum library loaded")
    except ImportError as e:
        print(f"⚠️ Cirq not available: {e}")

    try:
        import tensorflow as tf
        print("✅ TensorFlow-macOS loaded")
    except ImportError as e:
        print(f"⚠️ TensorFlow not available: {e}")

    try:
        from real_quantum_ml_system import RealQuantumMLSystem
        print("✅ Real Quantum-ML System loaded")
    except ImportError as e:
        print(f"⚠️ Real Quantum-ML System not available: {e}")
        
    print("✅ Verification complete - proceeding with build")

def clean_build_directories():
    """Clean previous build artifacts"""
    print("\n🧹 Cleaning build directories...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"🗑️ Removed {dir_name}")
    
    # Clean Python cache files
    run_command("find . -name '*.pyc' -delete", "Removing .pyc files", check=False)
    run_command("find . -name '__pycache__' -type d -exec rm -rf {} +", "Removing __pycache__ directories", check=False)

def build_app():
    """Build the macOS app bundle"""
    print("\n🏗️ Building macOS App Bundle...")
    
    # Build with py2app
    run_command(
        "python setup.py py2app",
        "Building app bundle with py2app"
    )
    
    # Check if build was successful
    app_path = "dist/PQS Framework 40-Qubit.app"
    if os.path.exists(app_path):
        print(f"✅ App bundle created: {app_path}")
        
        # Get app size
        size_result = run_command(f"du -sh '{app_path}'", "Checking app size", check=False)
        if size_result.returncode == 0:
            print(f"📦 App size: {size_result.stdout.strip()}")
        
        return True
    else:
        print("❌ App bundle creation failed")
        return False

def create_dmg():
    """Create a distributable .dmg file"""
    print("\n💿 Creating distributable .dmg file...")
    
    app_name = "PQS Framework 40-Qubit"
    dmg_name = f"{app_name} - Quantum-ML Edition.dmg"
    
    # Remove existing DMG
    if os.path.exists(dmg_name):
        os.remove(dmg_name)
        print(f"🗑️ Removed existing {dmg_name}")
    
    # Create DMG
    dmg_cmd = f"""
    hdiutil create -volname "{app_name}" -srcfolder "dist/{app_name}.app" -ov -format UDZO "{dmg_name}"
    """
    
    run_command(dmg_cmd, "Creating DMG file")
    
    if os.path.exists(dmg_name):
        print(f"✅ DMG created: {dmg_name}")
        
        # Get DMG size
        size_result = run_command(f"ls -lh '{dmg_name}'", "Checking DMG size", check=False)
        if size_result.returncode == 0:
            size_info = size_result.stdout.strip().split()
            if len(size_info) >= 5:
                print(f"📦 DMG size: {size_info[4]}")
        
        return True
    else:
        print("❌ DMG creation failed")
        return False

def test_app():
    """Test the built app"""
    print("\n🧪 Testing built app...")
    
    app_path = "dist/PQS Framework 40-Qubit.app"
    
    # Test app launch (non-blocking)
    test_cmd = f"open '{app_path}'"
    run_command(test_cmd, "Testing app launch", check=False)
    
    print("⏱️ App launched for testing. Check if it starts correctly.")
    print("💡 The app should appear in your menu bar.")

def main():
    """Main build process"""
    print("🚀 PQS Framework Quantum-ML App Builder")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Step 1: Check dependencies
        check_dependencies()
        
        # Step 2: Setup quantum-ML environment
        setup_quantum_ml_environment()
        
        # Step 3: Clean build directories
        clean_build_directories()
        
        # Step 4: Build app
        if not build_app():
            print("❌ Build failed")
            sys.exit(1)
        
        # Step 5: Create DMG
        if not create_dmg():
            print("❌ DMG creation failed")
            sys.exit(1)
        
        # Step 6: Test app
        test_app()
        
        # Success!
        build_time = time.time() - start_time
        print(f"\n🎉 Build completed successfully in {build_time:.1f} seconds!")
        print("\n📦 Deliverables:")
        print(f"   • App Bundle: dist/PQS Framework 40-Qubit.app")
        print(f"   • DMG File: PQS Framework 40-Qubit - Quantum-ML Edition.dmg")
        print("\n🚀 Ready for distribution!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()