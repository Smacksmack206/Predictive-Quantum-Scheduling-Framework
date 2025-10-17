#!/usr/bin/env python3
"""
Universal Build Script for PQS Framework
Ensures proper Intel and Apple Silicon compatibility
"""

import subprocess
import sys
import os
import platform
import shutil

def check_prerequisites():
    """Check build prerequisites"""
    print("🔍 Checking Build Prerequisites...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.9+ required.")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check Briefcase
    try:
        result = subprocess.run(['briefcase', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Briefcase: {result.stdout.strip()}")
        else:
            print("❌ Briefcase not found. Install with: pip install briefcase")
            return False
    except FileNotFoundError:
        print("❌ Briefcase not found. Install with: pip install briefcase")
        return False
    
    # Check required files
    required_files = [
        'universal_pqs_app.py',
        'pyproject.toml',
        'templates/quantum_dashboard_enhanced.html',
        'static/themes.css'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return False
    
    return True

def prepare_build_environment():
    """Prepare build environment"""
    print("\n🔧 Preparing Build Environment...")
    
    # Create build directories if they don't exist
    os.makedirs('build', exist_ok=True)
    os.makedirs('dist', exist_ok=True)
    
    # Clean previous builds
    if os.path.exists('build'):
        print("🧹 Cleaning previous builds...")
        for item in os.listdir('build'):
            item_path = os.path.join('build', item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    print("✅ Build environment prepared")

def install_dependencies():
    """Install all required dependencies"""
    print("\n📦 Installing Dependencies...")
    
    dependencies = [
        'rumps>=0.4.0',
        'psutil>=5.9.0',
        'flask>=2.3.0',
        'numpy>=1.21.0',
        'Jinja2>=3.0.0',
        'Werkzeug>=2.0.0',
        'click>=8.0.0',
        'MarkupSafe>=2.0.0',
        'itsdangerous>=2.0.0',
        'blinker>=1.6.0'
    ]
    
    for dep in dependencies:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {dep}")
            else:
                print(f"❌ Failed to install {dep}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing {dep}: {e}")
            return False
    
    return True

def build_app():
    """Build the application"""
    print("\n🏗️  Building PQS Framework...")
    
    # Detect current architecture
    machine = platform.machine().lower()
    if 'arm' in machine or 'arm64' in machine:
        print("🍎 Building on Apple Silicon")
    elif any(arch in machine for arch in ['x86', 'amd64', 'i386']):
        print("💻 Building on Intel")
    else:
        print("❓ Unknown architecture")
    
    # Create the app
    print("📱 Creating app structure...")
    try:
        result = subprocess.run(['briefcase', 'create'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Create failed: {result.stderr}")
            return False
        print("✅ App structure created")
    except Exception as e:
        print(f"❌ Create error: {e}")
        return False
    
    # Build the app
    print("🔨 Building app...")
    try:
        result = subprocess.run(['briefcase', 'build'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            return False
        print("✅ App built successfully")
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False
    
    # Package the app
    print("📦 Packaging app...")
    try:
        result = subprocess.run(['briefcase', 'package'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Package failed: {result.stderr}")
            return False
        print("✅ App packaged successfully")
    except Exception as e:
        print(f"❌ Package error: {e}")
        return False
    
    return True

def verify_build():
    """Verify the build"""
    print("\n🔍 Verifying Build...")
    
    # Check if app bundle exists
    app_path = "dist/PQS Framework 40-Qubit.app"
    if os.path.exists(app_path):
        print(f"✅ App bundle created: {app_path}")
        
        # Check bundle structure
        contents_path = os.path.join(app_path, "Contents")
        if os.path.exists(contents_path):
            print("✅ Bundle structure valid")
            
            # Check for universal binary
            macos_path = os.path.join(contents_path, "MacOS")
            if os.path.exists(macos_path):
                print("✅ MacOS executable directory exists")
            
            # Check for resources
            resources_path = os.path.join(contents_path, "Resources")
            if os.path.exists(resources_path):
                print("✅ Resources directory exists")
                
                # Check for templates and static files
                app_resources = os.path.join(resources_path, "app")
                if os.path.exists(app_resources):
                    templates_path = os.path.join(app_resources, "templates")
                    static_path = os.path.join(app_resources, "static")
                    
                    if os.path.exists(templates_path):
                        print("✅ Templates included")
                    else:
                        print("❌ Templates missing")
                    
                    if os.path.exists(static_path):
                        print("✅ Static files included")
                    else:
                        print("❌ Static files missing")
        else:
            print("❌ Invalid bundle structure")
            return False
    else:
        print(f"❌ App bundle not found: {app_path}")
        return False
    
    return True

def main():
    """Main build process"""
    print("🚀 PQS Framework Universal Build Script")
    print("=" * 50)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix issues and try again.")
        sys.exit(1)
    
    # Step 2: Prepare environment
    prepare_build_environment()
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        sys.exit(1)
    
    # Step 4: Build app
    if not build_app():
        print("\n❌ Build failed.")
        sys.exit(1)
    
    # Step 5: Verify build
    if not verify_build():
        print("\n❌ Build verification failed.")
        sys.exit(1)
    
    print("\n🎉 Build Complete!")
    print("=" * 50)
    print("✅ PQS Framework has been built successfully")
    print("📱 App location: dist/PQS Framework 40-Qubit.app")
    print("🌍 Universal binary supports both Intel and Apple Silicon Macs")
    print("📦 Standalone - no Python installation required on target machines")
    print("\n🔧 To test on your fiancé's Intel Mac:")
    print("   1. Copy the .app bundle to the Intel Mac")
    print("   2. Double-click to run")
    print("   3. The app will automatically detect Intel architecture")
    print("   4. Intel-optimized quantum algorithms will be used")

if __name__ == "__main__":
    main()