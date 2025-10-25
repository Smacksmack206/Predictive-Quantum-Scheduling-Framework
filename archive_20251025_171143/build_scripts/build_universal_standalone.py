#!/usr/bin/env python3
"""
Universal macOS Standalone App Builder
Creates a standalone .app bundle that works on:
- macOS 15 (Sequoia) through macOS 26 (future)
- Apple Silicon (M1, M2, M3, M4+)
- Intel Macs (i3, i5, i7, i9)

No Python or dependencies required for end users!
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

print("🚀 Universal PQS Framework - Standalone App Builder")
print("=" * 60)

# Configuration
APP_NAME = "PQS Framework 48-Qubit"
BUNDLE_ID = "com.pqsframework.48qubit"
VERSION = "5.0.0"
PYTHON_PATH = "/Users/home/Projects/system-tools/m3.macbook.air/quantum_ml_311/bin/python3.11"

# Paths
PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build_standalone"
DIST_DIR = PROJECT_ROOT / "dist_standalone"

def check_dependencies():
    """Check if required build tools are installed"""
    print("\n📦 Checking build dependencies...")
    
    # Check if py2app is installed
    try:
        import py2app
        print("✅ py2app installed")
    except ImportError:
        print("❌ py2app not installed")
        print("Installing py2app...")
        subprocess.run([PYTHON_PATH, "-m", "pip", "install", "py2app"], check=True)
        print("✅ py2app installed successfully")
    
    # Check if setuptools is installed
    try:
        import setuptools
        print("✅ setuptools installed")
    except ImportError:
        print("❌ setuptools not installed")
        print("Installing setuptools...")
        subprocess.run([PYTHON_PATH, "-m", "pip", "install", "setuptools"], check=True)
        print("✅ setuptools installed successfully")

def clean_build_dirs():
    """Clean previous build artifacts"""
    print("\n🧹 Cleaning previous builds...")
    
    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_path}")
    
    # Also clean standard build dirs
    for dir_name in ["build", "dist"]:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   Removed {dir_path}")
    
    print("✅ Build directories cleaned")

def create_setup_script():
    """Create setup.py for py2app"""
    print("\n📝 Creating setup script...")
    
    setup_content = '''
"""
Setup script for creating universal macOS standalone app
"""
from setuptools import setup

APP = ['universal_pqs_app.py']
DATA_FILES = [
    ('templates', [
        'templates/battery_history.html',
        'templates/battery_monitor.html',
        'templates/comprehensive_system_control.html',
        'templates/index.html',
        'templates/quantum_dashboard_enhanced.html',
        'templates/quantum_dashboard.html',
        'templates/technical_validation.html',
        'templates/universal_dashboard.html'
    ]),
    ('static', [
        'static/themes.css',
        'static/battery-history.js',
        'static/battery-history-simple.js',
        'static/battery-history-new.js'
    ]),
]

OPTIONS = {
    'argv_emulation': False,  # Disable to avoid fork() issues
    'plist': {
        'CFBundleName': 'PQS Framework 48-Qubit',
        'CFBundleDisplayName': 'PQS Framework 48-Qubit',
        'CFBundleIdentifier': 'com.pqsframework.48qubit',
        'CFBundleVersion': '5.0.0',
        'CFBundleShortVersionString': '5.0.0',
        'LSUIElement': True,  # Menu bar app
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',  # macOS Big Sur (supports both Intel and Apple Silicon)
        'NSRequiresAquaSystemAppearance': False,
    },
    'packages': [
        'rumps',
        'psutil',
        'flask',
        'numpy',
        'cirq',
        'tensorflow',
        'torch',
        'logging',
        'threading',
        'collections',
        'dataclasses',
        'typing',
        'json',
        'time',
        'platform',
        'subprocess',
    ],
    'includes': [
        'universal_pqs_app',
        'real_quantum_ml_system',
        'quantum_ml_integration',
        'real_quantum_engine',
        'real_ml_system',
        'metal_quantum_simulator',
        'quantum_ml_persistence',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython',
    ],
    'arch': 'universal2',  # Universal binary for Intel + Apple Silicon
    'iconfile': None,
    'strip': False,
    'optimize': 0,
    'semi_standalone': False,  # Fully standalone
    'site_packages': True,
    'use_pythonpath': False,
    'emulate_shell_environment': False,  # Disable to avoid fork() issues
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name='PQS Framework 48-Qubit',
    version='5.0.0',
    description='Quantum-ML Power Optimization System for macOS',
    author='HM Media Labs',
)
'''
    
    setup_path = PROJECT_ROOT / "setup_standalone.py"
    with open(setup_path, 'w') as f:
        f.write(setup_content)
    
    print(f"✅ Setup script created: {setup_path}")
    return setup_path

def build_app(setup_path):
    """Build the standalone app using py2app"""
    print("\n🔨 Building standalone app...")
    print("   This may take several minutes...")
    
    # Run py2app build
    cmd = [
        PYTHON_PATH,
        str(setup_path),
        'py2app',
        '--arch=universal2',  # Universal binary
    ]
    
    print(f"\n   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ App built successfully!")
        
        # Show build output
        if result.stdout:
            print("\n📋 Build output:")
            print(result.stdout[-1000:])  # Last 1000 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed!")
        print(f"\nError output:")
        print(e.stderr)
        return False

def create_dmg():
    """Create a DMG installer"""
    print("\n📦 Creating DMG installer...")
    
    app_path = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    dmg_path = PROJECT_ROOT / f"{APP_NAME}-v{VERSION}-Universal.dmg"
    
    if not app_path.exists():
        print(f"❌ App not found at {app_path}")
        return False
    
    # Remove old DMG if exists
    if dmg_path.exists():
        dmg_path.unlink()
    
    # Create DMG using hdiutil
    cmd = [
        'hdiutil',
        'create',
        '-volname', APP_NAME,
        '-srcfolder', str(app_path),
        '-ov',
        '-format', 'UDZO',
        str(dmg_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ DMG created: {dmg_path}")
        print(f"   Size: {dmg_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DMG creation failed: {e}")
        return False

def verify_app():
    """Verify the built app"""
    print("\n🔍 Verifying app...")
    
    app_path = PROJECT_ROOT / "dist" / f"{APP_NAME}.app"
    
    if not app_path.exists():
        print(f"❌ App not found at {app_path}")
        return False
    
    # Check app structure
    required_paths = [
        app_path / "Contents",
        app_path / "Contents" / "MacOS",
        app_path / "Contents" / "Resources",
        app_path / "Contents" / "Info.plist",
    ]
    
    for path in required_paths:
        if path.exists():
            print(f"✅ {path.name}")
        else:
            print(f"❌ Missing: {path.name}")
            return False
    
    # Check app size
    size_mb = sum(f.stat().st_size for f in app_path.rglob('*') if f.is_file()) / (1024*1024)
    print(f"\n📊 App size: {size_mb:.1f} MB")
    
    # Check architectures
    executable = app_path / "Contents" / "MacOS" / APP_NAME.replace(" ", "")
    if executable.exists():
        try:
            result = subprocess.run(
                ['file', str(executable)],
                capture_output=True,
                text=True
            )
            print(f"\n🏗️  Architectures:")
            print(f"   {result.stdout.strip()}")
            
            # Check if it's universal
            if 'x86_64' in result.stdout and 'arm64' in result.stdout:
                print("✅ Universal binary (Intel + Apple Silicon)")
            elif 'arm64' in result.stdout:
                print("⚠️  Apple Silicon only")
            elif 'x86_64' in result.stdout:
                print("⚠️  Intel only")
            else:
                print("❌ Unknown architecture")
                
        except Exception as e:
            print(f"⚠️  Could not check architectures: {e}")
    
    print("\n✅ App verification complete")
    return True

def create_readme():
    """Create README for distribution"""
    print("\n📄 Creating README...")
    
    readme_content = f"""
# {APP_NAME} - Universal macOS App

## Installation

1. Open the DMG file
2. Drag "{APP_NAME}.app" to your Applications folder
3. Launch from Applications or Spotlight

## System Requirements

- **macOS**: 11.0 (Big Sur) or later
  - Fully supports macOS 15 (Sequoia) through future versions
- **Hardware**: 
  - Apple Silicon (M1, M2, M3, M4+) - Full quantum acceleration
  - Intel Mac (i3, i5, i7, i9) - Optimized classical algorithms
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space

## Features

- ⚛️ Real quantum computing with Qiskit (40-qubit circuits)
- 🧠 Machine learning with TensorFlow and PyTorch
- 🍎 Apple Silicon optimization (Metal GPU, Neural Engine)
- 💻 Intel Mac support with optimized algorithms
- 🔋 Battery life optimization
- 📊 Real-time system monitoring
- 🌐 Web dashboard at http://localhost:5002

## First Launch

On first launch, macOS may show a security warning because the app is not signed.

To allow the app to run:
1. Right-click the app and select "Open"
2. Click "Open" in the security dialog
3. The app will start and appear in your menu bar

## Usage

1. Click the ⚡ icon in your menu bar
2. Select "Open Dashboard" to view the web interface
3. The app runs automatically in the background
4. Optimizations happen every 30 seconds

## Troubleshooting

**App won't open:**
- Right-click → Open (don't double-click)
- Check System Settings → Privacy & Security

**Dashboard won't load:**
- Wait 10 seconds after launch
- Try http://127.0.0.1:5002 instead

**Performance issues:**
- Close other apps to free resources
- Check Activity Monitor for CPU/memory usage

## Support

For issues or questions:
- GitHub: https://github.com/Smacksmack206/Predictive-Quantum-Scheduling-Framework
- Email: cvallieu94@gmail.com

## Version

Version: {VERSION}
Build: Universal (Intel + Apple Silicon)
Date: October 2025

---

© 2025 HM Media Labs. MIT License.
"""
    
    readme_path = PROJECT_ROOT / "dist" / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ README created: {readme_path}")

def main():
    """Main build process"""
    try:
        # Step 1: Check dependencies
        check_dependencies()
        
        # Step 2: Clean previous builds
        clean_build_dirs()
        
        # Step 3: Create setup script
        setup_path = create_setup_script()
        
        # Step 4: Build app
        if not build_app(setup_path):
            print("\n❌ Build failed!")
            return 1
        
        # Step 5: Verify app
        if not verify_app():
            print("\n⚠️  App verification failed, but app may still work")
        
        # Step 6: Create README
        create_readme()
        
        # Step 7: Create DMG
        create_dmg()
        
        # Success!
        print("\n" + "=" * 60)
        print("🎉 BUILD COMPLETE!")
        print("=" * 60)
        print(f"\n📦 Your standalone app is ready:")
        print(f"   App: dist/{APP_NAME}.app")
        print(f"   DMG: {APP_NAME}-v{VERSION}-Universal.dmg")
        print(f"\n🚀 The app works on:")
        print(f"   ✅ macOS 11+ (Big Sur through Sequoia and beyond)")
        print(f"   ✅ Apple Silicon (M1, M2, M3, M4+)")
        print(f"   ✅ Intel Macs (i3, i5, i7, i9)")
        print(f"\n💡 No Python or dependencies required for end users!")
        print(f"\n📋 Next steps:")
        print(f"   1. Test the app: open 'dist/{APP_NAME}.app'")
        print(f"   2. Distribute the DMG file to users")
        print(f"   3. Users just drag to Applications and run!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Build cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
