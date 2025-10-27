#!/usr/bin/env python3
"""
Build Universal macOS App for PQS Framework
Targets: macOS 15.0+ (Sequoia and beyond)
Architectures: x86_64 (Intel) and arm64 (Apple Silicon)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

APP_NAME = "PQS Framework"
BUNDLE_ID = "com.pqs.framework"
VERSION = "1.0.0"
MIN_MACOS_VERSION = "15.0"

def run_command(cmd, cwd=None):
    """Run a shell command"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def create_app_bundle():
    """Create the macOS app bundle structure"""
    print("📦 Creating app bundle structure...")
    
    app_path = Path(f"{APP_NAME}.app")
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    frameworks = contents / "Frameworks"
    
    # Clean existing
    if app_path.exists():
        shutil.rmtree(app_path)
    
    # Create directories
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    frameworks.mkdir(parents=True)
    
    print(f"✅ Created {app_path}")
    return app_path

def create_info_plist(app_path):
    """Create Info.plist"""
    print("📝 Creating Info.plist...")
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>pqs_launcher</string>
    <key>CFBundleIconFile</key>
    <string>pqs_icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>{BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>{APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>{VERSION}</string>
    <key>CFBundleVersion</key>
    <string>{VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>{MIN_MACOS_VERSION}</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 PQS Framework. All rights reserved.</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
"""
    
    plist_path = app_path / "Contents" / "Info.plist"
    plist_path.write_text(plist_content)
    print(f"✅ Created {plist_path}")

def create_launcher_script(app_path):
    """Create the launcher script"""
    print("🚀 Creating launcher script...")
    
    launcher_content = """#!/bin/bash
# PQS Framework Launcher
# Universal launcher for Intel and Apple Silicon

# Get the directory where the app bundle is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"

# Detect architecture
ARCH=$(uname -m)
echo "🖥️  Detected architecture: $ARCH"

# Set Python path based on architecture
if [ "$ARCH" = "arm64" ]; then
    echo "🍎 Running on Apple Silicon"
    PYTHON_PATH="$RESOURCES/python_arm64/bin/python3"
elif [ "$ARCH" = "x86_64" ]; then
    echo "💻 Running on Intel"
    PYTHON_PATH="$RESOURCES/python_x86_64/bin/python3"
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "⚠️  Python not found at $PYTHON_PATH"
    echo "Using system Python..."
    PYTHON_PATH="python3"
fi

# Set environment
export PYTHONPATH="$RESOURCES:$PYTHONPATH"
export PQS_RESOURCES="$RESOURCES"

# Launch the app
cd "$RESOURCES"
exec "$PYTHON_PATH" "$RESOURCES/universal_pqs_app.py" "$@"
"""
    
    launcher_path = app_path / "Contents" / "MacOS" / "pqs_launcher"
    launcher_path.write_text(launcher_content)
    launcher_path.chmod(0o755)
    print(f"✅ Created {launcher_path}")

def copy_resources(app_path):
    """Copy application resources"""
    print("📁 Copying resources...")
    
    resources = app_path / "Contents" / "Resources"
    
    # Copy Python files
    files_to_copy = [
        "universal_pqs_app.py",
        "native_window.py",
        "real_quantum_ml_system.py",
        "quantum_ml_integration.py",
        "quantum_battery_guardian.py",
        "auto_battery_protection.py",
        "aggressive_idle_manager.py",
    ]
    
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, resources / file)
            print(f"  ✓ {file}")
    
    # Copy templates directory
    if Path("templates").exists():
        shutil.copytree("templates", resources / "templates", dirs_exist_ok=True)
        print(f"  ✓ templates/")
    
    # Copy static directory if exists
    if Path("static").exists():
        shutil.copytree("static", resources / "static", dirs_exist_ok=True)
        print(f"  ✓ static/")
    
    # Copy icon
    if Path("pqs_icon.icns").exists():
        shutil.copy2("pqs_icon.icns", resources / "pqs_icon.icns")
        print(f"  ✓ pqs_icon.icns")
    
    print("✅ Resources copied")

def create_requirements_file():
    """Create requirements.txt for the app"""
    requirements = """# PQS Framework Requirements
cirq>=1.3.0
qiskit>=1.0.0
tensorflow-macos>=2.16.0; platform_machine == 'arm64'
tensorflow-metal>=1.1.0; platform_machine == 'arm64'
tensorflow>=2.16.0; platform_machine == 'x86_64'
torch>=2.0.0
numpy>=1.24.0
flask>=3.0.0
rumps>=0.4.0
psutil>=5.9.0
pyobjc-core>=10.0
pyobjc-framework-Cocoa>=10.0
pyobjc-framework-WebKit>=10.0
pillow>=10.0.0
"""
    
    Path("app_requirements.txt").write_text(requirements)
    print("✅ Created app_requirements.txt")

def create_build_script():
    """Create a script to build Python environments for both architectures"""
    script = """#!/bin/bash
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
"""
    
    Path("build_python_envs.sh").write_text(script)
    Path("build_python_envs.sh").chmod(0o755)
    print("✅ Created build_python_envs.sh")

def sign_app(app_path):
    """Code sign the app (optional, requires developer certificate)"""
    print("🔐 Attempting to sign app...")
    
    # Check if we have a signing identity
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True,
        text=True
    )
    
    if "0 valid identities found" in result.stdout:
        print("⚠️  No signing identity found - app will not be signed")
        print("   To sign, obtain an Apple Developer certificate")
        return False
    
    # Try to sign
    cmd = [
        "codesign",
        "--force",
        "--deep",
        "--sign",
        "-",  # Ad-hoc signing
        str(app_path)
    ]
    
    if run_command(cmd):
        print("✅ App signed (ad-hoc)")
        return True
    else:
        print("⚠️  Signing failed - app will still work but may show warnings")
        return False

def create_dmg(app_path):
    """Create a DMG installer"""
    print("💿 Creating DMG installer...")
    
    dmg_name = f"{APP_NAME.replace(' ', '_')}_v{VERSION}_Universal.dmg"
    
    # Remove existing DMG
    if Path(dmg_name).exists():
        Path(dmg_name).unlink()
    
    cmd = [
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", str(app_path),
        "-ov",
        "-format", "UDZO",
        dmg_name
    ]
    
    if run_command(cmd):
        print(f"✅ Created {dmg_name}")
        return True
    else:
        print("⚠️  DMG creation failed")
        return False

def main():
    """Main build process"""
    print("=" * 70)
    print("🚀 PQS Framework - Universal macOS App Builder")
    print("=" * 70)
    print(f"App Name: {APP_NAME}")
    print(f"Version: {VERSION}")
    print(f"Bundle ID: {BUNDLE_ID}")
    print(f"Min macOS: {MIN_MACOS_VERSION}")
    print(f"Architectures: x86_64 (Intel) + arm64 (Apple Silicon)")
    print("=" * 70)
    
    # Create app bundle
    app_path = create_app_bundle()
    
    # Create Info.plist
    create_info_plist(app_path)
    
    # Create launcher
    create_launcher_script(app_path)
    
    # Copy resources
    copy_resources(app_path)
    
    # Create requirements
    create_requirements_file()
    
    # Create build script
    create_build_script()
    
    # Sign app
    sign_app(app_path)
    
    # Create DMG
    create_dmg(app_path)
    
    print("\n" + "=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print(f"\n📦 App Bundle: {app_path}")
    print(f"💿 Installer: {APP_NAME.replace(' ', '_')}_v{VERSION}_Universal.dmg")
    print("\n📝 Next Steps:")
    print("1. Run ./build_python_envs.sh to create Python environments")
    print("2. Copy python_arm64/ and python_x86_64/ to app Resources/")
    print("3. Test the app on both Intel and Apple Silicon Macs")
    print("4. Distribute the DMG file")
    print("\n🎯 The app will automatically detect and use the correct")
    print("   Python environment based on the Mac's architecture!")
    print("=" * 70)

if __name__ == "__main__":
    main()
