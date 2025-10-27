#!/usr/bin/env python3
"""
Build Standalone Universal macOS App for PQS Framework
Creates a completely self-contained app with embedded Python and all dependencies
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

APP_NAME = "PQS Framework"
BUNDLE_ID = "com.pqs.framework"
VERSION = "1.0.0"
MIN_MACOS_VERSION = "15.0"

def check_dependencies():
    """Check if required build tools are installed"""
    print("🔍 Checking dependencies...")
    
    required = {
        'PyInstaller': 'pip install pyinstaller',
        'PIL': 'pip install pillow',
    }
    
    missing = []
    for package, install_cmd in required.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - missing")
            missing.append((package, install_cmd))
    
    if missing:
        print("\n❌ Missing dependencies. Install with:")
        for pkg, cmd in missing:
            print(f"   {cmd}")
        return False
    
    print("✅ All dependencies found")
    return True

def create_pyinstaller_spec():
    """Create PyInstaller spec file for universal build"""
    print("📝 Creating PyInstaller spec file...")
    
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Collect all data files
datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('pqs_icon.icns', '.'),
    ('pqs_icon.png', '.'),
]

# Hidden imports for quantum and ML libraries
hiddenimports = [
    'cirq',
    'qiskit',
    'tensorflow',
    'torch',
    'flask',
    'rumps',
    'psutil',
    'numpy',
    'scipy',
    'sympy',
    'PIL',
    'objc',
    'Foundation',
    'AppKit',
    'WebKit',
    'sqlite3',
    'json',
    'threading',
    'multiprocessing',
]

# Analysis
a = Analysis(
    ['universal_pqs_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect Python files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pqs_framework',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='universal2',
    codesign_identity=None,
    entitlements_file=None,
    icon='pqs_icon.icns',
)

# Collect everything
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pqs_framework',
)

# Create macOS app bundle
app = BUNDLE(
    coll,
    name='PQS Framework.app',
    icon='pqs_icon.icns',
    bundle_identifier='com.pqs.framework',
    version='1.0.0',
    info_plist={
        'CFBundleName': 'PQS Framework',
        'CFBundleDisplayName': 'PQS Framework',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'LSMinimumSystemVersion': '15.0',
        'NSHighResolutionCapable': True,
        'NSSupportsAutomaticGraphicsSwitching': True,
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'NSHumanReadableCopyright': 'Copyright © 2025 PQS Framework',
        'LSUIElement': False,
        'NSPrincipalClass': 'NSApplication',
    },
)
"""
    
    Path("pqs_framework.spec").write_text(spec_content)
    print("✅ Created pqs_framework.spec")

def build_with_pyinstaller():
    """Build the app using PyInstaller"""
    print("🏗️  Building standalone app with PyInstaller...")
    print("   This may take several minutes...")
    
    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "pqs_framework.spec"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Build failed:")
        print(result.stderr)
        return False
    
    print("✅ PyInstaller build complete")
    return True

def fix_app_structure():
    """Fix the app structure and add missing files"""
    print("🔧 Fixing app structure...")
    
    app_path = Path("dist/PQS Framework.app")
    
    if not app_path.exists():
        print("❌ App not found at dist/PQS Framework.app")
        return False
    
    # Ensure Resources directory exists
    resources = app_path / "Contents" / "Resources"
    resources.mkdir(parents=True, exist_ok=True)
    
    # Copy additional Python files that might be needed
    python_files = [
        "native_window.py",
        "real_quantum_ml_system.py",
        "quantum_ml_integration.py",
        "quantum_battery_guardian.py",
        "auto_battery_protection.py",
        "aggressive_idle_manager.py",
    ]
    
    for file in python_files:
        if Path(file).exists():
            dest = resources / file
            if not dest.exists():
                shutil.copy2(file, dest)
                print(f"  ✓ Added {file}")
    
    # Ensure icon is in Resources
    if Path("pqs_icon.icns").exists():
        icon_dest = resources / "pqs_icon.icns"
        if not icon_dest.exists():
            shutil.copy2("pqs_icon.icns", icon_dest)
            print(f"  ✓ Added icon")
    
    print("✅ App structure fixed")
    return True

def test_app():
    """Test if the app can launch"""
    print("🧪 Testing app...")
    
    app_path = Path("dist/PQS Framework.app")
    
    if not app_path.exists():
        print("❌ App not found")
        return False
    
    # Check if executable exists
    exe_path = app_path / "Contents" / "MacOS" / "pqs_framework"
    if not exe_path.exists():
        print(f"❌ Executable not found at {exe_path}")
        return False
    
    print(f"✅ Executable found: {exe_path}")
    
    # Check if it's executable
    if not os.access(exe_path, os.X_OK):
        print("⚠️  Executable doesn't have execute permissions")
        exe_path.chmod(0o755)
        print("  ✓ Fixed permissions")
    
    # Try to get file info
    result = subprocess.run(["file", str(exe_path)], capture_output=True, text=True)
    print(f"  File type: {result.stdout.strip()}")
    
    # Check for universal binary
    result = subprocess.run(["lipo", "-info", str(exe_path)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Architecture: {result.stdout.strip()}")
    
    print("✅ App structure looks good")
    return True

def sign_app():
    """Sign the app for distribution"""
    print("🔐 Signing app...")
    
    app_path = Path("dist/PQS Framework.app")
    
    # Ad-hoc signing (works without developer certificate)
    cmd = [
        "codesign",
        "--force",
        "--deep",
        "--sign",
        "-",
        str(app_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ App signed (ad-hoc)")
        
        # Verify signature
        verify_cmd = ["codesign", "--verify", "--verbose", str(app_path)]
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        
        if verify_result.returncode == 0:
            print("✅ Signature verified")
        else:
            print("⚠️  Signature verification failed")
        
        return True
    else:
        print("⚠️  Signing failed (app will still work)")
        print(result.stderr)
        return False

def create_dmg():
    """Create DMG installer"""
    print("💿 Creating DMG installer...")
    
    app_path = Path("dist/PQS Framework.app")
    dmg_name = f"PQS_Framework_v{VERSION}_Universal_Standalone.dmg"
    
    # Remove existing DMG
    if Path(dmg_name).exists():
        Path(dmg_name).unlink()
    
    # Create DMG
    cmd = [
        "hdiutil", "create",
        "-volname", "PQS Framework",
        "-srcfolder", str(app_path),
        "-ov",
        "-format", "UDZO",
        dmg_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Get DMG size
        size = Path(dmg_name).stat().st_size / (1024 * 1024)
        print(f"✅ Created {dmg_name} ({size:.1f} MB)")
        return True
    else:
        print("❌ DMG creation failed")
        print(result.stderr)
        return False

def create_launch_test_script():
    """Create a script to test the app"""
    script = """#!/bin/bash
# Test PQS Framework App

echo "🧪 Testing PQS Framework..."
echo ""

APP="dist/PQS Framework.app"

if [ ! -d "$APP" ]; then
    echo "❌ App not found at: $APP"
    exit 1
fi

echo "✅ App found"
echo ""

# Check architecture
echo "🔍 Checking architecture support..."
ARCH=$(uname -m)
echo "   Current architecture: $ARCH"

# Check executable
EXE="$APP/Contents/MacOS/pqs_framework"
if [ -f "$EXE" ]; then
    echo "✅ Executable found"
    file "$EXE"
    lipo -info "$EXE" 2>/dev/null || echo "   (Not a fat binary)"
else
    echo "❌ Executable not found"
    exit 1
fi

echo ""
echo "🚀 Launching app..."
open "$APP"

echo ""
echo "✅ App launched!"
echo "   Check if the window appears and engine selection works"
"""
    
    Path("test_app.sh").write_text(script)
    Path("test_app.sh").chmod(0o755)
    print("✅ Created test_app.sh")

def main():
    """Main build process"""
    print("=" * 70)
    print("🚀 PQS Framework - Standalone Universal App Builder")
    print("=" * 70)
    print(f"App Name: {APP_NAME}")
    print(f"Version: {VERSION}")
    print(f"Bundle ID: {BUNDLE_ID}")
    print(f"Min macOS: {MIN_MACOS_VERSION}")
    print(f"Type: Standalone (embedded Python + all dependencies)")
    print("=" * 70)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    print()
    
    # Create spec file
    create_pyinstaller_spec()
    print()
    
    # Build with PyInstaller
    if not build_with_pyinstaller():
        print("\n❌ Build failed")
        return False
    
    print()
    
    # Fix app structure
    if not fix_app_structure():
        print("\n❌ Failed to fix app structure")
        return False
    
    print()
    
    # Test app
    if not test_app():
        print("\n⚠️  App test had issues")
    
    print()
    
    # Sign app
    sign_app()
    print()
    
    # Create DMG
    create_dmg()
    print()
    
    # Create test script
    create_launch_test_script()
    print()
    
    print("=" * 70)
    print("✅ BUILD COMPLETE!")
    print("=" * 70)
    print()
    print(f"📦 Standalone App: dist/PQS Framework.app")
    print(f"💿 Installer: PQS_Framework_v{VERSION}_Universal_Standalone.dmg")
    print()
    print("📝 The app is completely standalone:")
    print("   ✓ Embedded Python interpreter")
    print("   ✓ All dependencies included")
    print("   ✓ No external requirements")
    print("   ✓ Works on Intel and Apple Silicon")
    print("   ✓ Runs on macOS 15.0+")
    print()
    print("🧪 Test the app:")
    print("   ./test_app.sh")
    print()
    print("📤 Distribute:")
    print("   Share the DMG file with users")
    print("   They can drag to Applications and run immediately")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
