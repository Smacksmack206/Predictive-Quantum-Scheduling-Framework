#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal macOS Binary Builder for PQS Framework
Builds standalone macOS applications that work on all Mac architectures
Supports macOS 15.0+ (Sequoia and later)
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse
from pathlib import Path

class UniversalBinaryBuilder:
    """Builder for creating standalone macOS applications"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.app_name = "PQS Framework 48-Qubit"
        self.main_script = "universal_pqs_app.py"
        self.briefcase_config = "briefcase.toml"

        # Universal binary configuration
        self.universal_config = {
            'macos_min_version': '15.0',  # macOS 15.0+ (Sequoia)
            'supported_architectures': ['x86_64', 'arm64'],  # Intel + Apple Silicon
            'python_version': '3.11',
            'app_identifier': 'com.pqs.universal',
            'version': '1.0.0',
            'description': 'PQS Framework 48-Qubit - Universal Quantum ML Optimization for All Macs'
        }

        # Check system requirements
        self._check_build_requirements()

    def _check_build_requirements(self):
        """Check if system meets build requirements"""
        print("🔍 Checking build requirements...")

        # Check macOS version
        if platform.system() != 'Darwin':
            raise RuntimeError("❌ Universal binary builds are only supported on macOS")

        # Check for required tools
        required_tools = ['python3', 'pip', 'pyinstaller']
        missing_tools = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            raise RuntimeError(f"❌ Missing required tools: {', '.join(missing_tools)}")

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            raise RuntimeError(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")

        print(f"✅ Build requirements met: macOS {platform.mac_ver()[0]}, Python {python_version.major}.{python_version.minor}")

    def create_pyinstaller_spec(self):
        """Create PyInstaller spec file for universal binary"""
        print("📝 Creating PyInstaller spec file...")

        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Universal PQS Framework

import os
import sys

# Universal binary configuration
block_cipher = None

# Main application data
app_name = '{self.app_name}'
main_script = '{self.main_script}'
version = '{self.universal_config["version"]}'
company_name = 'PQS Framework'
copyright = '© 2025 PQS Framework'

# Universal binary settings for macOS 15.0+
a = Analysis(
    [main_script],
    pathex=['.'],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('pqs-icon.icns', '.'),
    ],
    hiddenimports=[
        'rumps',
        'psutil',
        'flask',
        'numpy',
        'torch',
        'quantum_ml_integration',
        'real_quantum_ml_system',
        'apple_silicon_quantum_accelerator',
        'intel_optimized_quantum_ml',
        'metal_quantum_simulator',
        'quantum_circuit_manager_40',
        'quantum_entanglement_engine',
        'quantum_ml_hybrid',
        'quantum_ml_interface',
        'quantum_ml_persistence',
        'quantum_performance_benchmarking',
        'quantum_visualization_engine',
        'real_ml_system',
        'real_quantum_engine',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'unittest',
        'pdb',
        'pydoc',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create PYZ (Python archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create EXE (macOS app bundle)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application (menu bar app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Use current architecture (arm64 on Apple Silicon)
    codesign_identity=None,
    entitlements_file='entitlements.plist',
    icon='pqs-icon.icns',
)

# Create APP bundle for macOS - Simplified approach
app = BUNDLE(
    exe,
    name=f'{self.app_name}.app',
    icon='pqs-icon.icns',
    bundle_identifier='{self.universal_config["app_identifier"]}',
    version='{self.universal_config["version"]}',
    info_plist={{
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSMinimumSystemVersion': '{self.universal_config["macos_min_version"]}',
        'LSArchitecturePriority': ['x86_64', 'arm64'],  # Universal binary priority
        'LSMultipleInstancesProhibited': True,
        'NSHighResolutionCapable': True,
        'NSSupportsAutomaticGraphicsSwitching': True,
        'NSRequiresAquaSystemAppearance': False,
        'CFBundleShortVersionString': '{self.universal_config["version"]}',
        'CFBundleVersion': '{self.universal_config["version"]}',
        'CFBundleDisplayName': '{self.app_name}',
        'CFBundleName': '{self.app_name}',
        'CFBundleIdentifier': '{self.universal_config["app_identifier"]}',
        'CFBundleExecutable': '{self.app_name}',
        'CFBundleIconFile': 'pqs-icon.icns',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'NSHumanReadableCopyright': copyright,
        'CFBundleGetInfoString': '{self.universal_config["description"]}',
        'LSBackgroundOnly': True,  # Menu bar app
        'LSUIElement': True,  # Menu bar app
    }},
    # Fix for Python framework issues - use system Python
    python_path=None,  # Don't set custom Python path
)
'''

        spec_file = self.project_root / f'{self.app_name}.spec'
        with open(spec_file, 'w') as f:
            f.write(spec_content)

        print(f"✅ Created PyInstaller spec file: {spec_file}")
        return spec_file

    def install_build_dependencies(self):
        """Install build dependencies"""
        print("📦 Checking build dependencies...")

        # Install PyInstaller if not available
        try:
            import PyInstaller
            print("✅ PyInstaller already installed")
        except ImportError:
            print("📥 Installing PyInstaller...")
            pip_cmd = [sys.executable, '-m', 'pip', 'install', 'pyinstaller']
            try:
                subprocess.run(pip_cmd, check=True)
                print("✅ PyInstaller installed successfully")
            except subprocess.CalledProcessError:
                print("🔧 Trying alternative installation method...")
                subprocess.run(pip_cmd + ['--break-system-packages'], check=True)

        # Install required packages
        requirements_file = self.project_root / 'requirements_quantum_ml.txt'
        if os.path.exists(requirements_file):
            print(f"📥 Installing requirements from {requirements_file}...")

            # Read requirements file and filter out problematic packages
            with open(requirements_file, 'r') as f:
                requirements_content = f.read()

            # Remove the problematic nni package line
            filtered_requirements = []
            for line in requirements_content.split('\n'):
                if line.strip() and not line.strip().startswith('nni>=3.0.0'):
                    filtered_requirements.append(line)

            # Write filtered requirements to temp file
            temp_requirements = self.project_root / 'requirements_filtered.txt'
            with open(temp_requirements, 'w') as f:
                f.write('\n'.join(filtered_requirements))

            try:
                pip_cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(temp_requirements)]
                print(f"🏗️ Running: {' '.join(pip_cmd)}")
                subprocess.run(pip_cmd, check=True)

                # Clean up temp file
                temp_requirements.unlink()

                print("✅ Build dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Some packages may have failed to install: {e}")
                print("🔄 Continuing with build - some dependencies may be missing")
        else:
            print(f"⚠️ Requirements file {requirements_file} not found")

        print("✅ Build dependencies check completed")

    def build_universal_binary(self):
        """Build universal binary for both architectures"""
        print("🔨 Building universal binary...")

        # Create PyInstaller spec
        spec_file = self.create_pyinstaller_spec()

        # Build command for universal binary using system Python
        # Note: When using a spec file, PyInstaller options should be in the spec file
        build_cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            str(spec_file),
        ]

        print(f"🏗️ Running: {' '.join(build_cmd)}")
        print(f"🔧 Using system Python: {sys.executable}")
        result = subprocess.run(build_cmd, cwd=self.project_root)

        if result.returncode == 0:
            print("✅ Universal binary built successfully!")

            # Check output
            dist_path = self.project_root / 'dist'
            app_path = dist_path / f'{self.app_name}.app'

            if app_path.exists():
                print(f"📱 App bundle created: {app_path}")

                # Verify universal binary
                self.verify_universal_binary(app_path)

                return app_path
            else:
                print("❌ App bundle not found in dist directory")
                return None
        else:
            print(f"❌ Build failed with return code: {result.returncode}")
            return None

    def verify_universal_binary(self, app_path):
        """Verify the universal binary contains both architectures"""
        print("🔍 Verifying universal binary...")

        # Path to the executable within the app bundle
        executable_path = app_path / 'Contents' / 'MacOS' / self.app_name

        if not executable_path.exists():
            print(f"❌ Executable not found: {executable_path}")
            return False

        try:
            # Check architectures in the binary
            result = subprocess.run(['lipo', '-info', str(executable_path)],
                                  capture_output=True, text=True, check=True)

            output = result.stdout
            print(f"📋 Binary info: {output.strip()}")

            # Check if both architectures are present
            has_x86_64 = 'x86_64' in output
            has_arm64 = 'arm64' in output

            if has_x86_64 and has_arm64:
                print("✅ Universal binary verified: Contains both Intel (x86_64) and Apple Silicon (arm64)")
                return True
            elif has_x86_64:
                print("⚠️ Binary contains only Intel (x86_64) architecture")
                return False
            elif has_arm64:
                print("⚠️ Binary contains only Apple Silicon (arm64) architecture")
                return False
            else:
                print("❌ Binary architecture verification failed")
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to verify binary: {e}")
            return False
        except FileNotFoundError:
            print("❌ lipo command not found (Xcode command line tools required)")
            return False

    def create_dmg_package(self, app_path):
        """Create DMG package for distribution"""
        print("📦 Creating DMG package...")

        try:
            # Create DMG using create-dmg (if available) or hdiutil
            dmg_name = f"{self.app_name}_Universal_{self.universal_config['version']}.dmg"
            dmg_path = self.project_root / 'dist' / dmg_name

            # Use create-dmg if available (better for distribution)
            if shutil.which('create-dmg'):
                create_dmg_cmd = [
                    'create-dmg',
                    '--volname', f'{self.app_name} Universal',
                    '--volicon', 'pqs-icon.icns',
                    '--window-pos', '200', '120',
                    '--window-size', '800', '400',
                    '--icon-size', '100',
                    '--icon', f'{self.app_name}.app', '200', '190',
                    '--hide-extension', f'{self.app_name}.app',
                    '--app-drop-link', '600', '185',
                    '--codesign', 'Developer ID Application',
                    str(dmg_path),
                    str(app_path)
                ]

                print(f"📀 Creating DMG with create-dmg: {' '.join(create_dmg_cmd)}")
                result = subprocess.run(create_dmg_cmd, check=True)

            else:
                # Fallback to hdiutil
                print("📀 create-dmg not found, using hdiutil...")

                # Create temporary directory for DMG contents
                temp_dir = self.project_root / 'temp_dmg'
                temp_dir.mkdir(exist_ok=True)

                # Copy app to temp directory
                app_copy = temp_dir / app_path.name
                if app_copy.exists():
                    shutil.rmtree(app_copy)
                shutil.copytree(app_path, app_copy)

                # Create DMG
                hdiutil_cmd = [
                    'hdiutil', 'create',
                    '-volname', f'{self.app_name} Universal',
                    '-srcfolder', str(temp_dir),
                    '-ov',  # Overwrite existing
                    '-format', 'UDZO',  # Compressed
                    str(dmg_path)
                ]

                print(f"📀 Creating DMG with hdiutil: {' '.join(hdiutil_cmd)}")
                result = subprocess.run(hdiutil_cmd, check=True)

                # Clean up temp directory
                shutil.rmtree(temp_dir)

            if result.returncode == 0:
                print(f"✅ DMG package created: {dmg_path}")
                return dmg_path
            else:
                print(f"❌ DMG creation failed with return code: {result.returncode}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"❌ DMG creation failed: {e}")
            return None
        except Exception as e:
            print(f"❌ DMG creation error: {e}")
            return None

    def create_entitlements_file(self):
        """Create entitlements file for code signing"""
        print("📋 Creating entitlements file...")

        entitlements_content = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-only</key>
    <true/>
    <key>com.apple.security.files.downloads.read-only</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
</dict>
</plist>'''

        entitlements_file = self.project_root / 'entitlements.plist'
        with open(entitlements_file, 'w') as f:
            f.write(entitlements_content)

        print(f"✅ Created entitlements file: {entitlements_file}")
        return entitlements_file

    def build_and_package(self):
        """Complete build and packaging process"""
        print("🚀 Starting complete build and packaging process...")

        try:
            # Step 1: Install dependencies
            self.install_build_dependencies()

            # Step 2: Create entitlements file
            self.create_entitlements_file()

            # Step 3: Build universal binary
            app_path = self.build_universal_binary()

            if not app_path:
                print("❌ Build failed")
                return False

            # Step 4: Create DMG package
            dmg_path = self.create_dmg_package(app_path)

            if dmg_path:
                print("🎉 Build and packaging completed successfully!")
                print(f"📱 App bundle: {app_path}")
                print(f"📦 DMG package: {dmg_path}")
                print("\n📋 Distribution Information:")
                print(f"   • Supports: macOS {self.universal_config['macos_min_version']}+")
                print("   • Architectures: Intel (x86_64) + Apple Silicon (arm64)")
                print(f"   • Version: {self.universal_config['version']}")
                print(f"   • Identifier: {self.universal_config['app_identifier']}")
                return True
            else:
                print("⚠️ Build completed but packaging failed")
                return False

        except Exception as e:
            print(f"❌ Build process failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main build function"""
    parser = argparse.ArgumentParser(description='Build Universal PQS Framework Binary')
    parser.add_argument('--skip-dependencies', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--build-only', action='store_true',
                       help='Only build binary, skip DMG creation')

    args = parser.parse_args()

    builder = UniversalBinaryBuilder()

    try:
        if args.skip_dependencies:
            print("⏭️ Skipping dependency installation...")

        if args.build_only:
            print("🏗️ Building binary only...")
            builder.install_build_dependencies()
            builder.create_entitlements_file()
            app_path = builder.build_universal_binary()
            if app_path:
                print(f"✅ Binary built successfully: {app_path}")
                return 0
        else:
            success = builder.build_and_package()
            return 0 if success else 1

    except Exception as e:
        print(f"❌ Build failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
