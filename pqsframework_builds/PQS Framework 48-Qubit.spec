# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Universal PQS Framework

import os
import sys

# Universal binary configuration
block_cipher = None

# Main application data
app_name = 'PQS Framework 48-Qubit'
main_script = 'universal_pqs_app.py'
version = '1.0.0'
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
    hooksconfig={},
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
    name=f'PQS Framework 48-Qubit.app',
    icon='pqs-icon.icns',
    bundle_identifier='com.pqs.universal',
    version='1.0.0',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'LSMinimumSystemVersion': '15.0',
        'LSArchitecturePriority': ['x86_64', 'arm64'],  # Universal binary priority
        'LSMultipleInstancesProhibited': True,
        'NSHighResolutionCapable': True,
        'NSSupportsAutomaticGraphicsSwitching': True,
        'NSRequiresAquaSystemAppearance': False,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'CFBundleDisplayName': 'PQS Framework 48-Qubit',
        'CFBundleName': 'PQS Framework 48-Qubit',
        'CFBundleIdentifier': 'com.pqs.universal',
        'CFBundleExecutable': 'PQS Framework 48-Qubit',
        'CFBundleIconFile': 'pqs-icon.icns',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': '????',
        'NSHumanReadableCopyright': copyright,
        'CFBundleGetInfoString': 'PQS Framework 48-Qubit - Universal Quantum ML Optimization for All Macs',
        'LSBackgroundOnly': True,  # Menu bar app
        'LSUIElement': True,  # Menu bar app
    },
    # Fix for Python framework issues - use system Python
    python_path=None,  # Don't set custom Python path
)
