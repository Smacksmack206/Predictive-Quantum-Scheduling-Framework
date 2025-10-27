# -*- mode: python ; coding: utf-8 -*-

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
