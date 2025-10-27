"""
py2app setup script for PQS Framework
This creates a proper double-clickable macOS app
"""

from setuptools import setup

APP = ['pqs_framework/__main__.py']
DATA_FILES = [
    ('templates', ['pqs_framework/templates']),
    ('static', ['pqs_framework/static']),
]
OPTIONS = {
    'argv_emulation': False,
    'packages': [
        'cirq', 'qiskit', 'tensorflow', 'torch', 'flask', 'rumps',
        'psutil', 'numpy', 'objc', 'Foundation', 'AppKit', 'WebKit'
    ],
    'includes': [
        'pqs_framework.native_window',
        'pqs_framework.real_quantum_ml_system',
        'pqs_framework.quantum_ml_integration',
        'pqs_framework.quantum_battery_guardian',
        'pqs_framework.auto_battery_protection',
        'pqs_framework.aggressive_idle_manager',
    ],
    'iconfile': 'pqs_icon.icns',
    'plist': {
        'CFBundleName': 'PQS Framework',
        'CFBundleDisplayName': 'PQS Framework',
        'CFBundleIdentifier': 'com.pqs.framework',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '15.0',
        'NSHighResolutionCapable': True,
    }
}

setup(
    name='PQS Framework',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
