
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
