"""
Minimal PQS Framework 40-Qubit - macOS App Bundle Build Configuration
Focused on fixing the email module issue without including heavy dependencies
"""
from setuptools import setup
import os

# Main application entry point
APP = ['fixed_40_qubit_app.py']

# Minimal data files
DATA_FILES = [
    ('templates', [
        'templates/battery_history.html',
        'templates/working_enhanced_eas_dashboard.html',
        'templates/working_real_time_eas_monitor.html'
    ]),
    ('static', [
        'static/themes.css'
    ]),
    ('', [
        'apps.conf'
    ])
]

# Minimal py2app options focused on fixing the email module issue
OPTIONS = {
    'argv_emulation': False,
    
    # App bundle metadata
    'plist': {
        'CFBundleName': 'PQS Framework 40-Qubit',
        'CFBundleDisplayName': 'PQS Framework - 40-Qubit Quantum System',
        'CFBundleIdentifier': 'com.pqsframework.40qubit.production',
        'CFBundleVersion': '4.0.0',
        'CFBundleShortVersionString': '4.0.0',
        'LSUIElement': True,  # Menu bar app
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
        
        # Comprehensive Python runtime locations
        'PyRuntimeLocations': [
            '@executable_path/../Frameworks/Python.framework/Versions/Current/Python',
            '@executable_path/../Frameworks/Python.framework/Python',
            '/opt/homebrew/bin/python3',
            '/usr/local/bin/python3',
            '/usr/bin/python3'
        ]
    },
    
    # Only essential packages
    'packages': [
        'rumps',
        'psutil', 
        'flask',
        'numpy'
    ],
    
    # Explicitly include only what we need
    'includes': [
        'fixed_40_qubit_app',
        'platform',
        'subprocess',
        'psutil',
        'numpy',
        'email',
        'email.mime',
        'email.mime.text',
        'email.utils',
        'importlib',
        'importlib.util'
    ],
    
    # Exclude everything we don't need
    'excludes': [
        'tensorflow', 'torch', 'sklearn', 'scipy', 'matplotlib',
        'pandas', 'sympy', 'networkx', 'seaborn', 'plotly',
        'tkinter', 'jupyter', 'IPython', 'notebook', 'pytest',
        'keras', 'tensorboard', 'qiskit', 'cirq',
        'test', 'unittest', 'doctest', 'pydoc'
    ],
    
    # Build settings
    'alias': False,
    'semi_standalone': False,
    'optimize': 0,
    'compressed': False,
    'strip': False,
    'site_packages': True,
    'no_chdir': True,
    'emulate_shell_environment': False,  # Disable to avoid fork() issues
    'arch': 'universal2'
}

# Minimal setup
setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name='PQS Framework 40-Qubit',
    version='4.0.0'
)