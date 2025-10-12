"""
Setup script to create PQS Framework.app bundle
"""
from setuptools import setup

APP = ['enhanced_app.py']
DATA_FILES = [
    ('templates', ['templates/dashboard.html', 'templates/eas_dashboard.html', 'templates/quantum_dashboard.html', 'templates/battery_history.html', 'templates/enhanced_eas_dashboard.html', 'templates/index.html', 'templates/real_time_eas_monitor.html']),
    ('static', ['static/themes.css', 'static/battery-history.js', 'static/battery-history-simple.js', 'static/battery-history-new.js']),
    ('', ['requirements.txt', 'apps.conf'])
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'app_icon.icns',
    'plist': {
        'CFBundleName': 'PQS Framework',
        'CFBundleDisplayName': 'PQS Framework - Ultimate EAS Fixed',
        'CFBundleIdentifier': 'com.pqsframework.ultimate',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSUIElement': True,  # Hide from dock (menu bar only)
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '13.0',  # Ventura+ for M3 optimization
        'CFBundleDocumentTypes': [],
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'NSHumanReadableCopyright': 'Copyright © 2025 PQS Framework - HM-Media Labs'
    },
    'packages': ['rumps', 'psutil', 'flask', 'waitress'],
    'includes': ['sqlite3', 'json', 'threading', 'subprocess', 'time', 'os', 'signal', 'statistics', 'datetime', 'asyncio'],
    'excludes': [
        # Exclude heavy quantum/AI dependencies that cause recursion
        'cirq', 'tensorflow', 'torch', 'sklearn', 'numpy', 'scipy', 'matplotlib',
        'tkinter', 'jupyter', 'IPython', 'pandas', 'sympy', 'networkx',
        # Exclude quantum system files from bundle (they'll run from source)
        'ultimate_eas_system', 'pure_cirq_quantum_system', 'advanced_neural_system',
        'gpu_acceleration', 'advanced_quantum_features',
        # Exclude problematic packages causing conflicts
        'test', 'setuptools._vendor', 'pip._vendor'
    ],
    'optimize': 2,
    'compressed': True,
    'strip': True,
    'no_chdir': True
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name='PQS Framework',
    version='2.0.0',
    description='PQS Framework - Ultimate EAS Fixes: Default enabled, working toggle, view status',
    author='PQS Framework - HM-Media Labs',
    url='https://github.com/Smacksmack206/Predictive-Quantum-Scheduling-Framework'
)
