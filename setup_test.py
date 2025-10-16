"""
Minimal test setup for py2app build
"""
from setuptools import setup

APP = ['fixed_40_qubit_app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'packages': ['rumps', 'psutil', 'flask', 'numpy'],
    'excludes': ['tkinter'],
    'plist': {
        'CFBundleName': 'PQS Framework Test',
        'CFBundleDisplayName': 'PQS Framework Test',
        'CFBundleIdentifier': 'com.pqsframework.test',
        'CFBundleVersion': '1.0.0',
        'LSUIElement': True,
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)