"""
Setup script to create Battery Optimizer Pro.app bundle
"""
from setuptools import setup

APP = ['enhanced_app.py']
DATA_FILES = [
    ('templates', ['templates/dashboard.html', 'templates/eas_dashboard.html']),
    ('', ['requirements.txt', 'apps.conf'])
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'app_icon.icns',
    'plist': {
        'CFBundleName': 'Battery Optimizer Pro',
        'CFBundleDisplayName': 'Battery Optimizer Pro',
        'CFBundleIdentifier': 'com.batteryoptimizer.pro',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSUIElement': True,  # Hide from dock (menu bar only)
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '11.0',
        'CFBundleDocumentTypes': [],
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'NSHumanReadableCopyright': 'Copyright © 2025 HM-Media Labs'
    },
    'packages': ['rumps', 'psutil', 'flask', 'waitress'],
    'includes': ['sqlite3', 'json', 'threading', 'subprocess', 'time', 'os', 'signal', 'statistics', 'datetime'],
    'excludes': ['tkinter', 'matplotlib', 'numpy', 'scipy'],
    'optimize': 2,
    'compressed': True,
    'strip': True
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    name='Battery Optimizer Pro',
    version='1.0.0',
    description='Intelligent battery optimization for macOS',
    author='HM-Media Labs',
    url='https://github.com/Smacksmack206/Battery-Optimizer-Pro'
)
