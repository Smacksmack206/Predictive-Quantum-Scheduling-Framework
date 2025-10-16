"""
PQS Framework 40-Qubit - macOS App Bundle Build Configuration
Production-ready setup for creating distributable .app bundle
"""
from setuptools import setup
import os

# Main application entry point
APP = ['fixed_40_qubit_app.py']

# Include all necessary data files for the app bundle
DATA_FILES = [
    # Web dashboard templates (all existing files)
    ('templates', [
        'templates/battery_history.html',
        'templates/comprehensive_system_control.html',
        'templates/index.html',
        'templates/quantum_dashboard_enhanced.html',
        'templates/quantum_dashboard.html',
        'templates/technical_validation.html',
        'templates/working_enhanced_eas_dashboard.html',
        'templates/working_real_time_eas_monitor.html'
    ]),
    
    # Static assets for web interface (all existing files)
    ('static', [
        'static/themes.css',
        'static/battery-history.js',
        'static/battery-history-simple.js',
        'static/battery-history-new.js'
    ]),
    
    # Quantum component modules (only existing files)
    ('quantum_components', [
        'quantum_circuit_manager_40.py',
        'apple_silicon_quantum_accelerator.py',
        'quantum_ml_interface.py',
        'quantum_entanglement_engine.py',
        'quantum_visualization_engine.py',
        'quantum_performance_benchmarking.py'
    ]),
    
    # Enhanced app dependency
    ('dependencies', [
        'enhanced_app.py'
    ]),
    
    # Documentation and configuration (only existing files)
    ('docs', [
        'PROJECT_ARCHITECTURE.md',
        'PRODUCTION_READY_VISUAL_FEATURES.md',
        'VISUAL_FEATURES_DOCUMENTATION.md',
        'DISTRIBUTED_OPTIMIZATION_NETWORK.md',
        'FIXES_APPLIED.md'
    ]),
    
    # Configuration files (only existing files)
    ('', [
        'requirements.txt',
        'apps.conf'
    ])
]

# py2app build options for production-ready app bundle
OPTIONS = {
    'argv_emulation': False,
    
    # App bundle metadata
    'plist': {
        'CFBundleName': 'PQS Framework 40-Qubit',
        'CFBundleDisplayName': 'PQS Framework - 40-Qubit Quantum System',
        'CFBundleIdentifier': 'com.pqsframework.40qubit.production',
        'CFBundleVersion': '4.0.0',
        'CFBundleShortVersionString': '4.0.0',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSUIElement': True,  # Menu bar app (hide from dock)
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',  # macOS Catalina and later for Intel Mac support
        'CFBundleDocumentTypes': [],
        'NSRequiresAquaSystemAppearance': False,
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'NSHumanReadableCopyright': 'Copyright © 2025 PQS Framework',
        'CFBundleGetInfoString': 'PQS Framework - 40-Qubit Quantum Energy Management System',
        
        # Permissions for system access
        'NSSystemAdministrationUsageDescription': 'PQS Framework needs system access for quantum energy optimization.',
        'NSAppleEventsUsageDescription': 'PQS Framework uses Apple Events for system integration.',
        
        # Network access for distributed optimization
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': True
        },
        
        # Comprehensive Python runtime locations for standalone distribution
        'PyRuntimeLocations': [
            # Bundled Python (first priority)
            '@executable_path/../Frameworks/Python.framework/Versions/Current/Python',
            '@executable_path/../Frameworks/Python.framework/Python',
            '@executable_path/../Resources/lib/python3.13/lib-dynload',
            
            # Common Homebrew locations (Apple Silicon)
            '/opt/homebrew/bin/python3',
            '/opt/homebrew/bin/python3.13',
            '/opt/homebrew/Frameworks/Python.framework/Versions/3.13/Python',
            '/opt/homebrew/Frameworks/Python.framework/Versions/Current/Python',
            
            # Common Homebrew locations (Intel)
            '/usr/local/bin/python3',
            '/usr/local/bin/python3.13',
            '/usr/local/Frameworks/Python.framework/Versions/3.13/Python',
            '/usr/local/Frameworks/Python.framework/Versions/Current/Python',
            
            # System Python locations
            '/System/Library/Frameworks/Python.framework/Versions/3.13/Python',
            '/System/Library/Frameworks/Python.framework/Versions/Current/Python',
            '/usr/bin/python3',
            
            # Fallback locations
            '/Library/Frameworks/Python.framework/Versions/3.13/Python',
            '/Library/Frameworks/Python.framework/Versions/Current/Python'
        ]
    },
    
    # Required packages for quantum system
    'packages': [
        'rumps',           # Menu bar app framework
        'psutil',          # System monitoring (CRITICAL)
        'flask',           # Web dashboard server
        'numpy',           # Quantum calculations
        'email',           # Email module (needed by pkg_resources)
        'pkg_resources',   # Package resources (needed by py2app)
        'setuptools',      # Setup tools (needed by pkg_resources)
        'importlib'        # Import library (needed by pkg_resources)
    ],
    
    # Explicitly include core modules and all required standard library modules
    'includes': [
        # Core application modules
        'fixed_40_qubit_app',
        'enhanced_app',
        # Ensure Intel Mac compatibility components are included
        'platform',
        'subprocess',
        'psutil',
        'numpy',
        # Complete email module hierarchy
        'email',
        'email.mime',
        'email.mime.text',
        'email.mime.multipart',
        'email.mime.base',
        'email.utils',
        'email.parser',
        'email.message',
        'email.header',
        'email.encoders',
        # Package management modules
        'pkg_resources',
        'setuptools',
        'distutils',
        'distutils.util',
        # Import system modules
        'importlib',
        'importlib.util',
        'importlib.metadata',
        'importlib.machinery',
        # Other standard library modules that might be needed
        'collections',
        'collections.abc',
        'functools',
        'itertools',
        'operator',
        'types',
        'weakref'
    ],
    
    # Exclude problematic packages and heavy frameworks
    'excludes': [
        # Heavy ML frameworks (not needed for quantum operations)
        'tensorflow', 'torch', 'sklearn', 'scipy', 'matplotlib',
        'pandas', 'sympy', 'networkx', 'seaborn', 'plotly',
        
        # Development tools only
        'tkinter', 'jupyter', 'IPython', 'notebook', 'pytest',
        
        # Problematic packages causing build conflicts
        'setuptools._vendor', 'pip._vendor', 'wheel'
    ],
    
    # Build optimization - Fully self-contained for distribution
    'alias': False,           # Create full app bundle (not alias)
    'semi_standalone': False, # Don't use semi-standalone
    'optimize': 0,            # No optimization to avoid runtime issues
    'compressed': False,      # Don't compress to avoid runtime issues
    'strip': False,          # Don't strip to preserve runtime info
    'site_packages': True,   # Include site packages
    'no_chdir': True,        # Don't change directory to avoid path issues
    'iconfile': None,        # No custom icon for now
    
    # Universal binary support for all Mac architectures
    'arch': 'universal2',     # Build universal binary for Intel + Apple Silicon
    
    # Python framework bundling
    'use_pythonpath': False,   # Don't use system PYTHONPATH
    'alias': False,           # Don't create alias bundle
    'emulate_shell_environment': False,  # Disable to avoid fork() issues
    
    # Resource optimization
    'debug_modulegraph': False,
    'resources': [],          # Additional resources
    
    # Build directory configuration
    'bdist_base': 'build',
    'dist_dir': 'dist'
}

# Setup configuration for py2app
setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    
    # Package metadata
    name='PQS Framework 40-Qubit',
    version='4.0.0',
    description='PQS Framework - 40-Qubit Quantum Energy Management System with Real-Time Optimization',
    long_description='''
    PQS Framework 40-Qubit is a revolutionary quantum energy management system that provides:
    
    • Real-time MacBook performance optimization using 40-qubit quantum circuits
    • Measurable battery life improvements through quantum-enhanced process scheduling
    • Interactive quantum visualization with D3.js circuit diagrams
    • Technical validation dashboard proving 100% authentic data usage
    • Distributed optimization network for sharing quantum algorithms
    • Cross-platform compatibility (Apple Silicon + Intel Mac support)
    • Production-ready web dashboard with glass morphism UI
    
    The system delivers genuine performance improvements through quantum computing
    principles while maintaining complete transparency in all data sources.
    ''',
    author='PQS Framework Development Team',
    author_email='contact@pqsframework.com',
    url='https://github.com/Smacksmack206/Predictive-Quantum-Scheduling-Framework',
    license='MIT',
    
    # Classification
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: System :: Systems Administration',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Utilities'
    ],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Runtime dependencies (will be bundled)
    install_requires=[
        'rumps>=0.3.0',
        'psutil>=5.8.0', 
        'flask>=2.0.0',
        'numpy>=1.20.0'
    ],
    
    # Keywords for discovery
    keywords='quantum computing energy optimization macOS battery performance scheduling'
)
