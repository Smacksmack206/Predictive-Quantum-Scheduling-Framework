#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime hook for PyInstaller to fix Python framework issues
"""

import sys
import os

# Fix for Python framework loading issues
def fix_python_framework():
    """Fix Python framework loading for PyInstaller"""
    try:
        # Get the current executable path
        if hasattr(sys, '_MEIPASS'):
            app_path = sys._MEIPASS
        else:
            app_path = os.path.dirname(sys.executable)

        # Look for Python framework in the app bundle
        frameworks_path = os.path.join(app_path, '..', 'Frameworks')
        python_framework = os.path.join(frameworks_path, 'Python.framework', 'Versions', 'Current', 'Python')

        if os.path.exists(python_framework):
            print(f"🔧 Found Python framework: {python_framework}")
            # Add to sys.path if needed
            python_dir = os.path.dirname(python_framework)
            if python_dir not in sys.path:
                sys.path.insert(0, python_dir)
                print(f"✅ Added Python framework to sys.path: {python_dir}")
        else:
            print(f"⚠️ Python framework not found at: {python_framework}")

        # Also check for python3.13 directory
        python313_path = os.path.join(frameworks_path, 'python3.13')
        if os.path.exists(python313_path):
            if python313_path not in sys.path:
                sys.path.insert(0, python313_path)
                print(f"✅ Added python3.13 to sys.path: {python313_path}")

    except Exception as e:
        print(f"⚠️ Error in runtime hook: {e}")

# Run the fix
fix_python_framework()
