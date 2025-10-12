#!/usr/bin/env python3
"""
Battery Optimizer Pro - Direct Launch Version
Includes all Ultimate EAS fixes: default enabled, working toggle, view status
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Change to the script directory
os.chdir(current_dir)

print("🔋 Battery Optimizer Pro - Ultimate EAS Edition")
print("🚀 Ultimate EAS fixes included:")
print("   • Default enabled: ✅")
print("   • Working toggle: ✅") 
print("   • View status: ✅")
print("   • Auto-start optimization: ✅")
print()

# Import and run the enhanced app
try:
    import enhanced_app
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting app: {e}")
    sys.exit(1)