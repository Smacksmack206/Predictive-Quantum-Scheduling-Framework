#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQS Framework - Main Entry Point for Briefcase
"""

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Main entry point for PQS Framework"""
    try:
        # Import and run the universal PQS app main function
        from .universal_pqs_app import main as app_main
        
        # Run the app
        app_main()
        
    except Exception as e:
        print(f"❌ Failed to start PQS Framework: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
