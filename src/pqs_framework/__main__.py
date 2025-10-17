#!/usr/bin/env python3
"""
PQS Framework 40-Qubit - Main Entry Point
Universal compatibility for Intel and Apple Silicon Macs
"""

def main():
    """Main entry point for Briefcase"""
    # Import and run the main application from the local package
    from . import universal_pqs_app
    universal_pqs_app.main()

if __name__ == "__main__":
    main()