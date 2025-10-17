#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal PQS Framework Launcher
Automatically detects and optimizes for your Mac architecture
"""

import sys
import os
import subprocess
import platform

def check_requirements():
    """Check if required packages are available"""
    required_packages = ['rumps', 'psutil', 'flask']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip3 install {' '.join(missing_packages)}")
        return False
    
    return True

def detect_system():
    """Quick system detection for launch message"""
    try:
        machine = platform.machine().lower()
        
        if 'arm' in machine or 'arm64' in machine:
            # Apple Silicon
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    brand = result.stdout.strip().lower()
                    if 'm3' in brand:
                        return 'Apple M3', 'maximum'
                    elif 'm2' in brand:
                        return 'Apple M2', 'high'
                    elif 'm1' in brand:
                        return 'Apple M1', 'high'
                    else:
                        return 'Apple Silicon', 'high'
            except:
                pass
            return 'Apple Silicon', 'high'
            
        elif any(arch in machine for arch in ['x86', 'amd64', 'i386']):
            # Intel Mac
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    brand = result.stdout.strip()
                    if 'i3' in brand:
                        return 'Intel Core i3', 'basic'
                    elif 'i5' in brand:
                        return 'Intel Core i5', 'medium'
                    elif 'i7' in brand:
                        return 'Intel Core i7', 'medium'
                    elif 'i9' in brand:
                        return 'Intel Core i9', 'medium'
            except:
                pass
            return 'Intel Mac', 'medium'
        else:
            return 'Unknown', 'minimal'
            
    except:
        return 'Unknown', 'minimal'

def main():
    """Launch Universal PQS"""
    print("🌍 Universal PQS Framework Launcher")
    print("=" * 50)
    
    # Check system compatibility
    if platform.system() != 'Darwin':
        print("❌ This application requires macOS")
        sys.exit(1)
    
    # Detect system
    chip, tier = detect_system()
    print(f"🔍 Detected: {chip}")
    print(f"🎯 Optimization Tier: {tier}")
    
    # Special messages for different architectures
    if 'M3' in chip:
        print("🔥 M3 MacBook detected - ULTIMATE PERFORMANCE MODE!")
        print("   • Full 40-qubit quantum acceleration")
        print("   • Metal GPU + Neural Engine")
        print("   • Maximum energy optimization")
    elif 'Apple' in chip:
        print("🍎 Apple Silicon detected - Full quantum acceleration!")
        print("   • 40-qubit quantum simulation")
        print("   • GPU acceleration with Metal")
        print("   • Advanced energy optimization")
    elif 'i3' in chip:
        print("💻 Intel i3 detected - CPU-friendly optimizations!")
        print("   • 20-qubit quantum simulation (i3 optimized)")
        print("   • Lightweight algorithms")
        print("   • Power-efficient processing")
    elif 'Intel' in chip:
        print("💻 Intel Mac detected - Classical optimization!")
        print("   • 30-qubit quantum simulation")
        print("   • CPU-optimized algorithms")
        print("   • Standard energy optimization")
    else:
        print("❓ Unknown system - Basic compatibility mode")
        print("   • 10-qubit simulation")
        print("   • Minimal optimization")
    
    print()
    
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All requirements satisfied")
    
    # Launch application
    print("\n🚀 Launching Universal PQS Framework...")
    print("📱 Menu bar app will appear shortly")
    print("🌐 Dashboard: http://localhost:5003")
    print("\n" + "=" * 50)
    
    try:
        # Import and run the universal app
        from universal_pqs_app import main as run_universal_pqs
        run_universal_pqs()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure universal_pqs_app.py is in the same directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Universal PQS Framework stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()