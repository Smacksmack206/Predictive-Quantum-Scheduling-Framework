#!/usr/bin/env python3
"""
Intel Mac Compatibility Test for PQS Framework
Tests all critical functions on Intel Macs to ensure compatibility
"""

import platform
import subprocess
import sys
import os

def test_intel_compatibility():
    """Test Intel Mac compatibility"""
    print("🔍 Testing Intel Mac Compatibility...")
    
    # Test 1: Architecture detection
    print("\n1. Testing Architecture Detection:")
    machine = platform.machine().lower()
    processor = platform.processor().lower()
    print(f"   Machine: {machine}")
    print(f"   Processor: {processor}")
    
    is_intel = any(arch in machine for arch in ['x86', 'amd64', 'i386'])
    is_apple_silicon = 'arm' in machine or 'arm64' in machine
    
    print(f"   Intel detected: {is_intel}")
    print(f"   Apple Silicon detected: {is_apple_silicon}")
    
    # Test 2: System calls that work on Intel
    print("\n2. Testing Intel-compatible system calls:")
    
    # Test CPU brand string (works on Intel)
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"   ✅ CPU Brand: {result.stdout.strip()}")
        else:
            print("   ❌ CPU Brand detection failed")
    except Exception as e:
        print(f"   ❌ CPU Brand error: {e}")
    
    # Test memory detection (works on Intel)
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            mem_gb = int(result.stdout.strip()) // (1024**3)
            print(f"   ✅ Memory: {mem_gb} GB")
        else:
            print("   ❌ Memory detection failed")
    except Exception as e:
        print(f"   ❌ Memory error: {e}")
    
    # Test 3: Apple Silicon specific calls (should fail gracefully on Intel)
    print("\n3. Testing Apple Silicon specific calls (should fail on Intel):")
    
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"   ⚠️  P-cores detected: {result.stdout.strip()} (unexpected on Intel)")
        else:
            print("   ✅ P-cores detection failed (expected on Intel)")
    except Exception as e:
        print(f"   ✅ P-cores error (expected on Intel): {e}")
    
    try:
        result = subprocess.run(['sysctl', '-n', 'hw.perflevel1.logicalcpu'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print(f"   ⚠️  E-cores detected: {result.stdout.strip()} (unexpected on Intel)")
        else:
            print("   ✅ E-cores detection failed (expected on Intel)")
    except Exception as e:
        print(f"   ✅ E-cores error (expected on Intel): {e}")
    
    # Test 4: Import compatibility
    print("\n4. Testing Import Compatibility:")
    
    try:
        import rumps
        print("   ✅ rumps imported successfully")
    except ImportError as e:
        print(f"   ❌ rumps import failed: {e}")
    
    try:
        import psutil
        print("   ✅ psutil imported successfully")
        print(f"   CPU count: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    except ImportError as e:
        print(f"   ❌ psutil import failed: {e}")
    
    try:
        import flask
        print("   ✅ flask imported successfully")
    except ImportError as e:
        print(f"   ❌ flask import failed: {e}")
    
    try:
        import numpy
        print("   ✅ numpy imported successfully")
    except ImportError as e:
        print(f"   ⚠️  numpy import failed (fallback mode): {e}")
    
    # Test 5: File system compatibility
    print("\n5. Testing File System Compatibility:")
    
    required_files = [
        'universal_pqs_app.py',
        'templates/quantum_dashboard_enhanced.html',
        'static/themes.css',
        'pyproject.toml'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} exists")
        else:
            print(f"   ❌ {file_path} missing")
    
    print("\n🏁 Intel Compatibility Test Complete!")
    
    # Summary
    if is_intel:
        print("\n✅ Running on Intel Mac - all Intel-specific optimizations should work")
    elif is_apple_silicon:
        print("\n🍎 Running on Apple Silicon - Intel compatibility cannot be fully tested")
    else:
        print("\n❓ Unknown architecture - compatibility uncertain")

if __name__ == "__main__":
    test_intel_compatibility()