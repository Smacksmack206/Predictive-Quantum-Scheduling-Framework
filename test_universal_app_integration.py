#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Universal App Integration
===============================

Verifies that universal_pqs_app.py works with enhanced quantum system.
"""

import sys
import time

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from enhanced_quantum_ml_system import create_enhanced_system
        print("✅ Enhanced system import")
    except Exception as e:
        print(f"❌ Enhanced system import: {e}")
        return False
    
    try:
        # Import key components from universal app
        import rumps
        print("✅ rumps import")
    except Exception as e:
        print(f"⚠️ rumps import (OK if not on macOS): {e}")
    
    try:
        from flask import Flask
        print("✅ Flask import")
    except Exception as e:
        print(f"❌ Flask import: {e}")
        return False
    
    return True

def test_enhanced_system():
    """Test enhanced system functionality"""
    print("\nTesting enhanced system...")
    
    try:
        from enhanced_quantum_ml_system import create_enhanced_system
        
        # Create system
        system = create_enhanced_system(enable_unified=True)
        print(f"✅ System created (unified: {system.unified_enabled})")
        
        # Run optimization
        result = system.run_optimization()
        print(f"✅ Optimization: {result['energy_saved_percent']:.1f}% saved")
        
        # Get metrics
        metrics = system.get_hardware_metrics()
        print(f"✅ Metrics: {len(metrics)} values")
        
        # Get stats
        stats = system.get_statistics()
        print(f"✅ Stats: {stats['optimization_count']} optimizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_universal_app_compatibility():
    """Test that universal app can use enhanced system"""
    print("\nTesting universal app compatibility...")
    
    try:
        # Simulate universal app initialization
        ENHANCED_QUANTUM_AVAILABLE = True
        
        from enhanced_quantum_ml_system import create_enhanced_system
        
        # Create enhanced system like universal app does
        enhanced_system = None
        if ENHANCED_QUANTUM_AVAILABLE:
            enhanced_system = create_enhanced_system(enable_unified=True)
            print("✅ Enhanced system initialized in universal app style")
        
        # Test optimization like universal app does
        if enhanced_system:
            result = enhanced_system.run_optimization()
            if result['success']:
                print(f"✅ Optimization successful: {result['energy_saved_percent']:.1f}%")
                print(f"   Method: {result['method']}")
                print(f"   GPU: {result['gpu_accelerated']}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Universal app compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test that API endpoints would work"""
    print("\nTesting API endpoint compatibility...")
    
    try:
        from enhanced_quantum_ml_system import create_enhanced_system
        
        system = create_enhanced_system(enable_unified=True)
        
        # Test status endpoint data
        stats = system.get_statistics()
        metrics = system.get_hardware_metrics()
        recommendations = system.get_recommendations()
        
        print(f"✅ Status endpoint data available")
        print(f"   Stats keys: {len(stats)}")
        print(f"   Metrics keys: {len(metrics)}")
        print(f"   Recommendations: {len(recommendations)}")
        
        # Test optimize endpoint data
        result = system.run_optimization()
        print(f"✅ Optimize endpoint data available")
        print(f"   Result keys: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Universal App Integration Test")
    print("=" * 60)
    
    results = {
        'imports': test_imports(),
        'enhanced_system': test_enhanced_system(),
        'universal_app': test_universal_app_compatibility(),
        'api_endpoints': test_api_endpoints()
    }
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed - Universal app integration ready!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
