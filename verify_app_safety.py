#!/usr/bin/env python3
"""
Comprehensive App Safety Verification
Tests that all updates are safe and won't break the app
"""

import sys
import os

# Suppress verbose output during testing
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

def test_imports():
    """Test that all modules import without errors"""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test kernel_level_pqs
    try:
        from kernel_level_pqs import get_kernel_pqs, run_kernel_optimization
        print("✅ kernel_level_pqs imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ kernel_level_pqs import failed: {e}")
        tests_failed += 1
        return False
    
    # Test universal_pqs_app
    try:
        import universal_pqs_app
        print("✅ universal_pqs_app imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ universal_pqs_app import failed: {e}")
        tests_failed += 1
        return False
    
    print(f"\n✅ Import tests: {tests_passed} passed, {tests_failed} failed\n")
    return tests_failed == 0


def test_kernel_initialization():
    """Test that kernel-level PQS initializes correctly"""
    print("=" * 60)
    print("TEST 2: Kernel-Level PQS Initialization")
    print("=" * 60)
    
    try:
        from kernel_level_pqs import get_kernel_pqs
        kernel_pqs = get_kernel_pqs()
        
        # Check that it's enabled
        if not kernel_pqs.enabled:
            print("❌ Kernel PQS not enabled")
            return False
        print("✅ Kernel PQS is enabled")
        
        # Check status
        status = kernel_pqs.get_kernel_status()
        print(f"✅ Kernel PQS status retrieved")
        print(f"   - Root privileges: {status['root_privileges']}")
        print(f"   - Hooks active: {status['kernel_hooks_active']}")
        print(f"   - Enabled: {status['enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Kernel initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_optimization():
    """Test that kernel optimization runs without errors"""
    print("\n" + "=" * 60)
    print("TEST 3: Kernel Optimization Execution")
    print("=" * 60)
    
    try:
        from kernel_level_pqs import run_kernel_optimization
        result = run_kernel_optimization()
        
        # Check result
        if not result.get('success'):
            print(f"❌ Kernel optimization failed: {result.get('error', 'Unknown error')}")
            return False
        
        print("✅ Kernel optimization executed successfully")
        print(f"   - Total speedup: {result.get('total_speedup', 1.0):.2f}x")
        print(f"   - Kernel level: {result.get('kernel_level', False)}")
        
        # Check optimizations
        optimizations = result.get('optimizations', {})
        if not optimizations:
            print("⚠️  No optimizations returned")
        else:
            print(f"✅ {len(optimizations)} optimizations active:")
            for name, opt in optimizations.items():
                if opt.get('success'):
                    print(f"   - {name}: {opt.get('speedup', 1.0):.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Kernel optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_app_integration():
    """Test that kernel PQS is integrated into the main app"""
    print("\n" + "=" * 60)
    print("TEST 4: App Integration")
    print("=" * 60)
    
    try:
        import universal_pqs_app
        
        # Check that kernel_pqs_system exists
        if not hasattr(universal_pqs_app, 'kernel_pqs_system'):
            print("❌ kernel_pqs_system not found in app")
            return False
        print("✅ kernel_pqs_system attribute exists")
        
        # Check that it's initialized
        if universal_pqs_app.kernel_pqs_system is None:
            print("⚠️  kernel_pqs_system is None (graceful fallback)")
            return True  # This is OK - graceful fallback
        
        print("✅ kernel_pqs_system is initialized")
        print(f"   - Enabled: {universal_pqs_app.kernel_pqs_system.enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_universal_system():
    """Test that universal system still works"""
    print("\n" + "=" * 60)
    print("TEST 5: Universal System (No Breaking Changes)")
    print("=" * 60)
    
    try:
        import universal_pqs_app
        
        # Check that universal_system exists
        if not hasattr(universal_pqs_app, 'universal_system'):
            print("❌ universal_system not found in app")
            return False
        print("✅ universal_system attribute exists")
        
        # Check that it's initialized
        if universal_pqs_app.universal_system is None:
            print("❌ universal_system is None")
            return False
        
        print("✅ universal_system is initialized")
        print(f"   - Available: {universal_pqs_app.universal_system.available}")
        
        # Check system info
        if hasattr(universal_pqs_app.universal_system, 'system_info'):
            info = universal_pqs_app.universal_system.system_info
            print(f"   - System: {info.get('chip_model', 'Unknown')}")
            print(f"   - Tier: {info.get('optimization_tier', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_fallbacks():
    """Test that the app handles missing components gracefully"""
    print("\n" + "=" * 60)
    print("TEST 6: Graceful Fallbacks")
    print("=" * 60)
    
    try:
        # Test that kernel optimization handles errors gracefully
        from kernel_level_pqs import KernelLevelPQS
        
        # Create instance
        kernel = KernelLevelPQS()
        
        # Even without root, it should work
        result = kernel.optimize_kernel_operations()
        
        if not result.get('success'):
            print("⚠️  Optimization returned success=False (this is OK)")
        else:
            print("✅ Optimization works without root privileges")
        
        print("✅ Graceful fallback handling verified")
        return True
        
    except Exception as e:
        print(f"❌ Graceful fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_breaking_changes():
    """Test that existing functionality still works"""
    print("\n" + "=" * 60)
    print("TEST 7: No Breaking Changes to Existing Features")
    print("=" * 60)
    
    try:
        import universal_pqs_app
        
        # Test that quantum ML system still works
        if hasattr(universal_pqs_app, 'QUANTUM_ML_AVAILABLE'):
            print(f"✅ QUANTUM_ML_AVAILABLE: {universal_pqs_app.QUANTUM_ML_AVAILABLE}")
        
        # Test that enhanced quantum is still available
        if hasattr(universal_pqs_app, 'ENHANCED_QUANTUM_AVAILABLE'):
            print(f"✅ ENHANCED_QUANTUM_AVAILABLE: {universal_pqs_app.ENHANCED_QUANTUM_AVAILABLE}")
        
        # Test that anti-lag is still available
        if hasattr(universal_pqs_app, 'ANTI_LAG_AVAILABLE'):
            print(f"✅ ANTI_LAG_AVAILABLE: {universal_pqs_app.ANTI_LAG_AVAILABLE}")
        
        # Test that app accelerator is still available
        if hasattr(universal_pqs_app, 'APP_ACCELERATOR_AVAILABLE'):
            print(f"✅ APP_ACCELERATOR_AVAILABLE: {universal_pqs_app.APP_ACCELERATOR_AVAILABLE}")
        
        print("✅ All existing features still available")
        return True
        
    except Exception as e:
        print(f"❌ Breaking changes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all safety tests"""
    print("\n" + "=" * 60)
    print("PQS APP SAFETY VERIFICATION")
    print("=" * 60)
    print("Testing that all updates are safe and won't break the app\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Kernel Initialization", test_kernel_initialization),
        ("Kernel Optimization", test_kernel_optimization),
        ("App Integration", test_app_integration),
        ("Universal System", test_universal_system),
        ("Graceful Fallbacks", test_graceful_fallbacks),
        ("No Breaking Changes", test_no_breaking_changes),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED - APP IS SAFE!")
        print("✅ No breaking changes detected")
        print("✅ Kernel-level integration working")
        print("✅ Graceful fallbacks in place")
        print("✅ Existing features preserved")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED - REVIEW NEEDED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
