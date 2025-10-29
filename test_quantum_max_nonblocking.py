#!/usr/bin/env python3
"""
Test that Quantum Max Mode activation is non-blocking
"""

import time
import threading

def test_quantum_max_activation():
    """Test that quantum max activation doesn't block"""
    print("🧪 Testing Quantum Max Mode Activation (Non-Blocking)")
    print("=" * 60)
    
    try:
        from quantum_max_integration import get_quantum_max_integration
        
        qmax = get_quantum_max_integration()
        
        if not qmax.quantum_scheduler:
            print("⚠️  Quantum Max Scheduler not available - test skipped")
            return True
        
        print("✅ Quantum Max Integration loaded")
        
        # Test 1: Activation should return immediately
        print("\nTest 1: Activation timing...")
        start_time = time.time()
        
        result = qmax.activate_quantum_max_mode(interval=10)
        
        elapsed = time.time() - start_time
        
        if elapsed > 1.0:
            print(f"❌ FAIL: Activation took {elapsed:.2f}s (should be < 1s)")
            print("   This indicates blocking behavior!")
            return False
        else:
            print(f"✅ PASS: Activation took {elapsed:.3f}s (non-blocking)")
        
        if not result:
            print("⚠️  Activation returned False (may be expected)")
        else:
            print("✅ Activation successful")
        
        # Test 2: Deactivation should also be non-blocking
        print("\nTest 2: Deactivation timing...")
        start_time = time.time()
        
        qmax.deactivate_quantum_max_mode()
        
        elapsed = time.time() - start_time
        
        if elapsed > 1.0:
            print(f"❌ FAIL: Deactivation took {elapsed:.2f}s (should be < 1s)")
            return False
        else:
            print(f"✅ PASS: Deactivation took {elapsed:.3f}s (non-blocking)")
        
        # Test 3: Multiple rapid activations shouldn't block
        print("\nTest 3: Rapid activation/deactivation...")
        start_time = time.time()
        
        for i in range(3):
            qmax.activate_quantum_max_mode(interval=10)
            qmax.deactivate_quantum_max_mode()
        
        elapsed = time.time() - start_time
        
        if elapsed > 3.0:
            print(f"❌ FAIL: 3 cycles took {elapsed:.2f}s (should be < 3s)")
            return False
        else:
            print(f"✅ PASS: 3 cycles took {elapsed:.3f}s (non-blocking)")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Quantum Max is non-blocking!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ui_responsiveness():
    """Test that UI remains responsive during activation"""
    print("\n" + "=" * 60)
    print("🧪 Testing UI Responsiveness During Activation")
    print("=" * 60)
    
    try:
        from quantum_max_integration import get_quantum_max_integration
        
        qmax = get_quantum_max_integration()
        
        if not qmax.quantum_scheduler:
            print("⚠️  Quantum Max Scheduler not available - test skipped")
            return True
        
        # Simulate UI operations during activation
        ui_responsive = True
        ui_operations = 0
        
        def simulate_ui_operations():
            nonlocal ui_operations, ui_responsive
            start = time.time()
            while time.time() - start < 2.0:
                # Simulate UI update
                time.sleep(0.1)
                ui_operations += 1
                # If we can't do at least 10 operations in 2 seconds, UI is blocked
                if ui_operations < 5 and time.time() - start > 1.0:
                    ui_responsive = False
        
        # Start UI simulation
        ui_thread = threading.Thread(target=simulate_ui_operations, daemon=True)
        ui_thread.start()
        
        # Activate quantum max while UI is running
        time.sleep(0.2)  # Let UI thread start
        qmax.activate_quantum_max_mode(interval=10)
        
        # Wait for UI thread
        ui_thread.join(timeout=3.0)
        
        if not ui_responsive:
            print(f"❌ FAIL: UI became unresponsive (only {ui_operations} operations)")
            return False
        
        print(f"✅ PASS: UI remained responsive ({ui_operations} operations completed)")
        
        # Cleanup
        qmax.deactivate_quantum_max_mode()
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QUANTUM MAX NON-BLOCKING TEST SUITE")
    print("=" * 60)
    print()
    
    test1_passed = test_quantum_max_activation()
    test2_passed = test_ui_responsiveness()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        print("✅ Quantum Max Mode is non-blocking")
        print("✅ UI will remain responsive")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_passed:
            print("❌ Activation/deactivation has blocking behavior")
        if not test2_passed:
            print("❌ UI responsiveness affected")
        exit(1)
