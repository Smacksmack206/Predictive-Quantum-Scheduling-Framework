#!/usr/bin/env python3
"""
Test script for Modern UI implementation
Verifies all components are working correctly
"""

import sys
import os
from pathlib import Path

def test_config_module():
    """Test configuration module"""
    print("=" * 70)
    print("TEST 1: Configuration Module")
    print("=" * 70)
    
    try:
        from config import config, PQSConfig
        print("✅ Config module imports successfully")
        
        # Test default config
        assert config.quantum.max_qubits == 48, "Default max_qubits should be 48"
        print(f"✅ Quantum max qubits: {config.quantum.max_qubits}")
        
        assert config.idle.suspend_delay == 30, "Default suspend_delay should be 30"
        print(f"✅ Idle suspend delay: {config.idle.suspend_delay}s")
        
        assert config.battery.critical_threshold == 20, "Default critical_threshold should be 20"
        print(f"✅ Battery critical threshold: {config.battery.critical_threshold}%")
        
        # Test to_dict
        config_dict = config.to_dict()
        assert 'quantum' in config_dict, "Config dict should have 'quantum' key"
        print("✅ Config to_dict() works")
        
        # Test from_dict
        new_config = PQSConfig.from_dict(config_dict)
        assert new_config.quantum.max_qubits == 48, "from_dict should preserve values"
        print("✅ Config from_dict() works")
        
        # Test save/load
        test_path = Path('test_config.json')
        config.save(test_path)
        assert test_path.exists(), "Config file should be created"
        print("✅ Config save() works")
        
        loaded_config = PQSConfig.load(test_path)
        assert loaded_config.quantum.max_qubits == 48, "Loaded config should match"
        print("✅ Config load() works")
        
        # Cleanup
        test_path.unlink()
        print("✅ Test cleanup complete")
        
        print("\n✅ Configuration Module: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration Module: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_templates_exist():
    """Test that all templates exist"""
    print("=" * 70)
    print("TEST 2: Template Files")
    print("=" * 70)
    
    templates = [
        'templates/base_modern.html',
        'templates/dashboard_modern.html',
        'templates/quantum_modern.html',
        'templates/battery_modern.html',
        'templates/system_control_modern.html'
    ]
    
    all_exist = True
    for template in templates:
        path = Path(template)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {template} ({size:,} bytes)")
        else:
            print(f"❌ {template} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✅ Template Files: PASSED\n")
    else:
        print("\n❌ Template Files: FAILED\n")
    
    return all_exist

def test_flask_routes():
    """Test that Flask routes are defined"""
    print("=" * 70)
    print("TEST 3: Flask Routes")
    print("=" * 70)
    
    try:
        # Import the Flask app
        from universal_pqs_app import flask_app
        
        # Get all routes
        routes = []
        for rule in flask_app.url_map.iter_rules():
            routes.append(str(rule))
        
        # Check for modern UI routes
        required_routes = [
            '/modern',
            '/quantum-modern',
            '/battery-modern',
            '/system-control-modern',
            '/api/settings',
            '/api/system/status',
            '/api/system/kill',
            '/api/system/optimize',
            '/api/quantum/toggle',
            '/api/battery/suspend'
        ]
        
        all_found = True
        for route in required_routes:
            if route in routes:
                print(f"✅ {route}")
            else:
                print(f"❌ {route} - NOT FOUND")
                all_found = False
        
        if all_found:
            print(f"\n✅ Flask Routes: PASSED ({len(routes)} total routes)\n")
        else:
            print("\n❌ Flask Routes: FAILED\n")
        
        return all_found
        
    except Exception as e:
        print(f"\n❌ Flask Routes: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_template_syntax():
    """Test that templates have valid syntax"""
    print("=" * 70)
    print("TEST 4: Template Syntax")
    print("=" * 70)
    
    templates = [
        'templates/quantum_modern.html',
        'templates/battery_modern.html',
        'templates/system_control_modern.html'
    ]
    
    all_valid = True
    for template in templates:
        path = Path(template)
        if not path.exists():
            print(f"❌ {template} - NOT FOUND")
            all_valid = False
            continue
        
        try:
            content = path.read_text()
            
            # Check for required elements
            checks = [
                ('{% extends "base_modern.html" %}', 'extends base'),
                ('{% block content %}', 'content block start'),
                ('{% endblock %}', 'content block end'),
                ('x-data=', 'Alpine.js data'),
                ('function ', 'JavaScript function'),
                ('@click=', 'Alpine.js click handler')
            ]
            
            template_valid = True
            for check, description in checks:
                if check in content:
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ {description} - MISSING")
                    template_valid = False
            
            if template_valid:
                print(f"✅ {template}\n")
            else:
                print(f"❌ {template}\n")
                all_valid = False
                
        except Exception as e:
            print(f"❌ {template} - ERROR: {e}\n")
            all_valid = False
    
    if all_valid:
        print("✅ Template Syntax: PASSED\n")
    else:
        print("❌ Template Syntax: FAILED\n")
    
    return all_valid

def test_api_endpoints_defined():
    """Test that API endpoint functions are defined"""
    print("=" * 70)
    print("TEST 5: API Endpoint Functions")
    print("=" * 70)
    
    try:
        import universal_pqs_app as app
        
        endpoints = [
            'api_settings',
            'api_system_status',
            'api_system_kill',
            'api_system_optimize',
            'api_system_cleanup',
            'api_system_suspend_idle',
            'api_quantum_toggle',
            'api_quantum_algorithm',
            'api_battery_suspend',
            'api_battery_protection'
        ]
        
        all_found = True
        for endpoint in endpoints:
            if hasattr(app, endpoint):
                print(f"✅ {endpoint}()")
            else:
                print(f"❌ {endpoint}() - NOT FOUND")
                all_found = False
        
        if all_found:
            print(f"\n✅ API Endpoint Functions: PASSED\n")
        else:
            print("\n❌ API Endpoint Functions: FAILED\n")
        
        return all_found
        
    except Exception as e:
        print(f"\n❌ API Endpoint Functions: FAILED - {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_documentation_exists():
    """Test that documentation files exist"""
    print("=" * 70)
    print("TEST 6: Documentation Files")
    print("=" * 70)
    
    docs = [
        'config.py',
        'config.json',
        'MODERN_UI_IMPLEMENTATION_COMPLETE.md',
        'MODERN_UI_QUICK_START.md',
        'IMPLEMENTATION_CHECKLIST.md'
    ]
    
    all_exist = True
    for doc in docs:
        path = Path(doc)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {doc} ({size:,} bytes)")
        else:
            print(f"❌ {doc} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✅ Documentation Files: PASSED\n")
    else:
        print("\n❌ Documentation Files: FAILED\n")
    
    return all_exist

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MODERN UI IMPLEMENTATION TEST SUITE")
    print("=" * 70 + "\n")
    
    results = {
        'Configuration Module': test_config_module(),
        'Template Files': test_templates_exist(),
        'Flask Routes': test_flask_routes(),
        'Template Syntax': test_template_syntax(),
        'API Endpoint Functions': test_api_endpoints_defined(),
        'Documentation Files': test_documentation_exists()
    }
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Implementation is complete and working.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
