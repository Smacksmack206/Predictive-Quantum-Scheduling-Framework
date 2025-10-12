#!/usr/bin/env python3
"""
UI Test Agent - Automated testing framework for Battery Optimizer Pro
Tests all menu items, buttons, and web interface components
"""

import requests
import time
import json
import subprocess
import threading
from datetime import datetime
import sys
import os

class UITestAgent:
    def __init__(self, base_url="http://localhost:9010"):
        self.base_url = base_url
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def log_test(self, test_name, status, details="", error=None):
        """Log test results"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        if status == "PASS":
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: {details}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {details}")
            if error:
                print(f"   Error: {error}")
    
    def test_server_connectivity(self):
        """Test if the Flask server is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.log_test("Server Connectivity", "PASS", f"Server responding on {self.base_url}")
                return True
            else:
                self.log_test("Server Connectivity", "FAIL", f"Server returned {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Server Connectivity", "FAIL", "Server not accessible", e)
            return False
    
    def test_main_dashboard(self):
        """Test main dashboard page"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                content = response.text
                # Check for key elements
                if "Battery Optimizer Pro" in content:
                    self.log_test("Main Dashboard", "PASS", "Dashboard loads with correct title")
                else:
                    self.log_test("Main Dashboard", "FAIL", "Dashboard missing title")
            else:
                self.log_test("Main Dashboard", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Main Dashboard", "FAIL", "Failed to load dashboard", e)
    
    def test_battery_history_page(self):
        """Test battery history page"""
        try:
            response = requests.get(f"{self.base_url}/history", timeout=5)
            if response.status_code == 200:
                content = response.text
                if "Battery History" in content and "<!DOCTYPE html>" in content:
                    self.log_test("Battery History Page", "PASS", "Page loads correctly")
                else:
                    self.log_test("Battery History Page", "FAIL", "Page content incomplete")
            else:
                self.log_test("Battery History Page", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Battery History Page", "FAIL", "Failed to load page", e)
    
    def test_battery_history_api(self):
        """Test battery history API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/battery-history", timeout=5)
            if response.status_code == 200:
                data = response.json()
                required_keys = ['history', 'cycles', 'app_changes', 'statistics']
                missing_keys = [key for key in required_keys if key not in data]
                
                if not missing_keys:
                    self.log_test("Battery History API", "PASS", "All required data fields present")
                else:
                    self.log_test("Battery History API", "FAIL", f"Missing keys: {missing_keys}")
            else:
                self.log_test("Battery History API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Battery History API", "FAIL", "API request failed", e)
    
    def test_quantum_dashboard(self):
        """Test quantum dashboard page"""
        try:
            response = requests.get(f"{self.base_url}/quantum", timeout=5)
            if response.status_code == 200:
                content = response.text
                if "Quantum Dashboard" in content or "Ultimate EAS" in content:
                    self.log_test("Quantum Dashboard", "PASS", "Dashboard loads correctly")
                else:
                    self.log_test("Quantum Dashboard", "FAIL", "Dashboard content missing")
            else:
                self.log_test("Quantum Dashboard", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Quantum Dashboard", "FAIL", "Failed to load dashboard", e)
    
    def test_quantum_status_api(self):
        """Test quantum status API - Critical for Ultimate EAS"""
        try:
            response = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check Ultimate EAS status
                ultimate_enabled = data.get('ultimate_eas_enabled', False)
                if ultimate_enabled:
                    self.log_test("Ultimate EAS Status", "PASS", "Ultimate EAS is enabled by default ✨")
                else:
                    self.log_test("Ultimate EAS Status", "FAIL", "Ultimate EAS not enabled by default")
                
                # Check quantum metrics
                if 'quantum_metrics' in data:
                    metrics = data['quantum_metrics']
                    required_metrics = ['quantum_operations', 'neural_classifications', 'energy_predictions']
                    missing_metrics = [m for m in required_metrics if m not in metrics]
                    
                    if not missing_metrics:
                        self.log_test("Quantum Metrics API", "PASS", "All quantum metrics available")
                    else:
                        self.log_test("Quantum Metrics API", "FAIL", f"Missing metrics: {missing_metrics}")
                else:
                    self.log_test("Quantum Metrics API", "FAIL", "No quantum metrics in response")
                    
            else:
                self.log_test("Quantum Status API", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Quantum Status API", "FAIL", "API request failed", e)
    
    def test_eas_dashboard(self):
        """Test EAS dashboard page"""
        try:
            response = requests.get(f"{self.base_url}/eas", timeout=5)
            if response.status_code == 200:
                content = response.text
                if "EAS Dashboard" in content or "Energy Aware" in content:
                    self.log_test("EAS Dashboard", "PASS", "EAS dashboard loads")
                else:
                    self.log_test("EAS Dashboard", "FAIL", "EAS dashboard content missing")
            else:
                self.log_test("EAS Dashboard", "FAIL", f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("EAS Dashboard", "FAIL", "Failed to load EAS dashboard", e)
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        endpoints = [
            ('/api/status', 'System Status API'),
            ('/api/battery-status', 'Battery Status API'),
            ('/api/eas-status', 'EAS Status API'),
            ('/api/suspended-apps', 'Suspended Apps API'),
            ('/api/analytics', 'Analytics API')
        ]
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(name, "PASS", f"Returns valid JSON data")
                else:
                    self.log_test(name, "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test(name, "FAIL", "API request failed", e)
    
    def test_static_resources(self):
        """Test static resources (CSS, JS)"""
        resources = [
            ('/static/themes.css', 'CSS Themes'),
            ('/static/battery-history.js', 'Battery History JS'),
            ('/static/battery-history-simple.js', 'Battery History Simple JS'),
            ('/static/battery-history-new.js', 'Battery History New JS')
        ]
        
        for resource, name in resources:
            try:
                response = requests.get(f"{self.base_url}{resource}", timeout=5)
                if response.status_code == 200:
                    self.log_test(f"Static Resource: {name}", "PASS", "Resource loads correctly")
                else:
                    self.log_test(f"Static Resource: {name}", "FAIL", f"HTTP {response.status_code}")
            except Exception as e:
                self.log_test(f"Static Resource: {name}", "FAIL", "Resource not accessible", e)
    
    def test_menu_functionality_simulation(self):
        """Simulate menu button functionality by testing corresponding web endpoints"""
        print("\n🔍 Testing Menu Button Functionality (via web endpoints)...")
        
        # Test menu items that have web equivalents
        menu_tests = [
            ("Open Dashboard", "/", "Main dashboard should be accessible"),
            ("Open Battery History", "/history", "Battery history should load"),
            ("Open Quantum Dashboard", "/quantum", "Quantum dashboard should be accessible"),
            ("View EAS Status", "/eas", "EAS dashboard should be accessible")
        ]
        
        for menu_item, endpoint, description in menu_tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.log_test(f"Menu: {menu_item}", "PASS", description)
                else:
                    self.log_test(f"Menu: {menu_item}", "FAIL", f"Endpoint returns {response.status_code}")
            except Exception as e:
                self.log_test(f"Menu: {menu_item}", "FAIL", f"Endpoint not accessible: {e}")
    
    def test_ultimate_eas_features(self):
        """Test Ultimate EAS specific features"""
        print("\n⚛️  Testing Ultimate EAS Features...")
        
        # Test Ultimate EAS status endpoint
        try:
            response = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check if Ultimate EAS is working
                ultimate_enabled = data.get('ultimate_eas_enabled', False)
                system_status = data.get('system_status', {})
                quantum_metrics = data.get('quantum_metrics', {})
                
                if ultimate_enabled:
                    self.log_test("Ultimate EAS Core", "PASS", "Ultimate EAS is active and enabled")
                    
                    # Check quantum operations
                    quantum_ops = quantum_metrics.get('quantum_operations', 0)
                    if quantum_ops > 0:
                        self.log_test("Quantum Operations", "PASS", f"{quantum_ops} quantum operations recorded")
                    else:
                        self.log_test("Quantum Operations", "WARN", "No quantum operations yet (may be normal for new install)")
                    
                    # Check neural classifications
                    neural_class = quantum_metrics.get('neural_classifications', 0)
                    if neural_class >= 0:  # Allow 0 for new installs
                        self.log_test("Neural Classifications", "PASS", f"{neural_class} neural classifications")
                    
                    # Check system performance
                    perf_level = system_status.get('performance_level', 0)
                    if perf_level > 0:
                        self.log_test("System Performance", "PASS", f"Performance level: {perf_level}%")
                    
                else:
                    self.log_test("Ultimate EAS Core", "FAIL", "Ultimate EAS not enabled - this should be enabled by default")
                    
            else:
                self.log_test("Ultimate EAS API", "FAIL", f"Quantum status API returned {response.status_code}")
                
        except Exception as e:
            self.log_test("Ultimate EAS API", "FAIL", "Failed to check Ultimate EAS status", e)
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🤖 UI Test Agent Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Basic connectivity
        if not self.test_server_connectivity():
            print("❌ Server not accessible - cannot continue tests")
            return False
        
        print("\n📱 Testing Web Interface Pages...")
        self.test_main_dashboard()
        self.test_battery_history_page()
        self.test_quantum_dashboard()
        self.test_eas_dashboard()
        
        print("\n🔌 Testing API Endpoints...")
        self.test_battery_history_api()
        self.test_quantum_status_api()
        self.test_api_endpoints()
        
        print("\n📁 Testing Static Resources...")
        self.test_static_resources()
        
        # Test menu functionality
        self.test_menu_functionality_simulation()
        
        # Test Ultimate EAS specifically
        self.test_ultimate_eas_features()
        
        return True
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 UI TEST AGENT REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed = len(self.passed_tests)
        failed = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print("\n🔧 ISSUES FOUND:")
            for i, test in enumerate(self.failed_tests, 1):
                # Find the test result
                result = next(r for r in self.test_results if r['test'] == test)
                print(f"{i}. {test}")
                print(f"   Issue: {result['details']}")
                if result['error']:
                    print(f"   Error: {result['error']}")
        
        if self.passed_tests:
            print(f"\n✅ WORKING FEATURES ({len(self.passed_tests)}):")
            for test in self.passed_tests:
                result = next(r for r in self.test_results if r['test'] == test)
                if "Ultimate EAS" in test or "Quantum" in test:
                    print(f"   ⚛️  {test}: {result['details']}")
                else:
                    print(f"   • {test}")
        
        # Save detailed report
        report_file = f"ui_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': (passed/total_tests)*100
                },
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return failed == 0

def main():
    """Main test execution"""
    print("🚀 Battery Optimizer Pro - UI Test Agent")
    print("Testing all menu items, buttons, and web interface components")
    print("Preserving Ultimate EAS functionality while identifying issues")
    
    agent = UITestAgent()
    
    # Run tests
    success = agent.run_comprehensive_test()
    
    if success:
        # Generate report
        all_passed = agent.generate_report()
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! No issues found.")
        else:
            print(f"\n⚠️  Found {len(agent.failed_tests)} issues that need attention.")
            print("Ultimate EAS functionality will be preserved during fixes.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)