#!/usr/bin/env python3
"""
Comprehensive UI and Feature Test for Ultimate EAS System
Tests all buttons, features, and user interactions
"""

import time
import requests
import json
import subprocess
import os
import signal
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class UltimateEASUITester:
    def __init__(self):
        self.base_url = "http://localhost:9010"
        self.driver = None
        self.app_process = None
        
    def setup_browser(self):
        """Setup headless Chrome browser for testing"""
        print("🔧 Setting up browser for UI testing...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✅ Browser setup successful")
            return True
        except Exception as e:
            print(f"⚠️  Browser setup failed: {e}")
            print("   Continuing with API-only tests...")
            return False
    
    def start_app(self):
        """Start the PQS Framework app"""
        print("🚀 Starting PQS Framework app...")
        try:
            # Kill any existing instances
            subprocess.run(["pkill", "-f", "PQS Framework"], capture_output=True)
            time.sleep(2)
            
            # Start the app
            self.app_process = subprocess.Popen(
                ["./venv/bin/python", "launch_fixed_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for app to start
            time.sleep(8)
            
            # Test if server is responding
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ PQS Framework app started successfully")
                return True
            else:
                print(f"❌ App not responding: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start app: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🧪 Testing API Endpoints...")
        
        endpoints = [
            ("/", "Main Dashboard"),
            ("/api/status", "System Status API"),
            ("/api/quantum-status", "Quantum Status API"),
            ("/quantum", "Quantum Dashboard"),
            ("/history", "Battery History"),
            ("/static/themes.css", "CSS Themes")
        ]
        
        results = {}
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    print(f"   ✅ {name}: OK ({response.status_code})")
                    results[endpoint] = "PASS"
                else:
                    print(f"   ❌ {name}: Failed ({response.status_code})")
                    results[endpoint] = f"FAIL ({response.status_code})"
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                results[endpoint] = f"ERROR ({e})"
        
        return results
    
    def test_quantum_dashboard_ui(self):
        """Test Quantum Dashboard UI elements"""
        if not self.driver:
            print("⚠️  Skipping UI tests - no browser available")
            return {}
            
        print("\n🖥️  Testing Quantum Dashboard UI...")
        
        try:
            self.driver.get(f"{self.base_url}/quantum")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            ui_tests = {}
            
            # Test 1: Check if page title is correct
            try:
                title = self.driver.title
                if "Ultimate EAS" in title and "Quantum Dashboard" in title:
                    print("   ✅ Page title: Correct")
                    ui_tests["page_title"] = "PASS"
                else:
                    print(f"   ❌ Page title: Wrong - '{title}'")
                    ui_tests["page_title"] = f"FAIL - {title}"
            except Exception as e:
                print(f"   ❌ Page title: Error - {e}")
                ui_tests["page_title"] = f"ERROR - {e}"
            
            # Test 2: Check for system uptime element
            try:
                uptime_element = self.driver.find_element(By.ID, "system-uptime")
                uptime_text = uptime_element.text
                if uptime_text and "hours" in uptime_text:
                    print(f"   ✅ System uptime: {uptime_text}")
                    ui_tests["system_uptime"] = "PASS"
                else:
                    print(f"   ❌ System uptime: Invalid - '{uptime_text}'")
                    ui_tests["system_uptime"] = f"FAIL - {uptime_text}"
            except NoSuchElementException:
                print("   ❌ System uptime: Element not found")
                ui_tests["system_uptime"] = "FAIL - Element not found"
            except Exception as e:
                print(f"   ❌ System uptime: Error - {e}")
                ui_tests["system_uptime"] = f"ERROR - {e}"
            
            # Test 3: Check for quantum operations counter
            try:
                quantum_ops = self.driver.find_element(By.ID, "quantum-operations")
                ops_text = quantum_ops.text
                if ops_text and ops_text.isdigit() and int(ops_text) >= 0:
                    print(f"   ✅ Quantum operations: {ops_text}")
                    ui_tests["quantum_operations"] = "PASS"
                else:
                    print(f"   ❌ Quantum operations: Invalid - '{ops_text}'")
                    ui_tests["quantum_operations"] = f"FAIL - {ops_text}"
            except NoSuchElementException:
                print("   ❌ Quantum operations: Element not found")
                ui_tests["quantum_operations"] = "FAIL - Element not found"
            except Exception as e:
                print(f"   ❌ Quantum operations: Error - {e}")
                ui_tests["quantum_operations"] = f"ERROR - {e}"
            
            # Test 4: Check for optimization cycles
            try:
                opt_cycles = self.driver.find_element(By.ID, "optimization-cycles")
                cycles_text = opt_cycles.text
                if cycles_text and cycles_text.isdigit() and int(cycles_text) >= 0:
                    print(f"   ✅ Optimization cycles: {cycles_text}")
                    ui_tests["optimization_cycles"] = "PASS"
                else:
                    print(f"   ❌ Optimization cycles: Invalid - '{cycles_text}'")
                    ui_tests["optimization_cycles"] = f"FAIL - {cycles_text}"
            except NoSuchElementException:
                print("   ❌ Optimization cycles: Element not found")
                ui_tests["optimization_cycles"] = "FAIL - Element not found"
            except Exception as e:
                print(f"   ❌ Optimization cycles: Error - {e}")
                ui_tests["optimization_cycles"] = f"ERROR - {e}"
            
            # Test 5: Check for CSS styling
            try:
                body_bg = self.driver.execute_script(
                    "return window.getComputedStyle(document.body).backgroundColor"
                )
                if body_bg and body_bg != "rgba(0, 0, 0, 0)":
                    print(f"   ✅ CSS styling: Applied (bg: {body_bg})")
                    ui_tests["css_styling"] = "PASS"
                else:
                    print(f"   ❌ CSS styling: Not applied")
                    ui_tests["css_styling"] = "FAIL - No background"
            except Exception as e:
                print(f"   ❌ CSS styling: Error - {e}")
                ui_tests["css_styling"] = f"ERROR - {e}"
            
            # Test 6: Check for JavaScript errors
            try:
                js_errors = self.driver.get_log('browser')
                error_count = len([log for log in js_errors if log['level'] == 'SEVERE'])
                if error_count == 0:
                    print("   ✅ JavaScript: No errors")
                    ui_tests["javascript_errors"] = "PASS"
                else:
                    print(f"   ❌ JavaScript: {error_count} errors found")
                    ui_tests["javascript_errors"] = f"FAIL - {error_count} errors"
            except Exception as e:
                print(f"   ⚠️  JavaScript errors: Could not check - {e}")
                ui_tests["javascript_errors"] = f"UNKNOWN - {e}"
            
            return ui_tests
            
        except TimeoutException:
            print("   ❌ Dashboard UI: Page load timeout")
            return {"page_load": "FAIL - Timeout"}
        except Exception as e:
            print(f"   ❌ Dashboard UI: Error - {e}")
            return {"dashboard_ui": f"ERROR - {e}"}
    
    def test_main_dashboard_ui(self):
        """Test Main Dashboard UI elements"""
        if not self.driver:
            return {}
            
        print("\n🏠 Testing Main Dashboard UI...")
        
        try:
            self.driver.get(f"{self.base_url}/")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            ui_tests = {}
            
            # Test battery level display
            try:
                battery_elements = self.driver.find_elements(By.CLASS_NAME, "battery-level")
                if battery_elements:
                    battery_text = battery_elements[0].text
                    print(f"   ✅ Battery level: {battery_text}")
                    ui_tests["battery_level"] = "PASS"
                else:
                    print("   ❌ Battery level: Element not found")
                    ui_tests["battery_level"] = "FAIL - Not found"
            except Exception as e:
                print(f"   ❌ Battery level: Error - {e}")
                ui_tests["battery_level"] = f"ERROR - {e}"
            
            # Test navigation links
            nav_links = [
                ("Quantum Dashboard", "/quantum"),
                ("Battery History", "/history")
            ]
            
            for link_text, expected_href in nav_links:
                try:
                    links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, link_text)
                    if links:
                        href = links[0].get_attribute("href")
                        if expected_href in href:
                            print(f"   ✅ Navigation link '{link_text}': Working")
                            ui_tests[f"nav_{link_text.lower().replace(' ', '_')}"] = "PASS"
                        else:
                            print(f"   ❌ Navigation link '{link_text}': Wrong href - {href}")
                            ui_tests[f"nav_{link_text.lower().replace(' ', '_')}"] = f"FAIL - {href}"
                    else:
                        print(f"   ❌ Navigation link '{link_text}': Not found")
                        ui_tests[f"nav_{link_text.lower().replace(' ', '_')}"] = "FAIL - Not found"
                except Exception as e:
                    print(f"   ❌ Navigation link '{link_text}': Error - {e}")
                    ui_tests[f"nav_{link_text.lower().replace(' ', '_')}"] = f"ERROR - {e}"
            
            return ui_tests
            
        except Exception as e:
            print(f"   ❌ Main Dashboard UI: Error - {e}")
            return {"main_dashboard": f"ERROR - {e}"}
    
    def test_data_updates(self):
        """Test that data updates over time"""
        print("\n⏱️  Testing Data Updates...")
        
        try:
            # Get initial quantum status
            response1 = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            data1 = response1.json()
            
            print(f"   Initial quantum operations: {data1.get('quantum_operations', 0)}")
            print(f"   Initial uptime: {data1.get('system_uptime', 0):.3f} hours")
            
            # Wait 10 seconds
            print("   Waiting 10 seconds for data to update...")
            time.sleep(10)
            
            # Get updated quantum status
            response2 = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            data2 = response2.json()
            
            print(f"   Updated quantum operations: {data2.get('quantum_operations', 0)}")
            print(f"   Updated uptime: {data2.get('system_uptime', 0):.3f} hours")
            
            # Check if data increased
            tests = {}
            
            if data2.get('system_uptime', 0) > data1.get('system_uptime', 0):
                print("   ✅ System uptime: Increasing")
                tests["uptime_increase"] = "PASS"
            else:
                print("   ❌ System uptime: Not increasing")
                tests["uptime_increase"] = "FAIL"
            
            if data2.get('quantum_operations', 0) >= data1.get('quantum_operations', 0):
                print("   ✅ Quantum operations: Stable/Increasing")
                tests["quantum_ops_increase"] = "PASS"
            else:
                print("   ❌ Quantum operations: Decreasing")
                tests["quantum_ops_increase"] = "FAIL"
            
            # Check required fields exist
            required_fields = [
                'system_status', 'uptime_formatted', 'quantum_operations',
                'optimized_processes', 'gpu_name', 'average_speedup'
            ]
            
            for field in required_fields:
                if field == 'uptime_formatted':
                    if 'system_status' in data2 and 'uptime_formatted' in data2['system_status']:
                        print(f"   ✅ Field '{field}': Present")
                        tests[f"field_{field}"] = "PASS"
                    else:
                        print(f"   ❌ Field '{field}': Missing")
                        tests[f"field_{field}"] = "FAIL"
                else:
                    if field in data2:
                        print(f"   ✅ Field '{field}': Present")
                        tests[f"field_{field}"] = "PASS"
                    else:
                        print(f"   ❌ Field '{field}': Missing")
                        tests[f"field_{field}"] = "FAIL"
            
            return tests
            
        except Exception as e:
            print(f"   ❌ Data updates test: Error - {e}")
            return {"data_updates": f"ERROR - {e}"}
    
    def test_menu_bar_simulation(self):
        """Simulate menu bar interactions via API calls"""
        print("\n📱 Testing Menu Bar Features (API Simulation)...")
        
        tests = {}
        
        # Test 1: Simulate Toggle Ultimate EAS
        try:
            # Check initial state
            response = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            initial_data = response.json()
            initial_enabled = initial_data.get('ultimate_eas_enabled', False)
            
            print(f"   Initial Ultimate EAS state: {initial_enabled}")
            
            # The toggle would normally be done via menu bar, but we can check if the system
            # responds to having Ultimate EAS enabled by checking the quantum metrics
            if initial_data.get('quantum_operations', 0) > 0:
                print("   ✅ Ultimate EAS: System shows quantum activity")
                tests["ultimate_eas_activity"] = "PASS"
            else:
                print("   ⚠️  Ultimate EAS: No quantum activity yet")
                tests["ultimate_eas_activity"] = "PENDING"
                
        except Exception as e:
            print(f"   ❌ Ultimate EAS toggle test: Error - {e}")
            tests["ultimate_eas_toggle"] = f"ERROR - {e}"
        
        # Test 2: Check if system status API works (simulates "View Ultimate EAS Status")
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                if 'enabled' in status_data and 'battery_level' in status_data:
                    print("   ✅ System status: API working")
                    tests["system_status_api"] = "PASS"
                else:
                    print("   ❌ System status: Missing fields")
                    tests["system_status_api"] = "FAIL - Missing fields"
            else:
                print(f"   ❌ System status: API error {response.status_code}")
                tests["system_status_api"] = f"FAIL - {response.status_code}"
        except Exception as e:
            print(f"   ❌ System status test: Error - {e}")
            tests["system_status_api"] = f"ERROR - {e}"
        
        return tests
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🌟" + "=" * 60 + "🌟")
        print("🧪 COMPREHENSIVE ULTIMATE EAS UI & FEATURE TEST")
        print("🌟" + "=" * 60 + "🌟")
        
        all_results = {}
        
        # Start the app
        if not self.start_app():
            print("❌ Cannot start app - aborting tests")
            return False
        
        # Setup browser (optional)
        browser_available = self.setup_browser()
        
        try:
            # Test 1: API Endpoints
            all_results["api_endpoints"] = self.test_api_endpoints()
            
            # Test 2: Data Updates
            all_results["data_updates"] = self.test_data_updates()
            
            # Test 3: Menu Bar Simulation
            all_results["menu_bar"] = self.test_menu_bar_simulation()
            
            # Test 4: UI Tests (if browser available)
            if browser_available:
                all_results["quantum_dashboard_ui"] = self.test_quantum_dashboard_ui()
                all_results["main_dashboard_ui"] = self.test_main_dashboard_ui()
            
            # Generate report
            self.generate_test_report(all_results)
            
            return True
            
        finally:
            self.cleanup()
    
    def generate_test_report(self, results):
        """Generate comprehensive test report"""
        print("\n" + "🏆" + "=" * 60 + "🏆")
        print("📊 COMPREHENSIVE TEST REPORT")
        print("🏆" + "=" * 60 + "🏆")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        
        for category, tests in results.items():
            print(f"\n📋 {category.upper().replace('_', ' ')}:")
            
            for test_name, result in tests.items():
                total_tests += 1
                
                if result == "PASS":
                    print(f"   ✅ {test_name}: PASS")
                    passed_tests += 1
                elif result.startswith("FAIL"):
                    print(f"   ❌ {test_name}: {result}")
                    failed_tests += 1
                elif result.startswith("ERROR"):
                    print(f"   💥 {test_name}: {result}")
                    error_tests += 1
                else:
                    print(f"   ⚠️  {test_name}: {result}")
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   💥 Errors: {error_tests}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print(f"\n🎉 EXCELLENT! Ultimate EAS system is working well!")
        elif success_rate >= 60:
            print(f"\n👍 GOOD! Most features are working, minor issues to fix.")
        else:
            print(f"\n⚠️  NEEDS ATTENTION! Several issues need to be resolved.")
        
        # Save detailed report
        report_file = f"ultimate_eas_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'errors': error_tests,
                    'success_rate': success_rate
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.driver:
            try:
                self.driver.quit()
                print("   ✅ Browser closed")
            except:
                pass
        
        if self.app_process:
            try:
                self.app_process.terminate()
                self.app_process.wait(timeout=5)
                print("   ✅ App process terminated")
            except:
                try:
                    self.app_process.kill()
                except:
                    pass
        
        # Kill any remaining processes
        try:
            subprocess.run(["pkill", "-f", "PQS Framework"], capture_output=True)
        except:
            pass

def main():
    """Main test function"""
    tester = UltimateEASUITester()
    
    try:
        success = tester.run_comprehensive_test()
        return success
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.cleanup()
        return False
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        tester.cleanup()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)