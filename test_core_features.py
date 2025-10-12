#!/usr/bin/env python3
"""
Core Features Test for Ultimate EAS System
Tests essential functionality without requiring Selenium
"""

import time
import requests
import json
import subprocess
import os
import signal
import threading

class CoreFeaturesTester:
    def __init__(self):
        self.base_url = "http://localhost:9010"
        self.app_process = None
        
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
            print("   Waiting for app to initialize...")
            time.sleep(8)
            
            # Test if server is responding
            for attempt in range(5):
                try:
                    response = requests.get(f"{self.base_url}/", timeout=5)
                    if response.status_code == 200:
                        print("✅ PQS Framework app started successfully")
                        return True
                except:
                    if attempt < 4:
                        print(f"   Attempt {attempt + 1}/5 failed, retrying...")
                        time.sleep(3)
                    
            print("❌ App failed to start properly")
            return False
                
        except Exception as e:
            print(f"❌ Failed to start app: {e}")
            return False
    
    def test_essential_endpoints(self):
        """Test essential API endpoints"""
        print("\n🔗 Testing Essential Endpoints...")
        
        endpoints = [
            ("/", "Main Dashboard", "text/html"),
            ("/api/status", "System Status API", "application/json"),
            ("/api/quantum-status", "Quantum Status API", "application/json"),
            ("/quantum", "Quantum Dashboard", "text/html"),
            ("/static/themes.css", "CSS Themes", "text/css")
        ]
        
        results = {}
        
        for endpoint, name, expected_type in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                content_type = response.headers.get('content-type', '').lower()
                
                if response.status_code == 200:
                    if expected_type in content_type or endpoint == "/static/themes.css":
                        print(f"   ✅ {name}: OK ({response.status_code})")
                        results[endpoint] = "PASS"
                    else:
                        print(f"   ⚠️  {name}: Wrong content type - {content_type}")
                        results[endpoint] = f"WARN - Wrong content type"
                else:
                    print(f"   ❌ {name}: Failed ({response.status_code})")
                    results[endpoint] = f"FAIL ({response.status_code})"
                    
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                results[endpoint] = f"ERROR ({e})"
        
        return results
    
    def test_quantum_status_data(self):
        """Test quantum status API data structure"""
        print("\n⚛️  Testing Quantum Status Data...")
        
        try:
            response = requests.get(f"{self.base_url}/api/quantum-status", timeout=10)
            
            if response.status_code != 200:
                return {"quantum_api": f"FAIL - Status {response.status_code}"}
            
            data = response.json()
            tests = {}
            
            # Test required top-level fields
            required_fields = [
                'available', 'system_uptime', 'quantum_operations', 
                'optimized_processes', 'gpu_name', 'average_speedup'
            ]
            
            for field in required_fields:
                if field in data:
                    print(f"   ✅ Field '{field}': Present ({data[field]})")
                    tests[f"field_{field}"] = "PASS"
                else:
                    print(f"   ❌ Field '{field}': Missing")
                    tests[f"field_{field}"] = "FAIL"
            
            # Test system_status object (critical for dashboard)
            if 'system_status' in data:
                system_status = data['system_status']
                
                # Test uptime_formatted (this was the main issue)
                if 'uptime_formatted' in system_status:
                    uptime = system_status['uptime_formatted']
                    if 'hours' in uptime:
                        print(f"   ✅ system_status.uptime_formatted: '{uptime}'")
                        tests["uptime_formatted"] = "PASS"
                    else:
                        print(f"   ❌ system_status.uptime_formatted: Invalid format - '{uptime}'")
                        tests["uptime_formatted"] = f"FAIL - {uptime}"
                else:
                    print("   ❌ system_status.uptime_formatted: Missing")
                    tests["uptime_formatted"] = "FAIL - Missing"
                
                # Test other system_status fields
                status_fields = ['optimization_cycles', 'quantum_operations', 'system_id']
                for field in status_fields:
                    if field in system_status:
                        print(f"   ✅ system_status.{field}: {system_status[field]}")
                        tests[f"status_{field}"] = "PASS"
                    else:
                        print(f"   ❌ system_status.{field}: Missing")
                        tests[f"status_{field}"] = "FAIL"
            else:
                print("   ❌ system_status object: Missing")
                tests["system_status_object"] = "FAIL"
            
            return tests
            
        except Exception as e:
            print(f"   ❌ Quantum status test: Error - {e}")
            return {"quantum_status": f"ERROR - {e}"}
    
    def test_data_progression(self):
        """Test that metrics increase over time"""
        print("\n📈 Testing Data Progression...")
        
        try:
            # Get initial reading
            response1 = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            data1 = response1.json()
            
            initial_uptime = data1.get('system_uptime', 0)
            initial_ops = data1.get('quantum_operations', 0)
            
            print(f"   Initial uptime: {initial_uptime:.4f} hours")
            print(f"   Initial quantum ops: {initial_ops}")
            
            # Wait for progression
            print("   Waiting 15 seconds for metrics to update...")
            time.sleep(15)
            
            # Get updated reading
            response2 = requests.get(f"{self.base_url}/api/quantum-status", timeout=5)
            data2 = response2.json()
            
            updated_uptime = data2.get('system_uptime', 0)
            updated_ops = data2.get('quantum_operations', 0)
            
            print(f"   Updated uptime: {updated_uptime:.4f} hours")
            print(f"   Updated quantum ops: {updated_ops}")
            
            tests = {}
            
            # Test uptime progression
            if updated_uptime > initial_uptime:
                print("   ✅ System uptime: Progressing")
                tests["uptime_progression"] = "PASS"
            else:
                print("   ❌ System uptime: Not progressing")
                tests["uptime_progression"] = "FAIL"
            
            # Test quantum operations (should be stable or increasing)
            if updated_ops >= initial_ops:
                print("   ✅ Quantum operations: Stable/Increasing")
                tests["quantum_ops_progression"] = "PASS"
            else:
                print("   ❌ Quantum operations: Decreasing")
                tests["quantum_ops_progression"] = "FAIL"
            
            # Test that optimized processes are being tracked
            optimized = data2.get('optimized_processes', 0)
            if optimized >= 0:
                print(f"   ✅ Optimized processes: {optimized}")
                tests["optimized_processes"] = "PASS"
            else:
                print(f"   ❌ Optimized processes: Invalid - {optimized}")
                tests["optimized_processes"] = "FAIL"
            
            return tests
            
        except Exception as e:
            print(f"   ❌ Data progression test: Error - {e}")
            return {"data_progression": f"ERROR - {e}"}
    
    def test_dashboard_content(self):
        """Test dashboard HTML content"""
        print("\n🖥️  Testing Dashboard Content...")
        
        tests = {}
        
        # Test main dashboard
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                content = response.text
                
                # Check for essential elements
                if "Battery Optimizer" in content or "PQS Framework" in content:
                    print("   ✅ Main dashboard: Title present")
                    tests["main_title"] = "PASS"
                else:
                    print("   ❌ Main dashboard: Title missing")
                    tests["main_title"] = "FAIL"
                
                if "themes.css" in content:
                    print("   ✅ Main dashboard: CSS linked")
                    tests["main_css"] = "PASS"
                else:
                    print("   ❌ Main dashboard: CSS not linked")
                    tests["main_css"] = "FAIL"
            else:
                tests["main_dashboard"] = f"FAIL - {response.status_code}"
                
        except Exception as e:
            print(f"   ❌ Main dashboard test: Error - {e}")
            tests["main_dashboard"] = f"ERROR - {e}"
        
        # Test quantum dashboard
        try:
            response = requests.get(f"{self.base_url}/quantum", timeout=10)
            if response.status_code == 200:
                content = response.text
                
                # Check for quantum dashboard elements
                if "Ultimate EAS" in content and "Quantum Dashboard" in content:
                    print("   ✅ Quantum dashboard: Title present")
                    tests["quantum_title"] = "PASS"
                else:
                    print("   ❌ Quantum dashboard: Title missing")
                    tests["quantum_title"] = "FAIL"
                
                # Check for essential IDs that JavaScript will use
                essential_ids = ["system-uptime", "quantum-operations", "optimization-cycles"]
                for element_id in essential_ids:
                    if f'id="{element_id}"' in content:
                        print(f"   ✅ Quantum dashboard: Element '{element_id}' present")
                        tests[f"element_{element_id}"] = "PASS"
                    else:
                        print(f"   ❌ Quantum dashboard: Element '{element_id}' missing")
                        tests[f"element_{element_id}"] = "FAIL"
                
                # Check for JavaScript API calls
                if "/api/quantum-status" in content:
                    print("   ✅ Quantum dashboard: API calls configured")
                    tests["api_calls"] = "PASS"
                else:
                    print("   ❌ Quantum dashboard: API calls not configured")
                    tests["api_calls"] = "FAIL"
                    
            else:
                tests["quantum_dashboard"] = f"FAIL - {response.status_code}"
                
        except Exception as e:
            print(f"   ❌ Quantum dashboard test: Error - {e}")
            tests["quantum_dashboard"] = f"ERROR - {e}"
        
        return tests
    
    def test_ultimate_eas_availability(self):
        """Test Ultimate EAS system availability"""
        print("\n🚀 Testing Ultimate EAS Availability...")
        
        tests = {}
        
        try:
            # Check quantum status for Ultimate EAS indicators
            response = requests.get(f"{self.base_url}/api/quantum-status", timeout=10)
            data = response.json()
            
            # Check if Ultimate EAS features are present
            eas_indicators = [
                ('quantum_operations', 'Quantum operations counter'),
                ('gpu_name', 'GPU acceleration info'),
                ('average_speedup', 'Performance metrics'),
                ('quantum_volume', 'Quantum volume metrics')
            ]
            
            for field, description in eas_indicators:
                if field in data and data[field] is not None:
                    print(f"   ✅ {description}: Available ({data[field]})")
                    tests[f"eas_{field}"] = "PASS"
                else:
                    print(f"   ❌ {description}: Not available")
                    tests[f"eas_{field}"] = "FAIL"
            
            # Check system status for Ultimate EAS enabled flag
            if 'system_status' in data:
                eas_enabled = data['system_status'].get('ultimate_eas_enabled', False)
                if eas_enabled:
                    print("   ✅ Ultimate EAS: Enabled in system status")
                    tests["eas_enabled_flag"] = "PASS"
                else:
                    print("   ⚠️  Ultimate EAS: Not enabled (using fallback/mock system)")
                    tests["eas_enabled_flag"] = "WARN - Using fallback"
            
            # Check if quantum operations are actually running
            quantum_ops = data.get('quantum_operations', 0)
            if quantum_ops > 0:
                print(f"   ✅ Quantum operations: Active ({quantum_ops} operations)")
                tests["quantum_active"] = "PASS"
            else:
                print("   ⚠️  Quantum operations: Not yet active")
                tests["quantum_active"] = "PENDING"
            
            return tests
            
        except Exception as e:
            print(f"   ❌ Ultimate EAS availability test: Error - {e}")
            return {"eas_availability": f"ERROR - {e}"}
    
    def run_core_tests(self):
        """Run all core functionality tests"""
        print("🌟" + "=" * 60 + "🌟")
        print("🧪 CORE FEATURES TEST - ULTIMATE EAS SYSTEM")
        print("🌟" + "=" * 60 + "🌟")
        
        all_results = {}
        
        # Start the app
        if not self.start_app():
            print("❌ Cannot start app - aborting tests")
            return False
        
        try:
            # Run all tests
            all_results["endpoints"] = self.test_essential_endpoints()
            all_results["quantum_data"] = self.test_quantum_status_data()
            all_results["data_progression"] = self.test_data_progression()
            all_results["dashboard_content"] = self.test_dashboard_content()
            all_results["eas_availability"] = self.test_ultimate_eas_availability()
            
            # Generate report
            self.generate_report(all_results)
            
            return True
            
        finally:
            self.cleanup()
    
    def generate_report(self, results):
        """Generate test report"""
        print("\n" + "🏆" + "=" * 50 + "🏆")
        print("📊 CORE FEATURES TEST REPORT")
        print("🏆" + "=" * 50 + "🏆")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        
        for category, tests in results.items():
            print(f"\n📋 {category.upper().replace('_', ' ')}:")
            
            for test_name, result in tests.items():
                total_tests += 1
                
                if result == "PASS":
                    print(f"   ✅ {test_name}")
                    passed_tests += 1
                elif result.startswith("WARN"):
                    print(f"   ⚠️  {test_name}: {result}")
                    warnings += 1
                elif result == "PENDING":
                    print(f"   ⏳ {test_name}: {result}")
                    warnings += 1
                else:
                    print(f"   ❌ {test_name}: {result}")
                    failed_tests += 1
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ❌ Failed: {failed_tests}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        
        # Determine overall status
        if failed_tests == 0 and warnings <= 2:
            print(f"\n🎉 EXCELLENT! All core features are working properly!")
            status = "EXCELLENT"
        elif failed_tests <= 2 and success_rate >= 75:
            print(f"\n👍 GOOD! Core functionality is working with minor issues.")
            status = "GOOD"
        elif success_rate >= 50:
            print(f"\n⚠️  NEEDS ATTENTION! Some core features need fixing.")
            status = "NEEDS_ATTENTION"
        else:
            print(f"\n❌ CRITICAL! Major issues found in core functionality.")
            status = "CRITICAL"
        
        # Save report
        report = {
            'timestamp': time.time(),
            'status': status,
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warnings,
                'failed': failed_tests,
                'success_rate': success_rate
            },
            'results': results
        }
        
        report_file = f"core_features_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        
        return status == "EXCELLENT" or status == "GOOD"
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.app_process:
            try:
                self.app_process.terminate()
                self.app_process.wait(timeout=5)
                print("   ✅ App process terminated")
            except:
                try:
                    self.app_process.kill()
                    print("   ✅ App process killed")
                except:
                    pass
        
        # Kill any remaining processes
        try:
            subprocess.run(["pkill", "-f", "PQS Framework"], capture_output=True)
        except:
            pass

def main():
    """Main test function"""
    tester = CoreFeaturesTester()
    
    try:
        success = tester.run_core_tests()
        return success
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.cleanup()
        return False
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        tester.cleanup()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)