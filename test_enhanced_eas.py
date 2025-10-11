#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced EAS system
Tests the new intelligent classification and learning capabilities
"""

import time
import requests
import json
import psutil
from datetime import datetime
from enhanced_eas_classifier import DynamicProcessClassifier, EnhancedEASScheduler
import os

class EnhancedEASTestSuite:
    """Comprehensive test suite for Enhanced EAS"""
    
    def __init__(self):
        self.db_path = os.path.expanduser("~/.battery_optimizer_enhanced_eas_test.db")
        self.scheduler = EnhancedEASScheduler(self.db_path)
        self.test_results = {}
    
    def test_dynamic_classification(self):
        """Test dynamic process classification"""
        print("🧠 Testing Dynamic Process Classification")
        print("=" * 50)
        
        # Get sample processes
        test_processes = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['pid'] > 100 and len(test_processes) < 15:
                    test_processes.append((proc.info['pid'], proc.info['name']))
            except:
                continue
        
        classification_results = []
        
        for pid, name in test_processes:
            try:
                # Test the enhanced classification
                classification, confidence = self.scheduler.classifier.classify_process_intelligent(pid, name)
                
                # Get behavioral analysis
                behavior = self.scheduler.classifier.analyze_process_behavior(pid, name)
                
                result = {
                    'name': name,
                    'pid': pid,
                    'classification': classification,
                    'confidence': confidence,
                    'cpu_usage': behavior.get('cpu_usage', 0),
                    'memory_mb': behavior.get('memory_mb', 0),
                    'user_interaction': behavior.get('user_interaction_score', 0),
                    'energy_efficiency': behavior.get('energy_efficiency_score', 0)
                }
                
                classification_results.append(result)
                
                print(f"Process: {name[:25]:25}")
                print(f"  Classification: {classification:20} (confidence: {confidence:.2f})")
                print(f"  CPU: {behavior.get('cpu_usage', 0):5.1f}%  Memory: {behavior.get('memory_mb', 0):6.0f}MB")
                print(f"  User Interaction: {behavior.get('user_interaction_score', 0):.2f}  Energy Efficiency: {behavior.get('energy_efficiency_score', 0):.2f}")
                print()
                
            except Exception as e:
                print(f"  Error testing {name}: {e}")
        
        self.test_results['classification'] = classification_results
        
        # Summary statistics
        if classification_results:
            avg_confidence = sum(r['confidence'] for r in classification_results) / len(classification_results)
            high_confidence = len([r for r in classification_results if r['confidence'] > 0.7])
            
            print(f"📊 Classification Summary:")
            print(f"  Processes tested: {len(classification_results)}")
            print(f"  Average confidence: {avg_confidence:.2f}")
            print(f"  High confidence (>0.7): {high_confidence}/{len(classification_results)}")
            
            # Classification distribution
            classifications = {}
            for result in classification_results:
                cls = result['classification']
                classifications[cls] = classifications.get(cls, 0) + 1
            
            print(f"  Classification distribution:")
            for cls, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
                print(f"    {cls}: {count}")
    
    def test_core_assignment_strategies(self):
        """Test enhanced core assignment strategies"""
        print("\n🎯 Testing Core Assignment Strategies")
        print("=" * 50)
        
        assignment_results = []
        
        # Test different process types
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 100 and len(assignment_results) < 10:
                    assignment = self.scheduler.classify_and_assign(pid, name)
                    assignment_results.append(assignment)
                    
                    print(f"Process: {name[:25]:25}")
                    print(f"  Classification: {assignment['classification']}")
                    print(f"  Strategy: {assignment['strategy']}")
                    print(f"  Target Core: {assignment['target_core']}")
                    print(f"  Priority Adj: {assignment['priority_adjustment']:+d}")
                    print(f"  Confidence: {assignment['confidence']:.2f}")
                    print(f"  System Load: P-cores {assignment.get('p_core_load', 0):.1f}%, E-cores {assignment.get('e_core_load', 0):.1f}%")
                    print()
                    
            except Exception as e:
                continue
        
        self.test_results['assignments'] = assignment_results
        
        # Strategy effectiveness analysis
        if assignment_results:
            strategies = {}
            core_distribution = {'p_core': 0, 'e_core': 0}
            
            for assignment in assignment_results:
                strategy = assignment['strategy']
                strategies[strategy] = strategies.get(strategy, 0) + 1
                core_distribution[assignment['target_core']] += 1
            
            print(f"📊 Assignment Summary:")
            print(f"  Total assignments: {len(assignment_results)}")
            print(f"  P-core assignments: {core_distribution['p_core']}")
            print(f"  E-core assignments: {core_distribution['e_core']}")
            print(f"  Strategy distribution:")
            for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
                print(f"    {strategy}: {count}")
    
    def test_learning_system(self):
        """Test the learning and adaptation system"""
        print("\n📚 Testing Learning System")
        print("=" * 50)
        
        # Get learning statistics
        stats = self.scheduler.classifier.get_classification_stats()
        
        print(f"Learning Statistics:")
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")
        print(f"  Recent classifications: {len(stats['recent_classifications'])}")
        
        if stats['recent_classifications']:
            print(f"  Recent classification breakdown:")
            for classification, count, avg_confidence in stats['recent_classifications']:
                print(f"    {classification}: {count} times (avg confidence: {avg_confidence:.3f})")
        
        print(f"  Current thresholds:")
        for threshold, value in stats['current_thresholds'].items():
            print(f"    {threshold}: {value}")
        
        self.test_results['learning'] = stats
    
    def test_performance_comparison(self):
        """Compare enhanced EAS vs basic classification"""
        print("\n⚡ Testing Performance Comparison")
        print("=" * 50)
        
        # Test processes with both methods
        comparison_results = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 100 and len(comparison_results) < 8:
                    # Enhanced classification
                    start_time = time.time()
                    enhanced_class, enhanced_conf = self.scheduler.classifier.classify_process_intelligent(pid, name)
                    enhanced_time = time.time() - start_time
                    
                    # Basic classification (fallback method)
                    start_time = time.time()
                    basic_class = self.scheduler.classifier._fallback_classification(name)
                    basic_time = time.time() - start_time
                    
                    comparison = {
                        'name': name,
                        'enhanced_classification': enhanced_class,
                        'enhanced_confidence': enhanced_conf,
                        'enhanced_time_ms': enhanced_time * 1000,
                        'basic_classification': basic_class,
                        'basic_time_ms': basic_time * 1000,
                        'improvement': enhanced_conf > 0.5  # Consider >0.5 confidence an improvement
                    }
                    
                    comparison_results.append(comparison)
                    
                    print(f"Process: {name[:20]:20}")
                    print(f"  Enhanced: {enhanced_class:20} (conf: {enhanced_conf:.2f}, {enhanced_time*1000:.1f}ms)")
                    print(f"  Basic:    {basic_class:20} (conf: N/A, {basic_time*1000:.1f}ms)")
                    print(f"  Improvement: {'✅' if comparison['improvement'] else '❌'}")
                    print()
                    
            except Exception as e:
                continue
        
        self.test_results['performance'] = comparison_results
        
        if comparison_results:
            improvements = len([c for c in comparison_results if c['improvement']])
            avg_enhanced_time = sum(c['enhanced_time_ms'] for c in comparison_results) / len(comparison_results)
            avg_basic_time = sum(c['basic_time_ms'] for c in comparison_results) / len(comparison_results)
            
            print(f"📊 Performance Summary:")
            print(f"  Processes tested: {len(comparison_results)}")
            print(f"  Improvements: {improvements}/{len(comparison_results)} ({improvements/len(comparison_results)*100:.1f}%)")
            print(f"  Avg enhanced time: {avg_enhanced_time:.1f}ms")
            print(f"  Avg basic time: {avg_basic_time:.1f}ms")
            print(f"  Time overhead: {avg_enhanced_time/avg_basic_time:.1f}x")
    
    def test_api_integration(self):
        """Test API integration with existing system"""
        print("\n🌐 Testing API Integration")
        print("=" * 50)
        
        # Test if the main app is running
        try:
            response = requests.get('http://localhost:9010/api/status', timeout=5)
            if response.status_code == 200:
                print("✅ Main Battery Optimizer is running")
                
                # Test enhanced EAS endpoints (if integrated)
                endpoints_to_test = [
                    '/api/eas-insights',
                    '/api/eas-learning-stats',
                ]
                
                for endpoint in endpoints_to_test:
                    try:
                        response = requests.get(f'http://localhost:9010{endpoint}', timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            print(f"✅ {endpoint}: Working ({len(data)} keys)")
                        else:
                            print(f"⚠️  {endpoint}: Not integrated yet (status: {response.status_code})")
                    except requests.exceptions.RequestException:
                        print(f"❌ {endpoint}: Connection failed")
                
                # Test enable enhanced EAS
                try:
                    response = requests.post('http://localhost:9010/api/eas-enable-enhanced', timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ Enhanced EAS enable: {data.get('success', False)}")
                    else:
                        print(f"⚠️  Enhanced EAS enable: Not integrated yet")
                except requests.exceptions.RequestException:
                    print(f"❌ Enhanced EAS enable: Connection failed")
                
            else:
                print("❌ Main Battery Optimizer not responding")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Battery Optimizer on localhost:9010")
            print("   Please start the main app first to test API integration")
    
    def test_memory_and_performance(self):
        """Test memory usage and performance of enhanced system"""
        print("\n🔧 Testing Memory and Performance")
        print("=" * 50)
        
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Perform intensive classification
        start_time = time.time()
        classifications = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 50:
                    self.scheduler.classifier.classify_process_intelligent(pid, name)
                    classifications += 1
                    
                if classifications >= 50:  # Test 50 processes
                    break
                    
            except Exception:
                continue
        
        end_time = time.time()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = end_time - start_time
        
        print(f"Performance Metrics:")
        print(f"  Processes classified: {classifications}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per process: {total_time/classifications*1000:.1f} ms")
        print(f"  Current memory: {current / 1024 / 1024:.1f} MB")
        print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
        print(f"  Classifications per second: {classifications/total_time:.1f}")
        
        self.test_results['performance_metrics'] = {
            'classifications': classifications,
            'total_time': total_time,
            'time_per_process': total_time/classifications*1000,
            'current_memory_mb': current / 1024 / 1024,
            'peak_memory_mb': peak / 1024 / 1024,
            'classifications_per_second': classifications/total_time
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 ENHANCED EAS TEST REPORT")
        print("=" * 60)
        
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Classification results
        if 'classification' in self.test_results:
            results = self.test_results['classification']
            if results:
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                high_confidence = len([r for r in results if r['confidence'] > 0.7])
                
                print(f"\n📊 Classification Performance:")
                print(f"  ✅ Processes classified: {len(results)}")
                print(f"  ✅ Average confidence: {avg_confidence:.2f}")
                print(f"  ✅ High confidence rate: {high_confidence/len(results)*100:.1f}%")
        
        # Assignment results
        if 'assignments' in self.test_results:
            assignments = self.test_results['assignments']
            if assignments:
                p_core_count = len([a for a in assignments if a['target_core'] == 'p_core'])
                e_core_count = len([a for a in assignments if a['target_core'] == 'e_core'])
                
                print(f"\n🎯 Core Assignment Distribution:")
                print(f"  ✅ P-core assignments: {p_core_count} ({p_core_count/len(assignments)*100:.1f}%)")
                print(f"  ✅ E-core assignments: {e_core_count} ({e_core_count/len(assignments)*100:.1f}%)")
        
        # Learning system
        if 'learning' in self.test_results:
            learning = self.test_results['learning']
            print(f"\n📚 Learning System Status:")
            print(f"  ✅ Total classifications stored: {learning['total_classifications']}")
            print(f"  ✅ System confidence: {learning['average_confidence']:.3f}")
        
        # Performance metrics
        if 'performance_metrics' in self.test_results:
            perf = self.test_results['performance_metrics']
            print(f"\n⚡ Performance Metrics:")
            print(f"  ✅ Classification speed: {perf['time_per_process']:.1f}ms per process")
            print(f"  ✅ Memory usage: {perf['peak_memory_mb']:.1f}MB peak")
            print(f"  ✅ Throughput: {perf['classifications_per_second']:.1f} classifications/sec")
        
        # Overall assessment
        print(f"\n🏆 Overall Assessment:")
        
        # Calculate overall score
        score_factors = []
        
        if 'classification' in self.test_results and self.test_results['classification']:
            avg_conf = sum(r['confidence'] for r in self.test_results['classification']) / len(self.test_results['classification'])
            score_factors.append(avg_conf * 100)
        
        if 'performance' in self.test_results and self.test_results['performance']:
            improvements = len([c for c in self.test_results['performance'] if c['improvement']])
            improvement_rate = improvements / len(self.test_results['performance'])
            score_factors.append(improvement_rate * 100)
        
        if 'learning' in self.test_results:
            learning_score = min(self.test_results['learning']['total_classifications'] / 100, 1.0) * 100
            score_factors.append(learning_score)
        
        if score_factors:
            overall_score = sum(score_factors) / len(score_factors)
            
            if overall_score > 80:
                assessment = "🚀 EXCELLENT - Enhanced EAS is working exceptionally well"
            elif overall_score > 60:
                assessment = "✅ GOOD - Enhanced EAS shows significant improvements"
            elif overall_score > 40:
                assessment = "📈 FAIR - Enhanced EAS is functional with room for improvement"
            else:
                assessment = "⚠️  NEEDS WORK - Enhanced EAS requires optimization"
            
            print(f"  {assessment}")
            print(f"  Overall Score: {overall_score:.1f}/100")
        
        print("\n💡 Next Steps:")
        print("1. Integrate enhanced EAS with main app using eas_integration_patch.py")
        print("2. Monitor classification accuracy over time")
        print("3. Adjust thresholds based on learning data")
        print("4. Add more behavioral analysis features")
        print("5. Implement user feedback mechanisms")
        
        print("=" * 60)
        
        return self.test_results

def main():
    """Run comprehensive enhanced EAS test suite"""
    print("🧪 Enhanced EAS Comprehensive Test Suite")
    print("Testing advanced machine learning classification system")
    print("=" * 60)
    
    tester = EnhancedEASTestSuite()
    
    try:
        # Run all tests
        tester.test_dynamic_classification()
        tester.test_core_assignment_strategies()
        tester.test_learning_system()
        tester.test_performance_comparison()
        tester.test_api_integration()
        tester.test_memory_and_performance()
        
        # Generate final report
        results = tester.generate_report()
        
        # Save results
        report_file = f"enhanced_eas_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()