#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced EAS System
Tests all components and measures performance benchmarks
"""

import unittest
import time
import psutil
import numpy as np
from advanced_eas_main import AdvancedEASSystem
from ml_process_classifier import MLProcessClassifier
from behavior_predictor import ProcessBehaviorPredictor
from context_analyzer import ContextAnalyzer
from hardware_monitor import HardwareMonitor
from predictive_energy_manager import PredictiveEnergyManager
from rl_scheduler import RLSchedulerTrainer
from quantum_scheduler import QuantumInspiredScheduler, QuantumSchedulingProblem

class TestAdvancedEAS(unittest.TestCase):
    """Comprehensive test suite for Advanced EAS"""
    
    def setUp(self):
        """Set up test environment"""
        self.eas_system = AdvancedEASSystem()
        
    def tearDown(self):
        """Clean up after tests"""
        if self.eas_system.running:
            self.eas_system.stop_system()
    
    def test_ml_classifier_performance(self):
        """Test ML classifier performance and accuracy"""
        print("🧠 Testing ML Process Classifier...")
        
        classifier = MLProcessClassifier()
        
        # Test classification speed
        start_time = time.time()
        classifications = []
        
        process_count = 0
        for proc in psutil.process_iter(['pid', 'name']):
            if process_count >= 50:  # Test 50 processes
                break
                
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 10 and name:
                    classification, confidence = classifier.classify_process(pid, name)
                    classifications.append((name, classification, confidence))
                    process_count += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        classification_time = time.time() - start_time
        
        # Performance assertions
        self.assertGreater(len(classifications), 10, "Should classify at least 10 processes")
        self.assertLess(classification_time, 5.0, "Classification should complete within 5 seconds")
        
        # Speed benchmark
        processes_per_second = len(classifications) / classification_time
        self.assertGreater(processes_per_second, 10, "Should classify at least 10 processes per second")
        
        print(f"  ✅ Classified {len(classifications)} processes in {classification_time:.2f}s")
        print(f"  ⚡ Speed: {processes_per_second:.1f} processes/second")
        
        # Test confidence levels
        confidences = [conf for _, _, conf in classifications]
        avg_confidence = np.mean(confidences)
        self.assertGreater(avg_confidence, 0.3, "Average confidence should be reasonable")
        
        print(f"  📊 Average confidence: {avg_confidence:.3f}")
    
    def test_behavior_predictor(self):
        """Test LSTM behavior predictor"""
        print("🔮 Testing Behavior Predictor...")
        
        predictor = ProcessBehaviorPredictor()
        
        # Test with synthetic data
        sequences, targets = predictor.generate_synthetic_training_data(50)
        
        self.assertEqual(len(sequences), 50, "Should generate 50 sequences")
        self.assertEqual(len(targets), 50, "Should generate 50 targets")
        
        # Test prediction (without training for speed)
        from ml_process_classifier import ProcessFeatures
        
        dummy_features = ProcessFeatures(
            cpu_usage_history=[20, 25, 30] * 10,
            memory_usage_history=[100, 105, 110] * 10,
            io_read_rate=1000,
            io_write_rate=500,
            network_bytes_sent=100,
            network_bytes_recv=5,
            thread_count=4,
            file_descriptors=10,
            context_switches=500,
            voluntary_switches=400,
            involuntary_switches=100,
            page_faults=25,
            cpu_affinity=[0, 1],
            nice_value=0,
            process_age=1800
        )
        
        prediction = predictor.predict_behavior(dummy_features)
        
        self.assertIn('cpu_prediction', prediction)
        self.assertIn('memory_prediction', prediction)
        self.assertIn('confidence', prediction)
        
        print(f"  ✅ Prediction generated successfully")
        print(f"  📊 Confidence: {prediction['confidence']:.3f}")
    
    def test_context_analyzer(self):
        """Test context analyzer functionality"""
        print("🎯 Testing Context Analyzer...")
        
        analyzer = ContextAnalyzer()
        context = analyzer.get_system_context()
        
        # Verify context structure
        self.assertIsInstance(context.meeting_in_progress, bool)
        self.assertIsInstance(context.workflow_phase, str)
        self.assertIsInstance(context.user_focus_level, float)
        self.assertIsInstance(context.battery_level, float)
        self.assertIsInstance(context.thermal_state, str)
        
        # Verify reasonable values
        self.assertGreaterEqual(context.user_focus_level, 0.0)
        self.assertLessEqual(context.user_focus_level, 1.0)
        self.assertGreaterEqual(context.battery_level, 0.0)
        self.assertLessEqual(context.battery_level, 100.0)
        
        print(f"  ✅ Context analysis completed")
        print(f"  📊 Focus Level: {context.user_focus_level:.2f}")
        print(f"  🔋 Battery: {context.battery_level:.1f}%")
        print(f"  🏢 Workflow: {context.workflow_phase}")
    
    def test_hardware_monitor(self):
        """Test hardware monitoring functionality"""
        print("🔧 Testing Hardware Monitor...")
        
        monitor = HardwareMonitor(sampling_interval=0.5)
        monitor.start_monitoring()
        
        # Wait for some samples
        time.sleep(2)
        
        metrics = monitor.get_current_metrics()
        
        if metrics:
            self.assertIsInstance(metrics.cpu_temperature, float)
            self.assertIsInstance(metrics.thermal_pressure, float)
            self.assertIsInstance(metrics.power_consumption, dict)
            
            # Verify reasonable ranges
            self.assertGreaterEqual(metrics.cpu_temperature, 0)
            self.assertLessEqual(metrics.cpu_temperature, 150)
            self.assertGreaterEqual(metrics.thermal_pressure, 0.0)
            self.assertLessEqual(metrics.thermal_pressure, 1.0)
            
            print(f"  ✅ Hardware monitoring active")
            print(f"  🌡️  CPU Temperature: {metrics.cpu_temperature:.1f}°C")
            print(f"  ⚡ Thermal Pressure: {metrics.thermal_pressure:.2f}")
        else:
            print("  ⚠️  Hardware metrics not available (may need sudo)")
        
        monitor.stop_monitoring()
    
    def test_energy_manager(self):
        """Test predictive energy manager"""
        print("🔮 Testing Predictive Energy Manager...")
        
        manager = PredictiveEnergyManager("~/.test_energy.db")
        
        test_metrics = {
            'battery_level': 60.0,
            'cpu_usage': 40.0,
            'temperature': 55.0,
            'thermal_state': 'cool',
            'active_processes': ['chrome', 'terminal']
        }
        
        prediction = manager.predict_energy_state(test_metrics)
        
        # Verify prediction structure
        self.assertIsInstance(prediction.battery_life_hours, float)
        self.assertIsInstance(prediction.thermal_throttling_risk, float)
        self.assertIsInstance(prediction.confidence, float)
        self.assertIsInstance(prediction.recommended_actions, list)
        
        # Verify reasonable ranges
        self.assertGreater(prediction.battery_life_hours, 0)
        self.assertLessEqual(prediction.battery_life_hours, 24)
        self.assertGreaterEqual(prediction.thermal_throttling_risk, 0.0)
        self.assertLessEqual(prediction.thermal_throttling_risk, 1.0)
        
        print(f"  ✅ Energy prediction completed")
        print(f"  🔋 Predicted Battery Life: {prediction.battery_life_hours:.1f}h")
        print(f"  🌡️  Thermal Risk: {prediction.thermal_throttling_risk:.2f}")
        print(f"  📊 Confidence: {prediction.confidence:.2f}")
    
    def test_rl_scheduler(self):
        """Test reinforcement learning scheduler"""
        print("🤖 Testing RL Scheduler...")
        
        trainer = RLSchedulerTrainer()
        
        # Quick training test
        scores = trainer.train(episodes=10)  # Very short training for testing
        
        self.assertEqual(len(scores), 10, "Should have 10 training scores")
        
        # Test the scheduler
        test_scores = trainer.test_scheduler(num_tests=3)
        
        self.assertEqual(len(test_scores), 3, "Should have 3 test scores")
        
        print(f"  ✅ RL training completed")
        print(f"  📈 Final training score: {scores[-1]:.2f}")
        print(f"  🧪 Average test score: {np.mean(test_scores):.2f}")
    
    def test_quantum_scheduler(self):
        """Test quantum-inspired scheduler"""
        print("⚛️  Testing Quantum Scheduler...")
        
        scheduler = QuantumInspiredScheduler()
        
        # Create small test problem
        test_processes = [
            {'pid': 1, 'name': 'test1', 'classification': 'interactive_application', 'cpu_usage': 30, 'priority': 0.8},
            {'pid': 2, 'name': 'test2', 'classification': 'background_service', 'cpu_usage': 10, 'priority': 0.2},
        ]
        
        test_cores = [
            {'type': 'p_core', 'capacity': 100.0, 'performance_rating': 1.0},
            {'type': 'e_core', 'capacity': 80.0, 'performance_rating': 0.7},
        ]
        
        problem = QuantumSchedulingProblem(
            processes=test_processes,
            cores=test_cores,
            constraints={},
            objective_weights={'efficiency': 0.6, 'performance': 0.4}
        )
        
        solution = scheduler.solve_scheduling_problem(problem)
        
        # Verify solution structure
        self.assertIn('assignments', solution)
        self.assertIn('quality_score', solution)
        self.assertEqual(len(solution['assignments']), 2, "Should have 2 assignments")
        
        print(f"  ✅ Quantum optimization completed")
        print(f"  📊 Solution quality: {solution['quality_score']:.3f}")
        print(f"  ⚛️  Convergence iterations: {solution.get('convergence_iterations', 'N/A')}")
    
    def test_full_system_integration(self):
        """Test full system integration"""
        print("🚀 Testing Full System Integration...")
        
        # Test system startup
        self.eas_system.start_system()
        self.assertTrue(self.eas_system.running, "System should be running")
        
        # Test optimization cycle
        result = self.eas_system.optimize_system()
        
        # Verify result structure
        self.assertIn('optimized_processes', result)
        self.assertIn('optimization_time_ms', result)
        self.assertIn('strategy_used', result)
        self.assertIn('energy_prediction', result)
        
        # Performance benchmarks
        self.assertLess(result['optimization_time_ms'], 10000, "Optimization should complete within 10 seconds")
        self.assertGreater(result['optimized_processes'], 0, "Should optimize at least some processes")
        
        print(f"  ✅ System integration test passed")
        print(f"  📊 Optimized: {result['optimized_processes']} processes")
        print(f"  ⚡ Time: {result['optimization_time_ms']:.1f}ms")
        print(f"  🎯 Strategy: {result['strategy_used']}")
        
        # Test system stats
        stats = self.eas_system.get_system_stats()
        self.assertGreater(stats['optimization_cycles'], 0)
        
        # Test system shutdown
        self.eas_system.stop_system()
        self.assertFalse(self.eas_system.running, "System should be stopped")
    
    def test_performance_benchmarks(self):
        """Run performance benchmarks"""
        print("📊 Running Performance Benchmarks...")
        
        # Benchmark 1: Process analysis speed
        analyzer = self.eas_system.process_analyzer
        
        start_time = time.time()
        analyzed_count = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            if analyzed_count >= 100:
                break
                
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                if pid > 10 and name:
                    intel = analyzer.analyze_process_enhanced(pid, name)
                    analyzed_count += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        analysis_time = time.time() - start_time
        analysis_speed = analyzed_count / analysis_time
        
        print(f"  🧠 Process Analysis: {analysis_speed:.1f} processes/second")
        
        # Benchmark 2: Full optimization cycle speed
        self.eas_system.start_system()
        
        optimization_times = []
        for _ in range(5):
            start_time = time.time()
            result = self.eas_system.optimize_system()
            optimization_times.append(time.time() - start_time)
        
        avg_optimization_time = np.mean(optimization_times)
        
        print(f"  🚀 Optimization Cycle: {avg_optimization_time:.3f} seconds average")
        
        self.eas_system.stop_system()
        
        # Performance assertions
        self.assertGreater(analysis_speed, 50, "Should analyze at least 50 processes per second")
        self.assertLess(avg_optimization_time, 5.0, "Optimization should complete within 5 seconds")
        
        print(f"  ✅ Performance benchmarks passed")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🧪 Advanced EAS - Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedEAS)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Summary:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"  ✅ All tests passed!")
    else:
        print(f"  ❌ Some tests failed")
        
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)