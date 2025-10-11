#!/usr/bin/env python3
"""
Quick test of quantum scheduler with small problem
"""

from quantum_scheduler import QuantumInspiredScheduler, QuantumSchedulingProblem
import time

def test_small_quantum_problem():
    """Test quantum scheduler with small problem"""
    print("⚛️  Testing Quantum Scheduler (Small Problem)")
    print("=" * 50)
    
    scheduler = QuantumInspiredScheduler()
    
    # Small test problem - just 5 processes
    test_processes = [
        {'pid': 1, 'name': 'chrome', 'classification': 'interactive_application', 'cpu_usage': 30, 'priority': 0.8},
        {'pid': 2, 'name': 'vscode', 'classification': 'interactive_application', 'cpu_usage': 25, 'priority': 0.9},
        {'pid': 3, 'name': 'python', 'classification': 'compute_intensive', 'cpu_usage': 60, 'priority': 0.7},
        {'pid': 4, 'name': 'backupd', 'classification': 'background_service', 'cpu_usage': 10, 'priority': 0.2},
        {'pid': 5, 'name': 'spotlight', 'classification': 'background_service', 'cpu_usage': 15, 'priority': 0.3},
    ]
    
    test_cores = [
        {'type': 'p_core', 'capacity': 100.0, 'performance_rating': 1.0},
        {'type': 'p_core', 'capacity': 100.0, 'performance_rating': 1.0},
        {'type': 'e_core', 'capacity': 80.0, 'performance_rating': 0.7},
        {'type': 'e_core', 'capacity': 80.0, 'performance_rating': 0.7},
    ]
    
    problem = QuantumSchedulingProblem(
        processes=test_processes,
        cores=test_cores,
        constraints={},
        objective_weights={'efficiency': 0.6, 'performance': 0.4}
    )
    
    print(f"Problem size: {len(test_processes)} processes, {len(test_cores)} cores")
    
    # Solve the problem
    start_time = time.time()
    solution = scheduler.solve_scheduling_problem(problem)
    solve_time = time.time() - start_time
    
    print(f"\n🎯 Results:")
    print(f"  Solution Quality: {solution['quality_score']:.3f}")
    print(f"  Solve Time: {solve_time:.3f} seconds")
    print(f"  Convergence Iterations: {solution['convergence_iterations']}")
    
    print(f"\n📊 Assignments:")
    for assignment in solution['assignments']:
        print(f"  {assignment['process_name']:12} → {assignment['core_type']:6}")
    
    return solve_time < 10  # Should complete within 10 seconds

if __name__ == "__main__":
    success = test_small_quantum_problem()
    if success:
        print("\n✅ Quantum scheduler test passed!")
    else:
        print("\n❌ Quantum scheduler test failed!")