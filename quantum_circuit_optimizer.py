#!/usr/bin/env python3
"""
Quantum Circuit Optimizer
Optimizes quantum circuits for maximum efficiency
"""

import logging
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
try:
    from qiskit.transpiler.passes import (
        Optimize1qGatesDecomposition,
        CommutativeCancellation,
        Unroll3qOrMore
    )
    # Qiskit 2.x compatible
    Optimize1qGates = Optimize1qGatesDecomposition
except ImportError:
    # Fallback for older versions
    from qiskit.transpiler.passes import (
        Optimize1qGates,
        CommutativeCancellation,
        Unroll3qOrMore
    )

logger = logging.getLogger(__name__)


class QuantumCircuitOptimizer:
    """
    Optimizes quantum circuits for better performance
    - Reduces circuit depth
    - Minimizes gate count
    - Optimizes for target hardware
    """
    
    def __init__(self):
        self.optimization_level = 3  # Maximum optimization
        self.pass_manager = self._create_pass_manager()
        self.stats = {
            'circuits_optimized': 0,
            'total_depth_reduction': 0,
            'total_gate_reduction': 0
        }
        logger.info("✅ Quantum Circuit Optimizer initialized")
    
    def _create_pass_manager(self) -> PassManager:
        """Create optimized pass manager"""
        passes = [
            # Optimize single-qubit gates
            Optimize1qGates(),
            
            # Cancel commutative operations
            CommutativeCancellation(),
            
            # Unroll 3+ qubit gates
            Unroll3qOrMore(),
        ]
        
        return PassManager(passes)
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Optimize a quantum circuit
        
        Args:
            circuit: Input quantum circuit
            
        Returns:
            Dict with optimized circuit and statistics
        """
        try:
            # Get original stats
            original_depth = circuit.depth()
            original_size = circuit.size()
            
            # Optimize circuit
            optimized_circuit = self.pass_manager.run(circuit)
            
            # Get optimized stats
            optimized_depth = optimized_circuit.depth()
            optimized_size = optimized_circuit.size()
            
            # Calculate improvements
            depth_reduction = ((original_depth - optimized_depth) / original_depth * 100) if original_depth > 0 else 0
            gate_reduction = ((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0
            
            # Update stats
            self.stats['circuits_optimized'] += 1
            self.stats['total_depth_reduction'] += depth_reduction
            self.stats['total_gate_reduction'] += gate_reduction
            
            logger.debug(f"Circuit optimized: depth {original_depth}→{optimized_depth} ({depth_reduction:.1f}%), "
                        f"gates {original_size}→{optimized_size} ({gate_reduction:.1f}%)")
            
            return {
                'success': True,
                'optimized_circuit': optimized_circuit,
                'original_depth': original_depth,
                'optimized_depth': optimized_depth,
                'depth_reduction_percent': depth_reduction,
                'original_gates': original_size,
                'optimized_gates': optimized_size,
                'gate_reduction_percent': gate_reduction,
                'speedup_estimate': 1.0 + (depth_reduction / 100.0)  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Circuit optimization error: {e}")
            return {
                'success': False,
                'error': str(e),
                'optimized_circuit': circuit  # Return original on error
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        avg_depth_reduction = (self.stats['total_depth_reduction'] / self.stats['circuits_optimized'] 
                              if self.stats['circuits_optimized'] > 0 else 0)
        avg_gate_reduction = (self.stats['total_gate_reduction'] / self.stats['circuits_optimized']
                             if self.stats['circuits_optimized'] > 0 else 0)
        
        return {
            'circuits_optimized': self.stats['circuits_optimized'],
            'avg_depth_reduction_percent': avg_depth_reduction,
            'avg_gate_reduction_percent': avg_gate_reduction,
            'estimated_speedup': 1.0 + (avg_depth_reduction / 100.0)
        }


# Global instance
_circuit_optimizer = None


def get_circuit_optimizer() -> QuantumCircuitOptimizer:
    """Get or create circuit optimizer instance"""
    global _circuit_optimizer
    if _circuit_optimizer is None:
        _circuit_optimizer = QuantumCircuitOptimizer()
    return _circuit_optimizer


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🧪 Testing Quantum Circuit Optimizer...\n")
    
    # Create test circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.h(0)  # This can be optimized
    qc.h(0)  # Adjacent H gates cancel
    qc.cx(0, 1)  # This might cancel with earlier CX
    qc.cx(0, 1)  # Adjacent CX gates cancel
    
    print(f"Original circuit:")
    print(f"  Depth: {qc.depth()}")
    print(f"  Gates: {qc.size()}")
    print()
    
    # Optimize
    optimizer = get_circuit_optimizer()
    result = optimizer.optimize_circuit(qc)
    
    if result['success']:
        print(f"Optimized circuit:")
        print(f"  Depth: {result['optimized_depth']} ({result['depth_reduction_percent']:.1f}% reduction)")
        print(f"  Gates: {result['optimized_gates']} ({result['gate_reduction_percent']:.1f}% reduction)")
        print(f"  Estimated speedup: {result['speedup_estimate']:.2f}x")
        print()
        
        # Get stats
        stats = optimizer.get_stats()
        print(f"Optimizer stats:")
        print(f"  Circuits optimized: {stats['circuits_optimized']}")
        print(f"  Avg depth reduction: {stats['avg_depth_reduction_percent']:.1f}%")
        print(f"  Avg gate reduction: {stats['avg_gate_reduction_percent']:.1f}%")
        print(f"  Estimated speedup: {stats['estimated_speedup']:.2f}x")
    else:
        print(f"❌ Optimization failed: {result['error']}")
    
    print("\n✅ Circuit optimizer test complete!")
