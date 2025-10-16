#!/usr/bin/env python3
"""
Simplified 40-Qubit Quantum Circuit Manager for PQS Framework
Cirq-only implementation for quantum circuit management
"""

import cirq
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass


@dataclass
class QubitAllocation:
    """Represents dynamic qubit allocation for algorithms"""
    algorithm_name: str
    allocated_qubits: List[int]
    allocation_time: float
    expected_duration: float
    priority: int


class QuantumCircuitManager40:
    """
    Simplified 40-Qubit Quantum Circuit Manager
    
    Provides comprehensive circuit management and dynamic qubit allocation
    for 40-qubit quantum systems using Cirq.
    """
    
    def __init__(self, max_qubits: int = 40):
        """
        Initialize the 40-qubit quantum circuit manager
        
        Args:
            max_qubits: Maximum number of qubits (default 40)
        """
        self.max_qubits = max_qubits
        
        # Create 40-qubit grid layout (8x5 for optimal connectivity)
        self.qubits = cirq.GridQubit.rect(8, 5)[:self.max_qubits]
        
        # Qubit allocation tracking
        self.allocated_qubits: Dict[str, QubitAllocation] = {}
        self.available_qubits = set(range(max_qubits))
        
        # Performance metrics
        self.circuit_creation_times: List[float] = []
        
        print(f"🔧 Initialized QuantumCircuitManager40 with {max_qubits} qubits")
    
    def create_40_qubit_circuit(self, 
                               algorithm_type: str = "qaoa",
                               parameters: Optional[Dict] = None) -> cirq.Circuit:
        """
        Create a 40-qubit quantum circuit for specified algorithm
        
        Args:
            algorithm_type: Type of quantum algorithm ("qaoa", "grover", "custom")
            parameters: Algorithm-specific parameters
            
        Returns:
            Cirq quantum circuit
        """
        start_time = time.perf_counter()
        
        if parameters is None:
            parameters = {}
        
        print(f"🔨 Creating 40-qubit {algorithm_type} circuit...")
        
        circuit = cirq.Circuit()
        
        if algorithm_type == "qaoa":
            layers = parameters.get('layers', 8)
            problem_graph = parameters.get('problem_graph', {})
            
            # Initialize superposition
            circuit.append(cirq.H(qubit) for qubit in self.qubits[:40])
            
            # QAOA layers
            for layer in range(layers):
                # Problem Hamiltonian
                gamma = parameters.get('gamma', np.pi / (4 * (layer + 1)))
                self._add_problem_hamiltonian(circuit, problem_graph, gamma)
                
                # Mixer Hamiltonian
                beta = parameters.get('beta', np.pi / (2 * (layer + 2)))
                self._add_mixer_hamiltonian(circuit, beta)
            
            # Measurement
            circuit.append(cirq.measure(*self.qubits[:40], key='result'))
            
        elif algorithm_type == "grover":
            target_states = parameters.get('target_states', [0])
            iterations = parameters.get('iterations', 10)  # Limited for practical execution
            
            # Initialize superposition
            circuit.append(cirq.H(qubit) for qubit in self.qubits[:40])
            
            # Grover iterations
            for _ in range(iterations):
                # Oracle (simplified)
                circuit.append(cirq.Z(self.qubits[0]))
                
                # Diffusion operator (simplified)
                circuit.append(cirq.H(qubit) for qubit in self.qubits[:40])
                circuit.append(cirq.Z(qubit) for qubit in self.qubits[:40])
                circuit.append(cirq.H(qubit) for qubit in self.qubits[:40])
            
            # Measurement
            circuit.append(cirq.measure(*self.qubits[:40], key='result'))
            
        elif algorithm_type == "custom":
            # Custom circuit from parameters
            gates = parameters.get('gates', [])
            for gate_spec in gates:
                self._add_custom_gate(circuit, gate_spec)
        
        creation_time = time.perf_counter() - start_time
        self.circuit_creation_times.append(creation_time)
        
        print(f"✅ Circuit created in {creation_time:.3f}s")
        
        return circuit
    
    def _add_problem_hamiltonian(self, circuit: cirq.Circuit, 
                                problem_graph: Dict, gamma: float):
        """Add problem Hamiltonian to circuit"""
        # ZZ interactions for edges
        for edge in problem_graph.get('edges', []):
            i, j = edge['nodes']
            if i < 40 and j < 40:
                weight = edge.get('weight', 1.0)
                circuit.append(cirq.ZZ(self.qubits[i], self.qubits[j]) ** (gamma * weight))
        
        # Z rotations for nodes
        for node in problem_graph.get('nodes', []):
            if node['id'] < 40:
                cost = node.get('cost', 1.0)
                circuit.append(cirq.Z(self.qubits[node['id']]) ** (gamma * cost))
    
    def _add_mixer_hamiltonian(self, circuit: cirq.Circuit, beta: float):
        """Add mixer Hamiltonian to circuit"""
        for qubit in self.qubits[:40]:
            circuit.append(cirq.X(qubit) ** beta)
    
    def _add_custom_gate(self, circuit: cirq.Circuit, gate_spec: Dict):
        """Add custom gate to circuit"""
        gate_type = gate_spec.get('type')
        qubits = gate_spec.get('qubits', [])
        
        if gate_type == 'H':
            for q in qubits:
                if q < 40:
                    circuit.append(cirq.H(self.qubits[q]))
        elif gate_type == 'CNOT':
            if len(qubits) >= 2:
                circuit.append(cirq.CNOT(self.qubits[qubits[0]], self.qubits[qubits[1]]))
    
    def allocate_qubits(self, 
                       algorithm_name: str,
                       required_qubits: int,
                       priority: int = 0,
                       expected_duration: float = 1.0) -> List[int]:
        """
        Dynamically allocate qubits for quantum algorithms
        
        Args:
            algorithm_name: Name/ID of the algorithm requesting qubits
            required_qubits: Number of qubits needed
            priority: Priority level (higher = more important)
            expected_duration: Expected execution time in seconds
            
        Returns:
            List of allocated qubit indices
        """
        if required_qubits > len(self.available_qubits):
            # Try to free up qubits from lower priority algorithms
            self._free_low_priority_qubits(required_qubits, priority)
        
        if required_qubits > len(self.available_qubits):
            raise ValueError(f"Cannot allocate {required_qubits} qubits. "
                           f"Only {len(self.available_qubits)} available.")
        
        # Allocate qubits (prefer contiguous allocation for better connectivity)
        allocated = self._allocate_contiguous_qubits(required_qubits)
        
        # Record allocation
        allocation = QubitAllocation(
            algorithm_name=algorithm_name,
            allocated_qubits=allocated,
            allocation_time=time.perf_counter(),
            expected_duration=expected_duration,
            priority=priority
        )
        
        self.allocated_qubits[algorithm_name] = allocation
        self.available_qubits -= set(allocated)
        
        print(f"📍 Allocated {required_qubits} qubits to {algorithm_name}: {allocated}")
        
        return allocated
    
    def _allocate_contiguous_qubits(self, count: int) -> List[int]:
        """Allocate contiguous qubits for better connectivity"""
        available_list = sorted(list(self.available_qubits))
        
        # Try to find contiguous block
        for start_idx in range(len(available_list) - count + 1):
            candidate_qubits = available_list[start_idx:start_idx + count]
            
            # Check if qubits are contiguous
            if all(candidate_qubits[i] + 1 == candidate_qubits[i + 1] 
                   for i in range(len(candidate_qubits) - 1)):
                return candidate_qubits
        
        # If no contiguous block found, allocate any available qubits
        return available_list[:count]
    
    def _free_low_priority_qubits(self, needed: int, min_priority: int):
        """Free qubits from lower priority algorithms"""
        # Sort allocations by priority (ascending)
        sorted_allocations = sorted(
            self.allocated_qubits.items(),
            key=lambda x: x[1].priority
        )
        
        freed_count = 0
        to_remove = []
        
        for algo_name, allocation in sorted_allocations:
            if allocation.priority < min_priority and freed_count < needed:
                # Free this allocation
                self.available_qubits.update(allocation.allocated_qubits)
                freed_count += len(allocation.allocated_qubits)
                to_remove.append(algo_name)
                
                print(f"🔄 Freed {len(allocation.allocated_qubits)} qubits from {algo_name}")
        
        # Remove freed allocations
        for algo_name in to_remove:
            del self.allocated_qubits[algo_name]
    
    def deallocate_qubits(self, algorithm_name: str):
        """Deallocate qubits from a completed algorithm"""
        if algorithm_name in self.allocated_qubits:
            allocation = self.allocated_qubits[algorithm_name]
            self.available_qubits.update(allocation.allocated_qubits)
            del self.allocated_qubits[algorithm_name]
            
            print(f"🔓 Deallocated qubits from {algorithm_name}")
    
    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current qubit allocation status"""
        return {
            'total_qubits': self.max_qubits,
            'available_qubits': len(self.available_qubits),
            'allocated_qubits': self.max_qubits - len(self.available_qubits),
            'active_algorithms': list(self.allocated_qubits.keys()),
            'allocation_details': {
                name: {
                    'qubits': alloc.allocated_qubits,
                    'priority': alloc.priority,
                    'duration': time.perf_counter() - alloc.allocation_time
                }
                for name, alloc in self.allocated_qubits.items()
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for circuit management"""
        return {
            'total_circuits_created': len(self.circuit_creation_times),
            'average_creation_time': np.mean(self.circuit_creation_times) if self.circuit_creation_times else 0,
            'current_allocations': len(self.allocated_qubits),
            'qubit_utilization': (self.max_qubits - len(self.available_qubits)) / self.max_qubits
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the QuantumCircuitManager40
    print("🧪 Testing QuantumCircuitManager40...")
    
    # Initialize manager
    manager = QuantumCircuitManager40(max_qubits=40)
    
    # Test qubit allocation
    qubits1 = manager.allocate_qubits("test_algorithm_1", 10, priority=5)
    qubits2 = manager.allocate_qubits("test_algorithm_2", 15, priority=3)
    
    print("Allocation status:", manager.get_allocation_status())
    
    # Test circuit creation
    problem_graph = {
        'nodes': [{'id': i, 'cost': 0.5} for i in range(20)],
        'edges': [{'nodes': [i, i+1], 'weight': 0.3} for i in range(19)]
    }
    
    circuit = manager.create_40_qubit_circuit(
        algorithm_type="qaoa",
        parameters={'layers': 4, 'problem_graph': problem_graph}
    )
    
    print("✅ Circuit created successfully")
    print("Performance metrics:", manager.get_performance_metrics())
    
    # Clean up
    manager.deallocate_qubits("test_algorithm_1")
    manager.deallocate_qubits("test_algorithm_2")
    
    print("🎉 QuantumCircuitManager40 test completed!")