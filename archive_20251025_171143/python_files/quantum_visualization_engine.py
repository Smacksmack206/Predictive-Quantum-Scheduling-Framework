#!/usr/bin/env python3
"""
Quantum Visualization Engine
Real-time quantum circuit and state visualization for 40-qubit systems
"""

import cirq
import numpy as np
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import io

class VisualizationType(Enum):
    """Types of quantum visualizations"""
    CIRCUIT_DIAGRAM = "circuit_diagram"
    STATE_VECTOR = "state_vector"
    BLOCH_SPHERE = "bloch_sphere"
    ENTANGLEMENT_GRAPH = "entanglement_graph"
    MEASUREMENT_HISTOGRAM = "measurement_histogram"
    QUANTUM_PROCESS = "quantum_process"

@dataclass
class VisualizationMetadata:
    """Visualization metadata"""
    visualization_id: str
    creation_time: float
    qubit_count: int
    gate_count: int
    depth: int
    visualization_type: VisualizationType
    interactive: bool
    export_formats: List[str]

@dataclass
class InteractiveVisualization:
    """Interactive quantum visualization"""
    visualization_id: str
    html_content: str
    javascript_code: str
    css_styles: str
    metadata: VisualizationMetadata
    circuit_id: Optional[str] = None

@dataclass
class VisualizationStats:
    """Visualization engine statistics"""
    total_visualizations: int = 0
    interactive_visualizations: int = 0
    circuit_diagrams: int = 0
    state_visualizations: int = 0
    export_count: int = 0
    debug_sessions: int = 0
    export_formats_available: int = 8  # PNG, SVG, HTML, QASM, JSON, PDF, EPS, TIKZ

class QuantumVisualizationEngine:
    """
    Quantum Visualization Engine for 40-qubit systems
    Creates interactive visualizations of quantum circuits and states
    """
    
    def __init__(self):
        self.visualizations: Dict[str, InteractiveVisualization] = {}
        self.visualization_cache = {}
        
        # Statistics tracking
        self.stats = VisualizationStats()
        
        print("🎨 QuantumVisualizationEngine initialized")
        print("📊 Supporting interactive quantum visualizations")
    
    def create_interactive_circuit_diagram(self, 
                                         circuit: cirq.Circuit,
                                         title: str = "Quantum Circuit",
                                         interactive_features: List[str] = None) -> InteractiveVisualization:
        """
        Create interactive quantum circuit diagram
        
        Args:
            circuit: Quantum circuit to visualize
            title: Visualization title
            interactive_features: List of interactive features to enable
            
        Returns:
            Interactive circuit visualization
        """
        if interactive_features is None:
            interactive_features = ['zoom', 'gate_info', 'measurement_preview']
        
        print(f"🎨 Creating interactive circuit diagram: {title}")
        
        # Generate visualization ID
        viz_id = f"circuit_{int(time.time())}_{hash(str(circuit)) % 10000}"
        
        # Analyze circuit
        qubits = list(circuit.all_qubits())
        gates = list(circuit.all_operations())
        
        # Create metadata
        metadata = VisualizationMetadata(
            visualization_id=viz_id,
            creation_time=time.time(),
            qubit_count=len(qubits),
            gate_count=len(gates),
            depth=len(circuit),
            visualization_type=VisualizationType.CIRCUIT_DIAGRAM,
            interactive=True,
            export_formats=['html', 'svg', 'png', 'json']
        )
        
        # Generate HTML content
        html_content = self._generate_circuit_html(circuit, title, interactive_features)
        
        # Generate JavaScript for interactivity
        javascript_code = self._generate_circuit_javascript(circuit, interactive_features)
        
        # Generate CSS styles
        css_styles = self._generate_circuit_css()
        
        # Create visualization object
        visualization = InteractiveVisualization(
            visualization_id=viz_id,
            html_content=html_content,
            javascript_code=javascript_code,
            css_styles=css_styles,
            metadata=metadata,
            circuit_id=viz_id
        )
        
        # Store visualization
        self.visualizations[viz_id] = visualization
        
        # Update statistics
        self.stats.total_visualizations += 1
        self.stats.interactive_visualizations += 1
        self.stats.circuit_diagrams += 1
        
        print(f"✅ Interactive circuit diagram created: {viz_id}")
        return visualization
    
    def create_quantum_state_visualization(self, 
                                         state_vector: np.ndarray,
                                         qubit_labels: List[str] = None,
                                         visualization_type: VisualizationType = VisualizationType.STATE_VECTOR) -> InteractiveVisualization:
        """
        Create quantum state visualization
        
        Args:
            state_vector: Quantum state vector to visualize
            qubit_labels: Labels for qubits
            visualization_type: Type of state visualization
            
        Returns:
            Interactive state visualization
        """
        print(f"🎨 Creating quantum state visualization: {visualization_type.value}")
        
        # Generate visualization ID
        viz_id = f"state_{visualization_type.value}_{int(time.time())}"
        
        # Determine number of qubits
        num_qubits = int(np.log2(len(state_vector)))
        
        if qubit_labels is None:
            qubit_labels = [f"q{i}" for i in range(num_qubits)]
        
        # Create metadata
        metadata = VisualizationMetadata(
            visualization_id=viz_id,
            creation_time=time.time(),
            qubit_count=num_qubits,
            gate_count=0,
            depth=0,
            visualization_type=visualization_type,
            interactive=True,
            export_formats=['html', 'svg', 'json']
        )
        
        # Generate visualization based on type
        if visualization_type == VisualizationType.STATE_VECTOR:
            html_content = self._generate_state_vector_html(state_vector, qubit_labels)
            javascript_code = self._generate_state_vector_javascript(state_vector)
        elif visualization_type == VisualizationType.BLOCH_SPHERE:
            html_content = self._generate_bloch_sphere_html(state_vector, qubit_labels)
            javascript_code = self._generate_bloch_sphere_javascript(state_vector)
        else:
            html_content = self._generate_generic_state_html(state_vector, qubit_labels)
            javascript_code = self._generate_generic_state_javascript(state_vector)
        
        css_styles = self._generate_state_css()
        
        # Create visualization object
        visualization = InteractiveVisualization(
            visualization_id=viz_id,
            html_content=html_content,
            javascript_code=javascript_code,
            css_styles=css_styles,
            metadata=metadata
        )
        
        # Store visualization
        self.visualizations[viz_id] = visualization
        
        # Update statistics
        self.stats.total_visualizations += 1
        self.stats.interactive_visualizations += 1
        self.stats.state_visualizations += 1
        
        print(f"✅ Quantum state visualization created: {viz_id}")
        return visualization
    
    def create_entanglement_graph_visualization(self, 
                                              entanglement_data: Dict[Tuple[int, int], float],
                                              qubit_positions: Dict[int, Tuple[float, float]] = None) -> InteractiveVisualization:
        """
        Create entanglement graph visualization
        
        Args:
            entanglement_data: Dictionary mapping qubit pairs to entanglement strength
            qubit_positions: Optional positions for qubits in visualization
            
        Returns:
            Interactive entanglement graph visualization
        """
        print("🔗 Creating entanglement graph visualization")
        
        # Generate visualization ID
        viz_id = f"entanglement_{int(time.time())}"
        
        # Extract qubits from entanglement data
        qubits = set()
        for (q1, q2) in entanglement_data.keys():
            qubits.add(q1)
            qubits.add(q2)
        qubits = sorted(list(qubits))
        
        # Generate default positions if not provided
        if qubit_positions is None:
            qubit_positions = self._generate_circular_layout(qubits)
        
        # Create metadata
        metadata = VisualizationMetadata(
            visualization_id=viz_id,
            creation_time=time.time(),
            qubit_count=len(qubits),
            gate_count=len(entanglement_data),
            depth=1,
            visualization_type=VisualizationType.ENTANGLEMENT_GRAPH,
            interactive=True,
            export_formats=['html', 'svg', 'json']
        )
        
        # Generate HTML content
        html_content = self._generate_entanglement_graph_html(entanglement_data, qubit_positions)
        
        # Generate JavaScript for interactivity
        javascript_code = self._generate_entanglement_graph_javascript(entanglement_data, qubit_positions)
        
        # Generate CSS styles
        css_styles = self._generate_entanglement_graph_css()
        
        # Create visualization object
        visualization = InteractiveVisualization(
            visualization_id=viz_id,
            html_content=html_content,
            javascript_code=javascript_code,
            css_styles=css_styles,
            metadata=metadata
        )
        
        # Store visualization
        self.visualizations[viz_id] = visualization
        
        # Update statistics
        self.stats.total_visualizations += 1
        self.stats.interactive_visualizations += 1
        
        print(f"✅ Entanglement graph visualization created: {viz_id}")
        return visualization
    
    def create_measurement_histogram(self, 
                                   measurement_results: Dict[str, int],
                                   title: str = "Measurement Results") -> InteractiveVisualization:
        """
        Create interactive measurement histogram
        
        Args:
            measurement_results: Dictionary mapping measurement outcomes to counts
            title: Histogram title
            
        Returns:
            Interactive histogram visualization
        """
        print(f"📊 Creating measurement histogram: {title}")
        
        # Generate visualization ID
        viz_id = f"histogram_{int(time.time())}"
        
        # Determine number of qubits from measurement results
        if measurement_results:
            max_outcome = max(measurement_results.keys(), key=len)
            num_qubits = len(max_outcome)
        else:
            num_qubits = 1
        
        # Create metadata
        metadata = VisualizationMetadata(
            visualization_id=viz_id,
            creation_time=time.time(),
            qubit_count=num_qubits,
            gate_count=0,
            depth=0,
            visualization_type=VisualizationType.MEASUREMENT_HISTOGRAM,
            interactive=True,
            export_formats=['html', 'svg', 'json']
        )
        
        # Generate HTML content
        html_content = self._generate_histogram_html(measurement_results, title)
        
        # Generate JavaScript for interactivity
        javascript_code = self._generate_histogram_javascript(measurement_results)
        
        # Generate CSS styles
        css_styles = self._generate_histogram_css()
        
        # Create visualization object
        visualization = InteractiveVisualization(
            visualization_id=viz_id,
            html_content=html_content,
            javascript_code=javascript_code,
            css_styles=css_styles,
            metadata=metadata
        )
        
        # Store visualization
        self.visualizations[viz_id] = visualization
        
        # Update statistics
        self.stats.total_visualizations += 1
        self.stats.interactive_visualizations += 1
        
        print(f"✅ Measurement histogram created: {viz_id}")
        return visualization
    
    def export_visualization(self, 
                           visualization_id: str,
                           export_format: str = 'html',
                           filename: Optional[str] = None) -> str:
        """
        Export visualization to file
        
        Args:
            visualization_id: Visualization to export
            export_format: Export format ('html', 'svg', 'png', 'json')
            filename: Optional filename for export
            
        Returns:
            Path to exported file or content string
        """
        if visualization_id not in self.visualizations:
            raise ValueError(f"Visualization {visualization_id} not found")
        
        visualization = self.visualizations[visualization_id]
        
        if filename is None:
            filename = f"{visualization_id}.{export_format}"
        
        print(f"💾 Exporting visualization {visualization_id} as {export_format}")
        
        if export_format == 'html':
            content = self._export_as_html(visualization)
        elif export_format == 'svg':
            content = self._export_as_svg(visualization)
        elif export_format == 'json':
            content = self._export_as_json(visualization)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Update statistics
        self.stats.export_count += 1
        
        print(f"✅ Visualization exported: {filename}")
        return content
    
    def _generate_circuit_html(self, 
                             circuit: cirq.Circuit,
                             title: str,
                             interactive_features: List[str]) -> str:
        """Generate HTML content for circuit diagram"""
        qubits = list(circuit.all_qubits())
        operations = list(circuit.all_operations())
        
        # Create SVG representation
        svg_content = self._create_circuit_svg(circuit)
        
        html_template = f"""
        <div class="quantum-circuit-container">
            <h2 class="circuit-title">{title}</h2>
            <div class="circuit-info">
                <span>Qubits: {len(qubits)}</span>
                <span>Gates: {len(operations)}</span>
                <span>Depth: {len(circuit)}</span>
            </div>
            <div class="circuit-svg-container">
                {svg_content}
            </div>
            <div class="circuit-controls">
                {'<button onclick="zoomIn()">Zoom In</button>' if 'zoom' in interactive_features else ''}
                {'<button onclick="zoomOut()">Zoom Out</button>' if 'zoom' in interactive_features else ''}
                {'<button onclick="showGateInfo()">Gate Info</button>' if 'gate_info' in interactive_features else ''}
                {'<button onclick="previewMeasurement()">Preview</button>' if 'measurement_preview' in interactive_features else ''}
            </div>
            <div id="gate-info-panel" class="info-panel" style="display: none;">
                <h3>Gate Information</h3>
                <div id="gate-details"></div>
            </div>
        </div>
        """
        
        return html_template
    
    def _generate_circuit_javascript(self, 
                                   circuit: cirq.Circuit,
                                   interactive_features: List[str]) -> str:
        """Generate JavaScript for circuit interactivity"""
        operations = list(circuit.all_operations())
        
        # Create gate information data
        gate_info = []
        for i, op in enumerate(operations):
            gate_info.append({
                'id': i,
                'gate': str(op.gate),
                'qubits': [str(q) for q in op.qubits],
                'description': self._get_gate_description(op.gate)
            })
        
        javascript_template = f"""
        let currentZoom = 1.0;
        let gateInfoData = {json.dumps(gate_info)};
        
        function zoomIn() {{
            currentZoom *= 1.2;
            updateZoom();
        }}
        
        function zoomOut() {{
            currentZoom /= 1.2;
            updateZoom();
        }}
        
        function updateZoom() {{
            const svg = document.querySelector('.circuit-svg-container svg');
            if (svg) {{
                svg.style.transform = `scale(${{currentZoom}})`;
            }}
        }}
        
        function showGateInfo() {{
            const panel = document.getElementById('gate-info-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }}
        
        function displayGateInfo(gateId) {{
            const gate = gateInfoData[gateId];
            const details = document.getElementById('gate-details');
            details.innerHTML = `
                <p><strong>Gate:</strong> ${{gate.gate}}</p>
                <p><strong>Qubits:</strong> ${{gate.qubits.join(', ')}}</p>
                <p><strong>Description:</strong> ${{gate.description}}</p>
            `;
        }}
        
        function previewMeasurement() {{
            // Simulate measurement preview
            alert('Measurement preview: This would show expected measurement outcomes');
        }}
        
        // Add click handlers to gates
        document.addEventListener('DOMContentLoaded', function() {{
            const gates = document.querySelectorAll('.quantum-gate');
            gates.forEach((gate, index) => {{
                gate.addEventListener('click', () => displayGateInfo(index));
                gate.style.cursor = 'pointer';
            }});
        }});
        """
        
        return javascript_template
    
    def _generate_circuit_css(self) -> str:
        """Generate CSS styles for circuit visualization"""
        return """
        .quantum-circuit-container {
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
        }
        
        .circuit-title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .circuit-info {
            display: flex;
            justify-content: space-around;
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        
        .circuit-svg-container {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 20px;
            overflow: auto;
            text-align: center;
        }
        
        .circuit-controls {
            margin-top: 15px;
            text-align: center;
        }
        
        .circuit-controls button {
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .circuit-controls button:hover {
            background: #0056b3;
        }
        
        .info-panel {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin-top: 15px;
        }
        
        .quantum-gate {
            fill: #007bff;
            stroke: #0056b3;
            stroke-width: 2;
        }
        
        .quantum-gate:hover {
            fill: #0056b3;
        }
        
        .qubit-line {
            stroke: #6c757d;
            stroke-width: 2;
        }
        """
    
    def _create_circuit_svg(self, circuit: cirq.Circuit) -> str:
        """Create SVG representation of quantum circuit"""
        qubits = list(circuit.all_qubits())
        operations = list(circuit.all_operations())
        
        # Calculate dimensions
        width = max(800, len(circuit) * 80 + 200)
        height = len(qubits) * 60 + 100
        
        svg_elements = []
        
        # Add SVG header
        svg_elements.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
        
        # Draw qubit lines
        for i, qubit in enumerate(qubits):
            y = 50 + i * 60
            svg_elements.append(f'<line x1="50" y1="{y}" x2="{width-50}" y2="{y}" class="qubit-line"/>')
            svg_elements.append(f'<text x="20" y="{y+5}" font-family="monospace" font-size="12">{qubit}</text>')
        
        # Draw gates
        moment_x = 100
        for moment in circuit:
            for operation in moment:
                gate_qubits = [qubits.index(q) for q in operation.qubits]
                
                if len(gate_qubits) == 1:
                    # Single qubit gate
                    y = 50 + gate_qubits[0] * 60
                    svg_elements.append(f'<rect x="{moment_x-15}" y="{y-15}" width="30" height="30" class="quantum-gate"/>')
                    svg_elements.append(f'<text x="{moment_x}" y="{y+5}" text-anchor="middle" font-family="monospace" font-size="10" fill="white">{str(operation.gate)[:3]}</text>')
                
                elif len(gate_qubits) == 2:
                    # Two qubit gate
                    y1 = 50 + gate_qubits[0] * 60
                    y2 = 50 + gate_qubits[1] * 60
                    
                    # Control line
                    svg_elements.append(f'<line x1="{moment_x}" y1="{y1}" x2="{moment_x}" y2="{y2}" stroke="#dc3545" stroke-width="3"/>')
                    
                    # Control dot
                    svg_elements.append(f'<circle cx="{moment_x}" cy="{y1}" r="5" fill="#dc3545"/>')
                    
                    # Target gate
                    svg_elements.append(f'<circle cx="{moment_x}" cy="{y2}" r="15" fill="none" stroke="#dc3545" stroke-width="3"/>')
                    svg_elements.append(f'<line x1="{moment_x-10}" y1="{y2}" x2="{moment_x+10}" y2="{y2}" stroke="#dc3545" stroke-width="3"/>')
                    svg_elements.append(f'<line x1="{moment_x}" y1="{y2-10}" x2="{moment_x}" y2="{y2+10}" stroke="#dc3545" stroke-width="3"/>')
            
            moment_x += 80
        
        svg_elements.append('</svg>')
        
        return '\\n'.join(svg_elements)
    
    def _get_gate_description(self, gate) -> str:
        """Get description for quantum gate"""
        gate_descriptions = {
            'H': 'Hadamard gate - creates superposition',
            'X': 'Pauli-X gate - bit flip',
            'Y': 'Pauli-Y gate - bit and phase flip',
            'Z': 'Pauli-Z gate - phase flip',
            'CNOT': 'Controlled-NOT gate - entangling gate',
            'CZ': 'Controlled-Z gate - phase entangling gate',
            'T': 'T gate - π/8 phase rotation',
            'S': 'S gate - π/4 phase rotation'
        }
        
        gate_str = str(gate)
        for key, description in gate_descriptions.items():
            if key in gate_str:
                return description
        
        return f'Quantum gate: {gate_str}'
    
    def _generate_state_vector_html(self, state_vector: np.ndarray, qubit_labels: List[str]) -> str:
        """Generate HTML for state vector visualization"""
        num_qubits = len(qubit_labels)
        
        # Create amplitude bars
        amplitudes_html = []
        for i, amplitude in enumerate(state_vector):
            basis_state = format(i, f'0{num_qubits}b')
            probability = abs(amplitude) ** 2
            phase = np.angle(amplitude)
            
            amplitudes_html.append(f"""
            <div class="amplitude-bar" data-state="{basis_state}">
                <div class="basis-label">|{basis_state}⟩</div>
                <div class="amplitude-visual">
                    <div class="probability-bar" style="width: {probability*100}%"></div>
                    <div class="phase-indicator" style="transform: rotate({phase}rad)"></div>
                </div>
                <div class="amplitude-values">
                    <span>P: {probability:.3f}</span>
                    <span>φ: {phase:.2f}</span>
                </div>
            </div>
            """)
        
        html_template = f"""
        <div class="state-vector-container">
            <h2>Quantum State Vector</h2>
            <div class="qubit-labels">
                Qubits: {', '.join(qubit_labels)}
            </div>
            <div class="amplitudes-container">
                {''.join(amplitudes_html)}
            </div>
            <div class="state-info">
                <p>Total states: {len(state_vector)}</p>
                <p>Normalization: {np.linalg.norm(state_vector):.6f}</p>
            </div>
        </div>
        """
        
        return html_template
    
    def _generate_state_vector_javascript(self, state_vector: np.ndarray) -> str:
        """Generate JavaScript for state vector interactivity"""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            const amplitudeBars = document.querySelectorAll('.amplitude-bar');
            
            amplitudeBars.forEach(bar => {
                bar.addEventListener('mouseenter', function() {
                    const state = this.dataset.state;
                    this.style.backgroundColor = '#e3f2fd';
                    
                    // Show detailed information
                    const tooltip = document.createElement('div');
                    tooltip.className = 'amplitude-tooltip';
                    tooltip.innerHTML = `Basis state: |${state}⟩`;
                    document.body.appendChild(tooltip);
                });
                
                bar.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '';
                    const tooltip = document.querySelector('.amplitude-tooltip');
                    if (tooltip) tooltip.remove();
                });
            });
        });
        """
    
    def _generate_state_css(self) -> str:
        """Generate CSS for state visualizations"""
        return """
        .state-vector-container {
            font-family: Arial, sans-serif;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
        }
        
        .qubit-labels {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .amplitudes-container {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .amplitude-bar {
            display: flex;
            align-items: center;
            padding: 8px;
            border-bottom: 1px solid #dee2e6;
            transition: background-color 0.2s;
        }
        
        .basis-label {
            width: 80px;
            font-family: monospace;
            font-weight: bold;
        }
        
        .amplitude-visual {
            flex: 1;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            position: relative;
            margin: 0 10px;
        }
        
        .probability-bar {
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            border-radius: 10px;
            transition: width 0.3s;
        }
        
        .phase-indicator {
            position: absolute;
            right: 5px;
            top: 50%;
            width: 10px;
            height: 2px;
            background: #dc3545;
            transform-origin: left center;
        }
        
        .amplitude-values {
            width: 120px;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        
        .state-info {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            margin-top: 15px;
            text-align: center;
        }
        """
    
    def _generate_bloch_sphere_html(self, state_vector: np.ndarray, qubit_labels: List[str]) -> str:
        """Generate HTML for Bloch sphere visualization"""
        # For single qubit states only
        if len(state_vector) != 2:
            return "<div>Bloch sphere visualization only available for single qubit states</div>"
        
        # Calculate Bloch vector
        alpha, beta = state_vector[0], state_vector[1]
        
        # Bloch sphere coordinates
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = abs(alpha)**2 - abs(beta)**2
        
        html_template = f"""
        <div class="bloch-sphere-container">
            <h2>Bloch Sphere Visualization</h2>
            <div class="sphere-canvas" id="bloch-sphere">
                <svg width="300" height="300" viewBox="-150 -150 300 300">
                    <!-- Sphere outline -->
                    <circle cx="0" cy="0" r="100" fill="none" stroke="#dee2e6" stroke-width="2"/>
                    
                    <!-- Axes -->
                    <line x1="-120" y1="0" x2="120" y2="0" stroke="#6c757d" stroke-width="1"/>
                    <line x1="0" y1="-120" x2="0" y2="120" stroke="#6c757d" stroke-width="1"/>
                    
                    <!-- Axis labels -->
                    <text x="125" y="5" font-size="12" fill="#6c757d">X</text>
                    <text x="5" y="-125" font-size="12" fill="#6c757d">Z</text>
                    
                    <!-- State vector -->
                    <line x1="0" y1="0" x2="{x*100}" y2="{-z*100}" stroke="#dc3545" stroke-width="3"/>
                    <circle cx="{x*100}" cy="{-z*100}" r="5" fill="#dc3545"/>
                    
                    <!-- Projection lines -->
                    <line x1="{x*100}" y1="{-z*100}" x2="{x*100}" y2="0" stroke="#28a745" stroke-width="1" stroke-dasharray="5,5"/>
                    <line x1="{x*100}" y1="{-z*100}" x2="0" y2="{-z*100}" stroke="#28a745" stroke-width="1" stroke-dasharray="5,5"/>
                </svg>
            </div>
            <div class="bloch-coordinates">
                <p>Bloch Vector: ({x:.3f}, {y:.3f}, {z:.3f})</p>
                <p>|α|² = {abs(alpha)**2:.3f}, |β|² = {abs(beta)**2:.3f}</p>
            </div>
        </div>
        """
        
        return html_template
    
    def _generate_bloch_sphere_javascript(self, state_vector: np.ndarray) -> str:
        """Generate JavaScript for Bloch sphere interactivity"""
        return """
        // Bloch sphere interactivity would be implemented here
        console.log('Bloch sphere visualization loaded');
        """
    
    def _generate_generic_state_html(self, state_vector: np.ndarray, qubit_labels: List[str]) -> str:
        """Generate generic state visualization HTML"""
        return f"""
        <div class="generic-state-container">
            <h2>Quantum State Visualization</h2>
            <p>State vector dimension: {len(state_vector)}</p>
            <p>Number of qubits: {len(qubit_labels)}</p>
            <p>Qubits: {', '.join(qubit_labels)}</p>
        </div>
        """
    
    def _generate_generic_state_javascript(self, state_vector: np.ndarray) -> str:
        """Generate generic state JavaScript"""
        return "console.log('Generic state visualization loaded');"
    
    def _generate_entanglement_graph_html(self, 
                                        entanglement_data: Dict[Tuple[int, int], float],
                                        qubit_positions: Dict[int, Tuple[float, float]]) -> str:
        """Generate HTML for entanglement graph"""
        # Create SVG elements for the graph
        svg_elements = []
        
        # Draw edges (entanglement connections)
        for (q1, q2), strength in entanglement_data.items():
            if q1 in qubit_positions and q2 in qubit_positions:
                x1, y1 = qubit_positions[q1]
                x2, y2 = qubit_positions[q2]
                
                # Line thickness based on entanglement strength
                thickness = max(1, strength * 5)
                opacity = strength
                
                svg_elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#007bff" stroke-width="{thickness}" opacity="{opacity}"/>')
        
        # Draw nodes (qubits)
        for qubit, (x, y) in qubit_positions.items():
            svg_elements.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#28a745" stroke="#1e7e34" stroke-width="2"/>')
            svg_elements.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-family="monospace" font-size="12" fill="white">{qubit}</text>')
        
        html_template = f"""
        <div class="entanglement-graph-container">
            <h2>Quantum Entanglement Graph</h2>
            <div class="graph-svg-container">
                <svg width="400" height="400" viewBox="0 0 400 400">
                    {''.join(svg_elements)}
                </svg>
            </div>
            <div class="entanglement-legend">
                <p>Line thickness indicates entanglement strength</p>
                <p>Total entangled pairs: {len(entanglement_data)}</p>
            </div>
        </div>
        """
        
        return html_template
    
    def _generate_entanglement_graph_javascript(self, 
                                              entanglement_data: Dict[Tuple[int, int], float],
                                              qubit_positions: Dict[int, Tuple[float, float]]) -> str:
        """Generate JavaScript for entanglement graph interactivity"""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            const circles = document.querySelectorAll('circle');
            circles.forEach(circle => {
                circle.addEventListener('mouseenter', function() {
                    this.setAttribute('r', '20');
                    this.style.cursor = 'pointer';
                });
                
                circle.addEventListener('mouseleave', function() {
                    this.setAttribute('r', '15');
                });
            });
        });
        """
    
    def _generate_entanglement_graph_css(self) -> str:
        """Generate CSS for entanglement graph"""
        return """
        .entanglement-graph-container {
            font-family: Arial, sans-serif;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
            text-align: center;
        }
        
        .graph-svg-container {
            background: white;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .entanglement-legend {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
        }
        """
    
    def _generate_histogram_html(self, measurement_results: Dict[str, int], title: str) -> str:
        """Generate HTML for measurement histogram"""
        total_measurements = sum(measurement_results.values())
        
        bars_html = []
        for outcome, count in sorted(measurement_results.items()):
            probability = count / total_measurements if total_measurements > 0 else 0
            height = probability * 200  # Scale to 200px max height
            
            bars_html.append(f"""
            <div class="histogram-bar">
                <div class="bar" style="height: {height}px;" data-outcome="{outcome}" data-count="{count}"></div>
                <div class="bar-label">|{outcome}⟩</div>
                <div class="bar-value">{count}</div>
            </div>
            """)
        
        html_template = f"""
        <div class="histogram-container">
            <h2>{title}</h2>
            <div class="histogram-chart">
                {''.join(bars_html)}
            </div>
            <div class="histogram-info">
                <p>Total measurements: {total_measurements}</p>
                <p>Unique outcomes: {len(measurement_results)}</p>
            </div>
        </div>
        """
        
        return html_template
    
    def _generate_histogram_javascript(self, measurement_results: Dict[str, int]) -> str:
        """Generate JavaScript for histogram interactivity"""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            const bars = document.querySelectorAll('.bar');
            
            bars.forEach(bar => {
                bar.addEventListener('mouseenter', function() {
                    const outcome = this.dataset.outcome;
                    const count = this.dataset.count;
                    
                    this.style.backgroundColor = '#0056b3';
                    
                    // Show tooltip
                    const tooltip = document.createElement('div');
                    tooltip.className = 'histogram-tooltip';
                    tooltip.innerHTML = `Outcome: |${outcome}⟩<br>Count: ${count}`;
                    tooltip.style.position = 'absolute';
                    tooltip.style.background = '#333';
                    tooltip.style.color = 'white';
                    tooltip.style.padding = '5px';
                    tooltip.style.borderRadius = '3px';
                    tooltip.style.fontSize = '12px';
                    document.body.appendChild(tooltip);
                });
                
                bar.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '#007bff';
                    const tooltip = document.querySelector('.histogram-tooltip');
                    if (tooltip) tooltip.remove();
                });
            });
        });
        """
    
    def _generate_histogram_css(self) -> str:
        """Generate CSS for histogram"""
        return """
        .histogram-container {
            font-family: Arial, sans-serif;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
        }
        
        .histogram-chart {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            height: 250px;
            background: white;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .histogram-bar {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 5px;
        }
        
        .bar {
            background: #007bff;
            width: 30px;
            min-height: 5px;
            border-radius: 2px 2px 0 0;
            transition: background-color 0.2s;
            cursor: pointer;
        }
        
        .bar-label {
            margin-top: 5px;
            font-family: monospace;
            font-size: 12px;
        }
        
        .bar-value {
            font-size: 10px;
            color: #6c757d;
        }
        
        .histogram-info {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        """
    
    def _generate_circular_layout(self, qubits: List[int]) -> Dict[int, Tuple[float, float]]:
        """Generate circular layout for qubits"""
        positions = {}
        center_x, center_y = 200, 200
        radius = 150
        
        for i, qubit in enumerate(qubits):
            angle = 2 * np.pi * i / len(qubits)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions[qubit] = (x, y)
        
        return positions
    
    def _export_as_html(self, visualization: InteractiveVisualization) -> str:
        """Export visualization as complete HTML file"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Visualization - {visualization.visualization_id}</title>
            <style>
                {visualization.css_styles}
            </style>
        </head>
        <body>
            {visualization.html_content}
            <script>
                {visualization.javascript_code}
            </script>
        </body>
        </html>
        """
    
    def _export_as_svg(self, visualization: InteractiveVisualization) -> str:
        """Export visualization as SVG"""
        # Extract SVG content from HTML
        # This is a simplified implementation
        return f"<svg><!-- SVG export for {visualization.visualization_id} --></svg>"
    
    def _export_as_json(self, visualization: InteractiveVisualization) -> str:
        """Export visualization metadata as JSON"""
        return json.dumps({
            'visualization_id': visualization.visualization_id,
            'metadata': asdict(visualization.metadata),
            'created_at': time.ctime(visualization.metadata.creation_time)
        }, indent=2)
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get visualization engine statistics"""
        return {
            'total_visualizations': self.stats.total_visualizations,
            'interactive_visualizations': self.stats.interactive_visualizations,
            'circuit_diagrams': self.stats.circuit_diagrams,
            'state_visualizations': self.stats.state_visualizations,
            'export_count': self.stats.export_count,
            'debug_sessions': self.stats.debug_sessions,
            'available_visualizations': list(self.visualizations.keys())
        }

if __name__ == "__main__":
    # Test the Quantum Visualization Engine
    print("🧪 Testing Quantum Visualization Engine")
    
    engine = QuantumVisualizationEngine()
    
    # Create test circuit
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.measure(*qubits, key='result')
    )
    
    # Test circuit visualization
    circuit_viz = engine.create_interactive_circuit_diagram(circuit, "Test GHZ Circuit")
    print(f"✅ Circuit visualization created: {circuit_viz.visualization_id}")
    
    # Test state visualization
    state_vector = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
    state_viz = engine.create_quantum_state_visualization(state_vector, ['q0', 'q1', 'q2'])
    print(f"✅ State visualization created: {state_viz.visualization_id}")
    
    # Test entanglement graph
    entanglement_data = {(0, 1): 0.8, (1, 2): 0.9, (0, 2): 0.3}
    entanglement_viz = engine.create_entanglement_graph_visualization(entanglement_data)
    print(f"✅ Entanglement graph created: {entanglement_viz.visualization_id}")
    
    # Test measurement histogram
    measurement_results = {'000': 45, '111': 55}
    histogram_viz = engine.create_measurement_histogram(measurement_results)
    print(f"✅ Histogram created: {histogram_viz.visualization_id}")
    
    # Test export
    html_export = engine.export_visualization(circuit_viz.visualization_id, 'html')
    print(f"✅ HTML export completed: {len(html_export)} characters")
    
    # Get statistics
    stats = engine.get_visualization_stats()
    print(f"📊 Stats: {stats['total_visualizations']} total, {stats['interactive_visualizations']} interactive")
    
