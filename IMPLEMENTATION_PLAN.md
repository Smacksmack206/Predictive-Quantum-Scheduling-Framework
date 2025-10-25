# PQS Framework - Implementation Plan
## Code Quality + Modern UI (No Electron Overhead)

## Overview

This plan adds:
1. Type hints + docstrings (code quality)
2. Configuration system (flexibility)
3. Modern UI using Alpine.js + Tailwind (Electron-quality, zero overhead)
4. Error handling improvements
5. Testing framework

**Timeline:** 2-4 weeks
**Approach:** Incremental, non-breaking changes

---

## Phase 1: Code Quality Foundation (Week 1)

### 1.1 Type Hints & Docstrings

**Goal:** Add type safety without breaking existing code

**Implementation:**
```python
# File: real_quantum_ml_system.py
# Add at top
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Before
def run_comprehensive_optimization(self, system_state):
    # ...
    return result

# After
def run_comprehensive_optimization(
    self, 
    system_state: SystemState
) -> OptimizationResult:
    """
    Run comprehensive quantum-ML optimization.
    
    Args:
        system_state: Current system metrics (CPU, memory, battery)
        
    Returns:
        OptimizationResult with energy savings and quantum advantage
        
    Raises:
        QuantumOptimizationError: If quantum engine fails
        
    Example:
        >>> result = system.run_comprehensive_optimization(state)
        >>> print(f"Saved {result.energy_saved}%")
    """
    # ... existing code unchanged
    return result
```

**Files to Update:**
- `real_quantum_ml_system.py` (main priority)
- `aggressive_idle_manager.py`
- `quantum_max_scheduler.py`
- `quantum_ml_integration.py`

**Effort:** 1-2 days
**Risk:** Very low (type hints are optional in Python)

### 1.2 Configuration System

**Goal:** Centralized config without hardcoded values

**New File:** `config.py`


```python
"""
PQS Framework Configuration
Centralized settings with validation
"""
from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

@dataclass
class QuantumConfig:
    engine: str = 'qiskit'  # or 'cirq'
    max_qubits: int = 48
    optimization_interval: int = 10  # seconds
    
@dataclass
class IdleConfig:
    suspend_delay: int = 30  # seconds
    sleep_delay: int = 120
    cpu_idle_threshold: float = 5.0  # percent
    
@dataclass
class BatteryConfig:
    critical_threshold: int = 20  # percent
    low_threshold: int = 40
    aggressive_mode: bool = True

@dataclass
class PQSConfig:
    quantum: QuantumConfig
    idle: IdleConfig
    battery: BatteryConfig
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'PQSConfig':
        """Load config from YAML or use defaults"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        return cls.default()
    
    @classmethod
    def default(cls) -> 'PQSConfig':
        """Default configuration"""
        return cls(
            quantum=QuantumConfig(),
            idle=IdleConfig(),
            battery=BatteryConfig()
        )
    
    def save(self, config_path: Path):
        """Save config to YAML"""
        # Implementation
        pass

# Global config instance
config = PQSConfig.default()
```

**Config File:** `config.yaml`
```yaml
quantum:
  engine: qiskit
  max_qubits: 48
  optimization_interval: 10

idle:
  suspend_delay: 30
  sleep_delay: 120
  cpu_idle_threshold: 5.0

battery:
  critical_threshold: 20
  low_threshold: 40
  aggressive_mode: true
```

**Integration:**
```python
# In universal_pqs_app.py
from config import config

# Replace hardcoded values
# Before: idle_threshold = 120
# After: idle_threshold = config.idle.sleep_delay
```

**Effort:** 1 day
**Risk:** Low (defaults ensure backward compatibility)

---

## Phase 2: Modern UI (Week 2)

### 2.1 Technology Stack (No Electron!)

**Choice:** Alpine.js + Tailwind CSS + Flask

**Why:**
- ✅ **Zero overhead** (just HTML/CSS/JS)
- ✅ **Electron-quality UI** (modern, reactive)
- ✅ **Tiny size** (~50KB vs 100MB Electron)
- ✅ **Native performance** (no Chromium)
- ✅ **Easy integration** (works with Flask)

**Dependencies:**
```html
<!-- Add to templates/base.html -->
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script>
```

### 2.2 New Base Template

**File:** `templates/base_modern.html`


```html
<!DOCTYPE html>
<html lang="en" x-data="{ darkMode: true }" :class="{ 'dark': darkMode }">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}PQS Framework{% endblock %}</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        quantum: {
                          50: '#f0f9ff',
                          500: '#0ea5e9',
                          600: '#0284c7',
                          700: '#0369a1',
                        }
                    }
                }
            }
        }
    </script>
    
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js"></script>
    
    <!-- Chart.js for graphs -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        /* Custom animations */
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(14, 165, 233, 0.5); }
            50% { box-shadow: 0 0 40px rgba(14, 165, 233, 0.8); }
        }
        .quantum-glow {
            animation: pulse-glow 2s ease-in-out infinite;
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <!-- Navigation -->
    <nav class="bg-gray-800 border-b border-gray-700">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center">
                    <span class="text-2xl font-bold text-quantum-500">⚛️ PQS</span>
                    <div class="ml-10 flex items-baseline space-x-4">
                        <a href="/" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">Dashboard</a>
                        <a href="/quantum" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">Quantum</a>
                        <a href="/battery-monitor" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">Battery</a>
                        <a href="/system-control" class="px-3 py-2 rounded-md text-sm font-medium hover:bg-gray-700">Control</a>
                    </div>
                </div>
                
                <!-- Status indicator -->
                <div x-data="statusWidget()" class="flex items-center space-x-4">
                    <div class="flex items-center space-x-2">
                        <div class="w-2 h-2 rounded-full bg-green-500 quantum-glow"></div>
                        <span class="text-sm" x-text="status"></span>
                    </div>
                    <span class="text-sm text-gray-400" x-text="energySaved"></span>
                </div>
            </div>
        </div>
    </nav>
    
    <!-- Main content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {% block content %}{% endblock %}
    </main>
    
    <!-- Alpine.js components -->
    <script>
        function statusWidget() {
            return {
                status: 'Optimizing...',
                energySaved: '0%',
                
                init() {
                    this.fetchStatus();
                    setInterval(() => this.fetchStatus(), 5000);
                },
                
                async fetchStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        this.status = data.status || 'Active';
                        this.energySaved = `${(data.energy_saved || 0).toFixed(1)}% saved`;
                    } catch (error) {
                        console.error('Status fetch error:', error);
                    }
                }
            }
        }
    </script>
    
    {% block scripts %}{% endblock %}
</body>
</html>
```

### 2.3 Modern Dashboard Component

**File:** `templates/dashboard_modern.html`
