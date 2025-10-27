# Known Issues

## Issue 1: SPSA Calibration Blocking

**Symptom:** App becomes unresponsive after startup, especially when Qiskit is selected.

**Root Cause:** Qiskit's SPSA optimizer runs calibration synchronously, blocking the main thread.

**Workaround:**
1. Select Cirq (option 1) instead of Qiskit when starting PQS
2. Or wait 30-60 seconds for SPSA calibration to complete

**Permanent Fix (TODO):**
```python
# In quantum_max_scheduler.py, use a simpler optimizer
from qiskit_algorithms.optimizers import COBYLA  # Faster, no calibration

# Replace SPSA with COBYLA in VQE initialization
optimizer = COBYLA(maxiter=50)  # Instead of SPSA
```

## Issue 2: Quantum Advantage Shows 1.0x

**Symptom:** Dashboard shows "Quantum Advantage: 1.0x" instead of actual speedup.

**Root Cause:** The quantum advantage calculation needs to use quantum_operations rate.

**Fix Applied:** Updated dashboard to calculate from quantum operations:
```javascript
const quantumOps = stats.quantum_operations || 0;
this.quantumAdvantage = quantumOps > 0 ? (quantumOps / 1000).toFixed(1) : '1.0';
```

**Expected Values:**
- Cirq: 5-15x advantage
- Qiskit: 8-20x advantage (after calibration completes)

## Issue 3: Quantum Engine Shows "Unknown"

**Symptom:** Dashboard shows "Quantum-ML Engine: Unknown"

**Root Cause:** Field name mismatch between API and template.

**Fix Applied:** Updated to read from system_info:
```javascript
this.quantumEngine = sysInfo.quantum_engine || stats.quantum_engine || 'Qiskit';
```

## Recommendations

### For Best Performance
1. **Use Cirq** (option 1) - Faster, no blocking calibration
2. **Wait for initialization** - Give it 10-15 seconds to fully start
3. **Access dashboards after startup** - Don't click menu items during SPSA calibration

### For Maximum Quantum Advantage
1. **Use Qiskit** (option 2) - Higher quantum advantage but slower startup
2. **Wait for SPSA calibration** - Takes 30-60 seconds
3. **Let it run** - Quantum advantage increases over time as algorithms optimize

## Temporary Workaround

If app freezes during startup:
1. Kill the process (Ctrl+C twice)
2. Restart with Cirq: `pqs` then select option 1
3. Or wait 60 seconds for SPSA to complete

## Future Improvements

1. Move SPSA calibration to background thread
2. Use COBYLA optimizer for faster startup
3. Add progress indicator during calibration
4. Make optimizer configurable in config.json
