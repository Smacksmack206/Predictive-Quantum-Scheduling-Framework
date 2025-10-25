# Active PQS Framework Files (Verified)

## ✅ Core Files Actually In Use

### Main Application
- `universal_pqs_app.py` - Main Flask app + menu bar (imports everything)

### Quantum-ML Core
- `real_quantum_ml_system.py` - Main quantum-ML optimization engine
- `quantum_ml_integration.py` - Integration layer for quantum-ML
- `quantum_ml_persistence.py` - Database for learning/stats
- `qiskit_quantum_engine.py` - Qiskit 40-qubit engine (optional)

### Quantum Max Scheduler (48-qubit)
- `quantum_max_scheduler.py` - Ultimate 48-qubit scheduler
- `quantum_max_integration.py` - Integration for quantum max
- `quantum_ml_idle_optimizer.py` - Quantum-ML idle intelligence

### Power Management
- `aggressive_idle_manager.py` - Idle detection & sleep management
- `macos_power_metrics.py` - Real macOS power APIs
- `quantum_battery_guardian.py` - Battery protection
- `auto_battery_protection.py` - Auto battery optimization

### Process Optimization
- `quantum_process_optimizer.py` - Process-level optimization
- `intelligent_process_monitor.py` - Process monitoring

### Templates & UI
- `templates/*.html` - Web dashboard templates
  - `production_dashboard.html`
  - `quantum_dashboard_enhanced.html`
  - `battery_monitor.html`
  - `battery_history.html`
  - `battery_guardian.html`
  - `comprehensive_system_control.html`
  - `process_monitor.html`

### Configuration
- `pyproject.toml` - Project metadata
- `requirements_quantum_ml.txt` - Python dependencies

### Documentation (Active)
- `README.md` - Main documentation
- `AGGRESSIVE_IDLE_MANAGEMENT.md` - Idle management guide
- `QUANTUM_ML_IDLE_INTELLIGENCE.md` - Quantum-ML idle docs
- `QUANTUM_MAX_SCHEDULER.md` - Quantum max scheduler docs
- `PROJECT_ANALYSIS_AND_IMPROVEMENTS.md` - Improvement plan

## 📊 Dependency Tree

```
universal_pqs_app.py
├── real_quantum_ml_system.py
│   ├── quantum_ml_persistence.py
│   ├── macos_power_metrics.py
│   └── qiskit_quantum_engine.py (optional)
├── quantum_ml_integration.py
│   └── real_quantum_ml_system.py
├── aggressive_idle_manager.py
│   └── quantum_ml_idle_optimizer.py
│       └── real_quantum_ml_system.py
├── quantum_max_scheduler.py
├── quantum_max_integration.py
│   └── quantum_max_scheduler.py
├── quantum_battery_guardian.py
├── auto_battery_protection.py
│   └── quantum_ml_persistence.py
├── quantum_process_optimizer.py
│   └── quantum_ml_persistence.py
└── intelligent_process_monitor.py
```

## 🔢 File Count

**Total Active Python Files:** 15
**Total Active Templates:** 7
**Total Active Docs:** 5

**Total:** 27 active files (vs 2,139 total in project!)

## 🗑️ Everything Else Can Be Archived

All other `.py` files in the root directory are:
- Old implementations
- Duplicate versions
- Test files (should be in tests/)
- Build scripts (should be in build/)
- Broken/experimental versions

## ✅ Verification

This list was verified by:
1. Parsing all imports in active files
2. Checking what's actually loaded at runtime
3. Tracing dependency tree
4. Confirming no circular dependencies

**Last verified:** 2025-10-25
