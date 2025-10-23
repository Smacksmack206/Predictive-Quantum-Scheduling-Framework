# PQS Framework 40-Qubit - macOS Build Instructions

## Overview

This document describes how to build a complete macOS application bundle with ALL quantum-ML dependencies included.

## Build Method

We use a custom build script instead of Briefcase to ensure ALL dependencies are properly bundled:

- ✅ Real Quantum-ML System (fully integrated)
- ✅ Cirq quantum simulation library
- ✅ PyTorch ML models
- ✅ TensorFlow-macOS with Metal GPU acceleration
- ✅ NumPy for quantum calculations
- ✅ Flask web dashboard
- ✅ Rumps menu bar integration
- ✅ All templates and static files

## Prerequisites

1. **macOS 10.15 or later**
2. **Python 3.11 or later**
3. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

## Build Steps

### Step 1: Build the App Bundle

```bash
./build_macos_app.sh
```

This script will:
1. Create the app bundle structure
2. Copy all application code
3. Install ALL dependencies in a bundled virtual environment
4. Create launcher scripts
5. Generate Info.plist with proper permissions

**Build time**: 5-10 minutes (depending on internet speed for dependencies)

**Output**: `dist_macos/PQS Framework 40-Qubit.app`

### Step 2: Test the App

```bash
open "dist_macos/PQS Framework 40-Qubit.app"
```

The app should:
- Appear in your menu bar
- Show system detection (Apple Silicon or Intel)
- Start the quantum-ML optimization loop
- Launch the web dashboard at http://localhost:5002

### Step 3: Create DMG Installer

```bash
./create_dmg.sh
```

This creates a distributable DMG file with:
- The app bundle
- Applications folder symlink for easy installation
- README with instructions

**Output**: `dist_macos/PQS-Framework-40Qubit-v4.0.0-macOS.dmg`

## What's Included

### Core Application
- `src/pqs_framework/universal_pqs_app.py` - Main application with architecture detection
- `real_quantum_ml_system.py` - Real quantum-ML hybrid system
- `templates/` - Web dashboard templates
- `static/` - CSS and JavaScript files

### Dependencies (ALL Bundled)
- **Quantum**: cirq>=1.6.0
- **ML**: torch>=2.0.0, tensorflow-macos, tensorflow-metal
- **System**: psutil>=7.0.0, numpy>=2.0.0
- **UI**: rumps>=0.4.0, flask>=3.1.0, pyobjc-framework-Cocoa>=11.0

### Features Preserved
- ✅ 20-qubit quantum simulation
- ✅ Real-time ML predictions
- ✅ Apple Silicon GPU acceleration (Metal)
- ✅ Intel Mac compatibility
- ✅ Universal binary support
- ✅ Menu bar integration
- ✅ Web dashboard
- ✅ Background optimization loop
- ✅ System metrics monitoring

## App Bundle Structure

```
PQS Framework 40-Qubit.app/
├── Contents/
│   ├── MacOS/
│   │   └── PQS Framework 40-Qubit (launcher script)
│   ├── Resources/
│   │   ├── venv/ (bundled Python environment with ALL dependencies)
│   │   ├── pqs_framework/ (application code)
│   │   ├── real_quantum_ml_system.py
│   │   ├── templates/
│   │   └── static/
│   ├── Info.plist
│   └── PkgInfo
```

## Distribution

The DMG file can be distributed to users. Installation is simple:
1. Open the DMG
2. Drag the app to Applications
3. Right-click and "Open" to bypass Gatekeeper
4. Grant system permissions when prompted

## Troubleshooting

### Build Fails
- Ensure Python 3.11+ is installed: `python3 --version`
- Check internet connection (needed for pip installs)
- Try cleaning: `rm -rf build_macos dist_macos`

### App Won't Launch
- Check Console.app for error messages
- Verify permissions: `ls -la "dist_macos/PQS Framework 40-Qubit.app/Contents/MacOS"`
- Test launcher directly: `"dist_macos/PQS Framework 40-Qubit.app/Contents/MacOS/PQS Framework 40-Qubit"`

### Missing Dependencies
- The build script installs ALL dependencies
- If a dependency is missing, check the build log
- Manually install in the bundled venv if needed

## Verification

After building, verify all features work:

```bash
# Test the app
open "dist_macos/PQS Framework 40-Qubit.app"

# Check the dashboard
open http://localhost:5002

# View logs
tail -f ~/Library/Logs/PQS-Framework/app.log
```

## Notes

- **Universal Binary**: The app works on both Intel and Apple Silicon Macs
- **Self-Contained**: All dependencies are bundled, no external installations needed
- **Real Quantum-ML**: The full quantum-ML system is integrated and enabled by default
- **No Features Removed**: All quantum simulation, ML, and optimization features are preserved

## Version

- **App Version**: 4.0.0
- **Build Date**: 2025-10-17
- **Quantum-ML**: Fully Integrated
- **Architecture Support**: Universal (Intel + Apple Silicon)
