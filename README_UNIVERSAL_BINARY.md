# PQS Framework 48-Qubit - macOS Universal Binary

## Overview

This document describes the universal macOS binary build for the PQS Framework 48-Qubit, which supports both Intel and Apple Silicon Macs running macOS 15.0+ (Sequoia and later).

## Architecture Support

- **Intel Macs**: x86_64 architecture (i3, i5, i7, i9 processors)
- **Apple Silicon**: arm64 architecture (M1, M2, M3, M4 chips)
- **macOS Support**: 15.0+ (Sequoia and later)
- **Universal Binary**: Single app bundle that runs natively on both architectures

## Features

### Universal Compatibility
- Automatic architecture detection at runtime
- Architecture-specific optimizations
- Seamless experience across all supported Macs

### Apple Silicon Optimizations
- 40-qubit quantum circuits (M1/M2/M3/M4)
- Neural Engine acceleration for ML
- Metal GPU acceleration for quantum simulation
- Unified memory optimization

### Intel Optimizations
- 20-qubit quantum circuits (i3) / 30-qubit (i5/i7/i9)
- CPU-friendly optimizations for 2020 i3 MacBook Air
- Thermal management for systems without fans
- Memory-efficient algorithms for 8GB systems

## Build Requirements

### System Requirements
- macOS (for building universal binaries)
- Python 3.8+
- Xcode command line tools
- PyInstaller

### Dependencies
- rumps (menu bar app framework)
- psutil (system monitoring)
- flask (web interface)
- numpy (numerical computing)
- All quantum ML dependencies

## Build Process

### Quick Build
```bash
# Build universal binary and DMG package
python3 build_universal_binary.py

# Build only (skip DMG creation)
python3 build_universal_binary.py --build-only

# Skip dependency installation
python3 build_universal_binary.py --skip-dependencies
```

### Manual Build Steps

1. **Install Dependencies**
   ```bash
   pip install pyinstaller
   pip install -r requirements_quantum_ml.txt
   ```

2. **Create Entitlements**
   ```bash
   python3 build_universal_binary.py  # Creates entitlements.plist
   ```

3. **Build Universal Binary**
   ```bash
   pyinstaller --target-arch universal2 UniversalPQS.spec
   ```

4. **Verify Binary**
   ```bash
   lipo -info dist/UniversalPQS.app/Contents/MacOS/UniversalPQS
   ```

5. **Create DMG Package**
   ```bash
   # Using create-dmg (recommended)
   create-dmg --volname "UniversalPQS Universal" dist/UniversalPQS_Universal_1.0.0.dmg dist/UniversalPQS.app

   # Or using hdiutil
   hdiutil create -volname "UniversalPQS Universal" -srcfolder dist -ov -format UDZO dist/UniversalPQS_Universal_1.0.0.dmg
   ```

## Distribution

### App Bundle Structure
```
UniversalPQS.app/
├── Contents/
│   ├── Info.plist          # App metadata and configuration
│   ├── MacOS/
│   │   └── UniversalPQS    # Universal binary executable
│   ├── Resources/
│   │   └── pqs-icon.icns   # App icon
│   ├── _CodeSignature/     # Code signature (if signed)
│   └── Frameworks/         # Embedded frameworks
├── templates/              # Web templates
├── static/                 # Web static files
└── [other app resources]
```

### DMG Package
- Volume name: "UniversalPQS Universal"
- Format: UDZO (compressed)
- Contains app bundle and installation instructions
- Drag-and-drop installation

## Runtime Behavior

### Architecture Detection
The app automatically detects the system architecture and applies appropriate optimizations:

- **Apple Silicon (M1/M2/M3/M4)**: Maximum performance mode with Neural Engine and Metal GPU acceleration
- **Intel i3 (2020 MacBook Air)**: CPU-friendly mode with reduced resource usage
- **Intel i5/i7/i9**: Standard optimization mode

### Performance Characteristics

| Architecture | Quantum Circuits | ML Acceleration | Power Efficiency |
|-------------|-----------------|-----------------|-------------------|
| Apple M4    | 40 qubits       | Neural Engine   | Maximum          |
| Apple M3    | 40 qubits       | Neural Engine   | Maximum          |
| Apple M2    | 40 qubits       | Neural Engine   | High             |
| Apple M1    | 40 qubits       | Neural Engine   | High             |
| Intel i9    | 30 qubits       | CPU optimized   | Standard         |
| Intel i7    | 30 qubits       | CPU optimized   | Standard         |
| Intel i5    | 30 qubits       | CPU optimized   | Standard         |
| Intel i3    | 20 qubits       | CPU friendly    | Optimized        |

## Installation

### For Users
1. Download the DMG file
2. Double-click to mount
3. Drag `UniversalPQS.app` to Applications folder
4. Launch from Applications or Spotlight

### For Developers
1. Clone the repository
2. Run `python3 build_universal_binary.py`
3. Distribute the generated DMG file

## Troubleshooting

### Build Issues

**PyInstaller not found:**
```bash
pip install pyinstaller
```

**Missing Xcode tools:**
```bash
xcode-select --install
```

**Architecture verification fails:**
- Ensure both Intel and Apple Silicon builds are available
- Check that lipo command is available
- Verify the build completed successfully

### Runtime Issues

**App won't start:**
- Check macOS version compatibility (15.0+ required)
- Verify all dependencies are included in the bundle
- Check Console.app for crash logs

**Performance issues:**
- The app automatically adapts to system capabilities
- Check system resources (CPU, memory, disk)
- Verify architecture-specific optimizations are active

## Code Signing (Optional)

For distribution outside the Mac App Store:

1. **Get Developer ID Certificate**
   - Enroll in Apple Developer Program
   - Create Developer ID Application certificate

2. **Sign the App**
   ```bash
   codesign --deep --force --verify --verbose --sign "Developer ID Application" dist/UniversalPQS.app
   ```

3. **Verify Signature**
   ```bash
   codesign --verify --deep --verbose dist/UniversalPQS.app
   spctl --assess --type execute dist/UniversalPQS.app
   ```

4. **Notarize (for Gatekeeper)**
   ```bash
   xcrun notarytool submit dist/UniversalPQS_Universal_1.0.0.dmg --keychain-profile "notarytool-profile"
   ```

## Version History

- **v1.0.0**: Initial universal binary release
  - Support for macOS 15.0+
  - Universal binary for Intel + Apple Silicon
  - Architecture-specific optimizations
  - Quantum ML integration

## Support

For issues and questions:
- Check the troubleshooting section above
- Verify system requirements are met
- Test on supported macOS versions only
- Report issues with system specifications

## Technical Details

### Universal Binary Implementation
- Uses PyInstaller's `universal2` target architecture
- Single executable containing both x86_64 and arm64 code
- Runtime architecture detection for optimal performance
- Architecture-specific resource allocation

### Memory Management
- Apple Silicon: Unified memory optimization
- Intel i3: Memory-efficient algorithms for 8GB systems
- Dynamic resource allocation based on system capabilities
- Automatic cleanup and optimization

### Performance Optimization
- Real-time system monitoring
- Adaptive algorithm selection
- Background optimization processes
- Thermal management for all architectures
