# PQS Framework - Universal macOS App

## 🎯 Overview

A **Universal macOS Application** that runs natively on both Intel and Apple Silicon Macs, targeting macOS 15.0 (Sequoia) through macOS 26.0 and beyond.

## ✨ Features

- **Universal Binary**: Single app works on both Intel (x86_64) and Apple Silicon (arm64)
- **Native Performance**: Optimized for each architecture
- **Modern macOS**: Targets macOS 15.0+ with full support for latest features
- **Quantum-ML Integration**: Real quantum computing with ML optimization
- **Native GUI**: Beautiful native macOS window with menu bar integration
- **Auto-Detection**: Automatically uses the correct Python environment for your Mac

## 📦 What's Included

### App Bundle Structure
```
PQS Framework.app/
├── Contents/
│   ├── Info.plist              # App metadata
│   ├── MacOS/
│   │   └── pqs_launcher        # Universal launcher script
│   ├── Resources/
│   │   ├── universal_pqs_app.py
│   │   ├── native_window.py
│   │   ├── real_quantum_ml_system.py
│   │   ├── templates/          # Web dashboard templates
│   │   ├── static/             # Static assets
│   │   ├── pqs_icon.icns       # App icon
│   │   ├── python_arm64/       # Apple Silicon Python
│   │   └── python_x86_64/      # Intel Python
│   └── Frameworks/             # (Reserved for future use)
```

## 🏗️ Building the Universal App

### Prerequisites
- macOS 15.0 or later
- Xcode Command Line Tools
- Python 3.11+
- Both architectures available (for full universal build)

### Build Steps

1. **Build the app bundle**:
```bash
python3 build_universal_app.py
```

2. **Create Python environments** (optional - for standalone distribution):
```bash
./build_python_envs.sh
```

This creates:
- `python_arm64/` - Apple Silicon Python environment
- `python_x86_64/` - Intel Python environment

3. **Copy environments to app** (for standalone distribution):
```bash
cp -r python_arm64 "PQS Framework.app/Contents/Resources/"
cp -r python_x86_64 "PQS Framework.app/Contents/Resources/"
```

4. **Test the app**:
```bash
open "PQS Framework.app"
```

## 📥 Distribution

### DMG Installer
The build process creates: `PQS_Framework_v1.0.0_Universal.dmg`

Users can:
1. Download the DMG
2. Open it
3. Drag "PQS Framework.app" to Applications
4. Launch from Applications or Spotlight

### System Requirements
- **macOS Version**: 15.0 (Sequoia) or later
- **Architecture**: Intel (x86_64) or Apple Silicon (arm64)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 2GB for app + dependencies

## 🚀 How It Works

### Architecture Detection
The launcher script automatically detects your Mac's architecture:

```bash
ARCH=$(uname -m)
# arm64 = Apple Silicon
# x86_64 = Intel
```

### Python Environment Selection
Based on architecture, it uses the appropriate Python:
- **Apple Silicon**: `python_arm64/bin/python3`
- **Intel**: `python_x86_64/bin/python3`

### Fallback Behavior
If bundled Python isn't found, it falls back to system Python:
```bash
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="python3"
fi
```

## 🔧 Advanced Configuration

### Custom Python Environments

To use custom Python builds:

1. Create your environments:
```bash
# For Apple Silicon
arch -arm64 python3 -m venv custom_arm64
arch -arm64 custom_arm64/bin/pip install -r app_requirements.txt

# For Intel
arch -x86_64 python3 -m venv custom_x86_64
arch -x86_64 custom_x86_64/bin/pip install -r app_requirements.txt
```

2. Copy to app:
```bash
cp -r custom_arm64 "PQS Framework.app/Contents/Resources/python_arm64"
cp -r custom_x86_64 "PQS Framework.app/Contents/Resources/python_x86_64"
```

### Code Signing

For distribution outside the App Store:

1. **Get a Developer ID certificate** from Apple Developer Program

2. **Sign the app**:
```bash
codesign --force --deep --sign "Developer ID Application: Your Name" "PQS Framework.app"
```

3. **Notarize** (required for macOS 15+):
```bash
# Create a zip
ditto -c -k --keepParent "PQS Framework.app" "PQS Framework.zip"

# Submit for notarization
xcrun notarytool submit "PQS Framework.zip" \
    --apple-id "your@email.com" \
    --team-id "TEAMID" \
    --password "app-specific-password" \
    --wait

# Staple the ticket
xcrun stapler staple "PQS Framework.app"
```

## 🎨 Customization

### Change App Icon
1. Create a 1024x1024 PNG icon
2. Convert to .icns:
```bash
mkdir MyIcon.iconset
# Create required sizes (16, 32, 64, 128, 256, 512, 1024)
iconutil -c icns MyIcon.iconset -o pqs_icon.icns
```
3. Rebuild the app

### Modify Bundle ID
Edit `build_universal_app.py`:
```python
BUNDLE_ID = "com.yourcompany.pqs"
```

### Update Version
```python
VERSION = "2.0.0"
```

## 🧪 Testing

### Test on Current Architecture
```bash
open "PQS Framework.app"
```

### Test on Specific Architecture (Rosetta 2)
```bash
# Force Intel mode on Apple Silicon
arch -x86_64 open "PQS Framework.app"
```

### Verify Universal Binary
```bash
lipo -info "PQS Framework.app/Contents/MacOS/pqs_launcher"
# Should show: Non-fat file (it's a script, not a binary)

# Check Python environments
ls "PQS Framework.app/Contents/Resources/"
# Should show: python_arm64/ and python_x86_64/
```

## 📊 Performance

### Apple Silicon (M1/M2/M3)
- Native arm64 execution
- Metal GPU acceleration
- Neural Engine support
- 40-qubit quantum simulation

### Intel (x86_64)
- Native x86_64 execution
- AVX2 optimization
- 20-qubit quantum simulation
- Full compatibility mode

## 🐛 Troubleshooting

### App Won't Open
```bash
# Check permissions
xattr -d com.apple.quarantine "PQS Framework.app"

# Check signature
codesign -vvv "PQS Framework.app"
```

### Python Not Found
```bash
# Verify Python environments exist
ls "PQS Framework.app/Contents/Resources/python_*/bin/python3"

# Check launcher script
cat "PQS Framework.app/Contents/MacOS/pqs_launcher"
```

### Architecture Mismatch
```bash
# Check current architecture
uname -m

# Force specific architecture
arch -arm64 open "PQS Framework.app"  # Apple Silicon
arch -x86_64 open "PQS Framework.app" # Intel/Rosetta
```

## 📝 License

Copyright © 2025 PQS Framework. All rights reserved.

## 🤝 Contributing

To contribute to the universal app build:

1. Fork the repository
2. Make changes to `build_universal_app.py`
3. Test on both Intel and Apple Silicon
4. Submit a pull request

## 🔮 Future Enhancements

- [ ] Automatic updates via Sparkle framework
- [ ] Sandboxing for App Store distribution
- [ ] XCTest integration for automated testing
- [ ] CI/CD pipeline for universal builds
- [ ] Localization support (i18n)
- [ ] Plugin architecture for extensions

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review console logs: `Console.app` → Filter: "PQS"
- Check system requirements

---

**Built with ❤️ for the quantum computing community**
