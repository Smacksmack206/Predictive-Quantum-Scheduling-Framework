#!/bin/bash
# Create Standalone PQS Framework App
# This creates a self-contained app that works on both Intel and Apple Silicon

set -e

APP_NAME="PQS Framework"
APP_DIR="$APP_NAME.app"
VERSION="1.0.0"

echo "=========================================="
echo "🚀 Creating Standalone PQS Framework App"
echo "=========================================="
echo ""

# Clean previous build
if [ -d "$APP_DIR" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf "$APP_DIR"
fi

# Create app structure
echo "📦 Creating app bundle structure..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
mkdir -p "$APP_DIR/Contents/Frameworks"

# Create Info.plist
echo "📝 Creating Info.plist..."
cat > "$APP_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>pqs_launcher</string>
    <key>CFBundleIconFile</key>
    <string>pqs_icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.pqs.framework</string>
    <key>CFBundleName</key>
    <string>PQS Framework</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>
EOF

# Create launcher script
echo "🚀 Creating launcher script..."
cat > "$APP_DIR/Contents/MacOS/pqs_launcher" << 'EOF'
#!/bin/bash
# PQS Framework Launcher - Universal

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"

# Use system Python (user must have dependencies installed)
# Or use bundled Python if available
if [ -f "$RESOURCES/python/bin/python3" ]; then
    PYTHON="$RESOURCES/python/bin/python3"
else
    PYTHON="python3"
fi

# Set environment
export PYTHONPATH="$RESOURCES:$PYTHONPATH"
export PQS_RESOURCES="$RESOURCES"

# Change to resources directory
cd "$RESOURCES"

# Launch the app
exec "$PYTHON" "$RESOURCES/universal_pqs_app.py" "$@" 2>&1 | logger -t "PQS Framework"
EOF

chmod +x "$APP_DIR/Contents/MacOS/pqs_launcher"

# Copy Python files
echo "📁 Copying Python files..."
cp universal_pqs_app.py "$APP_DIR/Contents/Resources/"
cp native_window.py "$APP_DIR/Contents/Resources/"
cp real_quantum_ml_system.py "$APP_DIR/Contents/Resources/"
cp quantum_ml_integration.py "$APP_DIR/Contents/Resources/"
cp quantum_battery_guardian.py "$APP_DIR/Contents/Resources/"
cp auto_battery_protection.py "$APP_DIR/Contents/Resources/"
cp aggressive_idle_manager.py "$APP_DIR/Contents/Resources/"

# Copy templates and static
echo "📁 Copying templates and static files..."
cp -r templates "$APP_DIR/Contents/Resources/"
cp -r static "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Copy icon
echo "🎨 Copying icon..."
cp pqs_icon.icns "$APP_DIR/Contents/Resources/" 2>/dev/null || echo "⚠️  Icon not found"
cp pqs_icon.png "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Create requirements file in app
echo "📝 Creating requirements.txt..."
cat > "$APP_DIR/Contents/Resources/requirements.txt" << 'EOF'
cirq>=1.3.0
qiskit>=1.0.0
tensorflow-macos>=2.16.0; platform_machine == 'arm64'
tensorflow-metal>=1.1.0; platform_machine == 'arm64'
tensorflow>=2.16.0; platform_machine == 'x86_64'
torch>=2.0.0
numpy>=1.24.0
flask>=3.0.0
rumps>=0.4.0
psutil>=5.9.0
pyobjc-core>=10.0
pyobjc-framework-Cocoa>=10.0
pyobjc-framework-WebKit>=10.0
pillow>=10.0.0
EOF

# Create README in app
cat > "$APP_DIR/Contents/Resources/README.txt" << 'EOF'
PQS Framework - Quantum-ML Power Management System

REQUIREMENTS:
- macOS 15.0 or later
- Python 3.11 or later
- Dependencies listed in requirements.txt

FIRST TIME SETUP:
1. Open Terminal
2. Run: pip3 install -r "/Applications/PQS Framework.app/Contents/Resources/requirements.txt"
3. Launch the app from Applications

The app will work on both Intel and Apple Silicon Macs.
EOF

# Sign the app (ad-hoc)
echo "🔐 Signing app..."
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || echo "⚠️  Signing skipped"

# Create DMG
echo "💿 Creating DMG..."
DMG_NAME="PQS_Framework_v${VERSION}_Universal.dmg"
rm -f "$DMG_NAME"
hdiutil create -volname "PQS Framework" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_NAME" > /dev/null 2>&1

# Get sizes
APP_SIZE=$(du -sh "$APP_DIR" | cut -f1)
DMG_SIZE=$(du -sh "$DMG_NAME" | cut -f1)

echo ""
echo "=========================================="
echo "✅ BUILD COMPLETE!"
echo "=========================================="
echo ""
echo "📦 App Bundle: $APP_DIR ($APP_SIZE)"
echo "💿 Installer: $DMG_NAME ($DMG_SIZE)"
echo ""
echo "🧪 Test the app:"
echo "   open '$APP_DIR'"
echo ""
echo "📤 Distribute:"
echo "   Share $DMG_NAME with users"
echo ""
echo "⚠️  IMPORTANT:"
echo "   Users must install Python dependencies first:"
echo "   pip3 install -r requirements.txt"
echo ""
echo "=========================================="
