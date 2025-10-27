#!/bin/bash
# Build Working Standalone PQS Framework App
# Uses system Python but bundles all Python dependencies

set -e

APP_NAME="PQS Framework"
APP_DIR="$APP_NAME.app"
VERSION="1.0.0"

echo "=========================================="
echo "🚀 Building Standalone PQS Framework"
echo "=========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Please activate your virtual environment first:"
    echo "   source quantum_ml_311/bin/activate"
    exit 1
fi

# Clean previous build
rm -rf "$APP_DIR" 2>/dev/null || true

# Create app structure
echo "📦 Creating app bundle..."
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources/lib"

# Create Info.plist
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
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Copy all site-packages (dependencies)
echo "📚 Bundling Python dependencies..."
SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.11/site-packages"
if [ -d "$SITE_PACKAGES" ]; then
    cp -R "$SITE_PACKAGES"/* "$APP_DIR/Contents/Resources/lib/" 2>/dev/null || true
    echo "   ✓ Dependencies bundled"
else
    echo "   ⚠️  Site-packages not found"
fi

# Create launcher
cat > "$APP_DIR/Contents/MacOS/pqs_launcher" << 'EOF'
#!/bin/bash
# PQS Framework Launcher

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"

# Find Python 3
PYTHON=""
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v $cmd &> /dev/null; then
        PYTHON=$(command -v $cmd)
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Python Required" message "PQS Framework requires Python 3.11 or later.\n\nInstall with:\nbrew install python@3.11\n\nOr download from python.org" as critical'
    exit 1
fi

# Set Python path to use bundled dependencies
export PYTHONPATH="$RESOURCES:$RESOURCES/lib:$PYTHONPATH"
export PQS_RESOURCES="$RESOURCES"

# Launch
cd "$RESOURCES"
exec "$PYTHON" "$RESOURCES/universal_pqs_app.py" "$@" 2>&1
EOF

chmod +x "$APP_DIR/Contents/MacOS/pqs_launcher"

# Copy application files
echo "📁 Copying application files..."
cp universal_pqs_app.py "$APP_DIR/Contents/Resources/"
cp native_window.py "$APP_DIR/Contents/Resources/"
cp real_quantum_ml_system.py "$APP_DIR/Contents/Resources/"
cp quantum_ml_integration.py "$APP_DIR/Contents/Resources/"
cp quantum_battery_guardian.py "$APP_DIR/Contents/Resources/"
cp auto_battery_protection.py "$APP_DIR/Contents/Resources/"
cp aggressive_idle_manager.py "$APP_DIR/Contents/Resources/"

# Copy templates and static
cp -r templates "$APP_DIR/Contents/Resources/"
cp -r static "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Copy icon
cp pqs_icon.icns "$APP_DIR/Contents/Resources/" 2>/dev/null || true
cp pqs_icon.png "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Sign
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true

# Create DMG
echo "💿 Creating DMG..."
DMG_NAME="PQS_Framework_v${VERSION}_Standalone.dmg"
rm -f "$DMG_NAME"

hdiutil create -volname "PQS Framework" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG_NAME" > /dev/null 2>&1

APP_SIZE=$(du -sh "$APP_DIR" | cut -f1)
DMG_SIZE=$(du -sh "$DMG_NAME" | cut -f1)

echo ""
echo "=========================================="
echo "✅ BUILD COMPLETE!"
echo "=========================================="
echo ""
echo "📦 App: $APP_DIR ($APP_SIZE)"
echo "💿 DMG: $DMG_NAME ($DMG_SIZE)"
echo ""
echo "🎯 This app bundles all Python dependencies"
echo "   Users need: Python 3.11+ (system Python)"
echo ""
echo "🧪 Test: open '$APP_DIR'"
echo "=========================================="
