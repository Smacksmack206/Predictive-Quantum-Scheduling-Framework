#!/bin/bash
# Build Final Working PQS Framework App
# This will work when users double-click it - no special setup needed

set -e

APP_NAME="PQS Framework"
APP_DIR="$APP_NAME.app"
VERSION="1.0.0"

echo "🚀 Building PQS Framework App..."

# Clean
rm -rf "$APP_DIR" 2>/dev/null || true

# Create structure
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Create PkgInfo (REQUIRED for double-click to work)
echo "APPL????" > "$APP_DIR/Contents/PkgInfo"

# Create Info.plist
cat > "$APP_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>PQS_Framework</string>
    <key>CFBundleIconFile</key>
    <string>pqs_icon</string>
    <key>CFBundleIdentifier</key>
    <string>com.pqs.framework</string>
    <key>CFBundleName</key>
    <string>PQS Framework</string>
    <key>CFBundleDisplayName</key>
    <string>PQS Framework</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

# Create executable launcher (no .sh extension)
cat > "$APP_DIR/Contents/MacOS/PQS_Framework" << 'LAUNCHER'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES="$DIR/../Resources"

# Find Python
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v $cmd &> /dev/null; then
        PYTHON=$cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Python Required" message "Please install Python 3.11+:\n\nbrew install python@3.11" as critical'
    exit 1
fi

export PYTHONPATH="$RESOURCES:$RESOURCES/lib:$PYTHONPATH"
export PQS_RESOURCES="$RESOURCES"

cd "$RESOURCES"
exec $PYTHON "$RESOURCES/universal_pqs_app.py" "$@"
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/PQS_Framework"

# Copy files
echo "📁 Copying files..."
cp universal_pqs_app.py "$APP_DIR/Contents/Resources/"
cp native_window.py "$APP_DIR/Contents/Resources/"
cp real_quantum_ml_system.py "$APP_DIR/Contents/Resources/"
cp quantum_ml_integration.py "$APP_DIR/Contents/Resources/"
cp quantum_battery_guardian.py "$APP_DIR/Contents/Resources/"
cp auto_battery_protection.py "$APP_DIR/Contents/Resources/"
cp aggressive_idle_manager.py "$APP_DIR/Contents/Resources/"
cp -r templates "$APP_DIR/Contents/Resources/"
cp -r static "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Copy icon (without extension in Resources)
if [ -f "pqs_icon.icns" ]; then
    cp pqs_icon.icns "$APP_DIR/Contents/Resources/pqs_icon.icns"
fi
if [ -f "pqs_icon.png" ]; then
    cp pqs_icon.png "$APP_DIR/Contents/Resources/"
fi

# Bundle dependencies if in venv
if [ -n "$VIRTUAL_ENV" ]; then
    echo "📚 Bundling dependencies..."
    mkdir -p "$APP_DIR/Contents/Resources/lib"
    SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.11/site-packages"
    if [ -d "$SITE_PACKAGES" ]; then
        cp -R "$SITE_PACKAGES"/* "$APP_DIR/Contents/Resources/lib/" 2>/dev/null || true
    fi
fi

# Sign
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true

# Set proper attributes
xattr -cr "$APP_DIR" 2>/dev/null || true
chmod -R u+w "$APP_DIR"
chmod +x "$APP_DIR/Contents/MacOS/PQS_Framework"

# Create DMG
echo "💿 Creating DMG..."
DMG="PQS_Framework_v${VERSION}.dmg"
rm -f "$DMG"
hdiutil create -volname "PQS Framework" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG" > /dev/null 2>&1

echo ""
echo "✅ BUILD COMPLETE!"
echo ""
echo "📦 App: $APP_DIR"
echo "💿 DMG: $DMG"
echo ""
echo "🧪 Test: open '$APP_DIR'"
echo ""
echo "Users can:"
echo "  1. Double-click the app"
echo "  2. Drag DMG to Applications"
echo "  3. Launch from Spotlight"
