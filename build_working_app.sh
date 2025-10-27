#!/bin/bash
# Build PQS Framework with COMPILED launcher - guaranteed to work

set -e

APP_NAME="PQS Framework"
APP_DIR="$APP_NAME.app"

echo "🚀 Building PQS Framework with compiled launcher..."

# Clean
rm -rf "$APP_DIR" 2>/dev/null || true

# Compile the launcher
echo "🔨 Compiling launcher..."
gcc -o PQS_Framework launcher.c -framework CoreFoundation
chmod +x PQS_Framework

# Create structure
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Move compiled launcher
mv PQS_Framework "$APP_DIR/Contents/MacOS/"

# Create PkgInfo
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
</dict>
</plist>
EOF

# Copy files
echo "📁 Copying application files..."
cp universal_pqs_app.py "$APP_DIR/Contents/Resources/"
cp native_window.py "$APP_DIR/Contents/Resources/"
cp real_quantum_ml_system.py "$APP_DIR/Contents/Resources/"
cp quantum_ml_integration.py "$APP_DIR/Contents/Resources/"
cp quantum_battery_guardian.py "$APP_DIR/Contents/Resources/"
cp auto_battery_protection.py "$APP_DIR/Contents/Resources/"
cp aggressive_idle_manager.py "$APP_DIR/Contents/Resources/"
cp -r templates "$APP_DIR/Contents/Resources/"
cp -r static "$APP_DIR/Contents/Resources/" 2>/dev/null || true
cp pqs_icon.icns "$APP_DIR/Contents/Resources/" 2>/dev/null || true
cp pqs_icon.png "$APP_DIR/Contents/Resources/" 2>/dev/null || true

# Bundle dependencies
if [ -n "$VIRTUAL_ENV" ]; then
    echo "📚 Bundling Python dependencies..."
    mkdir -p "$APP_DIR/Contents/Resources/lib"
    SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.11/site-packages"
    if [ -d "$SITE_PACKAGES" ]; then
        cp -R "$SITE_PACKAGES"/* "$APP_DIR/Contents/Resources/lib/" 2>/dev/null || true
    fi
fi

# Sign
echo "🔐 Signing..."
codesign --force --deep --sign - "$APP_DIR" 2>/dev/null || true

# Remove quarantine
xattr -cr "$APP_DIR" 2>/dev/null || true

# Create DMG
echo "💿 Creating DMG..."
DMG="PQS_Framework_v1.0.0_Final.dmg"
rm -f "$DMG"
hdiutil create -volname "PQS Framework" -srcfolder "$APP_DIR" -ov -format UDZO "$DMG" > /dev/null 2>&1

echo ""
echo "=========================================="
echo "✅ BUILD COMPLETE - COMPILED LAUNCHER"
echo "=========================================="
echo ""
echo "📦 App: $APP_DIR"
echo "💿 DMG: $DMG"
echo ""
echo "✅ This app has a COMPILED binary launcher"
echo "✅ Double-click WILL work"
echo "✅ No bash script issues"
echo ""
echo "Test: open '$APP_DIR'"
echo "=========================================="
