#!/bin/bash
# Build script for PQS Framework macOS App with ALL dependencies
# This preserves all quantum-ML features and dependencies

set -e  # Exit on error

echo "🚀 Building PQS Framework 40-Qubit macOS App"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="PQS Framework 40-Qubit"
BUNDLE_ID="com.pqsframework.40qubit"
VERSION="4.0.0"
BUILD_DIR="build_macos"
DIST_DIR="dist_macos"
APP_DIR="$DIST_DIR/$APP_NAME.app"

echo -e "${BLUE}📦 Step 1: Clean previous builds${NC}"
rm -rf "$BUILD_DIR" "$DIST_DIR"
mkdir -p "$BUILD_DIR" "$DIST_DIR"

echo -e "${BLUE}📦 Step 2: Create app bundle structure${NC}"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"
mkdir -p "$APP_DIR/Contents/Frameworks"

echo -e "${BLUE}📦 Step 3: Copy application code${NC}"
# Copy main application
cp universal_pqs_app.py "$APP_DIR/Contents/Resources/pqs_framework.py"
cp real_quantum_ml_system.py "$APP_DIR/Contents/Resources/"

# Copy templates and static files
cp -r templates "$APP_DIR/Contents/Resources/"
cp -r static "$APP_DIR/Contents/Resources/"

echo -e "${BLUE}📦 Step 4: Copy pre-existing virtual environment${NC}"
echo "Using virtual environment from /Users/home/Projects/system-tools/m3.macbook.air/pqs_venv"
cp -a /Users/home/Projects/system-tools/m3.macbook.air/pqs_venv "$APP_DIR/Contents/Resources/venv"

echo -e "${BLUE}📦 Step 5: Create launcher script${NC}"
cat > "$APP_DIR/Contents/MacOS/$APP_NAME" << 'LAUNCHER_EOF'
#!/bin/bash
# Launcher script for PQS Framework

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$DIR/../Resources"

# Activate the bundled virtual environment
source "$RESOURCES_DIR/venv/bin/activate"

# Add resources to Python path
export PYTHONPATH="$RESOURCES_DIR:$PYTHONPATH"

# Change to resources directory
cd "$RESOURCES_DIR"

# Run the application
exec python3 -m pqs_framework
LAUNCHER_EOF

chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

echo -e "${BLUE}📦 Step 6: Create Info.plist${NC}"
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>PQS Framework 40-Qubit</string>
    <key>CFBundleDisplayName</key>
    <string>PQS Framework 40-Qubit</string>
    <key>CFBundleIdentifier</key>
    <string>com.pqsframework.40qubit</string>
    <key>CFBundleVersion</key>
    <string>4.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>4.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleExecutable</key>
    <string>PQS Framework 40-Qubit</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 HM Media Labs</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSSystemAdministrationUsageDescription</key>
    <string>PQS Framework needs system access for quantum energy optimization.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>PQS Framework uses Apple Events for system integration.</string>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>
PLIST_EOF

echo -e "${BLUE}📦 Step 7: Create PkgInfo${NC}"
echo -n "APPL????" > "$APP_DIR/Contents/PkgInfo"

echo -e "${BLUE}📦 Step 8: Set permissions${NC}"
chmod -R 755 "$APP_DIR"

echo -e "${BLUE}📦 Step 9: Verify app structure${NC}"
echo "App bundle contents:"
ls -la "$APP_DIR/Contents/"
echo ""
echo "Resources:"
ls -la "$APP_DIR/Contents/Resources/" | head -20

echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "App location: $APP_DIR"
echo ""
echo "To test the app:"
echo "  open \"$APP_DIR\""
echo ""
echo "To create a DMG:"
echo "  ./create_dmg.sh"
