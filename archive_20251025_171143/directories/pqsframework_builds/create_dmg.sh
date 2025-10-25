#!/bin/bash
# Create DMG installer for PQS Framework

set -e

echo "📀 Creating DMG installer for PQS Framework"
echo "==========================================="

# Configuration
APP_NAME="PQS Framework 40-Qubit"
DMG_NAME="PQS-Framework-40Qubit-v4.0.0-macOS"
DIST_DIR="dist_macos"
APP_PATH="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/$DMG_NAME.dmg"
TEMP_DMG="$DIST_DIR/temp.dmg"
VOLUME_NAME="PQS Framework 40-Qubit"

# Check if app exists
if [ ! -d "$APP_PATH" ]; then
    echo "❌ Error: App not found at $APP_PATH"
    echo "Run ./build_macos_app.sh first"
    exit 1
fi

echo "📦 Step 1: Clean previous DMG"
rm -f "$DMG_PATH" "$TEMP_DMG"

echo "📦 Step 2: Create temporary DMG"
# Calculate size needed (app size + 50MB buffer)
APP_SIZE=$(du -sm "$APP_PATH" | cut -f1)
DMG_SIZE=$((APP_SIZE + 50))

hdiutil create -size ${DMG_SIZE}m -fs HFS+ -volname "$VOLUME_NAME" "$TEMP_DMG"

echo "📦 Step 3: Mount temporary DMG"
MOUNT_DIR=$(hdiutil attach "$TEMP_DMG" | grep Volumes | awk '{print $3}')

echo "📦 Step 4: Copy app to DMG"
cp -R "$APP_PATH" "$MOUNT_DIR/"

echo "📦 Step 5: Create Applications symlink"
ln -s /Applications "$MOUNT_DIR/Applications"

echo "📦 Step 6: Create README"
cat > "$MOUNT_DIR/README.txt" << 'README_EOF'
PQS Framework 40-Qubit - Quantum Energy Management System
==========================================================

Installation:
1. Drag "PQS Framework 40-Qubit.app" to the Applications folder
2. Open the app from Applications
3. The app will appear in your menu bar

Features:
• Real-time quantum-ML energy optimization
• 20-qubit quantum simulation with Cirq
• TensorFlow-macOS GPU acceleration (Apple Silicon)
• PyTorch ML models for predictive optimization
• Interactive web dashboard at http://localhost:5002
• Universal binary (Intel + Apple Silicon)

System Requirements:
• macOS 10.15 (Catalina) or later
• 8GB RAM minimum (16GB recommended)
• Apple Silicon (M1/M2/M3/M4) or Intel processor

First Launch:
• Right-click the app and select "Open" to bypass Gatekeeper
• Grant system permissions when prompted
• The menu bar icon will appear in the top-right

Support:
• GitHub: https://github.com/Smacksmack206/Predictive-Quantum-Scheduling-Framework
• Email: contact@hmmedia.dev

Copyright © 2025 HM Media Labs
Licensed under MIT License
README_EOF

echo "📦 Step 7: Set DMG appearance"
# Create .DS_Store for custom view (optional)
# This would require additional tools like dmgbuild

echo "📦 Step 8: Unmount temporary DMG"
hdiutil detach "$MOUNT_DIR"

echo "📦 Step 9: Convert to compressed DMG"
hdiutil convert "$TEMP_DMG" -format UDZO -o "$DMG_PATH"

echo "📦 Step 10: Clean up"
rm -f "$TEMP_DMG"

echo "✅ DMG created successfully!"
echo ""
echo "DMG location: $DMG_PATH"
echo "DMG size: $(du -h "$DMG_PATH" | cut -f1)"
echo ""
echo "To test the DMG:"
echo "  open \"$DMG_PATH\""
