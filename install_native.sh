#!/bin/bash

# Battery Optimizer Pro - Native macOS App Installer
# Installs the native .app bundle to /Applications

APP_NAME="Battery Optimizer Pro"
SOURCE_APP="dist/${APP_NAME}.app"
TARGET_DIR="/Applications"
TARGET_APP="${TARGET_DIR}/${APP_NAME}.app"

echo "🔋 Battery Optimizer Pro - Native macOS Installer"
echo "================================================"

# Check if source app exists
if [ ! -d "$SOURCE_APP" ]; then
    echo "❌ Error: ${SOURCE_APP} not found. Please run 'python setup.py py2app' first."
    exit 1
fi

# Check if target already exists
if [ -d "$TARGET_APP" ]; then
    echo "⚠️  ${APP_NAME} already exists in Applications folder."
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    echo "🗑️  Removing existing installation..."
    rm -rf "$TARGET_APP"
fi

# Copy app to Applications
echo "📦 Installing ${APP_NAME} to Applications folder..."
cp -R "$SOURCE_APP" "$TARGET_DIR/"

if [ $? -eq 0 ]; then
    echo "✅ Installation successful!"
    echo ""
    echo "🚀 You can now:"
    echo "   • Find '${APP_NAME}' in your Applications folder"
    echo "   • Launch it from Spotlight (Cmd+Space)"
    echo "   • Look for the ⚡ icon in your menu bar"
    echo "   • Access the dashboard at http://localhost:9010"
    echo ""
    echo "💡 The app will start automatically and run in the background."
else
    echo "❌ Installation failed. Please check permissions."
    exit 1
fi
