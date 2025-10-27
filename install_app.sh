#!/bin/bash
# Install PQS Framework to Applications

set -e

APP_NAME="PQS Framework.app"
DEST="/Applications/$APP_NAME"

echo "=========================================="
echo "📦 Installing PQS Framework"
echo "=========================================="
echo ""

# Check if app exists
if [ ! -d "$APP_NAME" ]; then
    echo "❌ $APP_NAME not found in current directory"
    exit 1
fi

# Remove old version
if [ -d "$DEST" ]; then
    echo "🗑️  Removing old version..."
    rm -rf "$DEST"
fi

# Copy to Applications
echo "📁 Copying to Applications..."
cp -R "$APP_NAME" "$DEST"

# Remove quarantine
echo "🔓 Removing quarantine attribute..."
xattr -cr "$DEST" 2>/dev/null || true

# Make executable
echo "🔧 Setting permissions..."
chmod +x "$DEST/Contents/MacOS/pqs_launcher"

# Register with Launch Services
echo "📝 Registering with macOS..."
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST"

# Refresh Finder
killall Finder 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ INSTALLATION COMPLETE!"
echo "=========================================="
echo ""
echo "📍 Installed to: $DEST"
echo ""
echo "🚀 Launch methods:"
echo "   1. Spotlight: Press Cmd+Space, type 'PQS Framework'"
echo "   2. Applications folder: Open Finder → Applications → PQS Framework"
echo "   3. Command line: open '/Applications/PQS Framework.app'"
echo ""
echo "=========================================="
