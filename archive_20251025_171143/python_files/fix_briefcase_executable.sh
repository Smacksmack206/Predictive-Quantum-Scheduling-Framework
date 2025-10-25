#!/bin/bash
# Fix Briefcase executable name issue

set -e

echo "🔧 Fixing Briefcase executable name..."

APP_DIR="build/pqs-framework/macos/app/PQS Framework 40-Qubit.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"

# Check if Stub exists
if [ -f "$MACOS_DIR/Stub" ]; then
    echo "   Found Stub executable, renaming..."
    mv "$MACOS_DIR/Stub" "$MACOS_DIR/PQS Framework 40-Qubit"
    chmod +x "$MACOS_DIR/PQS Framework 40-Qubit"
    echo "✅ Executable renamed to 'PQS Framework 40-Qubit'"
elif [ -f "$MACOS_DIR/PQS Framework 40-Qubit" ]; then
    echo "✅ Executable already correctly named"
else
    echo "❌ No executable found in $MACOS_DIR"
    exit 1
fi

# Verify the fix
if [ -f "$MACOS_DIR/PQS Framework 40-Qubit" ]; then
    echo ""
    echo "✅ Fix complete! App is ready to launch:"
    echo "   open '$APP_DIR'"
else
    echo "❌ Fix failed"
    exit 1
fi
