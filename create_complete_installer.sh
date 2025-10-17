#!/bin/bash
# Create complete DMG installer with instructions

echo "🔧 Creating complete DMG installer..."

# Create temporary directory for DMG contents
mkdir -p dmg_complete
cp -R "dist/PQS Framework 40-Qubit.app" dmg_complete/
cp "Installation Instructions.txt" dmg_complete/

# Create Applications symlink for drag-and-drop
ln -s /Applications dmg_complete/Applications

# Create DMG with custom settings
hdiutil create -volname "PQS Framework 40-Qubit" \
    -srcfolder dmg_complete \
    -ov -format UDZO \
    -imagekey zlib-level=9 \
    "PQS Framework 40-Qubit - Complete Installer.dmg"

# Clean up
rm -rf dmg_complete

echo "✅ Complete installer created: PQS Framework 40-Qubit - Complete Installer.dmg"
echo ""
echo "📦 What your fiancé gets:"
echo "• Drag-and-drop installer (no Terminal needed)"
echo "• Complete installation instructions"
echo "• Universal app that works on Intel Macs"
echo "• Self-contained - no dependencies needed"