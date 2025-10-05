#!/bin/bash

echo "🚀 Building Battery Optimizer Pro.app"

# Check if we're in the right directory
if [ ! -f "enhanced_app.py" ]; then
    echo "❌ Please run this script from the project directory"
    exit 1
fi

# Install py2app if not already installed
echo "📦 Installing py2app..."
./venv/bin/pip install py2app

# Create icon from SVG
echo "🎨 Creating app icon..."
if command -v rsvg-convert >/dev/null 2>&1; then
    # Use rsvg-convert if available
    rsvg-convert -w 512 -h 512 app_icon.svg -o app_icon.png
elif command -v inkscape >/dev/null 2>&1; then
    # Use inkscape if available
    inkscape -w 512 -h 512 app_icon.svg -o app_icon.png
else
    echo "⚠️  No SVG converter found. Using default icon."
    # Create a simple PNG icon using ImageMagick if available
    if command -v convert >/dev/null 2>&1; then
        convert -size 512x512 xc:green -fill white -gravity center -pointsize 72 -annotate +0+0 "⚡" app_icon.png
    else
        echo "⚠️  No image tools found. App will use default icon."
        touch app_icon.png
    fi
fi

# Convert PNG to ICNS
if [ -f "app_icon.png" ]; then
    echo "🔄 Converting to ICNS format..."
    mkdir -p app_icon.iconset
    
    # Create different sizes for the iconset
    if command -v sips >/dev/null 2>&1; then
        sips -z 16 16 app_icon.png --out app_icon.iconset/icon_16x16.png
        sips -z 32 32 app_icon.png --out app_icon.iconset/icon_16x16@2x.png
        sips -z 32 32 app_icon.png --out app_icon.iconset/icon_32x32.png
        sips -z 64 64 app_icon.png --out app_icon.iconset/icon_32x32@2x.png
        sips -z 128 128 app_icon.png --out app_icon.iconset/icon_128x128.png
        sips -z 256 256 app_icon.png --out app_icon.iconset/icon_128x128@2x.png
        sips -z 256 256 app_icon.png --out app_icon.iconset/icon_256x256.png
        sips -z 512 512 app_icon.png --out app_icon.iconset/icon_256x256@2x.png
        sips -z 512 512 app_icon.png --out app_icon.iconset/icon_512x512.png
        cp app_icon.png app_icon.iconset/icon_512x512@2x.png
        
        # Create ICNS file
        iconutil -c icns app_icon.iconset
        echo "✅ Icon created: app_icon.icns"
    else
        echo "⚠️  sips not available. Using PNG as icon."
        cp app_icon.png app_icon.icns
    fi
    
    # Clean up
    rm -rf app_icon.iconset app_icon.png
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/

# Build the app
echo "🔨 Building macOS app..."
./venv/bin/python setup.py py2app

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📱 App created: dist/Battery Optimizer Pro.app"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Test the app: open 'dist/Battery Optimizer Pro.app'"
    echo "2. Move to Applications: cp -r 'dist/Battery Optimizer Pro.app' /Applications/"
    echo "3. Create DMG: hdiutil create -volname 'Battery Optimizer Pro' -srcfolder dist -ov -format UDZO BatteryOptimizerPro.dmg"
    echo ""
    echo "📋 App Info:"
    ls -la "dist/Battery Optimizer Pro.app"
else
    echo "❌ Build failed. Check the error messages above."
    exit 1
fi
