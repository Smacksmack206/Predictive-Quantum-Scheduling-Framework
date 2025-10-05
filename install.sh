#!/bin/bash

echo "🔋 Installing Battery Optimizer Pro..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Make scripts executable
chmod +x optimizer.sh
chmod +x enhanced_app.py

# Create LaunchAgent directory if it doesn't exist
mkdir -p ~/Library/LaunchAgents

# Update plist file with current directory
CURRENT_DIR=$(pwd)
sed "s|/Users/home/m3.macbook.air|$CURRENT_DIR|g" com.user.batteryoptimizer.plist > ~/Library/LaunchAgents/com.user.batteryoptimizer.plist

# Load the service
echo "🚀 Loading background service..."
launchctl unload ~/Library/LaunchAgents/com.user.batteryoptimizer.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.user.batteryoptimizer.plist

echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. The service is now running in the background"
echo "2. Look for the ⚡ icon in your menu bar"
echo "3. Open the dashboard at: http://localhost:9010"
echo "4. Configure your apps in the dashboard"
echo ""
echo "📊 Features:"
echo "• Smart battery optimization with ML"
echo "• Beautiful Material UI dashboard"
echo "• Real-time analytics and insights"
echo "• Amphetamine mode for developers"
echo "• Automatic learning from usage patterns"
