# PQS Framework 40-Qubit - Intel Mac Installation Guide

## 🍎 For 2020 MacBook Air Intel i3 with macOS Sequoia 15.5

This version of PQS Framework 40-Qubit is specifically designed to work perfectly on Intel Macs, including your 2020 MacBook Air.

## 📦 What You Received

- **PQS_Framework_40_Qubit_Simple.dmg** - The main installer
- This contains a simple, reliable app that doesn't require complex dependencies

## 🚀 Installation Instructions

### Step 1: Install the App
1. **Double-click** `PQS_Framework_40_Qubit_Simple.dmg`
2. **Drag** the "PQS Framework 40-Qubit" app to your Applications folder
3. **Eject** the DMG by clicking the eject button in Finder

### Step 2: First Launch
1. **Go to Applications folder**
2. **Right-click** on "PQS Framework 40-Qubit"
3. **Select "Open"** (this bypasses macOS security for the first time)
4. **Click "Open"** in the security dialog

### Step 3: Automatic Setup
The app will automatically:
- ✅ Detect your Intel Mac architecture
- ✅ Find your Python installation (or help you install it)
- ✅ Install required dependencies automatically
- ✅ Show helpful error messages if anything is missing

## 🔧 If Python is Missing

If you see a dialog saying "Python 3 not found", follow these steps:

### Option 1: Install Python from Official Website (Recommended)
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12 (latest stable)
3. Run the installer
4. Make sure to check "Add Python to PATH" during installation

### Option 2: Install using Homebrew (Advanced)
1. Open Terminal
2. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
3. Install Python: `brew install python`

## 🎯 Intel Mac Optimizations

This app is specifically optimized for Intel Macs:

- **Classical Optimization**: Uses quantum-inspired algorithms optimized for Intel processors
- **Intel Python Paths**: Automatically finds Python in `/usr/local/bin/python3`
- **Fallback Systems**: Includes Intel Mac-specific quantum simulation
- **Battery Optimization**: Tailored for Intel Mac power management
- **System Monitoring**: Uses Intel-optimized system monitoring

## 🚨 Troubleshooting

### App Won't Launch
1. **Check Python**: Make sure Python 3.8+ is installed
2. **Check Permissions**: Right-click app → "Open" (don't double-click first time)
3. **Check Console**: Open Console.app to see any error messages

### Missing Dependencies Error
The app will automatically try to install missing packages. If this fails:
1. Open Terminal
2. Run: `python3 -m pip install --user rumps psutil flask numpy`

### "App is Damaged" Error
1. Open Terminal
2. Run: `xattr -cr "/Applications/PQS Framework 40-Qubit.app"`
3. Try launching the app again

## 📊 What the App Does

Once running, PQS Framework 40-Qubit will:

1. **Menu Bar Icon**: Shows a quantum icon in your menu bar
2. **Battery Optimization**: Improves battery life through intelligent process scheduling
3. **Web Dashboard**: Opens a local web interface at http://localhost:5000
4. **Real-time Monitoring**: Tracks system performance and optimization
5. **Intel Mac Mode**: Automatically uses classical algorithms optimized for Intel processors

## 🔍 Verification

To verify the app is working:
1. Look for the PQS icon in your menu bar (top right)
2. Click the icon to see the menu
3. Select "Open Dashboard" to see the web interface
4. Check that it shows "Intel Mac Compatible Mode" in the dashboard

## 📞 Support

If you have any issues:
1. The app includes detailed error messages and suggestions
2. Check the Console.app for detailed logs
3. The app is designed to be self-diagnosing and helpful

## 🎉 Success!

Once installed, PQS Framework 40-Qubit will:
- ✅ Run silently in the background
- ✅ Optimize your MacBook's performance
- ✅ Improve battery life
- ✅ Provide real-time system insights
- ✅ Work perfectly on your Intel Mac

The app is specifically designed for Intel Macs and will automatically detect and optimize for your 2020 MacBook Air Intel i3 processor.

---

**Note**: This version uses a simple, reliable installation method that avoids the complex py2app issues. It's designed to "just work" on Intel Macs without any hassle.