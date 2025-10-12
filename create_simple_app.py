#!/usr/bin/env python3
"""
Create a simple .app bundle that works
"""

import os
import shutil

def create_simple_app():
    app_name = "BatteryOptimizerPro"
    bundle_name = f"{app_name}.app"
    
    # Remove existing
    if os.path.exists(bundle_name):
        shutil.rmtree(bundle_name)
    
    # Create structure
    contents_dir = f"{bundle_name}/Contents"
    macos_dir = f"{contents_dir}/MacOS"
    resources_dir = f"{contents_dir}/Resources"
    
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    # Copy all Python files
    python_files = [f for f in os.listdir('.') if f.endswith('.py') and not f.startswith('test_') and not f.startswith('build_') and not f.startswith('create_')]
    
    for file in python_files:
        shutil.copy2(file, macos_dir)
    
    # Copy web files
    for item in ['templates', 'static']:
        if os.path.exists(item):
            shutil.copytree(item, f"{macos_dir}/{item}")
    
    # Create simple Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>enhanced_app.py</string>
    <key>CFBundleIdentifier</key>
    <string>com.batteryoptimizer.pro</string>
    <key>CFBundleName</key>
    <string>Battery Optimizer Pro</string>
    <key>CFBundleVersion</key>
    <string>2.0.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>Battery Optimizer Pro needs access to manage system processes.</string>
    <key>NSSystemAdministrationUsageDescription</key>
    <string>Battery Optimizer Pro requires administrator access for Ultimate EAS features.</string>
</dict>
</plist>"""
    
    with open(f"{contents_dir}/Info.plist", "w") as f:
        f.write(info_plist)
    
    # Make the main script executable
    os.chmod(f"{macos_dir}/enhanced_app.py", 0o755)
    
    print(f"✅ Simple {bundle_name} created!")
    return bundle_name

if __name__ == "__main__":
    create_simple_app()