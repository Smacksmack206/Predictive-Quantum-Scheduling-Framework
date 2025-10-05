#!/usr/bin/env python3
"""
Fix incorrect battery level reading
"""

import subprocess
import time
import requests
import psutil

def kill_app():
    subprocess.run(['pkill', '-f', 'enhanced_app.py'], check=False)
    time.sleep(2)

def check_actual_battery():
    """Check what the actual battery level is"""
    battery = psutil.sensors_battery()
    if battery:
        print(f"Actual battery level (psutil): {battery.percent}%")
        print(f"Plugged in (psutil): {battery.power_plugged}")
        return battery.percent, battery.power_plugged
    return None, None

def fix_battery_detection():
    """Fix the battery level detection in the app"""
    
    with open('enhanced_app.py', 'r') as f:
        content = f.read()
    
    # Look for battery level detection issues
    # The issue might be in the EAS metrics update
    old_pattern = "self.current_metrics['battery_level'] = battery.percent"
    
    if old_pattern in content:
        # Add debugging and force refresh
        new_pattern = """# Force fresh battery reading
                battery = psutil.sensors_battery()
                if battery:
                    self.current_metrics['battery_level'] = battery.percent
                    print(f"DEBUG: Battery level updated to {battery.percent}%")
                else:
                    print("DEBUG: Could not read battery level")"""
        
        content = content.replace(
            "self.current_metrics['battery_level'] = battery.percent",
            new_pattern
        )
        
        with open('enhanced_app.py', 'w') as f:
            f.write(content)
        
        print("✅ Added battery level debugging")
        return True
    else:
        print("⚠️  Could not find battery level detection code")
        return False

def test_after_fix():
    """Start app and test battery reading"""
    
    # Check actual battery first
    actual_level, actual_plugged = check_actual_battery()
    
    # Start app
    process = subprocess.Popen(['python3', 'enhanced_app.py'], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    time.sleep(5)
    
    try:
        # Test API
        response = requests.get('http://localhost:9010/api/eas-status', timeout=3)
        if response.status_code == 200:
            data = response.json()
            battery = data.get('advanced_battery', {})
            
            api_level = battery.get('battery_level', 0)
            api_plugged = battery.get('plugged', False)
            charge_rate = battery.get('current_ma_charge', 0)
            
            print(f"\nComparison:")
            print(f"  Actual battery: {actual_level}% (plugged: {actual_plugged})")
            print(f"  API reports: {api_level}% (plugged: {api_plugged})")
            print(f"  Charge rate: {charge_rate}mA")
            
            if abs(actual_level - api_level) > 1:
                print(f"❌ MISMATCH: API shows {api_level}% but actual is {actual_level}%")
            elif actual_level < 100 and actual_plugged and charge_rate == 0:
                print(f"❌ ISSUE: At {actual_level}% plugged in, should show charging rate")
            elif actual_level < 100 and actual_plugged and charge_rate > 0:
                print(f"✅ SUCCESS: Showing {charge_rate}mA at {actual_level}%")
            else:
                print(f"✅ Battery reading appears correct")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    finally:
        try:
            process.terminate()
            process.wait(timeout=3)
        except:
            process.kill()

def main():
    print("🔧 Fixing Battery Level Detection")
    print("=" * 40)
    
    kill_app()
    fix_battery_detection()
    test_after_fix()

if __name__ == "__main__":
    main()