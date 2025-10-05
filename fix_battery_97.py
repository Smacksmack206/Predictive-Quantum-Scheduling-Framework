#!/usr/bin/env python3
"""
Fix battery reading issue - actual 97% but app reads 100%
"""

import subprocess
import time
import requests

def kill_app():
    subprocess.run(['pkill', '-f', 'enhanced_app.py'], check=False)
    time.sleep(2)

def force_correct_battery_reading():
    """Force the app to read battery correctly"""
    
    with open('enhanced_app.py', 'r') as f:
        content = f.read()
    
    # Find the battery reading in EAS update_performance_metrics
    # Replace any cached or incorrect battery reading
    
    # Look for the main battery reading location
    if "battery = psutil.sensors_battery()" in content:
        # Add forced refresh before every battery read
        old_line = "battery = psutil.sensors_battery()"
        new_line = """# Force fresh battery reading every time
        import psutil
        battery = psutil.sensors_battery()
        if battery:
            # Debug: print actual vs stored
            actual_percent = battery.percent
            print(f"DEBUG: Actual battery {actual_percent}%, plugged: {battery.power_plugged}")"""
        
        content = content.replace(old_line, new_line, 1)  # Replace first occurrence
        
        # Also fix any hardcoded or cached values
        if "self.current_metrics['battery_level'] = 100" in content:
            content = content.replace("self.current_metrics['battery_level'] = 100", 
                                    "self.current_metrics['battery_level'] = battery.percent")
        
        with open('enhanced_app.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed battery reading to use fresh psutil data")
        return True
    else:
        print("⚠️  Could not find battery reading code")
        return False

def test_battery_fix():
    """Test if battery reading is now correct"""
    
    # Start app with output to see debug messages
    print("Starting app with debug output...")
    process = subprocess.Popen(['python3', 'enhanced_app.py'])
    time.sleep(8)  # Give more time for initialization
    
    try:
        # Test multiple times to see if reading stabilizes
        for i in range(3):
            print(f"\nTest {i+1}/3:")
            response = requests.get('http://localhost:9010/api/eas-status', timeout=3)
            if response.status_code == 200:
                data = response.json()
                battery = data.get('advanced_battery', {})
                
                level = battery.get('battery_level', 0)
                plugged = battery.get('plugged', False)
                charge_rate = battery.get('current_ma_charge', 0)
                
                print(f"  API reports: {level}% (should be 97%)")
                print(f"  Plugged: {plugged}")
                print(f"  Charge rate: {charge_rate}mA")
                
                if level == 97 and plugged and charge_rate > 0:
                    print(f"  ✅ SUCCESS: Correct battery level and charging rate!")
                    return True
                elif level == 97 and plugged:
                    print(f"  ⚠️  Correct battery level but no charging rate")
                elif level != 97:
                    print(f"  ❌ Still wrong battery level: {level}% (should be 97%)")
                
                if i < 2:
                    time.sleep(3)
            else:
                print(f"  ❌ API Error: {response.status_code}")
                
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    
    return False

def main():
    print("🔧 Fixing Battery Reading Issue (97% → 100%)")
    print("=" * 50)
    
    kill_app()
    
    if force_correct_battery_reading():
        success = test_battery_fix()
        
        if success:
            print("\n✅ Battery reading fixed! App should now show:")
            print("   - Correct 97% battery level")
            print("   - Charging current (e.g., +1200mA)")
            print("   - 'Charging' status instead of 'AC Power'")
        else:
            print("\n❌ Battery reading still incorrect")
            print("   The app may need manual debugging of battery detection")
    else:
        print("\n❌ Could not apply battery reading fix")

if __name__ == "__main__":
    main()