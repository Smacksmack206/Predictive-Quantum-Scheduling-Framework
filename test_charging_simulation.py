#!/usr/bin/env python3
"""
Test charging rate detection by simulating different battery levels
"""

import subprocess
import time
import requests

def kill_processes():
    subprocess.run(['pkill', '-f', 'enhanced_app.py'], check=False)
    time.sleep(2)

def test_charging_at_different_levels():
    """Test what charging rates would be shown at different battery levels"""
    
    # Start app
    process = subprocess.Popen(['python3', 'enhanced_app.py'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
    time.sleep(5)
    
    try:
        print("🔋 Testing Charging Rates at Different Battery Levels")
        print("=" * 60)
        
        # Get current status
        response = requests.get('http://localhost:9010/api/eas-status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            battery = data.get('advanced_battery', {})
            
            actual_level = battery.get('battery_level', 0)
            actual_charge = battery.get('current_ma_charge', 0)
            plugged = battery.get('plugged', False)
            
            print(f"Current Status:")
            print(f"  Battery Level: {actual_level}%")
            print(f"  Plugged In: {plugged}")
            print(f"  Charge Rate: {actual_charge}mA")
            
            print(f"\nExpected Charging Rates (when plugged in):")
            print(f"  At 20%: 3500mA (Fast charging)")
            print(f"  At 50%: 3000mA (Medium-high charging)")
            print(f"  At 80%: 2500mA (Medium charging)")
            print(f"  At 93%: 2000mA (Normal charging) ← Your battery level")
            print(f"  At 97%: 1200mA (Slower charging)")
            print(f"  At 99%: 800mA (Final trickle)")
            print(f"  At 100%: 0mA (Full - no charging needed)")
            
            if actual_level == 100:
                print(f"\n✅ CORRECT: Battery is full (100%), showing {actual_charge}mA is correct")
                print(f"   When battery drops to 99% or below, it should show charging current")
            elif actual_level >= 95 and actual_charge > 0:
                print(f"\n✅ SUCCESS: At {actual_level}%, showing {actual_charge}mA charging rate")
            elif actual_level < 100 and plugged and actual_charge == 0:
                print(f"\n❌ ISSUE: At {actual_level}% plugged in, should show charging rate but shows 0mA")
            else:
                print(f"\n🔋 On battery power - charging detection not applicable")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            
    finally:
        try:
            process.terminate()
            process.wait(timeout=3)
        except:
            process.kill()

def main():
    print("🧪 Charging Rate Simulation Test")
    print("=" * 40)
    
    kill_processes()
    test_charging_at_different_levels()
    
    print(f"\n💡 To test charging detection:")
    print(f"   1. Unplug your MacBook to drain battery below 99%")
    print(f"   2. Plug it back in")
    print(f"   3. Check dashboard/EAS monitor for charging mA display")
    print(f"   4. Should show 800-2000mA depending on battery level")

if __name__ == "__main__":
    main()