#!/usr/bin/env python3
"""
Test responsive battery metrics - should update every 5 seconds
"""

import time
import requests
import psutil
from datetime import datetime

def test_responsive_updates():
    print("🔋 Testing Responsive Battery Updates")
    print("=" * 60)
    print("This will monitor battery metrics for 30 seconds")
    print("You should see Current Draw update every 5 seconds")
    print("=" * 60)
    
    # Get initial battery status for comparison
    battery = psutil.sensors_battery()
    if battery:
        print(f"Initial Battery: {battery.percent}% ({'Plugged' if battery.power_plugged else 'On Battery'})")
    
    print("\nTime     | API Status | Current Draw | Predicted | Plugged Status")
    print("-" * 65)
    
    start_time = time.time()
    last_values = {}
    
    try:
        while time.time() - start_time < 30:  # Run for 30 seconds
            try:
                # Test the EAS status API
                response = requests.get('http://localhost:9010/api/eas-status', timeout=2)
                
                if response.status_code == 200:
                    data = response.json()
                    battery_data = data.get('advanced_battery', {})
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    current_draw = battery_data.get('current_ma_drain', 0)
                    current_charge = battery_data.get('current_ma_charge', 0)
                    predicted_hours = battery_data.get('predicted_battery_hours', 0)
                    plugged = battery_data.get('plugged', False)
                    
                    # Show current draw or charge rate
                    if plugged:
                        draw_display = f"+{current_charge:.0f}mA" if current_charge > 0 else "AC Power"
                    else:
                        draw_display = f"{current_draw:.0f}mA"
                    
                    plugged_display = "Plugged" if plugged else "Battery"
                    
                    # Check if values changed (for responsiveness test)
                    changed = ""
                    if 'draw' in last_values and last_values['draw'] != draw_display:
                        changed += " [DRAW CHANGED]"
                    if 'plugged' in last_values and last_values['plugged'] != plugged:
                        changed += " [STATUS CHANGED]"
                    
                    print(f"{current_time} | ✅ OK      | {draw_display:11} | {predicted_hours:7.1f}h | {plugged_display:7}{changed}")
                    
                    # Store for change detection
                    last_values = {'draw': draw_display, 'plugged': plugged}
                    
                else:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"{current_time} | ❌ ERROR   | API Error {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"{current_time} | ❌ TIMEOUT | Connection failed")
            
            time.sleep(5)  # Wait 5 seconds between checks
            
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("\n💡 Expected behavior:")
    print("• Current Draw should show immediately (not 0mA)")
    print("• Values should update every 5 seconds")
    print("• Plugged status should be accurate")
    print("• No false positives for charging status")

def test_power_status_accuracy():
    """Test power status detection accuracy"""
    print("\n🔌 Testing Power Status Detection Accuracy")
    print("=" * 50)
    
    try:
        # Test multiple detection methods
        battery = psutil.sensors_battery()
        psutil_status = battery.power_plugged if battery else False
        
        # Test pmset
        try:
            import subprocess
            pmset_output = subprocess.check_output(['pmset', '-g', 'batt'], text=True, timeout=2)
            pmset_status = 'AC Power' in pmset_output
        except:
            pmset_status = "Error"
        
        print(f"psutil detection: {psutil_status}")
        print(f"pmset detection:  {pmset_status}")
        
        # Test API
        response = requests.get('http://localhost:9010/api/battery-debug', timeout=5)
        if response.status_code == 200:
            data = response.json()
            api_status = data.get('current_metrics', {}).get('plugged', False)
            print(f"API detection:    {api_status}")
            
            if psutil_status == pmset_status == api_status:
                print("✅ All detection methods agree!")
            else:
                print("⚠️  Detection methods disagree - this may cause false positives")
        
    except Exception as e:
        print(f"❌ Power status test failed: {e}")

if __name__ == "__main__":
    test_responsive_updates()
    test_power_status_accuracy()