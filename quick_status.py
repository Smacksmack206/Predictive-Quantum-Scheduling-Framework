#!/usr/bin/env python3
"""
Quick status check without hanging
"""

import requests
import json

def quick_check():
    try:
        # Quick timeout to avoid hanging
        response = requests.get('http://localhost:9010/api/eas-status', timeout=2)
        if response.status_code == 200:
            data = response.json()
            battery = data.get('advanced_battery', {})
            
            level = battery.get('battery_level', 0)
            plugged = battery.get('plugged', False)
            charge_rate = battery.get('current_ma_charge', 0)
            drain_rate = battery.get('current_ma_drain', 0)
            
            print(f"Battery: {level}% | Plugged: {plugged} | Charge: {charge_rate}mA | Drain: {drain_rate}mA")
            
            if plugged and level < 100 and charge_rate == 0:
                print("❌ ISSUE: Should show charging rate but shows 0mA")
                return False
            elif plugged and level == 100:
                print("✅ CORRECT: Battery full, AC Power is correct")
                return True
            elif plugged and charge_rate > 0:
                print(f"✅ SUCCESS: Showing {charge_rate}mA charging rate")
                return True
            else:
                print("🔋 On battery power")
                return True
        else:
            print(f"❌ API Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    quick_check()