#!/usr/bin/env python3
"""Test Aggressive Idle Manager"""

from aggressive_idle_manager import AggressiveIdleManager
import time

print("🔋 Testing Aggressive Idle Manager")
print("=" * 60)

manager = AggressiveIdleManager()

# Get current state
print("\n📊 Getting current activity state...")
state = manager.get_activity_state()

if state:
    print(f"\n✅ Activity State:")
    print(f"   CPU Active: {'🔴 YES' if state.cpu_active else '🟢 NO'}")
    print(f"   User Input Recent: {'🔴 YES' if state.user_input_recent else '🟢 NO'}")
    print(f"   Media Playing: {'🔴 YES' if state.media_playing else '🟢 NO'}")
    print(f"   Network Active: {'🔴 YES' if state.network_active else '🟢 NO'}")
    print(f"   Disk Active: {'🔴 YES' if state.disk_active else '🟢 NO'}")
    print(f"   Active Workload: {'🔴 YES' if state.active_workload else '🟢 NO'}")
    print(f"   Lid Open: {'🟢 YES' if state.lid_open else '🔴 NO'}")
    print(f"   Power Plugged: {'🔌 YES' if state.power_plugged else '🔋 NO'}")
    print(f"   Battery: {state.battery_percent:.0f}%")
    
    is_idle = manager.is_truly_idle(state)
    print(f"\n💤 System is truly idle: {'YES ✅' if is_idle else 'NO ❌'}")
    
    if is_idle:
        print("\n⚠️  System is idle - would trigger:")
        print("   1. Suspend battery-draining apps after 30s")
        print("   2. Force sleep after 2 minutes (on battery)")
        print("   3. Immediate sleep if lid closed for 30s")
    else:
        print("\n✅ System is active - no sleep actions")
    
    # Check for sleep-preventing apps
    print("\n🔍 Checking for sleep-preventing apps...")
    import psutil
    found_preventers = []
    
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name']
            if any(prev.lower() in name.lower() for prev in manager.sleep_preventers):
                found_preventers.append(name)
        except:
            continue
    
    if found_preventers:
        print(f"   Found {len(found_preventers)} sleep preventers:")
        for app in found_preventers:
            print(f"   - {app} (would be suspended when idle)")
    else:
        print("   No sleep-preventing apps found")
    
    # Show status
    print("\n📈 Manager Status:")
    status = manager.get_status()
    print(f"   Monitoring: {status.get('monitoring', False)}")
    print(f"   Suspended Apps: {status.get('suspended_apps', 0)}")
    
    print("\n✅ Test complete!")
    print("\nTo enable aggressive idle management:")
    print("   manager.start_monitoring()")
    print("\nThis will:")
    print("   ✓ Suspend Amphetamine, Kiro, and other idle apps")
    print("   ✓ Force sleep when truly idle on battery")
    print("   ✓ Immediate sleep when lid closed")
    print("   ✓ Detect real workloads and media playback")
    
else:
    print("❌ Failed to get activity state")
