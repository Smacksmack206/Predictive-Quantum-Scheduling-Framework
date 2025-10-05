#!/usr/bin/env python3
"""
Test script to verify battery data collection is working
"""

import sqlite3
import os
from datetime import datetime

def test_database_exists():
    """Check if the database file exists and has data"""
    db_file = os.path.expanduser("~/.battery_optimizer.db")
    
    print("🔍 Testing Battery Data Collection")
    print("=" * 50)
    
    if not os.path.exists(db_file):
        print(f"❌ Database file not found: {db_file}")
        print("   The app may not have run long enough to create data")
        return False
    
    print(f"✅ Database file exists: {db_file}")
    
    try:
        conn = sqlite3.connect(db_file)
        
        # Check if tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"✅ Tables found: {', '.join(tables)}")
        
        if 'battery_events' in tables:
            # Check how many records we have
            cursor = conn.execute("SELECT COUNT(*) FROM battery_events")
            count = cursor.fetchone()[0]
            print(f"✅ Battery events in database: {count}")
            
            if count > 0:
                # Show recent records
                cursor = conn.execute("""
                    SELECT timestamp, battery_level, power_source, suspended_apps 
                    FROM battery_events 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                print("\n📊 Recent battery events:")
                for row in cursor.fetchall():
                    timestamp, battery_level, power_source, suspended_apps = row
                    dt = datetime.fromisoformat(timestamp)
                    suspended_count = len(eval(suspended_apps)) if suspended_apps and suspended_apps != '[]' else 0
                    print(f"  {dt.strftime('%H:%M:%S')}: {battery_level}% ({power_source}) - {suspended_count} apps suspended")
                
                return True
            else:
                print("⚠️  No battery events recorded yet")
                print("   The app needs to run for a few minutes to collect data")
                return False
        else:
            print("❌ battery_events table not found")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    finally:
        conn.close()

def test_current_data_via_api():
    """Test if we can get current data via API"""
    import requests
    
    print("\n🌐 Testing Current Data via API")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            
            print(f"✅ API responded successfully")
            print(f"✅ History points returned: {len(history)}")
            
            if history:
                latest = history[-1]
                print(f"✅ Latest data point:")
                print(f"   Time: {latest.get('timestamp', 'N/A')}")
                print(f"   Battery: {latest.get('battery_level', 'N/A')}%")
                print(f"   Current Draw: {latest.get('current_draw', 'N/A')}mA")
                print(f"   EAS Active: {latest.get('eas_active', 'N/A')}")
                return True
            else:
                print("⚠️  No history data returned from API")
                print("   This is normal for a new installation")
                return False
        else:
            print(f"❌ API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    print("🔋 Battery Optimizer Pro - Data Collection Test")
    print("=" * 60)
    
    db_ok = test_database_exists()
    api_ok = test_current_data_via_api()
    
    print("\n" + "=" * 60)
    
    if db_ok and api_ok:
        print("✅ Data collection is working properly!")
        print("   The battery history dashboard should show real data")
    elif api_ok:
        print("⚠️  API is working but no historical data yet")
        print("   Let the app run for a few minutes to collect data")
    else:
        print("❌ Data collection issues detected")
        print("   Make sure the Battery Optimizer app is running")
    
    print("\n💡 Tips:")
    print("• The app collects data every 5 seconds when running")
    print("• Data is stored in ~/.battery_optimizer.db")
    print("• Battery history shows real data as it's collected")
    print("• Try unplugging/plugging power to see immediate changes")

if __name__ == "__main__":
    main()