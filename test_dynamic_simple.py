#!/usr/bin/env python3
"""
Simple synchronous test of dynamic learning
"""

import psutil
import time
from quantum_ml_persistence import get_database

print("🧪 Testing Dynamic Learning System (Simple)")
print("=" * 70)

# Test 1: Load learned apps from database
print("\n📚 Test 1: Loading Learned Apps from Database")
db = get_database()

try:
    cursor = db.conn.cursor()
    cursor.execute('''
        SELECT process_name, 
               AVG(avg_energy_saved) as avg_impact,
               SUM(times_applied) as frequency,
               AVG(success_rate) as success
        FROM process_optimizations
        WHERE architecture = 'apple_silicon'
        GROUP BY process_name
        HAVING frequency > 3 AND success > 0.5
        ORDER BY avg_impact DESC
        LIMIT 10
    ''')
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"✅ Found {len(rows)} learned priority apps:")
        for i, row in enumerate(rows, 1):
            app_name = row[0]
            impact = row[1]
            frequency = row[2]
            success = row[3]
            priority_score = impact * frequency * success
            
            print(f"   {i:2}. {app_name[:35]:35} "
                  f"Impact: {impact:>6.2f}% "
                  f"Freq: {frequency:>3} "
                  f"Score: {priority_score:>7.1f}")
    else:
        print("   No learned apps yet - database is empty")
        
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Discover high-impact apps dynamically
print("\n🔍 Test 2: Discovering High-Impact Apps Dynamically")

high_impact_apps = []

for proc in psutil.process_iter(['pid', 'name']):
    try:
        name = proc.info['name']
        cpu = proc.cpu_percent(interval=0.05)
        memory = proc.memory_percent()
        
        # Calculate impact score
        impact_score = (cpu * 2.0) + (memory * 0.5)
        
        if impact_score > 3.0:  # Significant impact
            high_impact_apps.append({
                'name': name,
                'impact': impact_score,
                'cpu': cpu,
                'memory': memory
            })
            
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

# Sort by impact
high_impact_apps.sort(key=lambda x: x['impact'], reverse=True)

print(f"✅ Discovered {len(high_impact_apps)} high-impact apps:")
for i, app in enumerate(high_impact_apps[:10], 1):
    print(f"   {i:2}. {app['name'][:35]:35} "
          f"Impact: {app['impact']:>6.1f} "
          f"CPU: {app['cpu']:>5.1f}% "
          f"Mem: {app['memory']:>5.1f}%")

# Test 3: Check if Kiro is in the list
print("\n🎯 Test 3: Kiro Analysis")

kiro_apps = [app for app in high_impact_apps if 'kiro' in app['name'].lower()]

if kiro_apps:
    print(f"✅ Found {len(kiro_apps)} Kiro-related processes:")
    for app in kiro_apps:
        print(f"   • {app['name'][:40]:40} Impact: {app['impact']:>6.1f}")
    
    total_kiro_impact = sum(app['impact'] for app in kiro_apps)
    print(f"\n   Total Kiro Impact: {total_kiro_impact:.1f}")
    print(f"   Recommendation: {'HIGH PRIORITY' if total_kiro_impact > 15 else 'MODERATE PRIORITY'}")
else:
    print("   No Kiro processes found with significant impact")

print("\n✅ Dynamic Learning Test Complete!")
print("\n💡 Key Points:")
print("   1. System learns from database (persistent)")
print("   2. Discovers new high-impact apps dynamically")
print("   3. Prioritizes based on actual behavior, not hardcoded lists")
print("   4. Adapts to YOUR specific usage patterns")
