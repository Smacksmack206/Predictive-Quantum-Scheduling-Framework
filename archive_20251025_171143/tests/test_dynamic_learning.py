#!/usr/bin/env python3
"""
Quick test of dynamic learning system
"""

from auto_battery_protection import AutoBatteryProtectionService
import time

print("🧪 Testing Dynamic Learning System")
print("=" * 70)

# Create service
service = AutoBatteryProtectionService(check_interval=10)

print(f"\n📚 Initial State:")
print(f"   Learned priority apps from DB: {service.stats['apps_learned']}")
if service.priority_apps:
    print(f"   Top 5 priorities: {', '.join(service.priority_apps[:5])}")
else:
    print(f"   No learned priorities - will discover dynamically")

print(f"\n🔍 Discovering high-impact apps...")
service._discover_high_impact_apps()

print(f"\n📊 After Discovery:")
print(f"   Apps analyzed: {len(service.app_impact_scores)}")

if service.app_impact_scores:
    print(f"\n🎯 Top 10 High-Impact Apps (Discovered Dynamically):")
    sorted_apps = sorted(
        service.app_impact_scores.items(),
        key=lambda x: x[1]['priority_score'],
        reverse=True
    )
    
    for i, (app_name, scores) in enumerate(sorted_apps[:10], 1):
        print(f"   {i:2}. {app_name[:35]:35} "
              f"Impact: {scores['impact']:>6.1f} "
              f"Score: {scores['priority_score']:>7.1f}")

# Update priority list
print(f"\n🔄 Updating dynamic priority list...")
service._update_priority_apps()

print(f"\n✅ Dynamic Priority Apps (Top 10):")
for i, app in enumerate(service.priority_apps[:10], 1):
    print(f"   {i:2}. {app}")

# Test app insights for Kiro
print(f"\n🔍 Kiro Insights:")
kiro_insights = service.get_app_insights('Kiro')
print(f"   Is priority: {kiro_insights['is_priority']}")
print(f"   Has impact data: {kiro_insights['has_impact_data']}")
print(f"   Has pattern data: {kiro_insights['has_pattern_data']}")

if kiro_insights.get('impact_data'):
    impact = kiro_insights['impact_data']
    print(f"   Battery impact: {impact['impact']:.1f}")
    print(f"   Priority score: {impact['priority_score']:.1f}")

if kiro_insights.get('recommendations'):
    print(f"   Recommendations:")
    for rec in kiro_insights['recommendations']:
        print(f"      • {rec}")

print(f"\n✅ Dynamic Learning Test Complete!")
print(f"\n💡 Key Insight: Priority apps are now learned from actual behavior,")
print(f"   not hardcoded. System adapts to YOUR specific usage patterns!")
