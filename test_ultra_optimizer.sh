#!/bin/bash
# Test Ultra Idle Battery Optimizer

echo "🧪 Testing Ultra Idle Battery Optimizer"
echo "========================================"
echo ""

# Check if app is running
echo "1. Checking if PQS Framework is running..."
if pgrep -f "pqs_framework" > /dev/null; then
    echo "   ✅ PQS Framework is running"
else
    echo "   ❌ PQS Framework is not running"
    echo "   Start it with: python3 -m pqs_framework"
    exit 1
fi

echo ""
echo "2. Checking Ultra Optimizer status via API..."
curl -s http://localhost:5002/api/ultra-optimizer/status | python3 -m json.tool

echo ""
echo "3. Checking system idle state..."
python3 << 'EOF'
import sys
sys.path.insert(0, 'pqs_framework')
from ultra_idle_battery_optimizer import get_ultra_optimizer

optimizer = get_ultra_optimizer()
state = optimizer.get_idle_state()

if state:
    print(f"   Is Idle: {state.is_idle}")
    print(f"   Idle Duration: {state.idle_duration:.0f}s")
    print(f"   Battery: {state.battery_percent:.0f}%")
    print(f"   Power Plugged: {state.power_plugged}")
    print(f"   CPU: {state.cpu_percent:.1f}%")
    print(f"   Memory: {state.memory_percent:.1f}%")
else:
    print("   ❌ Could not get idle state")
EOF

echo ""
echo "4. Checking optimizer status..."
python3 << 'EOF'
import sys
sys.path.insert(0, 'pqs_framework')
from ultra_idle_battery_optimizer import get_ultra_optimizer

optimizer = get_ultra_optimizer()
status = optimizer.get_status()

print(f"   Enabled: {status['enabled']}")
print(f"   Running: {status['running']}")
print(f"   Optimizations Applied: {status['optimizations_applied']}")
print(f"   Battery Saved (estimate): {status['battery_saved_estimate']}")
print(f"   Suspended Apps: {status['suspended_apps']}")
print(f"   Disabled Services: {status['disabled_services']}")
EOF

echo ""
echo "5. Checking logs..."
echo "   Recent activity:"
tail -20 /tmp/pqs_framework.log 2>/dev/null | grep -i "ultra\|idle\|suspend\|optim" || echo "   No logs found"

echo ""
echo "✅ Test complete!"
echo ""
echo "To monitor in real-time:"
echo "  tail -f /tmp/pqs_framework.log | grep -i ultra"
echo ""
echo "To check API:"
echo "  curl http://localhost:5002/api/ultra-optimizer/status | jq"
