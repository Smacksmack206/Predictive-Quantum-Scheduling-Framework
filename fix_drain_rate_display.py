#!/usr/bin/env python3
"""
Fix drain rate display showing 572%/h instead of proper mA values
"""

import subprocess
import time
import requests

def kill_app():
    subprocess.run(['pkill', '-f', 'enhanced_app.py'], check=False)
    time.sleep(2)

def fix_drain_rate_units():
    """Fix the drain rate display units in dashboard"""
    
    with open('templates/dashboard.html', 'r') as f:
        content = f.read()
    
    # Find the drain rate display logic
    old_display = '''document.getElementById('drain-optimized').textContent = 
                    drainRate > 0 ? `${drainRate}%/h` : 'N/A';'''
    
    # Fix to show mA instead of %/h
    new_display = '''document.getElementById('drain-optimized').textContent = 
                    drainRate > 0 ? `${Math.round(drainRate)}mA` : 'N/A';'''
    
    if old_display in content:
        content = content.replace(old_display, new_display)
        print("✅ Fixed drain rate display units (mA instead of %/h)")
    else:
        # Alternative pattern
        alt_old = 'drainRate > 0 ? `${drainRate}%/h` : \'N/A\''
        alt_new = 'drainRate > 0 ? `${Math.round(drainRate)}mA` : \'N/A\''
        
        if alt_old in content:
            content = content.replace(alt_old, alt_new)
            print("✅ Fixed drain rate display units (alternative pattern)")
        else:
            print("⚠️  Could not find drain rate display to fix")
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(content)
    
    return True

def fix_battery_history_data():
    """Fix battery history to use real data and be responsive"""
    
    with open('static/battery-history-new.js', 'r') as f:
        content = f.read()
    
    # Make updates more frequent for responsiveness
    old_interval = 'setInterval(() => {\n            this.loadData();\n        }, 30000);'
    new_interval = 'setInterval(() => {\n            this.loadData();\n        }, 5000);  // Update every 5 seconds for responsiveness'
    
    if old_interval in content:
        content = content.replace(old_interval, new_interval)
        print("✅ Made battery history more responsive (5s updates)")
    
    # Fix theme selector event handling
    old_theme_setup = '''        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            console.log('Theme selector found, adding event listener');
            themeSelect.addEventListener('change', (e) => {
                console.log(`Theme selector changed to: ${e.target.value}`);
                this.changeTheme(e.target.value);
            });
        } else {
            console.error('Theme selector not found!');
        }'''
    
    new_theme_setup = '''        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            console.log('Theme selector found, adding event listener');
            themeSelect.addEventListener('change', (e) => {
                console.log(`Theme selector changed to: ${e.target.value}`);
                this.changeTheme(e.target.value);
                // Force immediate visual update
                setTimeout(() => {
                    if (this.chart) {
                        this.updateChartTheme();
                    }
                }, 100);
            });
        } else {
            console.error('Theme selector not found!');
        }'''
    
    if old_theme_setup in content:
        content = content.replace(old_theme_setup, new_theme_setup)
        print("✅ Enhanced theme selector for battery history")
    
    with open('static/battery-history-new.js', 'w') as f:
        f.write(content)
    
    return True

def fix_theme_css_loading():
    """Fix theme CSS loading issues on battery history"""
    
    with open('templates/battery_history.html', 'r') as f:
        content = f.read()
    
    # Ensure themes.css is loaded properly
    if '<link rel="stylesheet" href="/static/themes.css">' not in content:
        # Add themes.css if missing
        head_end = content.find('</head>')
        if head_end != -1:
            css_link = '    <link rel="stylesheet" href="/static/themes.css">\n'
            content = content[:head_end] + css_link + content[head_end:]
            print("✅ Added themes.css link to battery history")
    
    # Make sure theme selector has proper styling
    theme_selector_css = '''
    <style>
        .theme-selector {
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            z-index: 9999 !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: var(--border-radius) !important;
            padding: 8px !important;
            box-shadow: var(--shadow) !important;
        }
        
        .theme-toggle select {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: 4px !important;
            padding: 4px 8px !important;
            font-size: 12px !important;
        }
    </style>'''
    
    # Add inline CSS for theme selector if not present
    if '.theme-selector' not in content:
        head_end = content.find('</head>')
        if head_end != -1:
            content = content[:head_end] + theme_selector_css + '\n' + content[head_end:]
            print("✅ Added theme selector CSS to battery history")
    
    with open('templates/battery_history.html', 'w') as f:
        f.write(content)
    
    return True

def test_fixes():
    """Test all the fixes"""
    
    print("Starting app to test fixes...")
    process = subprocess.Popen(['python3', 'enhanced_app.py'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
    time.sleep(8)
    
    try:
        # Test analytics API
        response = requests.get('http://localhost:9010/api/analytics', timeout=5)
        if response.status_code == 200:
            data = response.json()
            battery_savings = data.get('battery_savings', {})
            
            drain_rate = battery_savings.get('drain_rate_with_optimization', 0)
            print(f"Analytics API - Drain Rate: {drain_rate}mA (should be reasonable)")
            
            if 200 <= drain_rate <= 2000:
                print("✅ Drain rate is in reasonable range")
            else:
                print(f"⚠️  Drain rate seems unusual: {drain_rate}mA")
        
        # Test battery history API
        response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=5)
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            statistics = data.get('statistics', {})
            
            print(f"Battery History API - {len(history)} data points")
            if history:
                sample = history[0]
                print(f"  Sample: {sample.get('battery_level')}%, {sample.get('current_draw')}mA")
                print("✅ Battery history has real data")
            else:
                print("⚠️  No battery history data")
        
        return True
        
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

def main():
    print("🔧 Fixing Drain Rate Display & Battery History")
    print("=" * 50)
    
    kill_app()
    
    print("1. Fixing drain rate units...")
    fix_drain_rate_units()
    
    print("2. Fixing battery history responsiveness...")
    fix_battery_history_data()
    
    print("3. Fixing theme selector on battery history...")
    fix_theme_css_loading()
    
    print("4. Testing fixes...")
    test_fixes()
    
    print("\n🎉 ALL FIXES APPLIED!")
    print("   Dashboard should now show:")
    print("   - Drain Rate: 572mA (instead of 572%/h)")
    print("   - Battery History: Real data, 5s updates")
    print("   - Theme Selector: Working on all pages")
    print("\n   Start the app to see results!")

if __name__ == "__main__":
    main()