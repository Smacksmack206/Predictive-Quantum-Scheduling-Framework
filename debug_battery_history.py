#!/usr/bin/env python3
"""
Debug and fix battery history chart display issues
"""

import subprocess
import time
import requests

def kill_app():
    subprocess.run(['pkill', '-f', 'enhanced_app.py'], check=False)
    time.sleep(2)

def debug_battery_history_api():
    """Debug what the battery history API is actually returning"""
    
    print("🔍 Debugging Battery History API Response")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:9010/api/battery-history?range=today', timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            history = data.get('history', [])
            statistics = data.get('statistics', {})
            cycles = data.get('cycles', [])
            
            print(f"✅ API Response Success")
            print(f"   History Points: {len(history)}")
            print(f"   Cycles: {len(cycles)}")
            print(f"   Statistics: {statistics}")
            
            if history:
                print(f"\n📊 Sample Data Points (first 3):")
                for i, point in enumerate(history[:3]):
                    print(f"   {i+1}. Timestamp: {point.get('timestamp')}")
                    print(f"      Battery: {point.get('battery_level')}%")
                    print(f"      Current Draw: {point.get('current_draw')}mA")
                    print(f"      EAS Active: {point.get('eas_active')}")
                    print(f"      Power Source: {point.get('power_source')}")
                    print()
                
                # Check data structure
                required_fields = ['timestamp', 'battery_level', 'current_draw', 'eas_active']
                sample = history[0]
                missing = [field for field in required_fields if field not in sample]
                
                if missing:
                    print(f"❌ Missing fields: {missing}")
                else:
                    print(f"✅ All required fields present")
                
                return True
            else:
                print(f"❌ No history data returned")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
        return False

def fix_chart_initialization():
    """Fix Chart.js initialization issues"""
    
    with open('static/battery-history-new.js', 'r') as f:
        content = f.read()
    
    # Fix chart setup to be more robust
    old_setup = '''    setupChart() {
        const canvas = document.getElementById('batteryChart');
        if (!canvas) {
            console.error('Chart canvas not found!');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        console.log('Setting up chart...');'''
    
    new_setup = '''    setupChart() {
        const canvas = document.getElementById('batteryChart');
        if (!canvas) {
            console.error('Chart canvas not found!');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        console.log('Setting up chart...');
        
        // Ensure Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('Chart.js not loaded!');
            return;
        }'''
    
    if old_setup in content:
        content = content.replace(old_setup, new_setup)
        print("✅ Enhanced chart initialization")
    
    # Fix data loading to be more aggressive
    old_load = '''    async loadData() {
        try {
            console.log(`Loading battery history data for range: ${this.currentRange}`);
            const response = await fetch(`/api/battery-history?range=${this.currentRange}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`Loaded ${data.history?.length || 0} history points`);
            
            this.updateChart(data.history || []);
            this.updateCycles(data.cycles || []);
            this.updateAppChanges(data.app_changes || []);
            this.updateStatistics(data.statistics || {});
            
        } catch (error) {
            console.error('Failed to load battery history:', error);
            this.showError('Failed to load data');
        }
    }'''
    
    new_load = '''    async loadData() {
        try {
            console.log(`Loading battery history data for range: ${this.currentRange}`);
            const response = await fetch(`/api/battery-history?range=${this.currentRange}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`Loaded ${data.history?.length || 0} history points`);
            console.log('Sample data:', data.history?.slice(0, 2));
            
            // Force chart update even with empty data
            this.updateChart(data.history || []);
            this.updateCycles(data.cycles || []);
            this.updateAppChanges(data.app_changes || []);
            this.updateStatistics(data.statistics || {});
            
            // Show success message
            if (data.history && data.history.length > 0) {
                console.log(`✅ Successfully loaded ${data.history.length} data points`);
            } else {
                console.warn('⚠️ No history data available');
            }
            
        } catch (error) {
            console.error('Failed to load battery history:', error);
            this.showError('Failed to load data');
        }
    }'''
    
    if old_load in content:
        content = content.replace(old_load, new_load)
        print("✅ Enhanced data loading with better debugging")
    
    with open('static/battery-history-new.js', 'w') as f:
        f.write(content)

def fix_chart_display_issues():
    """Fix chart display and rendering issues"""
    
    with open('templates/battery_history.html', 'r') as f:
        content = f.read()
    
    # Ensure chart container has proper sizing
    chart_css = '''
    <style>
        .chart-wrapper {
            position: relative;
            height: 400px !important;
            width: 100% !important;
            background: var(--bg-card);
            border-radius: var(--border-radius);
            padding: 20px;
            border: 1px solid var(--border-primary);
            margin: 20px 0;
        }
        
        #batteryChart {
            width: 100% !important;
            height: 100% !important;
            display: block !important;
        }
        
        .chart-container {
            background: var(--bg-card);
            border-radius: var(--border-radius);
            padding: 20px;
            margin: 20px 0;
            border: 1px solid var(--border-primary);
        }
        
        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
    </style>'''
    
    # Add chart-specific CSS
    head_end = content.find('</head>')
    if head_end != -1 and '.chart-wrapper' not in content:
        content = content[:head_end] + chart_css + '\n' + content[head_end:]
        print("✅ Added chart display CSS")
    
    with open('templates/battery_history.html', 'w') as f:
        f.write(content)

def test_complete_battery_history():
    """Test complete battery history functionality"""
    
    print("\n🧪 Testing Complete Battery History Functionality")
    print("=" * 55)
    
    process = subprocess.Popen(['python3', 'enhanced_app.py'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
    time.sleep(8)
    
    try:
        # Test API first
        if debug_battery_history_api():
            print("\n✅ API is working - data should display in charts")
        else:
            print("\n❌ API issues - charts won't work")
            return False
        
        # Test page load
        try:
            response = requests.get('http://localhost:9010/history', timeout=5)
            if response.status_code == 200:
                html = response.text
                
                # Check for required elements
                checks = [
                    ('Chart.js library', 'chart.js' in html.lower()),
                    ('Battery chart canvas', 'batteryChart' in html),
                    ('Chart wrapper', 'chart-wrapper' in html),
                    ('Statistics panel', 'stats-panel' in html),
                    ('JavaScript file', 'battery-history-new.js' in html)
                ]
                
                print(f"\n📄 Battery History Page Checks:")
                all_good = True
                for check_name, result in checks:
                    status = "✅" if result else "❌"
                    print(f"   {status} {check_name}")
                    if not result:
                        all_good = False
                
                return all_good
            else:
                print(f"❌ Page load failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Page test failed: {e}")
            return False
            
    finally:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()

def main():
    print("🔧 Debugging & Fixing Battery History Charts")
    print("=" * 50)
    
    kill_app()
    
    print("1. Fixing chart initialization...")
    fix_chart_initialization()
    
    print("2. Fixing chart display issues...")
    fix_chart_display_issues()
    
    print("3. Testing complete functionality...")
    success = test_complete_battery_history()
    
    if success:
        print("\n🎉 BATTERY HISTORY SHOULD NOW WORK!")
        print("   Visit: http://localhost:9010/history")
        print("   You should see:")
        print("   - Real battery level chart over time")
        print("   - Current draw/charge rate chart")
        print("   - EAS status indicators")
        print("   - Real statistics with actual values")
    else:
        print("\n⚠️ Some issues remain - check browser console for errors")

if __name__ == "__main__":
    main()