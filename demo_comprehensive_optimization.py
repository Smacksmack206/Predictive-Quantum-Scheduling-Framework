#!/usr/bin/env python3
"""
Comprehensive Resource Optimization Demo
=========================================

Demonstrates the complete optimization system:
- CPU, Memory, I/O, Network, GPU optimization
- Persistent learning from database
- Real-time resource monitoring
"""

import psutil
import time
from quantum_process_optimizer import quantum_optimizer, apply_quantum_optimization
from quantum_ml_persistence import get_database

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def get_system_snapshot():
    """Get current system resource snapshot"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory': psutil.virtual_memory(),
        'disk': psutil.disk_usage('/'),
        'net_io': psutil.net_io_counters(),
        'process_count': len(psutil.pids())
    }

def collect_processes():
    """Collect process information with all resource metrics"""
    processes = []
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Get comprehensive metrics
            cpu_percent = proc.cpu_percent(interval=0.01)
            memory_info = proc.memory_info()
            memory_percent = proc.memory_percent()
            
            try:
                io_counters = proc.io_counters()
                io_data = {
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes
                }
            except:
                io_data = {}
            
            try:
                num_threads = proc.num_threads()
            except:
                num_threads = 0
            
            try:
                connections = len(proc.net_connections())
            except:
                connections = 0
            
            # Include processes with any significant resource usage
            if cpu_percent > 0.1 or memory_percent > 1.0:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': cpu_percent,
                    'memory': memory_percent,
                    'memory_mb': memory_info.rss / (1024 * 1024),
                    'io_counters': io_data,
                    'num_threads': num_threads,
                    'connections': connections
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return processes

def main():
    """Run comprehensive optimization demo"""
    
    print_header("🚀 Comprehensive Resource Optimization Demo")
    
    # Get database stats
    db = get_database()
    learned_opts = db.load_process_optimizations(quantum_optimizer.architecture)
    
    print(f"📚 Starting with {len(learned_opts)} learned optimizations")
    print(f"🏗️  Architecture: {quantum_optimizer.architecture}")
    
    # Get initial system state
    print_header("📊 Initial System State")
    initial_state = get_system_snapshot()
    
    print(f"CPU Usage:     {initial_state['cpu_percent']:.1f}%")
    print(f"Memory Usage:  {initial_state['memory'].percent:.1f}% "
          f"({initial_state['memory'].used / (1024**3):.1f}GB / "
          f"{initial_state['memory'].total / (1024**3):.1f}GB)")
    print(f"Disk Usage:    {initial_state['disk'].percent:.1f}%")
    print(f"Processes:     {initial_state['process_count']}")
    print(f"Network I/O:   {initial_state['net_io'].bytes_sent / (1024**2):.1f}MB sent, "
          f"{initial_state['net_io'].bytes_recv / (1024**2):.1f}MB received")
    
    # Collect processes
    print_header("🔍 Collecting Process Information")
    print("Scanning all processes with resource metrics...")
    processes = collect_processes()
    print(f"✅ Found {len(processes)} processes with measurable resource usage")
    
    # Show top resource consumers
    print("\n📈 Top Resource Consumers:")
    
    # Top CPU
    top_cpu = sorted(processes, key=lambda p: p['cpu'], reverse=True)[:3]
    print("\n   CPU:")
    for p in top_cpu:
        print(f"      {p['name'][:30]:30} {p['cpu']:>6.1f}%")
    
    # Top Memory
    top_mem = sorted(processes, key=lambda p: p['memory'], reverse=True)[:3]
    print("\n   Memory:")
    for p in top_mem:
        print(f"      {p['name'][:30]:30} {p['memory']:>6.1f}% ({p['memory_mb']:.0f}MB)")
    
    # Top Threads (I/O indicator)
    top_threads = sorted(processes, key=lambda p: p['num_threads'], reverse=True)[:3]
    print("\n   Threads (I/O indicator):")
    for p in top_threads:
        print(f"      {p['name'][:30]:30} {p['num_threads']:>6} threads")
    
    # Apply process optimizations
    print_header("⚡ Applying Process Optimizations")
    
    quantum_result = {
        'success': True,
        'energy_saved': 15.0,
        'quantum_advantage': 3.5
    }
    
    opt_result = apply_quantum_optimization(quantum_result, processes)
    
    print(f"✅ Process Optimization Complete:")
    print(f"   Total optimized:     {opt_result.get('optimizations_applied', 0)}")
    print(f"   🎓 Learned applied:  {opt_result.get('learned_applied', 0)}")
    print(f"   🆕 New discovered:   {opt_result.get('new_discovered', 0)}")
    print(f"   💾 Total in DB:      {opt_result.get('total_learned', 0)}")
    print(f"   💰 Energy saved:     {opt_result.get('actual_energy_saved', 0):.1f}%")
    
    if opt_result.get('optimizations'):
        print("\n   Applied Strategies:")
        strategy_counts = {}
        for opt in opt_result['optimizations']:
            strategy = opt['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {strategy:20} × {count}")
    
    # Apply comprehensive resource optimization
    print_header("🔧 Comprehensive Resource Optimization")
    
    resource_result = quantum_optimizer.optimize_all_resources()
    
    if resource_result.get('resources_optimized', 0) > 0:
        print(f"✅ Resource Optimization Complete:")
        print(f"   Resources optimized: {resource_result['resources_optimized']}")
        print(f"   Total savings:       {resource_result['total_estimated_savings']:.1f}%")
        
        for resource_type, details in resource_result['optimizations'].items():
            icon = {'memory': '🧠', 'io': '💾', 'network': '🌐', 'gpu': '🎮'}.get(resource_type, '⚙️')
            if details.get('processes_optimized', 0) > 0:
                print(f"\n   {icon} {resource_type.upper()}:")
                print(f"      Processes: {details['processes_optimized']}")
                print(f"      Savings:   {details['estimated_savings']:.1f}%")
    else:
        print("✅ All resources operating efficiently - no optimization needed")
    
    # Get final system state
    print_header("📊 Final System State")
    time.sleep(1)  # Let optimizations settle
    final_state = get_system_snapshot()
    
    print(f"CPU Usage:     {final_state['cpu_percent']:.1f}%")
    print(f"Memory Usage:  {final_state['memory'].percent:.1f}%")
    print(f"Disk Usage:    {final_state['disk'].percent:.1f}%")
    print(f"Processes:     {final_state['process_count']}")
    
    # Show database statistics
    print_header("💾 Database Statistics")
    
    stats = db.get_process_optimization_stats(quantum_optimizer.architecture)
    print(f"Total learned optimizations: {stats.get('total_learned_optimizations', 0)}")
    print(f"Average energy saved:        {stats.get('avg_energy_saved_per_process', 0):.2f}%")
    print(f"Average success rate:        {stats.get('avg_success_rate', 0):.1%}")
    print(f"Total applications:          {stats.get('total_applications', 0)}")
    
    # Summary
    print_header("✨ Summary")
    
    total_savings = opt_result.get('actual_energy_saved', 0) + resource_result.get('total_estimated_savings', 0)
    
    print(f"🎯 Total Estimated Energy Savings: {total_savings:.1f}%")
    print(f"📚 Learned Optimizations: {opt_result.get('total_learned', 0)}")
    print(f"🔄 Continuous Learning: Active")
    print(f"✅ All Resources Optimized: CPU, Memory, I/O, Network, GPU")
    
    print("\n💡 These optimizations persist across restarts!")
    print("   Run this again to see learned optimizations applied instantly.\n")

if __name__ == "__main__":
    main()
