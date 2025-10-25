#!/usr/bin/env python3
"""
Show Learned Process Optimizations
===================================

Display all learned process optimizations from the database
"""

from quantum_ml_persistence import get_database
import platform

def show_learned_optimizations():
    """Display learned optimizations"""
    
    # Determine architecture
    if platform.system() == 'Darwin':
        import subprocess
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        architecture = 'apple_silicon' if 'arm' in result.stdout.lower() else 'intel'
    else:
        architecture = 'x86_64'
    
    print(f"🎓 Learned Process Optimizations ({architecture})")
    print("=" * 80)
    
    db = get_database()
    
    # Get all learned optimizations
    optimizations = db.load_process_optimizations(architecture, min_success_rate=0.0)
    
    if not optimizations:
        print("No learned optimizations yet. Run the optimizer to start learning!")
        return
    
    # Sort by energy saved
    sorted_opts = sorted(optimizations.items(), 
                        key=lambda x: x[1]['avg_energy_saved'], 
                        reverse=True)
    
    print(f"\nTotal learned optimizations: {len(sorted_opts)}\n")
    
    # Display table header
    print(f"{'Process Name':<30} {'Strategy':<20} {'Applied':<10} {'Avg Savings':<15} {'Success Rate'}")
    print("-" * 95)
    
    total_applications = 0
    total_savings = 0.0
    
    for proc_name, opt in sorted_opts:
        times = opt['times_applied']
        savings = opt['avg_energy_saved']
        success = opt['success_rate']
        strategy = opt['strategy']
        
        total_applications += times
        total_savings += savings
        
        # Color code by success rate
        if success >= 0.8:
            success_icon = "✅"
        elif success >= 0.5:
            success_icon = "⚠️"
        else:
            success_icon = "❌"
        
        print(f"{proc_name[:29]:<30} {strategy:<20} {times:<10} {savings:>6.2f}%        {success_icon} {success:.1%}")
    
    # Summary
    print("-" * 95)
    print(f"\n📊 Summary:")
    print(f"   Total optimizations learned: {len(sorted_opts)}")
    print(f"   Total times applied: {total_applications}")
    print(f"   Average energy saved per process: {total_savings / len(sorted_opts):.2f}%")
    
    # Get database stats
    stats = db.get_process_optimization_stats(architecture)
    print(f"\n💾 Database Stats:")
    print(f"   Total applications: {stats.get('total_applications', 0)}")
    print(f"   Average success rate: {stats.get('avg_success_rate', 0):.1%}")
    
    print("\n💡 Tip: These optimizations are automatically applied on every restart!")

if __name__ == "__main__":
    show_learned_optimizations()
