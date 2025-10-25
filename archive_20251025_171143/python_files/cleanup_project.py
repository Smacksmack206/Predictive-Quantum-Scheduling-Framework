#!/usr/bin/env python3
"""
PQS Framework Project Cleanup Script
Removes redundant, obsolete, and replaced files safely
"""

import os
import shutil
import time
from pathlib import Path

class ProjectCleanup:
    """Safe project cleanup with categorized file removal"""
    
    def __init__(self):
        self.files_to_remove = {
            'obsolete_apps': [
                'BatteryOptimizerPro_Fixed.py',
                'app_validator.py',
                'quantum_dashboard.py',
                'start_complete_dashboard.py',
                'deploy_production_40_qubit.py',
                'pqs_40_qubit_controller.py',
                'launch_universal_pqs.py',
                'ui_test_agent.py'
            ],
            'redundant_optimizers': [
                'comprehensive_system_optimizer.py',
                'm3_optimizer.py',
                'real_time_quantum_optimizer.py',
                'predictive_energy_manager.py',
                'quantum_neural_eas.py',
                'distributed_quantum_eas.py'
            ],
            'old_quantum_components': [
                'quantum_circuit_manager_40_simple.py',
                'quantum_circuit_partitioner.py',
                'quantum_correlation_analyzer.py',
                'quantum_error_correction.py',
                'quantum_fallback_system.py',
                'quantum_gate_optimizer.py',
                'quantum_scheduler.py',
                'quantum_system_monitor.py',
                'pure_cirq_quantum_system.py',
                'mock_quantum_components.py'
            ],
            'old_eas_components': [
                'macos_eas.py',
                'enhanced_eas_classifier.py',
                'lightweight_eas_classifier.py',
                'eas_activity_logger.py',
                'eas_integration_patch.py',
                'check_enhanced_eas_status.py'
            ],
            'old_ml_components': [
                'ml_process_classifier.py',
                'behavior_predictor.py',
                'context_analyzer.py',
                'enhanced_process_analyzer.py',
                'rl_scheduler.py'
            ],
            'old_build_scripts': [
                'build_standalone_app.py',
                'build_universal_app.py',
                'create_complete_installer.sh',
                'install_universal_pqs.sh',
                'intel_mac_launcher.sh',
                'launch_pqs.sh',
                'push_to_github.sh',
                'run_tests.sh'
            ],
            'old_config_files': [
                'advanced_eas_config.json',
                'apps.conf',
                'pyproject_simple.toml',
                'requirements.in'
            ],
            'debug_and_temp_files': [
                'debug_chart.html',
                'debug_report.json',
                'battery',
                'files',
                'saver',
                '__init__.py',
                '__main__.py'
            ],
            'old_text_files': [
                'Installation Instructions.txt',
                'project-context.md',
                'doc.md'
            ],
            'old_icons': [
                'app_icon.icns',
                'app_icon.svg',
                'create_icon.py',
                'icon_1024x1024.png',
                'icon_128x128.png',
                'icon_16x16.png',
                'icon_256x256.png',
                'icon_32x32.png',
                'icon_512x512.png',
                'icon_64x64.png',
                'icon.png'
            ],
            'old_monitoring': [
                'hardware_monitor.py',
                'permission_manager.py',
                'real_time_eas_monitor.py'
            ],
            'force_fix_scripts': [
                'force_analytics_fix.py',
                'force_battery_history_fix.py',
                'force_charging_fix.py'
            ],
            'old_test_files': [
                'universal_compatibility_test.py'
            ],
            'duplicate_files': [
                'pqs_framework.py',  # Replaced by universal_pqs_app.py
                'intel_macbook_benchmark.py'  # Replaced by intel_2020_benchmark.py
            ]
        }
        
        self.directories_to_remove = [
            '__pycache__',
            'logs',
            'models'
        ]
        
        self.files_to_keep = {
            'core_app': [
                'universal_pqs_app.py'
            ],
            'build_system': [
                'pyproject.toml',
                'setup.py',
                'requirements.txt',
                'requirements-briefcase.txt',
                'requirements-testing.txt'
            ],
            'templates_and_static': [
                'templates/',
                'static/'
            ],
            'briefcase_project': [
                'pqsframework/',
                'src/'
            ],
            'icons': [
                'pqs-icon.icns',
                'pqs-icon.iconset/'
            ],
            'documentation': [
                'README.md',
                'REAL_DATA_IMPLEMENTATION.md',
                'IMPLEMENTATION_PLAN_REAL_TIME_OPTIMIZATION.md',
                'INTEL_COMPATIBILITY_GUIDE.md',
                'DEVELOPMENT_PATTERNS_SUCCESS.md',
                'PROJECT_ARCHITECTURE.md',
                'PROJECT_STATUS_FINAL.md',
                'ACADEMIC_VALIDATION_PLAN.md',
                'ADVANCED_BATTERY_ANALYTICS.md',
                'ai_implementation.md',
                'APPLE_SILICON_OPTIMIZATION_PLAN.md',
                'BATTERY_METRICS_FIX.md',
                'BRIEFCASE_BUILD_SUCCESS.md',
                'CURRENT_DRAW_EXPLANATION.md',
                'DASHBOARD_TROUBLESHOOTING_GUIDE.md',
                'DISTRIBUTED_OPTIMIZATION_NETWORK.md',
                'ENHANCED_EAS_INTEGRATION_GUIDE.md',
                'feature_roadmap.md',
                'IMPLEMENTATION_STATUS.md',
                'IMPROVEMENT_ROADMAP.md',
                'INTEL_MAC_INSTRUCTIONS.md',
                'INVESTOR_PITCH_DECK.md',
                'LICENSE.md',
                'MACOS_SEQUOIA_COMPATIBILITY.md',
                'PRODUCTION_READY_VISUAL_FEATURES.md',
                'PROJECT_CLEANUP_AND_ORGANIZATION.md',
                'PQS_FRAMEWORK_STATUS.md',
                'QUICK_START_GUIDE.md',
                'REVOLUTIONARY_REFACTORING_COMPLETE.md',
                'roadmap.md',
                'THREADING_TROUBLESHOOTING_GUIDE.md',
                'UNIVERSAL_COMPATIBILITY_REPORT.md',
                'VISUAL_FEATURES_DOCUMENTATION.md',
                'WARP.md',
                'WHITEPAPER_ENHANCEMENTS.md'
            ],
            'test_and_benchmark': [
                'intel_2020_benchmark.py',
                'intel_compatibility_test.py',
                'test_intel_optimizations.py',
                'build_universal.py'
            ],
            'quantum_components': [
                'apple_silicon_quantum_accelerator.py',
                'quantum_circuit_manager_40.py',
                'quantum_entanglement_engine.py',
                'quantum_ml_interface.py',
                'quantum_performance_benchmarking.py',
                'quantum_visualization_engine.py'
            ],
            'gpu_acceleration': [
                'gpu_acceleration.py'
            ],
            'specs': [
                '.kiro/'
            ]
        }
    
    def analyze_files(self):
        """Analyze which files will be removed vs kept"""
        print("📊 Project Cleanup Analysis")
        print("=" * 50)
        
        total_to_remove = 0
        for category, files in self.files_to_remove.items():
            existing_files = [f for f in files if os.path.exists(f)]
            total_to_remove += len(existing_files)
            print(f"{category}: {len(existing_files)} files")
        
        print(f"\nDirectories to remove: {len(self.directories_to_remove)}")
        print(f"Total files to remove: {total_to_remove}")
        
        # Show what will be kept
        print(f"\n✅ Files that will be KEPT:")
        for category, files in self.files_to_keep.items():
            print(f"  {category}: {len(files)} items")
        
        return total_to_remove
    
    def create_backup(self):
        """Create backup of important files before cleanup"""
        backup_dir = f"backup_{int(time.time())}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup core files
        important_files = [
            'universal_pqs_app.py',
            'pyproject.toml',
            'setup.py',
            'README.md'
        ]
        
        for file in important_files:
            if os.path.exists(file):
                shutil.copy2(file, backup_dir)
        
        print(f"💾 Backup created: {backup_dir}")
        return backup_dir
    
    def remove_files_safely(self, dry_run=True):
        """Remove files safely with dry run option"""
        removed_count = 0
        
        print(f"\n{'🔍 DRY RUN - ' if dry_run else '🗑️  REMOVING '}Files:")
        print("-" * 50)
        
        for category, files in self.files_to_remove.items():
            print(f"\n{category}:")
            for file in files:
                if os.path.exists(file):
                    if dry_run:
                        print(f"  Would remove: {file}")
                    else:
                        try:
                            os.remove(file)
                            print(f"  ✅ Removed: {file}")
                            removed_count += 1
                        except Exception as e:
                            print(f"  ❌ Failed to remove {file}: {e}")
                else:
                    print(f"  ⚪ Not found: {file}")
        
        # Remove directories
        print(f"\nDirectories:")
        for directory in self.directories_to_remove:
            if os.path.exists(directory):
                if dry_run:
                    print(f"  Would remove directory: {directory}")
                else:
                    try:
                        shutil.rmtree(directory)
                        print(f"  ✅ Removed directory: {directory}")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to remove directory {directory}: {e}")
            else:
                print(f"  ⚪ Directory not found: {directory}")
        
        return removed_count
    
    def verify_core_files(self):
        """Verify that core files are still present after cleanup"""
        print(f"\n🔍 Verifying Core Files:")
        
        core_files = [
            'universal_pqs_app.py',
            'pyproject.toml',
            'setup.py',
            'README.md',
            'pqsframework/',
            'templates/',
            'static/',
            'pqs-icon.icns'
        ]
        
        all_present = True
        for file in core_files:
            if os.path.exists(file):
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ MISSING: {file}")
                all_present = False
        
        return all_present
    
    def run_cleanup(self, dry_run=True, create_backup=True):
        """Run the complete cleanup process"""
        print("🧹 PQS Framework Project Cleanup")
        print("=" * 50)
        
        # Analyze what will be removed
        total_files = self.analyze_files()
        
        if total_files == 0:
            print("\n✅ No files to remove - project is already clean!")
            return
        
        # Create backup if requested
        if create_backup and not dry_run:
            self.create_backup()
        
        # Remove files
        removed_count = self.remove_files_safely(dry_run=dry_run)
        
        if not dry_run:
            # Verify core files are still there
            if self.verify_core_files():
                print(f"\n✅ Cleanup complete! Removed {removed_count} items.")
                print("🎯 Project is now clean and optimized.")
            else:
                print(f"\n❌ WARNING: Some core files are missing after cleanup!")
        else:
            print(f"\n🔍 Dry run complete. Would remove {total_files} items.")
            print("Run with dry_run=False to actually remove files.")

def main():
    """Run project cleanup"""
    import time
    
    cleanup = ProjectCleanup()
    
    print("Choose cleanup mode:")
    print("1. Dry run (show what would be removed)")
    print("2. Full cleanup (actually remove files)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        cleanup.run_cleanup(dry_run=True)
    elif choice == "2":
        print("\n⚠️  WARNING: This will permanently remove files!")
        confirm = input("Type 'YES' to confirm: ").strip()
        if confirm == "YES":
            cleanup.run_cleanup(dry_run=False, create_backup=True)
        else:
            print("Cleanup cancelled.")
    else:
        print("Invalid choice. Running dry run...")
        cleanup.run_cleanup(dry_run=True)

if __name__ == "__main__":
    main()