#!/usr/bin/env python3
"""
PQS Framework Cleanup and Organization Script
==============================================
Organizes the project structure and archives old files
"""

import os
import shutil
from pathlib import Path

# Active core files (keep in root for now, will organize later)
ACTIVE_CORE_FILES = {
    'universal_pqs_app.py',
    'real_quantum_ml_system.py',
    'aggressive_idle_manager.py',
    'quantum_max_scheduler.py',
    'quantum_max_integration.py',
    'qiskit_quantum_engine.py',
    'quantum_ml_integration.py',
    'quantum_ml_persistence.py',
    'macos_power_metrics.py',
    'quantum_battery_guardian.py',
    'auto_battery_protection.py',
    'quantum_process_optimizer.py',
    'intelligent_process_monitor.py',
}

# Files to archive
ARCHIVE_PATTERNS = {
    'broken': ['*_broken.py'],
    'old_implementations': ['real_ml_system.py', 'quantum_ml_hybrid.py', 'intel_optimized_quantum_ml.py'],
    'old_builds': ['build_*.py', 'setup_*.py', 'prepare_build.sh', 'fix_*.sh'],
    'old_tests': ['test_*.py'],  # Will move to tests/ directory
    'backups': ['backup_*'],
    'old_docs': ['*_PLAN.md', '*_GUIDE.md', '*_SUMMARY.md', '*_INSTRUCTIONS.md'],
}

def create_archive_structure():
    """Create archive directory structure"""
    print("📁 Creating archive structure...")
    
    archive_dirs = [
        'archive/old_implementations',
        'archive/old_builds',
        'archive/old_tests',
        'archive/old_docs',
        'archive/backups',
    ]
    
    for dir_path in archive_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {dir_path}")

def archive_files(dry_run=True):
    """Archive old files"""
    print(f"\n🗂️  {'DRY RUN: Would archive' if dry_run else 'Archiving'} files...")
    
    archived_count = 0
    
    # Archive broken files
    for pattern in ARCHIVE_PATTERNS['broken']:
        for file in Path('.').glob(pattern):
            if file.is_file():
                dest = Path('archive/old_implementations') / file.name
                print(f"   {'Would move' if dry_run else 'Moving'}: {file} -> {dest}")
                if not dry_run:
                    shutil.move(str(file), str(dest))
                archived_count += 1
    
    # Archive old implementations
    for file_name in ARCHIVE_PATTERNS['old_implementations']:
        file = Path(file_name)
        if file.exists() and file.is_file():
            dest = Path('archive/old_implementations') / file.name
            print(f"   {'Would move' if dry_run else 'Moving'}: {file} -> {dest}")
            if not dry_run:
                shutil.move(str(file), str(dest))
            archived_count += 1
    
    # Archive backups
    for pattern in ARCHIVE_PATTERNS['backups']:
        for item in Path('.').glob(pattern):
            if item.is_dir():
                dest = Path('archive/backups') / item.name
                print(f"   {'Would move' if dry_run else 'Moving'}: {item} -> {dest}")
                if not dry_run:
                    shutil.move(str(item), str(dest))
                archived_count += 1
    
    print(f"\n   {'Would archive' if dry_run else 'Archived'} {archived_count} items")

def organize_tests(dry_run=True):
    """Move test files to tests/ directory"""
    print(f"\n🧪 {'DRY RUN: Would organize' if dry_run else 'Organizing'} tests...")
    
    tests_dir = Path('tests')
    if not dry_run:
        tests_dir.mkdir(exist_ok=True)
    
    test_count = 0
    for test_file in Path('.').glob('test_*.py'):
        if test_file.is_file():
            dest = tests_dir / test_file.name
            print(f"   {'Would move' if dry_run else 'Moving'}: {test_file} -> {dest}")
            if not dry_run:
                shutil.move(str(test_file), str(dest))
            test_count += 1
    
    print(f"   {'Would move' if dry_run else 'Moved'} {test_count} test files")

def create_active_files_doc():
    """Create documentation of active files"""
    print("\n📝 Creating ACTIVE_FILES.md...")
    
    content = """# Active PQS Framework Files

## Core Application
- `universal_pqs_app.py` - Main Flask application and menu bar app
- `real_quantum_ml_system.py` - Quantum-ML optimization engine

## Quantum Engines
- `qiskit_quantum_engine.py` - Qiskit 40-qubit engine
- `quantum_max_scheduler.py` - 48-qubit ultimate scheduler
- `quantum_max_integration.py` - Quantum max integration layer

## ML & Optimization
- `quantum_ml_integration.py` - ML integration layer
- `quantum_ml_persistence.py` - Database persistence
- `quantum_process_optimizer.py` - Process-level optimization

## Power Management
- `aggressive_idle_manager.py` - Idle detection and sleep management
- `macos_power_metrics.py` - Real macOS power APIs
- `quantum_battery_guardian.py` - Battery protection
- `auto_battery_protection.py` - Automatic battery optimization

## System Monitoring
- `intelligent_process_monitor.py` - Process monitoring and anomaly detection

## Templates & UI
- `templates/` - HTML templates for web dashboard
- `static/` - Static assets (if any)

## Configuration
- `pyproject.toml` - Project configuration
- `requirements_quantum_ml.txt` - Python dependencies

## Documentation
- `README.md` - Main documentation
- `PROJECT_ANALYSIS_AND_IMPROVEMENTS.md` - Improvement plan
- `AGGRESSIVE_IDLE_MANAGEMENT.md` - Idle management docs
- `QUANTUM_MAX_SCHEDULER.md` - Quantum scheduler docs

## Archived
- `archive/` - Old implementations, tests, and documentation
"""
    
    with open('ACTIVE_FILES.md', 'w') as f:
        f.write(content)
    
    print("   ✅ Created ACTIVE_FILES.md")

def show_summary():
    """Show project summary"""
    print("\n" + "="*70)
    print("📊 PROJECT SUMMARY")
    print("="*70)
    
    # Count files
    py_files = len(list(Path('.').glob('*.py')))
    md_files = len(list(Path('.').glob('*.md')))
    
    print(f"   Python files in root: {py_files}")
    print(f"   Markdown files in root: {md_files}")
    print(f"   Active core files: {len(ACTIVE_CORE_FILES)}")
    
    # Show what would be archived
    would_archive = 0
    for pattern_list in ARCHIVE_PATTERNS.values():
        for pattern in pattern_list:
            would_archive += len(list(Path('.').glob(pattern)))
    
    print(f"   Files to archive: ~{would_archive}")
    print("="*70)

def main():
    """Main cleanup function"""
    print("\n" + "="*70)
    print("🧹 PQS FRAMEWORK CLEANUP & ORGANIZATION")
    print("="*70)
    
    # Show current state
    show_summary()
    
    # Ask for confirmation
    print("\n⚠️  This will:")
    print("   1. Create archive/ directory structure")
    print("   2. Move old/broken files to archive/")
    print("   3. Organize test files into tests/")
    print("   4. Create ACTIVE_FILES.md documentation")
    
    response = input("\n   Run in DRY RUN mode first? (Y/n): ").strip().lower()
    dry_run = response != 'n'
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be moved")
    else:
        confirm = input("\n   Are you sure you want to move files? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("   ❌ Cancelled")
            return
    
    # Execute cleanup
    create_archive_structure()
    archive_files(dry_run=dry_run)
    organize_tests(dry_run=dry_run)
    
    if not dry_run:
        create_active_files_doc()
    
    # Show results
    print("\n" + "="*70)
    if dry_run:
        print("✅ DRY RUN COMPLETE")
        print("\n   Review the changes above, then run again with 'n' to execute")
    else:
        print("✅ CLEANUP COMPLETE")
        print("\n   Next steps:")
        print("   1. Review ACTIVE_FILES.md")
        print("   2. Test that everything still works")
        print("   3. Commit changes to git")
        print("   4. Continue with module organization")
    print("="*70)

if __name__ == "__main__":
    main()
