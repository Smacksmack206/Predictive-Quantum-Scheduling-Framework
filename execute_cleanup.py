#!/usr/bin/env python3
"""
Execute PQS Framework Cleanup
==============================
Archives non-active files based on ACTIVE_FILES_VERIFIED.md
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Active files from ACTIVE_FILES_VERIFIED.md
ACTIVE_FILES = {
    # Core Python files
    'universal_pqs_app.py',
    'real_quantum_ml_system.py',
    'quantum_ml_integration.py',
    'quantum_ml_persistence.py',
    'qiskit_quantum_engine.py',
    'quantum_max_scheduler.py',
    'quantum_max_integration.py',
    'quantum_ml_idle_optimizer.py',
    'aggressive_idle_manager.py',
    'macos_power_metrics.py',
    'quantum_battery_guardian.py',
    'auto_battery_protection.py',
    'quantum_process_optimizer.py',
    'intelligent_process_monitor.py',
    
    # Configuration
    'pyproject.toml',
    'requirements_quantum_ml.txt',
    '.gitignore',
    '.python-version',
    
    # Active documentation
    'README.md',
    'AGGRESSIVE_IDLE_MANAGEMENT.md',
    'QUANTUM_ML_IDLE_INTELLIGENCE.md',
    'QUANTUM_MAX_SCHEDULER.md',
    'PROJECT_ANALYSIS_AND_IMPROVEMENTS.md',
    'ACTIVE_FILES_VERIFIED.md',
    
    # Keep cleanup scripts
    'cleanup_and_organize.py',
    'execute_cleanup.py',
    
    # Keep __init__ and __main__
    '__init__.py',
    '__main__.py',
}

# Directories to keep entirely
KEEP_DIRS = {
    'templates',
    'static',
    '.git',
    '.kiro',
    'quantum_ml_311',  # Virtual environment
    'pqs_venv',
    'quantum_ml_venv',
}

# Directories to archive
ARCHIVE_DIRS = {
    'backup_1760701064',
    'build',
    'build_macos',
    'dist',
    'dist_macos',
    '.eggs',
    '__pycache__',
    'pqsframework_builds',  # Has its own copy
    'req',
    'resources',
    'tests',  # Will reorganize later
}

def create_archive_structure():
    """Create archive directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_base = Path(f'archive_{timestamp}')
    
    dirs = [
        archive_base / 'python_files',
        archive_base / 'docs',
        archive_base / 'build_scripts',
        archive_base / 'tests',
        archive_base / 'directories',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return archive_base

def should_keep_file(file_path: Path) -> bool:
    """Check if file should be kept"""
    # Keep if in active files list
    if file_path.name in ACTIVE_FILES:
        return True
    
    # Keep if in a keep directory
    for keep_dir in KEEP_DIRS:
        if keep_dir in file_path.parts:
            return True
    
    return False

def archive_file(file_path: Path, archive_base: Path):
    """Archive a single file"""
    # Determine archive subdirectory
    if file_path.suffix == '.py':
        if file_path.name.startswith('test_'):
            dest_dir = archive_base / 'tests'
        elif file_path.name.startswith('build_') or file_path.name.startswith('setup_'):
            dest_dir = archive_base / 'build_scripts'
        else:
            dest_dir = archive_base / 'python_files'
    elif file_path.suffix == '.md':
        dest_dir = archive_base / 'docs'
    else:
        dest_dir = archive_base / 'python_files'
    
    dest_path = dest_dir / file_path.name
    
    # Handle duplicates
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1
    
    shutil.move(str(file_path), str(dest_path))
    return dest_path

def main():
    """Execute cleanup"""
    print("\n" + "="*70)
    print("🧹 PQS FRAMEWORK CLEANUP - EXECUTION")
    print("="*70)
    
    # Create archive
    print("\n📁 Creating archive directory...")
    archive_base = create_archive_structure()
    print(f"   ✅ Created: {archive_base}")
    
    # Archive directories first
    print("\n📦 Archiving directories...")
    archived_dirs = 0
    for dir_name in ARCHIVE_DIRS:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            dest = archive_base / 'directories' / dir_name
            print(f"   Moving: {dir_name}")
            shutil.move(str(dir_path), str(dest))
            archived_dirs += 1
    print(f"   ✅ Archived {archived_dirs} directories")
    
    # Archive root Python files
    print("\n🐍 Processing Python files in root...")
    archived_py = 0
    for py_file in Path('.').glob('*.py'):
        if not should_keep_file(py_file):
            dest = archive_file(py_file, archive_base)
            archived_py += 1
            if archived_py <= 5:  # Show first 5
                print(f"   Archived: {py_file.name}")
    
    if archived_py > 5:
        print(f"   ... and {archived_py - 5} more")
    print(f"   ✅ Archived {archived_py} Python files")
    
    # Archive markdown files
    print("\n📝 Processing markdown files...")
    archived_md = 0
    for md_file in Path('.').glob('*.md'):
        if not should_keep_file(md_file):
            dest = archive_file(md_file, archive_base)
            archived_md += 1
            if archived_md <= 5:
                print(f"   Archived: {md_file.name}")
    
    if archived_md > 5:
        print(f"   ... and {archived_md - 5} more")
    print(f"   ✅ Archived {archived_md} markdown files")
    
    # Archive other files
    print("\n📄 Processing other files...")
    archived_other = 0
    for file_path in Path('.').glob('*'):
        if file_path.is_file() and not should_keep_file(file_path):
            if file_path.suffix not in ['.py', '.md']:
                dest = archive_file(file_path, archive_base)
                archived_other += 1
    print(f"   ✅ Archived {archived_other} other files")
    
    # Create summary
    print("\n📊 Creating archive summary...")
    summary_file = archive_base / 'ARCHIVE_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write(f"PQS Framework Archive\n")
        f.write(f"Created: {datetime.now()}\n")
        f.write(f"\nArchived:\n")
        f.write(f"  - {archived_dirs} directories\n")
        f.write(f"  - {archived_py} Python files\n")
        f.write(f"  - {archived_md} markdown files\n")
        f.write(f"  - {archived_other} other files\n")
        f.write(f"\nTotal: {archived_dirs + archived_py + archived_md + archived_other} items\n")
        f.write(f"\nActive files kept in root:\n")
        for active_file in sorted(ACTIVE_FILES):
            if Path(active_file).exists():
                f.write(f"  - {active_file}\n")
    
    print(f"   ✅ Created: {summary_file}")
    
    # Show results
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70)
    print(f"\n📦 Archive location: {archive_base}")
    print(f"\n📊 Summary:")
    print(f"   - Archived {archived_dirs} directories")
    print(f"   - Archived {archived_py} Python files")
    print(f"   - Archived {archived_md} markdown files")
    print(f"   - Archived {archived_other} other files")
    print(f"   - Total: {archived_dirs + archived_py + archived_md + archived_other} items")
    
    # Count remaining files
    remaining_py = len(list(Path('.').glob('*.py')))
    remaining_md = len(list(Path('.').glob('*.md')))
    print(f"\n✨ Root directory now has:")
    print(f"   - {remaining_py} Python files (active)")
    print(f"   - {remaining_md} markdown files (active docs)")
    print(f"   - templates/ directory (7 HTML files)")
    
    print("\n🎯 Next steps:")
    print("   1. Test that the app still works: pqs")
    print("   2. Review archive if needed")
    print("   3. Commit changes to git")
    print("="*70)

if __name__ == "__main__":
    # Safety check
    if not Path('universal_pqs_app.py').exists():
        print("❌ Error: universal_pqs_app.py not found!")
        print("   Make sure you're in the project root directory")
        exit(1)
    
    print("\n⚠️  This will archive ~2,100 files to a timestamped archive directory")
    print("   Active files will remain in the root directory")
    print("   The archive can be restored if needed")
    
    response = input("\n   Continue? (yes/no): ").strip().lower()
    
    if response == 'yes':
        main()
    else:
        print("\n❌ Cancelled")
