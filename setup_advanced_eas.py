#!/usr/bin/env python3
"""
Setup Script for Advanced EAS System
Installs dependencies and configures the system
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    dependencies = [
        "tensorflow>=2.10.0",
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0",
        "psutil>=5.9.0",
        "joblib>=1.1.0",
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("🖥️  Checking system requirements...")
    
    system = platform.system()
    print(f"  Operating System: {system}")
    
    if system == "Darwin":  # macOS
        print("✅ macOS detected - full functionality available")
        
        # Check for powermetrics (requires sudo)
        try:
            result = subprocess.run(["which", "powermetrics"], capture_output=True)
            if result.returncode == 0:
                print("✅ powermetrics available for hardware monitoring")
            else:
                print("⚠️  powermetrics not found - hardware monitoring may be limited")
        except:
            print("⚠️  Could not check for powermetrics")
            
    elif system == "Linux":
        print("✅ Linux detected - most functionality available")
        print("⚠️  Some macOS-specific features may not work")
        
    else:
        print(f"⚠️  {system} detected - limited functionality expected")
        print("   This system is optimized for macOS and Linux")
    
    # Check available CPU cores
    cpu_count = os.cpu_count()
    print(f"  CPU Cores: {cpu_count}")
    
    if cpu_count >= 8:
        print("✅ Sufficient CPU cores for P-core/E-core simulation")
    else:
        print("⚠️  Limited CPU cores - some features may be simulated")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    home = Path.home()
    
    # Create database directory
    db_dir = home / ".advanced_eas"
    db_dir.mkdir(exist_ok=True)
    print(f"  Created: {db_dir}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"  Created: {models_dir}")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"  Created: {logs_dir}")
    
    print("✅ Directories created successfully")
    return True

def setup_permissions():
    """Setup necessary permissions"""
    print("🔐 Setting up permissions...")
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("  macOS detected - checking permissions...")
        
        # Check if running as root (needed for some hardware monitoring)
        if os.geteuid() == 0:
            print("✅ Running as root - full hardware access available")
        else:
            print("⚠️  Not running as root - some hardware monitoring may require sudo")
            print("   To enable full hardware monitoring, run with sudo or configure sudoers")
            
        # Check for System Extension permissions
        print("  Note: System Extension features require additional setup")
        print("  See documentation for macOS System Extension configuration")
        
    elif system == "Linux":
        print("  Linux detected - checking permissions...")
        
        # Check for necessary capabilities
        if os.geteuid() == 0:
            print("✅ Running as root - full system access available")
        else:
            print("⚠️  Not running as root - some features may require elevated privileges")
            
    print("✅ Permission check completed")
    return True

def run_initial_tests():
    """Run initial functionality tests"""
    print("🧪 Running initial tests...")
    
    try:
        # Test basic imports
        print("  Testing imports...")
        import tensorflow as tf
        import sklearn
        import numpy as np
        import psutil
        print("✅ All imports successful")
        
        # Test TensorFlow
        print("  Testing TensorFlow...")
        tf_version = tf.__version__
        print(f"  TensorFlow version: {tf_version}")
        
        # Test basic functionality
        print("  Testing basic functionality...")
        
        # Test process enumeration
        process_count = len(list(psutil.process_iter()))
        print(f"  Detected {process_count} processes")
        
        # Test system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"  CPU Usage: {cpu_percent}%")
        print(f"  Memory Usage: {memory.percent}%")
        
        print("✅ Initial tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Initial tests failed: {e}")
        return False

def create_sample_config():
    """Create sample configuration file"""
    print("⚙️  Creating sample configuration...")
    
    config = {
        "optimization_mode": "adaptive",
        "optimization_interval": 30,
        "hardware_monitoring": True,
        "ml_classification": True,
        "behavior_prediction": True,
        "quantum_optimization": False,
        "rl_training": False,
        "log_level": "INFO",
        "database_path": "~/.advanced_eas/eas.db",
        "model_path": "models/",
        "thresholds": {
            "thermal_warning": 80.0,
            "thermal_critical": 90.0,
            "battery_low": 20.0,
            "battery_critical": 10.0
        }
    }
    
    import json
    
    config_file = Path("advanced_eas_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration created: {config_file}")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Advanced EAS Setup Complete!")
    print("=" * 50)
    print("Next Steps:")
    print()
    print("1. 🧪 Run tests:")
    print("   python test_advanced_eas.py")
    print()
    print("2. 🚀 Start the system:")
    print("   python advanced_eas_main.py")
    print()
    print("3. 🔄 Run continuous optimization:")
    print("   python advanced_eas_main.py continuous")
    print()
    print("4. ⚙️  Customize configuration:")
    print("   Edit advanced_eas_config.json")
    print()
    print("5. 📚 Read documentation:")
    print("   See roadmap.md for detailed implementation guide")
    print()
    print("🎯 Key Features Available:")
    print("  ✅ ML-based process classification")
    print("  ✅ LSTM behavior prediction")
    print("  ✅ Context-aware scheduling")
    print("  ✅ Hardware performance monitoring")
    print("  ✅ Predictive energy management")
    print("  ✅ Reinforcement learning optimization")
    print("  ✅ Quantum-inspired scheduling")
    print()
    print("⚠️  Note: Some features may require additional setup:")
    print("  - Hardware monitoring may need sudo privileges")
    print("  - System Extension features need macOS configuration")
    print("  - Full functionality optimized for macOS Apple Silicon")

def main():
    """Main setup function"""
    print("🚀 Advanced EAS System Setup")
    print("=" * 40)
    print("Setting up Next-Generation Energy Aware Scheduling")
    print()
    
    success = True
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("System Requirements", check_system_requirements),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("Permissions", setup_permissions),
        ("Initial Tests", run_initial_tests),
        ("Configuration", create_sample_config),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        
        if not step_func():
            print(f"❌ {step_name} failed")
            success = False
            break
        
        print(f"✅ {step_name} completed")
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)