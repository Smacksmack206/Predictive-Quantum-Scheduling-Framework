#!/usr/bin/env python3
"""
Production Deployment Script for 40-Qubit PQS Framework
Fixes all issues and ensures production readiness
"""

import os
import sys
import subprocess
import json
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check and install required dependencies"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'rumps', 'psutil', 'flask', 'numpy', 'qiskit', 'torch', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"  ❌ {package} - missing")
    
    if missing_packages:
        logger.info(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            logger.info("✅ All dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def create_production_config():
    """Create production configuration"""
    logger.info("⚙️  Creating production configuration...")
    
    config = {
        "quantum_enabled": True,
        "auto_optimize": True,
        "optimization_interval": 60,  # Longer interval for production
        "dashboard_port": 5001,
        "max_optimization_time": 15,
        "thermal_threshold": 75,  # Conservative thermal threshold
        "energy_threshold": 3.0,  # Higher threshold for production
        "debug_mode": False,
        "log_level": "INFO",
        "max_log_size_mb": 50,
        "backup_stats": True,
        "health_check_interval": 300,  # 5 minutes
        "auto_restart_on_error": True,
        "max_memory_usage_mb": 2048
    }
    
    config_file = os.path.expanduser("~/.pqs_40_qubit_config.json")
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Configuration created: {config_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
        return False

def setup_logging():
    """Setup production logging"""
    logger.info("📝 Setting up production logging...")
    
    log_dir = os.path.expanduser("~/.pqs_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log rotation script
    log_rotation_script = f"""#!/bin/bash
# PQS 40-Qubit Log Rotation Script
LOG_DIR="{log_dir}"
MAX_SIZE=52428800  # 50MB

for log_file in "$LOG_DIR"/*.log; do
    if [ -f "$log_file" ] && [ $(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null) -gt $MAX_SIZE ]; then
        mv "$log_file" "$log_file.old"
        touch "$log_file"
        echo "$(date): Rotated $log_file" >> "$LOG_DIR/rotation.log"
    fi
done
"""
    
    rotation_script_path = os.path.expanduser("~/.pqs_log_rotation.sh")
    with open(rotation_script_path, 'w') as f:
        f.write(log_rotation_script)
    
    os.chmod(rotation_script_path, 0o755)
    logger.info("✅ Log rotation setup complete")
    return True

def create_startup_script():
    """Create startup script for production"""
    logger.info("🚀 Creating startup script...")
    
    current_dir = os.getcwd()
    startup_script = f"""#!/bin/bash
# PQS 40-Qubit Production Startup Script

export PYTHONPATH="{current_dir}:$PYTHONPATH"
export PQS_PRODUCTION=1
export PQS_LOG_LEVEL=INFO

cd "{current_dir}"

# Check if already running
if pgrep -f "fixed_production_40_qubit_app.py" > /dev/null; then
    echo "PQS 40-Qubit is already running"
    exit 1
fi

# Start the application
echo "Starting PQS 40-Qubit Production Framework..."
python3 fixed_production_40_qubit_app.py &

# Store PID
echo $! > ~/.pqs_40_qubit.pid

echo "PQS 40-Qubit started successfully"
echo "Dashboard: http://localhost:5001"
echo "Logs: ~/.pqs_40_qubit.log"
"""
    
    startup_script_path = os.path.expanduser("~/start_pqs_40_qubit.sh")
    with open(startup_script_path, 'w') as f:
        f.write(startup_script)
    
    os.chmod(startup_script_path, 0o755)
    logger.info(f"✅ Startup script created: {startup_script_path}")
    return True

def create_stop_script():
    """Create stop script for production"""
    logger.info("⏹️  Creating stop script...")
    
    stop_script = """#!/bin/bash
# PQS 40-Qubit Production Stop Script

PID_FILE=~/.pqs_40_qubit.pid

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping PQS 40-Qubit (PID: $PID)..."
        kill -TERM "$PID"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "PQS 40-Qubit stopped successfully"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if necessary
        echo "Force stopping PQS 40-Qubit..."
        kill -KILL "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "PQS 40-Qubit force stopped"
    else
        echo "PQS 40-Qubit is not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    # Try to find and kill by process name
    if pgrep -f "fixed_production_40_qubit_app.py" > /dev/null; then
        echo "Stopping PQS 40-Qubit processes..."
        pkill -f "fixed_production_40_qubit_app.py"
        echo "PQS 40-Qubit stopped"
    else
        echo "PQS 40-Qubit is not running"
    fi
fi
"""
    
    stop_script_path = os.path.expanduser("~/stop_pqs_40_qubit.sh")
    with open(stop_script_path, 'w') as f:
        f.write(stop_script)
    
    os.chmod(stop_script_path, 0o755)
    logger.info(f"✅ Stop script created: {stop_script_path}")
    return True

def create_health_check_script():
    """Create health check script"""
    logger.info("🏥 Creating health check script...")
    
    health_check_script = """#!/usr/bin/env python3
# PQS 40-Qubit Health Check Script

import requests
import psutil
import json
import sys
import os
from datetime import datetime

def check_dashboard():
    \"\"\"Check if dashboard is responding\"\"\"
    try:
        response = requests.get('http://localhost:5001/api/quantum/status', timeout=5)
        return response.status_code == 200
    except:
        return False

def check_process():
    \"\"\"Check if main process is running\"\"\"
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'fixed_production_40_qubit_app.py' in ' '.join(proc.info['cmdline'] or []):
                return True
        except:
            continue
    return False

def check_memory_usage():
    \"\"\"Check memory usage\"\"\"
    memory = psutil.virtual_memory()
    return memory.percent < 90  # Alert if memory usage > 90%

def check_disk_space():
    \"\"\"Check disk space\"\"\"
    disk = psutil.disk_usage('/')
    return disk.percent < 90  # Alert if disk usage > 90%

def main():
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'dashboard_responding': check_dashboard(),
        'process_running': check_process(),
        'memory_ok': check_memory_usage(),
        'disk_ok': check_disk_space(),
        'overall_healthy': True
    }
    
    # Determine overall health
    health_status['overall_healthy'] = all([
        health_status['dashboard_responding'],
        health_status['process_running'],
        health_status['memory_ok'],
        health_status['disk_ok']
    ])
    
    # Save health status
    health_file = os.path.expanduser('~/.pqs_health_status.json')
    with open(health_file, 'w') as f:
        json.dump(health_status, f, indent=2)
    
    # Exit with appropriate code
    sys.exit(0 if health_status['overall_healthy'] else 1)

if __name__ == '__main__':
    main()
"""
    
    health_script_path = os.path.expanduser("~/check_pqs_health.py")
    with open(health_script_path, 'w') as f:
        f.write(health_check_script)
    
    os.chmod(health_script_path, 0o755)
    logger.info(f"✅ Health check script created: {health_script_path}")
    return True

def create_launchd_plist():
    """Create macOS LaunchAgent plist for auto-start"""
    logger.info("🍎 Creating macOS LaunchAgent plist...")
    
    home_dir = os.path.expanduser("~")
    startup_script = f"{home_dir}/start_pqs_40_qubit.sh"
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pqs.40qubit.framework</string>
    <key>ProgramArguments</key>
    <array>
        <string>{startup_script}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{home_dir}/.pqs_logs/launchd.out</string>
    <key>StandardErrorPath</key>
    <string>{home_dir}/.pqs_logs/launchd.err</string>
    <key>WorkingDirectory</key>
    <string>{os.getcwd()}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONPATH</key>
        <string>{os.getcwd()}</string>
    </dict>
</dict>
</plist>"""
    
    launchagents_dir = os.path.expanduser("~/Library/LaunchAgents")
    os.makedirs(launchagents_dir, exist_ok=True)
    
    plist_path = f"{launchagents_dir}/com.pqs.40qubit.framework.plist"
    
    with open(plist_path, 'w') as f:
        f.write(plist_content)
    
    logger.info(f"✅ LaunchAgent plist created: {plist_path}")
    logger.info("   To enable auto-start: launchctl load ~/Library/LaunchAgents/com.pqs.40qubit.framework.plist")
    return True

def validate_quantum_components():
    """Validate quantum components are working"""
    logger.info("🔬 Validating quantum components...")
    
    try:
        # Test imports
        from quantum_circuit_manager_40 import QuantumCircuitManager40
        from quantum_entanglement_engine import QuantumEntanglementEngine
        from apple_silicon_quantum_accelerator import AppleSiliconQuantumAccelerator
        from quantum_ml_interface import QuantumMLInterface
        from quantum_visualization_engine import QuantumVisualizationEngine
        from quantum_performance_benchmarking import QuantumPerformanceBenchmarking
        
        logger.info("  ✅ All quantum components imported successfully")
        
        # Test basic functionality
        circuit_manager = QuantumCircuitManager40(max_qubits=40)
        test_circuit = circuit_manager.create_40_qubit_circuit()
        logger.info("  ✅ 40-qubit circuit creation working")
        
        entanglement_engine = QuantumEntanglementEngine()
        test_pairs = entanglement_engine.create_entangled_pairs([0, 1, 2, 3])
        logger.info("  ✅ Entanglement engine working")
        
        accelerator = AppleSiliconQuantumAccelerator()
        backend_config = accelerator.initialize_metal_quantum_backend()
        logger.info(f"  ✅ Apple Silicon acceleration: {backend_config['device']}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Quantum component validation failed: {e}")
        return False

def run_production_tests():
    """Run production readiness tests"""
    logger.info("🧪 Running production readiness tests...")
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Dependencies
    if check_dependencies():
        logger.info("  ✅ Test 1: Dependencies check passed")
        tests_passed += 1
    else:
        logger.error("  ❌ Test 1: Dependencies check failed")
    
    # Test 2: Quantum components
    if validate_quantum_components():
        logger.info("  ✅ Test 2: Quantum components validation passed")
        tests_passed += 1
    else:
        logger.error("  ❌ Test 2: Quantum components validation failed")
    
    # Test 3: Configuration
    if os.path.exists(os.path.expanduser("~/.pqs_40_qubit_config.json")):
        logger.info("  ✅ Test 3: Configuration file exists")
        tests_passed += 1
    else:
        logger.error("  ❌ Test 3: Configuration file missing")
    
    # Test 4: Scripts
    startup_script = os.path.expanduser("~/start_pqs_40_qubit.sh")
    if os.path.exists(startup_script) and os.access(startup_script, os.X_OK):
        logger.info("  ✅ Test 4: Startup script exists and is executable")
        tests_passed += 1
    else:
        logger.error("  ❌ Test 4: Startup script missing or not executable")
    
    # Test 5: Main app file
    if os.path.exists("fixed_production_40_qubit_app.py"):
        logger.info("  ✅ Test 5: Main application file exists")
        tests_passed += 1
    else:
        logger.error("  ❌ Test 5: Main application file missing")
    
    logger.info(f"📊 Production tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def main():
    """Main deployment function"""
    logger.info("🚀 Starting PQS 40-Qubit Production Deployment")
    logger.info("=" * 60)
    
    deployment_steps = [
        ("Check Dependencies", check_dependencies),
        ("Create Production Config", create_production_config),
        ("Setup Logging", setup_logging),
        ("Create Startup Script", create_startup_script),
        ("Create Stop Script", create_stop_script),
        ("Create Health Check", create_health_check_script),
        ("Create LaunchAgent", create_launchd_plist),
        ("Run Production Tests", run_production_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function in deployment_steps:
        logger.info(f"\n📋 {step_name}...")
        try:
            if step_function():
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"💥 {step_name} error: {e}")
            failed_steps.append(step_name)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    if not failed_steps:
        logger.info("🎉 PQS 40-Qubit Production Deployment Complete!")
        logger.info("✅ All deployment steps successful")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Run: ~/start_pqs_40_qubit.sh")
        logger.info("2. Check dashboard: http://localhost:5001")
        logger.info("3. Monitor logs: tail -f ~/.pqs_40_qubit.log")
        logger.info("4. Health check: ~/check_pqs_health.py")
        logger.info("5. Stop: ~/stop_pqs_40_qubit.sh")
        logger.info("\n🔧 Optional: Enable auto-start with:")
        logger.info("   launchctl load ~/Library/LaunchAgents/com.pqs.40qubit.framework.plist")
        return True
    else:
        logger.error("❌ PQS 40-Qubit Production Deployment Failed")
        logger.error(f"Failed steps: {', '.join(failed_steps)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)