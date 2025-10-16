#!/usr/bin/env python3
"""
Startup script for the fixed 40-qubit system
Handles port conflicts and ensures clean startup
"""

import subprocess
import time
import os
import signal
import psutil
import sys

def kill_existing_processes():
    """Kill any existing 40-qubit processes"""
    print("🔄 Cleaning up existing processes...")
    
    # Kill processes by pattern
    patterns = [
        "python.*40.*qubit",
        "python.*perfect_40_qubit",
        "python.*production.*40",
        "python.*fixed_40_qubit"
    ]
    
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pkill", "-f", pattern], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"   ✅ Killed processes matching: {pattern}")
            else:
                print(f"   ℹ️  No processes found for: {pattern}")
        except Exception as e:
            print(f"   ⚠️  Error killing {pattern}: {e}")
    
    # Kill processes using ports 5001-5005
    for port in range(5001, 5006):
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], 
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"   ✅ Killed process {pid} using port {port}")
                    except:
                        pass
        except:
            pass
    
    # Wait for cleanup
    time.sleep(2)
    print("✅ Cleanup complete")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        'rumps', 'psutil', 'flask', 'numpy'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"   ❌ {module} - MISSING")
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def find_available_port(start_port=5002):
    """Find an available port"""
    import socket
    
    for port in range(start_port, start_port + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def start_system():
    """Start the fixed 40-qubit system"""
    print("🚀 Starting Fixed 40-Qubit PQS Framework...")
    
    # Check if file exists
    if not os.path.exists('fixed_40_qubit_app.py'):
        print("❌ fixed_40_qubit_app.py not found!")
        print("Make sure you're in the correct directory.")
        return False
    
    # Find available port
    port = find_available_port()
    if port:
        print(f"🌐 Using port {port}")
    else:
        print("⚠️  No available ports found, using default 5002")
        port = 5002
    
    try:
        # Start the application
        print("🎯 Launching application...")
        
        # Use subprocess to start the app
        process = subprocess.Popen([
            sys.executable, 'fixed_40_qubit_app.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Application started successfully!")
            print(f"🌐 Dashboard: http://localhost:{port}")
            print("📱 Menu bar app should be running")
            print("🎯 All API endpoints should be working")
            print("\n📋 To stop the application:")
            print("   Press Ctrl+C or run: pkill -f 'python.*fixed_40_qubit'")
            
            # Wait for the process to complete
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping application...")
                process.terminate()
                process.wait()
                print("✅ Application stopped")
            
            return True
        else:
            # Process died, get error output
            stdout, stderr = process.communicate()
            print("❌ Application failed to start!")
            if stderr:
                print(f"Error: {stderr}")
            if stdout:
                print(f"Output: {stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def main():
    """Main startup function"""
    print("🔧 40-Qubit System Startup Script")
    print("=" * 40)
    
    # Step 1: Kill existing processes
    kill_existing_processes()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start due to missing dependencies")
        return
    
    # Step 3: Start the system
    success = start_system()
    
    if success:
        print("\n🎉 System started successfully!")
    else:
        print("\n❌ System failed to start")
        print("\n🔧 Troubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Make sure no other processes are using the ports")
        print("3. Check the error messages above")
        print("4. Try running manually: python3 fixed_40_qubit_app.py")

if __name__ == "__main__":
    main()