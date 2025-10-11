#!/usr/bin/env python3
"""
Permission Manager - Handle elevated permissions at startup
Ensures all required permissions are obtained upfront
"""

import subprocess
import os
import sys
import getpass
import time
from typing import Optional, Dict, List

class PermissionManager:
    """Manages system permissions for Ultimate EAS"""
    
    def __init__(self):
        self.sudo_available = False
        self.powermetrics_available = False
        self.system_extensions_available = False
        self.cached_sudo_session = False
        
    def check_and_request_permissions(self) -> Dict[str, bool]:
        """Check and request all required permissions at startup"""
        print("🔐 Checking system permissions...")
        
        permissions = {
            'sudo': self._check_sudo_access(),
            'powermetrics': self._check_powermetrics_access(),
            'system_extensions': self._check_system_extensions(),
            'hardware_monitoring': False
        }
        
        # Request sudo if needed
        if not permissions['sudo']:
            print("⚠️  Some features require elevated permissions")
            print("   This enables advanced hardware monitoring and optimization")
            
            if self._request_sudo_permission():
                permissions['sudo'] = True
                permissions['powermetrics'] = True
                permissions['hardware_monitoring'] = True
                print("✅ Elevated permissions granted")
            else:
                print("⚠️  Running with limited permissions")
                print("   Some advanced features will be disabled")
        else:
            permissions['hardware_monitoring'] = True
            
        return permissions
    
    def _check_sudo_access(self) -> bool:
        """Check if we have sudo access without prompting"""
        try:
            # Check if we're already root
            if os.geteuid() == 0:
                self.sudo_available = True
                return True
                
            # Check if sudo is available and cached
            result = subprocess.run(
                ['sudo', '-n', 'true'], 
                capture_output=True, 
                timeout=2
            )
            
            if result.returncode == 0:
                self.sudo_available = True
                self.cached_sudo_session = True
                return True
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return False
    
    def _request_sudo_permission(self) -> bool:
        """Request sudo permission with user consent"""
        try:
            print("🔑 Requesting elevated permissions for hardware monitoring...")
            print("   Enter your password when prompted:")
            
            # Test sudo access with a simple command
            result = subprocess.run(
                ['sudo', '-v'], 
                timeout=30,
                input='',
                text=True
            )
            
            if result.returncode == 0:
                self.sudo_available = True
                self.cached_sudo_session = True
                
                # Keep sudo session alive
                self._maintain_sudo_session()
                return True
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, KeyboardInterrupt):
            print("❌ Permission request cancelled or failed")
            
        return False
    
    def _maintain_sudo_session(self):
        """Keep sudo session alive to avoid repeated prompts"""
        def refresh_sudo():
            while self.cached_sudo_session:
                try:
                    subprocess.run(['sudo', '-v'], 
                                 capture_output=True, 
                                 timeout=5)
                    time.sleep(240)  # Refresh every 4 minutes
                except:
                    break
        
        import threading
        sudo_thread = threading.Thread(target=refresh_sudo, daemon=True)
        sudo_thread.start()
    
    def _check_powermetrics_access(self) -> bool:
        """Check if powermetrics is available"""
        try:
            result = subprocess.run(['which', 'powermetrics'], 
                                  capture_output=True, 
                                  timeout=5)
            self.powermetrics_available = result.returncode == 0
            return self.powermetrics_available
        except:
            return False
    
    def _check_system_extensions(self) -> bool:
        """Check system extension permissions"""
        try:
            # Check if we can access system information
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                  capture_output=True, 
                                  timeout=10)
            self.system_extensions_available = result.returncode == 0
            return self.system_extensions_available
        except:
            return False
    
    def execute_with_sudo(self, command: List[str], timeout: int = 10) -> Optional[subprocess.CompletedProcess]:
        """Execute command with sudo if available"""
        if not self.sudo_available:
            return None
            
        try:
            full_command = ['sudo'] + command
            return subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
    
    def get_permission_status(self) -> Dict[str, str]:
        """Get current permission status"""
        return {
            'sudo': '✅ Available' if self.sudo_available else '❌ Not available',
            'powermetrics': '✅ Available' if self.powermetrics_available else '❌ Not available',
            'system_extensions': '✅ Available' if self.system_extensions_available else '❌ Not available',
            'hardware_monitoring': '✅ Full access' if self.sudo_available else '⚠️  Limited access'
        }

# Global permission manager instance
permission_manager = PermissionManager()