#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Privilege Manager
=================
Ensures PQS Framework has elevated privileges for all operations.
Handles sudo authentication and privilege escalation safely.
"""

import os
import sys
import subprocess
import getpass
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivilegeManager:
    """Manages elevated privileges for PQS Framework"""
    
    def __init__(self):
        self.has_root = False
        self.sudo_password = None
        self._check_privileges()
    
    def _check_privileges(self):
        """Check if we have root privileges"""
        self.has_root = os.geteuid() == 0
        
        if self.has_root:
            logger.info("✅ Running with root privileges")
        else:
            logger.info("⚠️ Running without root privileges")
    
    def ensure_privileges(self) -> bool:
        """Ensure we have elevated privileges (silent, non-intrusive)"""
        if self.has_root:
            return True
        
        # Silently check if sudo is available
        return self._check_sudo_silent()
    
    def _check_sudo_silent(self) -> bool:
        """Silently check if sudo access is available (no prompts)"""
        try:
            # Test sudo access without prompting
            result = subprocess.run(
                ['sudo', '-n', 'true'],
                capture_output=True,
                timeout=1
            )
            
            if result.returncode == 0:
                logger.debug("✅ Sudo access available (cached)")
                # Keep sudo alive in background
                self._keep_sudo_alive()
                return True
            else:
                logger.debug("⚠️ Sudo not available (no cached credentials)")
                return False
        
        except Exception as e:
            logger.debug(f"Sudo check failed: {e}")
            return False
    
    def _keep_sudo_alive(self):
        """Keep sudo privileges alive in background"""
        import threading
        
        def refresh_sudo():
            while True:
                try:
                    # Refresh sudo every 4 minutes (sudo timeout is usually 5 min)
                    import time
                    time.sleep(240)
                    subprocess.run(
                        ['sudo', '-n', '-v'],
                        capture_output=True,
                        timeout=1
                    )
                except:
                    break
        
        thread = threading.Thread(target=refresh_sudo, daemon=True)
        thread.start()
        logger.info("🔄 Sudo keep-alive started")
    
    def run_privileged_command(self, cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
        """Run a command with elevated privileges"""
        try:
            if self.has_root:
                # Already root, run directly
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                # Use sudo
                sudo_cmd = ['sudo', '-n'] + cmd
                return subprocess.run(
                    sudo_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Error running privileged command: {e}")
            raise
    
    def run_privileged_command_safe(self, cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
        """Run a privileged command safely (returns None on error)"""
        try:
            return self.run_privileged_command(cmd, timeout)
        except Exception as e:
            logger.debug(f"Privileged command failed (safe): {e}")
            return None
    
    def restart_with_privileges(self):
        """Restart the application with elevated privileges"""
        try:
            logger.info("🔄 Restarting with elevated privileges...")
            
            # Get current script path
            script_path = sys.argv[0]
            
            # Build sudo command
            sudo_cmd = ['sudo', sys.executable] + sys.argv
            
            # Execute with sudo
            os.execvp('sudo', sudo_cmd)
        
        except Exception as e:
            logger.error(f"❌ Error restarting with privileges: {e}")
            sys.exit(1)
    
    def check_and_request_privileges(self) -> bool:
        """Check privileges silently (non-intrusive for standalone app)"""
        if self.has_root:
            return True
        
        # Silently check if sudo is available
        return self._check_sudo_silent()


# Global instance
_privilege_manager = None


def get_privilege_manager() -> PrivilegeManager:
    """Get or create global privilege manager"""
    global _privilege_manager
    
    if _privilege_manager is None:
        _privilege_manager = PrivilegeManager()
    
    return _privilege_manager


def ensure_privileges() -> bool:
    """Ensure the application has elevated privileges"""
    manager = get_privilege_manager()
    return manager.ensure_privileges()


def run_privileged(cmd: List[str], timeout: int = 5) -> subprocess.CompletedProcess:
    """Run a command with elevated privileges"""
    manager = get_privilege_manager()
    return manager.run_privileged_command(cmd, timeout)


def run_privileged_safe(cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
    """Run a privileged command safely (returns None on error)"""
    manager = get_privilege_manager()
    return manager.run_privileged_command_safe(cmd, timeout)


if __name__ == "__main__":
    print("🔐 Privilege Manager Test")
    print("="*60)
    
    manager = PrivilegeManager()
    
    print(f"\n📊 Current Status:")
    print(f"   Has Root: {manager.has_root}")
    print(f"   UID: {os.geteuid()}")
    print(f"   User: {getpass.getuser()}")
    
    if not manager.has_root:
        print("\n🔐 Testing sudo access...")
        if manager.ensure_privileges():
            print("✅ Sudo access granted")
            
            # Test privileged command
            print("\n🧪 Testing privileged command...")
            result = manager.run_privileged_command_safe(['whoami'])
            if result:
                print(f"   Running as: {result.stdout.strip()}")
        else:
            print("❌ Sudo access denied")
    
    print("\n✅ Test complete")
