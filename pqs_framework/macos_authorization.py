#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS Authorization Helper
==========================
Handles elevated privileges for standalone macOS app using native APIs.
Non-intrusive, uses macOS Security framework for proper authorization.
"""

import subprocess
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import macOS Security framework
try:
    from Foundation import NSBundle
    from Security import (
        AuthorizationCreate,
        AuthorizationExecuteWithPrivileges,
        kAuthorizationFlagDefaults,
        kAuthorizationFlagInteractionAllowed,
        kAuthorizationFlagPreAuthorize,
        kAuthorizationFlagExtendRights
    )
    SECURITY_FRAMEWORK_AVAILABLE = True
except ImportError:
    SECURITY_FRAMEWORK_AVAILABLE = False
    logger.debug("macOS Security framework not available")


class MacOSAuthorization:
    """
    Handles elevated privileges using macOS Security framework.
    Non-intrusive - only prompts when actually needed.
    """
    
    def __init__(self):
        self.authorization = None
        self.has_authorization = False
        
        if SECURITY_FRAMEWORK_AVAILABLE:
            self._initialize_authorization()
    
    def _initialize_authorization(self):
        """Initialize authorization reference"""
        try:
            # Create authorization reference
            status, self.authorization = AuthorizationCreate(
                None,
                None,
                kAuthorizationFlagDefaults,
                None
            )
            
            if status == 0:
                self.has_authorization = True
                logger.debug("✅ Authorization reference created")
            else:
                logger.debug(f"⚠️ Authorization creation failed: {status}")
        
        except Exception as e:
            logger.debug(f"Authorization initialization error: {e}")
    
    def run_with_privileges(self, cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
        """
        Run command with elevated privileges using macOS Security framework.
        Only prompts user when actually needed.
        """
        if not SECURITY_FRAMEWORK_AVAILABLE:
            return self._run_with_sudo_fallback(cmd, timeout)
        
        try:
            # Try to execute with privileges
            # This will prompt the user if needed
            status = AuthorizationExecuteWithPrivileges(
                self.authorization,
                cmd[0],
                kAuthorizationFlagDefaults,
                cmd[1:],
                None
            )
            
            if status == 0:
                logger.debug(f"✅ Executed with privileges: {' '.join(cmd)}")
                return subprocess.CompletedProcess(cmd, 0, '', '')
            else:
                logger.debug(f"⚠️ Execution failed: {status}")
                return None
        
        except Exception as e:
            logger.debug(f"Privileged execution error: {e}")
            return self._run_with_sudo_fallback(cmd, timeout)
    
    def _run_with_sudo_fallback(self, cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
        """Fallback to sudo -n (non-interactive)"""
        try:
            # Try sudo without prompting
            sudo_cmd = ['sudo', '-n'] + cmd
            result = subprocess.run(
                sudo_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.debug(f"✅ Executed with sudo: {' '.join(cmd)}")
                return result
            else:
                logger.debug(f"⚠️ Sudo failed (no cached credentials)")
                return None
        
        except Exception as e:
            logger.debug(f"Sudo fallback error: {e}")
            return None
    
    def run_safe(self, cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
        """
        Run command safely - tries privileged execution, falls back gracefully.
        Never prompts user, never blocks.
        """
        try:
            # First try without privileges
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result
            
            # If failed, try with privileges (non-interactive)
            return self._run_with_sudo_fallback(cmd, timeout)
        
        except Exception as e:
            logger.debug(f"Safe command execution error: {e}")
            return None


# Global instance
_macos_auth = None


def get_macos_authorization() -> MacOSAuthorization:
    """Get or create global macOS authorization instance"""
    global _macos_auth
    
    if _macos_auth is None:
        _macos_auth = MacOSAuthorization()
    
    return _macos_auth


def run_privileged_macos(cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
    """Run command with macOS authorization (non-intrusive)"""
    auth = get_macos_authorization()
    return auth.run_safe(cmd, timeout)


if __name__ == "__main__":
    print("🔐 macOS Authorization Test")
    print("="*60)
    
    auth = MacOSAuthorization()
    
    print(f"\n📊 Status:")
    print(f"   Security Framework: {SECURITY_FRAMEWORK_AVAILABLE}")
    print(f"   Has Authorization: {auth.has_authorization}")
    
    # Test safe command
    print("\n🧪 Testing safe command (pmset -g)...")
    result = auth.run_safe(['pmset', '-g', 'batt'])
    
    if result:
        print(f"✅ Success")
        print(f"   Output: {result.stdout[:100]}...")
    else:
        print("⚠️ Failed (may need privileges)")
    
    print("\n✅ Test complete")
