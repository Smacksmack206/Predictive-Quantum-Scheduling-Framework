#!/bin/bash
# Setup PQS Framework with elevated privileges
# This configures the system to always run PQS with necessary permissions

set -e

echo "🔐 PQS Framework - Elevated Privileges Setup"
echo "============================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run with sudo"
    echo "   Usage: sudo ./setup_elevated_privileges.sh"
    exit 1
fi

echo "✅ Running with root privileges"
echo ""

# Get the actual user (not root)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

echo "📋 Configuration:"
echo "   User: $ACTUAL_USER"
echo "   Home: $ACTUAL_HOME"
echo ""

# 1. Add user to necessary groups
echo "👥 Adding user to system groups..."
dseditgroup -o edit -a "$ACTUAL_USER" -t user wheel 2>/dev/null || true
echo "   ✅ Added to wheel group"

# 2. Configure sudoers for passwordless sudo (for specific commands)
echo ""
echo "🔧 Configuring sudoers..."
SUDOERS_FILE="/etc/sudoers.d/pqs_framework"

cat > "$SUDOERS_FILE" << EOF
# PQS Framework - Passwordless sudo for specific commands
# This allows PQS Framework to perform system optimizations

# Power management
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/pmset
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/sbin/sysctl

# Process management
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/renice
$ACTUAL_USER ALL=(ALL) NOPASSWD: /bin/kill

# Service management
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/mdutil
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/tmutil

# Network management
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/sbin/networksetup

# System utilities
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/bin/purge
$ACTUAL_USER ALL=(ALL) NOPASSWD: /usr/sbin/ioreg
EOF

chmod 0440 "$SUDOERS_FILE"
echo "   ✅ Sudoers configured: $SUDOERS_FILE"

# 3. Verify sudoers syntax
echo ""
echo "🔍 Verifying sudoers syntax..."
if visudo -c -f "$SUDOERS_FILE"; then
    echo "   ✅ Sudoers syntax valid"
else
    echo "   ❌ Sudoers syntax error - removing file"
    rm -f "$SUDOERS_FILE"
    exit 1
fi

# 4. Set up LaunchAgent (optional)
echo ""
read -p "📦 Install LaunchAgent to run at login? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LAUNCH_AGENTS_DIR="$ACTUAL_HOME/Library/LaunchAgents"
    mkdir -p "$LAUNCH_AGENTS_DIR"
    
    # Copy plist
    if [ -f "com.pqs.framework.plist" ]; then
        cp com.pqs.framework.plist "$LAUNCH_AGENTS_DIR/"
        chown "$ACTUAL_USER:staff" "$LAUNCH_AGENTS_DIR/com.pqs.framework.plist"
        echo "   ✅ LaunchAgent installed"
        
        # Load the agent
        sudo -u "$ACTUAL_USER" launchctl load "$LAUNCH_AGENTS_DIR/com.pqs.framework.plist" 2>/dev/null || true
        echo "   ✅ LaunchAgent loaded"
    else
        echo "   ⚠️ com.pqs.framework.plist not found"
    fi
fi

# 5. Test sudo access
echo ""
echo "🧪 Testing sudo access..."
if sudo -u "$ACTUAL_USER" sudo -n pmset -g 2>/dev/null; then
    echo "   ✅ Passwordless sudo working"
else
    echo "   ⚠️ Passwordless sudo may not be working"
    echo "   Try logging out and back in"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 What was configured:"
echo "   1. User added to wheel group"
echo "   2. Passwordless sudo for PQS commands"
echo "   3. Sudoers file: $SUDOERS_FILE"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   4. LaunchAgent installed"
fi
echo ""
echo "🚀 You can now run PQS Framework with:"
echo "   python3 -m pqs_framework"
echo ""
echo "   Or use the convenience script:"
echo "   ./run_pqs_with_sudo.sh"
echo ""
echo "⚠️  Note: You may need to log out and back in for all changes to take effect"
echo ""
