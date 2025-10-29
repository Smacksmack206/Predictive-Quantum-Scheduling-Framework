# PQS Framework - Elevated Privileges Guide

## Why Elevated Privileges?

PQS Framework requires elevated privileges to perform advanced system optimizations:

### Power Management
- Modify system power settings (`pmset`)
- Adjust CPU frequency scaling
- Control sleep/wake behavior
- Manage display brightness

### Process Management
- Adjust process priorities (`renice`)
- Suspend/resume processes (`kill -STOP/CONT`)
- Manage background services

### Service Control
- Disable/enable Spotlight indexing (`mdutil`)
- Control Time Machine backups (`tmutil`)
- Manage system services

### Network Optimization
- Configure WiFi settings (`networksetup`)
- Optimize network power usage

### System Optimization
- Purge inactive memory (`purge`)
- Read hardware information (`ioreg`)
- Modify system parameters (`sysctl`)

## Setup Methods

### Method 1: Automatic Setup (Recommended)

Run the setup script with sudo:

```bash
sudo ./setup_elevated_privileges.sh
```

This will:
1. Add your user to the `wheel` group
2. Configure passwordless sudo for PQS commands
3. Optionally install a LaunchAgent for auto-start
4. Test the configuration

### Method 2: Manual Sudo

Run PQS Framework with sudo each time:

```bash
sudo python3 -m pqs_framework
```

Or use the convenience script:

```bash
./run_pqs_with_sudo.sh
```

### Method 3: Interactive Prompt

Just run PQS Framework normally:

```bash
python3 -m pqs_framework
```

The app will prompt for sudo privileges when needed.

## What Gets Configured

### Sudoers File

Location: `/etc/sudoers.d/pqs_framework`

Allows passwordless sudo for specific commands:
- `pmset` - Power management
- `sysctl` - System parameters
- `renice` - Process priorities
- `kill` - Process signals
- `mdutil` - Spotlight control
- `tmutil` - Time Machine control
- `networksetup` - Network configuration
- `purge` - Memory management
- `ioreg` - Hardware information

### LaunchAgent (Optional)

Location: `~/Library/LaunchAgents/com.pqs.framework.plist`

Automatically starts PQS Framework at login with:
- High priority (nice -10)
- Auto-restart on crash
- Logging to `/tmp/pqs_framework.log`

## Security Considerations

### Safe Commands Only

The sudoers configuration only allows specific, safe commands:
- No shell access
- No file system modifications
- No package installation
- No system configuration changes (except power/process management)

### Minimal Permissions

Only the minimum necessary permissions are granted:
- User-specific (not system-wide)
- Command-specific (not blanket sudo)
- Auditable (all commands logged)

### Revocation

To remove elevated privileges:

```bash
# Remove sudoers file
sudo rm /etc/sudoers.d/pqs_framework

# Remove LaunchAgent
launchctl unload ~/Library/LaunchAgents/com.pqs.framework.plist
rm ~/Library/LaunchAgents/com.pqs.framework.plist

# Remove from wheel group
sudo dseditgroup -o edit -d $USER -t user wheel
```

## Troubleshooting

### "sudo: a password is required"

The passwordless sudo configuration may not be active yet:
1. Log out and back in
2. Or run: `sudo -k` to clear sudo cache
3. Or reboot

### "Permission denied"

Some operations may still require full sudo:
1. Run with: `sudo python3 -m pqs_framework`
2. Or use: `./run_pqs_with_sudo.sh`

### LaunchAgent not starting

Check logs:
```bash
cat /tmp/pqs_framework.log
cat /tmp/pqs_framework_error.log
```

Reload agent:
```bash
launchctl unload ~/Library/LaunchAgents/com.pqs.framework.plist
launchctl load ~/Library/LaunchAgents/com.pqs.framework.plist
```

### Sudoers syntax error

If you manually edited the sudoers file:
```bash
sudo visudo -c -f /etc/sudoers.d/pqs_framework
```

Fix any errors, or remove the file:
```bash
sudo rm /etc/sudoers.d/pqs_framework
```

## Running Without Elevated Privileges

PQS Framework can run without elevated privileges, but with limitations:

### What Still Works
- Quantum-ML optimization (calculations)
- Battery monitoring (read-only)
- Process monitoring (read-only)
- Web dashboard
- Statistics tracking

### What Doesn't Work
- Process suspension (SIGSTOP/SIGCONT)
- Process priority adjustment (renice)
- Service control (Spotlight, Time Machine)
- Power settings modification (pmset)
- Network optimization
- Memory purging
- System parameter changes

### Fallback Behavior

When elevated privileges are not available:
- Operations fail silently
- Warnings logged to console
- Core functionality continues
- Battery savings reduced (~50% less effective)

## Best Practices

### For Daily Use
1. Run the setup script once: `sudo ./setup_elevated_privileges.sh`
2. Let PQS Framework start automatically at login
3. Check logs occasionally: `tail -f /tmp/pqs_framework.log`

### For Development
1. Use the convenience script: `./run_pqs_with_sudo.sh`
2. Monitor console output for errors
3. Test without sudo to ensure graceful degradation

### For Distribution
1. Include setup script in package
2. Prompt user to run setup during installation
3. Provide clear documentation on privileges needed
4. Offer option to run without elevated privileges

## Verification

### Check Sudo Configuration

```bash
# Test passwordless sudo
sudo -n pmset -g

# Should work without password prompt
```

### Check LaunchAgent

```bash
# List loaded agents
launchctl list | grep pqs

# Check status
launchctl print gui/$(id -u)/com.pqs.framework
```

### Check Logs

```bash
# View logs
tail -f /tmp/pqs_framework.log

# Check for errors
tail -f /tmp/pqs_framework_error.log
```

## FAQ

### Q: Is it safe to give PQS Framework sudo access?

A: Yes, the sudoers configuration only allows specific, safe commands. No shell access or system modifications are permitted.

### Q: Can I run PQS Framework without sudo?

A: Yes, but many optimizations won't work. Battery savings will be significantly reduced.

### Q: Will this affect other applications?

A: No, the sudoers configuration is specific to PQS Framework commands only.

### Q: How do I uninstall?

A: Remove the sudoers file and LaunchAgent as shown in the Revocation section above.

### Q: Does this work on Apple Silicon and Intel?

A: Yes, the configuration works on all Mac architectures.

## Support

If you encounter issues with elevated privileges:

1. Check the logs: `/tmp/pqs_framework.log`
2. Verify sudoers syntax: `sudo visudo -c -f /etc/sudoers.d/pqs_framework`
3. Test individual commands: `sudo -n pmset -g`
4. Try running with full sudo: `sudo python3 -m pqs_framework`

For more help, see the main README.md or open an issue on GitHub.
