#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_FILE="$SCRIPT_DIR/apps.conf"
STATE_FILE="/tmp/m3_macbook_air_suspended_pids.txt"

# --- Configuration ---
# The number of seconds of inactivity before apps are suspended.
IDLE_TIMEOUT_SECONDS=300 # 5 minutes

# --- Helper Functions ---

# Check if the system is on battery power
is_on_battery() {
    pmset -g batt | grep -q 'Now drawing from ''Battery Power'''
}

# Get the system idle time in seconds
get_idle_time() {
    ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1000000000; exit}'
}

# Log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# --- Core Logic ---

suspend_apps() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR: Configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    log "Idle time exceeded. Suspending configured applications..."
    touch "$STATE_FILE"

    # Read each line from the config file
    while IFS= read -r app_name || [[ -n "$app_name" ]]; do
        # Skip empty lines or comments
        if [[ -z "$app_name" ]] || [[ "$app_name" == \#* ]]; then
            continue
        fi

        # Find PIDs for the application, excluding grep itself and this script
        pids=$(pgrep -fil "$app_name" | grep -v -e "grep" -e "optimizer.sh" | awk '{print $1}')

        if [ -n "$pids" ]; then
            for pid in $pids; do
                # Check if the PID is not already suspended by us
                if ! grep -q "^$pid$" "$STATE_FILE"; then
                    log "Suspending '$app_name' (PID: $pid)..."
                    kill -STOP "$pid"
                    echo "$pid" >> "$STATE_FILE" # Record that we suspended this PID
                fi
            done
        fi
    done < "$CONFIG_FILE"
}

resume_apps() {
    if [ ! -f "$STATE_FILE" ]; then
        return # Nothing to resume
    fi

    log "User is active. Resuming suspended applications..."
    
    while IFS= read -r pid || [[ -n "$pid" ]]; do
        if [ -n "$pid" ]; then
            # Check if the process still exists before trying to resume
            if ps -p "$pid" > /dev/null; then
                log "Resuming PID: $pid..."
                kill -CONT "$pid"
            fi
        fi
    done < "$STATE_FILE"

    # Clear the state file after resuming all
    rm "$STATE_FILE"
}

# --- Main Execution ---

main() {
    idle_time=$(get_idle_time)

    # Main condition: on battery and idle for longer than the timeout
    if is_on_battery && (( $(echo "$idle_time > $IDLE_TIMEOUT_SECONDS" | bc -l) )); then
        # If apps are already suspended, do nothing.
        if [ -f "$STATE_FILE" ]; then
            exit 0
        fi
        suspend_apps
    else
        # If not idle or plugged in, ensure apps are resumed.
        if [ -f "$STATE_FILE" ]; then
            resume_apps
        fi
    fi
}

# Run the main function
main
