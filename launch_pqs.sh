#!/bin/bash
# PQS Framework 40-Qubit Launcher Script
# Ensures proper startup on Intel and Apple Silicon Macs

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_DIR="$(dirname "$SCRIPT_DIR")"

echo "PQS Framework 40-Qubit Starting..."
echo "Script directory: $SCRIPT_DIR"
echo "App directory: $APP_DIR"

# Detect system architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Set Python paths based on architecture
if [[ "$ARCH" == "arm64" ]]; then
    # Apple Silicon paths
    PYTHON_PATHS=(
        "/opt/homebrew/bin/python3"
        "/opt/homebrew/bin/python3.13"
        "/opt/homebrew/bin/python3.12"
        "/opt/homebrew/bin/python3.11"
        "/usr/bin/python3"
        "/usr/local/bin/python3"
        "python3"
    )
else
    # Intel Mac paths
    PYTHON_PATHS=(
        "/usr/local/bin/python3"
        "/usr/local/bin/python3.13"
        "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11"
        "/usr/bin/python3"
        "/opt/homebrew/bin/python3"
        "python3"
    )
fi

# Try to find a working Python installation
PYTHON_CMD=""
for python_path in "${PYTHON_PATHS[@]}"; do
    if command -v "$python_path" &> /dev/null; then
        echo "Found Python at: $python_path"
        PYTHON_CMD="$python_path"
        break
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Error: No Python installation found!"
    echo "Please install Python 3.8 or later"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using Python: $PYTHON_VERSION"

# Set environment variables
export PYTHONPATH="$APP_DIR/Resources:$APP_DIR/Resources/lib/python3.13/site-packages:$PYTHONPATH"
export DYLD_LIBRARY_PATH="$APP_DIR/Frameworks:$DYLD_LIBRARY_PATH"

# Change to the app directory
cd "$APP_DIR/Resources" || cd "$SCRIPT_DIR"

# Try to run the main application
if [[ -f "fixed_40_qubit_app.py" ]]; then
    echo "Running main application..."
    exec "$PYTHON_CMD" "fixed_40_qubit_app.py" "$@"
elif [[ -f "app_launcher.py" ]]; then
    echo "Running app launcher..."
    exec "$PYTHON_CMD" "app_launcher.py" "$@"
else
    echo "Error: Main application files not found!"
    exit 1
fi