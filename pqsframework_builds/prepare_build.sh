#!/bin/bash
# This script prepares the build directory by copying all necessary files.

set -e

DEST_DIR="pqsframework_builds"
SRC_DIR="."

echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Copying files..."

# Copy main application files
cp "universal_pqs_app.py" "$DEST_DIR/"
cp "$SRC_DIR/app.py" "$DEST_DIR/"
cp "$SRC_DIR/__main__.py" "$DEST_DIR/"

# Copy quantum implementation
cp "$SRC_DIR/quantum_ml_hybrid.py" "$DEST_DIR/"
cp "$SRC_DIR/real_quantum_engine.py" "$DEST_DIR/"
cp "$SRC_DIR/real_ml_system.py" "$DEST_DIR/"
cp "$SRC_DIR/metal_quantum_simulator.py" "$DEST_DIR/"

# Copy web assets
cp -r "static" "$DEST_DIR/"
cp -r "templates" "$DEST_DIR/"

# Create __init__.py
touch "$DEST_DIR/__init__.py"

echo "All files copied successfully."
