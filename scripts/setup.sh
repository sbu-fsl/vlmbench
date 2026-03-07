#!/usr/bin/env bash

set -e  # Exit on error

# Default values
VENV_NAME=${1:-venv}
REQUIREMENTS_FILE=${2:-requirements.txt}

echo "Using virtual environment name: $VENV_NAME"
echo "Using requirements file: $REQUIREMENTS_FILE"

# Check if python3 exists
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Detected $PYTHON_VERSION"

# Create venv if not exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment already exists."
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements if file exists
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "No requirements file found. Skipping dependency installation."
fi

echo "Done!"
echo "To activate later, run:"
echo "source $VENV_NAME/bin/activate"
