#!/bin/bash
# Install the package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Uninstall uvloop if it's installed
echo "Uninstalling uvloop..."
pip uninstall uvloop -y

# Check if USE_TORCH=1 is already set in .env
if grep -q '^USE_TORCH=1$' .env; then
    echo "USE_TORCH=1 is already set in .env, skipping..."
else
    echo "Adding USE_TORCH=1 to .env..."
    # Ensure the file ends with a newline (helps avoid concatenating on the last line).
    sed -i -e '$a\' .env
    echo "USE_TORCH=1" >> .env
fi

# Verify if USE_TORCH=1 is set
if grep -q '^USE_TORCH=1$' .env; then
    echo "Successfully set USE_TORCH=1"
else
    echo "Failed to set USE_TORCH=1"
    echo "Please set USE_TORCH=1 manually in the .env file"
fi

# source the .env file
source .env

echo "Setup complete!"
