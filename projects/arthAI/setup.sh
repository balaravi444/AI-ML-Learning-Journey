#!/bin/bash

# 1. Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# 2. Activate the environment
# Note: Users need to run 'source venv/bin/activate' manually after this script
source venv/bin/activate

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "-----------------------------------------------"
echo "Setup complete! ✅"
echo "To start your environment, run: source venv/bin/activate"
echo "-----------------------------------------------"
