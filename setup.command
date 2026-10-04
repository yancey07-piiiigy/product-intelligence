#!/bin/zsh
set -e

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

echo "Setup complete. Double-click start.command to run ShoeLens."
