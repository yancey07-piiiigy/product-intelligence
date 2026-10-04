#!/bin/zsh
set -e

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

MODEL_FILE="models/clip-vit-base-patch32/pytorch_model.bin"
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "CLIP model not found. Downloading openai/clip-vit-base-patch32..."
  python prepare_project.py download
else
  echo "CLIP model already exists. Skipping download."
fi

echo "Setup complete. Double-click start.command to run ShoeLens."
