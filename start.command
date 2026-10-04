#!/bin/zsh
set -e

cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run setup.command first."
  exit 1
fi

source .venv/bin/activate
python server.py
