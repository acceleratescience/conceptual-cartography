#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/"

INSTALL_DEV=false
if [[ "$1" == "--dev" ]]; then
  INSTALL_DEV=true
fi

if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  uv venv
fi

source .venv/bin/activate

if $INSTALL_DEV; then
  echo "Installing with dev dependencies..."
  uv pip install -e ".[dev]"
else
  echo "Installing core dependencies only..."
  uv pip install -e .
fi
