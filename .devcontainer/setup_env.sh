#!/usr/bin/env sh
set -eu
if [ ! -f /workspace/.env ]; then
  echo "📝  Generating default .env from template"
  cp /workspace/.devcontainer/.env.template /workspace/.env
fi

