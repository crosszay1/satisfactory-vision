#!/usr/bin/env bash

uv tool install ruff # Install ruff
curl https://cursor.com/install -fsS | bash # Install cursor

uv sync
