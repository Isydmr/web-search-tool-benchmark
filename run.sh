#!/bin/bash

# Set the Python path
export PYTHONPATH=/app
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8080}"

# Start the application
echo "Starting application..."
python -u app/main.py
