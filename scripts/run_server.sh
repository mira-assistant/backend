#!/bin/bash

# Mira Backend Server Startup Script

# Set default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
RELOAD=${RELOAD:-"false"}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start the server
echo "Starting Mira Backend server on $HOST:$PORT..."
if [ "$RELOAD" = "true" ]; then
    uvicorn app.main:app --host $HOST --port $PORT --reload
else
    uvicorn app.main:app --host $HOST --port $PORT --workers $WORKERS
fi

