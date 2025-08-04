#!/bin/bash
# load_env.sh - Load environment variables from .env file

# Method 1: Export variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded"
else
    echo "❌ No .env file found"
    echo "Create a .env file with:"
    echo "  OPENAI_API_KEY=sk-your-key-here"
fi

# Verify the key is loaded
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OPENAI_API_KEY is set (length: ${#OPENAI_API_KEY})"
else
    echo "❌ OPENAI_API_KEY is not set"
fi