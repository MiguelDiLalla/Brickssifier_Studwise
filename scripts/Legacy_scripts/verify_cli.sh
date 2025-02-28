#!/bin/bash
# Simple script to verify CLI functionality

echo "Testing LEGO Bricks ML Vision CLI..."
python lego_cli.py --help

if [ $? -eq 0 ]; then
    echo "✓ CLI is working correctly!"
else
    echo "✗ CLI has errors. Check logs for details."
fi
