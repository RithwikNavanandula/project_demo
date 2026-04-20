#!/bin/bash

# Hindi OCR Web Application - Startup Script

echo "🚀 Hindi OCR Web Application"
echo "=============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔗 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p model_hindi

# Start the application
echo ""
echo "✅ Setup complete!"
echo "🌐 Starting Flask application..."
echo ""
echo "📍 Open your browser and go to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
