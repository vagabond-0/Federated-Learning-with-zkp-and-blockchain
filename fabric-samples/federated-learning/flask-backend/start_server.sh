#!/bin/bash

cd /home/amalendu/college/federatedLearning/fabric-samples/federated-learning/flask-backend

# Remove old venv
rm -rf venv

# Create new venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install only necessary packages
pip install Flask==3.0.0 Flask-CORS==4.0.0

echo "✅ Starting Flask server..."
python app.py