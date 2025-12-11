#!/bin/bash
# Setup script for Crop Price Prediction Streamlit App
# This script prepares the environment and extracts necessary metadata

set -e

echo "🌾 Crop Price Prediction - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if model file exists
echo ""
echo "✓ Checking for model file..."
if [ -f "random_forest_model.joblib" ]; then
    echo "  ✓ Model file found: random_forest_model.joblib"
else
    echo "  ✗ Model file not found!"
    echo "  Please ensure random_forest_model.joblib is in the current directory"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Extract and save metadata
echo ""
echo "✓ Extracting model metadata..."
python -c "
from metadata_utils import extract_model_features, load_metadata
import os

print('  Extracting model information...')
metadata = extract_model_features('random_forest_model.joblib')
if metadata:
    print(f'    - Features: {metadata[\"n_features\"]}')
    print(f'    - Estimators: {metadata[\"n_estimators\"]}')
    print(f'    - Max Depth: {metadata[\"max_depth\"]}')
else:
    print('  Warning: Could not extract model metadata')
"

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run the app: streamlit run app.py"
echo "2. Open your browser to: http://localhost:8501"
echo ""
