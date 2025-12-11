import json
import pickle
import pandas as pd
import joblib
from pathlib import Path

# This script extracts metadata from the notebook's variables
# Run this in your notebook environment to generate the metadata files

# Assuming these variables exist in your notebook:
# - X_train: the training feature DataFrame
# - df_raw: the raw dataset
# - rf_model: the trained Random Forest model

# For local setup, we'll create a mock extraction based on the notebook structure
# In production, you would run this in the notebook after training

def extract_and_save_metadata():
    """
    Extract metadata from notebook variables and save to files
    """
    
    # These would be obtained from your notebook after training
    # For now, we'll create a placeholder that will be filled in
    metadata = {
        "feature_columns": None,  # Will be X_train.columns.tolist()
        "categorical_features": {
            "cmname": "Crop name",
            "unit": "Unit of measurement",
            "category": "Product category",
            "admname": "Administrative region",
            "mktname": "Market name"
        },
        "numerical_features": {
            "year": {"min": None, "max": None},
            "month": {"min": 1, "max": 12},
            "day_of_week": {"min": 0, "max": 6}
        },
        "currency": "XOF",
        "country": "Senegal"
    }
    
    # Save metadata to JSON
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Metadata template created. Update 'model_metadata.json' with actual values from notebook.")

if __name__ == "__main__":
    extract_and_save_metadata()
