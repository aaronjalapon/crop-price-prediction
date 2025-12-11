"""
Metadata extraction utility for Crop Price Prediction model.
This module extracts and manages categorical options from the training data.
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
from pathlib import Path


def extract_model_features(model_path='random_forest_model.joblib'):
    """
    Extract feature information from the trained model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        dict: Dictionary with model metadata including n_features, feature_names, etc.
    """
    try:
        model = joblib.load(model_path)
        
        metadata = {
            'n_features': model.n_features_in_,
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'model_type': type(model).__name__,
        }
        
        # Try to get feature names if available (sklearn 1.0+)
        if hasattr(model, 'feature_names_in_'):
            metadata['feature_names'] = list(model.feature_names_in_)
        
        return metadata
    except Exception as e:
        print(f"Error extracting model features: {e}")
        return None


def create_feature_encoder(csv_path):
    """
    Create a feature encoder from the raw dataset.
    Extracts categorical unique values for one-hot encoding.
    
    Args:
        csv_path: Path to the CSV dataset file
        
    Returns:
        dict: Dictionary with categorical options for each feature
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Extract unique values for each categorical column
        categorical_columns = ['cmname', 'unit', 'category', 'admname', 'mktname']
        
        encoder = {}
        for col in categorical_columns:
            if col in df.columns:
                # Filter out header-like entries
                unique_vals = [v for v in df[col].unique() 
                              if isinstance(v, str) and not v.startswith('#')]
                encoder[col] = sorted(unique_vals)
        
        # Add fixed values
        encoder['currency'] = ['XOF']
        encoder['country'] = ['Senegal']
        
        # Add temporal ranges
        encoder['year_range'] = {
            'min': int(df['date'].min().year) if 'date' in df.columns else 2020,
            'max': int(df['date'].max().year) if 'date' in df.columns else 2024,
        }
        
        return encoder
    except Exception as e:
        print(f"Error creating feature encoder: {e}")
        return None


def create_one_hot_encoded_input(selections, encoder_info, feature_columns):
    """
    Create a properly encoded input DataFrame for model prediction.
    
    Args:
        selections: dict with selected values for each feature
        encoder_info: dict with categorical options
        feature_columns: list of expected feature column names from model
        
    Returns:
        pd.DataFrame: Properly formatted input for model prediction
    """
    try:
        # Initialize with zeros
        input_dict = {col: 0 for col in feature_columns}
        
        # Set categorical one-hot encoded features
        for col in ['cmname', 'unit', 'category', 'admname', 'mktname']:
            if col in selections:
                encoded_col = f"{col}_{selections[col]}"
                if encoded_col in input_dict:
                    input_dict[encoded_col] = 1
        
        # Set fixed categorical features
        if 'currency_XOF' in input_dict:
            input_dict['currency_XOF'] = 1
        if 'country_Senegal' in input_dict:
            input_dict['country_Senegal'] = 1
        
        # Set numerical features
        temporal_features = ['year', 'month', 'day_of_week']
        for feat in temporal_features:
            if feat in selections and feat in input_dict:
                input_dict[feat] = selections[feat]
        
        # Create DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Ensure column order matches feature_columns
        df_input = df_input[feature_columns]
        
        return df_input
    except Exception as e:
        print(f"Error creating encoded input: {e}")
        return None


def save_metadata(encoder_dict, filepath='model_metadata.pkl'):
    """
    Save encoder metadata for later use in the Streamlit app.
    
    Args:
        encoder_dict: Dictionary with categorical options
        filepath: Path to save the metadata file
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_dict, f)
        print(f"Metadata saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False


def load_metadata(filepath='model_metadata.pkl'):
    """
    Load encoder metadata for use in the Streamlit app.
    
    Args:
        filepath: Path to the metadata file
        
    Returns:
        dict: Categorical options and encoder information
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Metadata file not found: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def get_feature_columns_from_notebook():
    """
    Extract exact feature column names from the training notebook.
    This requires the notebook to have been executed and variables saved.
    
    Returns:
        list: Feature column names in the order used for training
    """
    # This would typically load X_train columns from a saved state
    # For now, return placeholder - would be dynamically extracted in production
    return None


if __name__ == "__main__":
    # Example usage
    print("Crop Price Model Metadata Extractor")
    print("=" * 50)
    
    # Extract model features
    print("\n1. Extracting model metadata...")
    model_meta = extract_model_features('random_forest_model.joblib')
    if model_meta:
        print(f"   Model Type: {model_meta['model_type']}")
        print(f"   Features: {model_meta['n_features']}")
        print(f"   Trees: {model_meta['n_estimators']}")
    
    # Create feature encoder (if CSV available)
    print("\n2. Creating feature encoder...")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        encoder = create_feature_encoder(csv_files[0])
        if encoder:
            print(f"   Categorical features found: {list(encoder.keys())}")
            
            # Save metadata
            print("\n3. Saving metadata...")
            save_metadata(encoder)
            print("   ✓ Metadata saved successfully")
    else:
        print("   No CSV files found in current directory")
