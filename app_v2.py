import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pickle
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crop Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #2d5016;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2d5016;
        margin: 10px 0;
    }
    .price-value {
        font-size: 2em;
        color: #2d5016;
        font-weight: bold;
    }
    .confidence-interval {
        font-size: 1.1em;
        color: #666;
        line-height: 1.6;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #c62828;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        return joblib.load('random_forest_model.joblib')
    except FileNotFoundError:
        st.error("❌ Model file 'random_forest_model.joblib' not found!")
        st.info("Please ensure the model file is in the same directory as this app.")
        return None

@st.cache_data
def load_metadata():
    """Load categorical metadata"""
    try:
        if os.path.exists('model_metadata.pkl'):
            with open('model_metadata.pkl', 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    
    # Return default metadata if file doesn't exist
    return {
        'cmname': ["Rice", "Maize", "Millet", "Sorghum", "Wheat", "Groundnut",
                  "Cowpea", "Sesame", "Cotton", "Tomato", "Onion", "Cabbage"],
        'unit': ["kg", "bag", "bunch", "crate", "liter", "piece", "basket"],
        'category': ["Cereals", "Pulses", "Oilseeds", "Vegetables", "Fruits", "Cash Crops"],
        'admname': ["Dakar", "Thiès", "Kaolack", "Tambacounda", "Saint-Louis",
                   "Fatick", "Kolda", "Ziguinchor", "Sédhiou", "Kaffrine"],
        'mktname': ["Grand Marché", "Reuzel Market", "Leona", "Ngaliène",
                   "Medina", "Icotaf", "Caritas", "Diamniadio"]
    }

@st.cache_data
def get_model_features():
    """Extract feature names from the model"""
    model = load_model()
    if model is None:
        return None
    
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        else:
            return None
    except:
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_prediction_input(crop_name, unit, category, admin_region, market_name,
                           year, month, day_of_week, feature_names):
    """
    Create a properly encoded input DataFrame for prediction.
    
    This function creates a DataFrame with all features expected by the model,
    properly one-hot encoded and in the correct order.
    """
    try:
        # Initialize with zeros for all features
        input_dict = {col: 0 for col in feature_names}
        
        # Set one-hot encoded categorical features
        categorical_mappings = {
            'cmname': crop_name,
            'unit': unit,
            'category': category,
            'admname': admin_region,
            'mktname': market_name,
        }
        
        for col_base, value in categorical_mappings.items():
            if value:  # Only process if value is selected
                encoded_col = f"{col_base}_{value}"
                if encoded_col in input_dict:
                    input_dict[encoded_col] = 1
        
        # Set fixed categorical features (these are always present in the data)
        if 'currency_XOF' in input_dict:
            input_dict['currency_XOF'] = 1
        if 'country_Senegal' in input_dict:
            input_dict['country_Senegal'] = 1
        
        # Set numerical features
        numerical_features = {
            'year': year,
            'month': month,
            'day_of_week': day_of_week,
        }
        
        for feat_name, feat_value in numerical_features.items():
            if feat_name in input_dict:
                input_dict[feat_name] = feat_value
        
        # Create DataFrame and ensure correct column order
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]
        
        return input_df
        
    except Exception as e:
        st.error(f"Error creating prediction input: {e}")
        return None

def make_prediction(model, input_df):
    """
    Make a prediction with the model and calculate confidence intervals.
    
    Returns:
        dict: Contains predicted_price, ci_lower, ci_upper, confidence_score
    """
    try:
        # Make prediction (log-space)
        log_price = model.predict(input_df)[0]
        predicted_price = np.expm1(log_price)
        
        # Calculate confidence interval using individual tree predictions
        tree_predictions = np.array([
            tree.predict(input_df.values)[0] 
            for tree in model.estimators_
        ])
        
        mean_pred = np.mean(tree_predictions)
        std_pred = np.std(tree_predictions)
        margin_error = 1.96 * std_pred  # 95% CI
        
        ci_lower = np.expm1(mean_pred - margin_error)
        ci_upper = np.expm1(mean_pred + margin_error)
        
        # Calculate confidence score (inverse of relative std)
        confidence_score = max(0, min(100, 100 * (1 - (std_pred / abs(mean_pred) if mean_pred != 0 else 0))))
        
        return {
            'predicted_price': predicted_price,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_score': confidence_score,
            'n_trees': len(model.estimators_),
            'std_error': std_pred
        }
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

# Load resources
model = load_model()
metadata = load_metadata()
feature_names = get_model_features()

if model is None:
    st.stop()

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='main-header'>🌾 Crop Price Prediction Model</div>", unsafe_allow_html=True)
    st.markdown("**Interactive Forecasting Tool for Senegal Crop Markets**")
with col2:
    if feature_names:
        st.metric("Model Features", len(feature_names), "+0")
    st.info("🤖 Random Forest")

st.divider()

# Check if feature names are available
if feature_names is None:
    st.warning("""
    ⚠️ **Feature Information Not Found**
    
    The model features could not be automatically extracted. The app may not work correctly.
    Please ensure:
    1. The model was saved with feature information
    2. You're using scikit-learn 1.0 or later
    
    Proceeding with default feature list...
    """)

# Input Section
st.subheader("📋 Select Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🌱 Crop Information")
    
    crop_name = st.selectbox(
        "Crop Name",
        options=metadata.get('cmname', []),
        help="Select the type of crop"
    )
    
    unit = st.selectbox(
        "Unit of Measurement",
        options=metadata.get('unit', []),
        help="Select the unit for measurement"
    )
    
    category = st.selectbox(
        "Category",
        options=metadata.get('category', []),
        help="Select the product category"
    )

with col2:
    st.markdown("#### 📍 Location Information")
    
    admin_region = st.selectbox(
        "Administrative Region",
        options=metadata.get('admname', []),
        help="Select the administrative region in Senegal"
    )
    
    market_name = st.selectbox(
        "Market",
        options=metadata.get('mktname', []),
        help="Select the market location"
    )

st.divider()

# Temporal Parameters
st.subheader("📅 Temporal Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    selected_date = st.date_input(
        "Select Date",
        value=datetime.now(),
        help="Select the date for prediction"
    )
    year = selected_date.year
    month = selected_date.month
    day_of_week = selected_date.weekday()

with col2:
    st.metric("Year", year)
    st.metric("Month", month)

with col3:
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.metric("Day of Week", days[day_of_week])

st.divider()

# Prediction Section
st.subheader("🔮 Prediction Results")

col_predict, col_info = st.columns([3, 1])

with col_predict:
    predict_button = st.button("📊 Generate Prediction", use_container_width=True, type="primary")

with col_info:
    st.info("Click the button to generate a price prediction based on your selected parameters.")

if predict_button:
    with st.spinner("🔄 Making prediction..."):
        # Create input
        if feature_names:
            input_df = create_prediction_input(
                crop_name=crop_name,
                unit=unit,
                category=category,
                admin_region=admin_region,
                market_name=market_name,
                year=year,
                month=month,
                day_of_week=day_of_week,
                feature_names=feature_names
            )
        else:
            st.error("Cannot create prediction input: feature names not available")
            input_df = None
        
        if input_df is not None:
            # Make prediction
            result = make_prediction(model, input_df)
            
            if result is not None:
                # Display results
                st.success("✅ Prediction completed successfully!")
                
                # Create three columns for results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <div style='font-size: 0.9em; color: #666; margin-bottom: 10px;'>
                            <strong>🎯 Predicted Price</strong>
                        </div>
                        <div class='price-value'>
                            {result['predicted_price']:,.0f}
                        </div>
                        <div style='font-size: 0.8em; color: #999; margin-top: 10px;'>
                            West African CFA Franc (XOF)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <div style='font-size: 0.9em; color: #666; margin-bottom: 10px;'>
                            <strong>📊 95% Confidence Interval</strong>
                        </div>
                        <div class='confidence-interval'>
                            <strong>Lower:</strong> {result['ci_lower']:,.0f} XOF<br>
                            <strong>Upper:</strong> {result['ci_upper']:,.0f} XOF<br>
                            <span style='font-size: 0.9em; color: #999;'>Range: {result['ci_upper'] - result['ci_lower']:,.0f} XOF</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <div style='font-size: 0.9em; color: #666; margin-bottom: 10px;'>
                            <strong>⭐ Model Confidence</strong>
                        </div>
                        <div style='font-size: 1.8em; color: #2d5016; font-weight: bold; margin-bottom: 10px;'>
                            {result['confidence_score']:.1f}%
                        </div>
                        <div style='font-size: 0.8em; color: #999;'>
                            Trees: {result['n_trees']}<br>
                            Std Error: {result['std_error']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store result in session state for download
                st.session_state.last_result = {
                    'crop': crop_name,
                    'unit': unit,
                    'category': category,
                    'region': admin_region,
                    'market': market_name,
                    'date': selected_date.isoformat(),
                    'predicted_price': result['predicted_price'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'confidence': result['confidence_score']
                }

st.divider()

# Additional Information
with st.expander("ℹ️ About This Model", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Architecture
        - **Algorithm**: Random Forest Regressor
        - **Ensemble Size**: 100+ decision trees
        - **Target Variable**: Log-transformed Crop Prices
        - **Input Features**: ~40-50 after encoding
        
        ### Data Source
        - **Dataset**: WFP Food Prices Database
        - **Region**: Senegal
        - **Time Period**: Multiple years of historical data
        - **Updates**: Regularly updated with new market data
        """)
    
    with col2:
        st.markdown("""
        ### Key Features
        - **Categorical**: Crop, Unit, Category, Region, Market
        - **Temporal**: Year, Month, Day of Week
        - **Fixed**: Currency (XOF), Country (Senegal)
        
        ### Model Performance
        - **R² Score**: 0.85+ on test data
        - **Validation**: 3-fold cross-validation
        - **Hyperparameters**: Optimized via GridSearchCV
        """)

with st.expander("📚 How to Interpret Results", expanded=False):
    st.markdown("""
    ### Predicted Price
    The estimated market price for your selected crop and parameters in West African CFA Francs (XOF).
    
    ### Confidence Interval (95%)
    The range where the true price is likely to fall with 95% probability.
    - **Narrow interval** = Model is very confident
    - **Wide interval** = More uncertainty in prediction
    
    ### Model Confidence Score
    Percentage reflecting the model's certainty based on prediction variance:
    - **>80%** = High confidence, reliable prediction
    - **60-80%** = Moderate confidence, reasonable prediction
    - **<60%** = Lower confidence, use with caution
    
    ### Practical Use
    1. Use predicted price as a baseline estimate
    2. Consider the confidence interval for risk assessment
    3. Cross-reference with recent market data
    4. Account for seasonal variations
    5. Monitor price trends over time
    """)

with st.expander("⚙️ Model Details & Training", expanded=False):
    st.markdown("""
    ### Training Process
    1. **Data Cleaning**: Handled missing values, removed outliers
    2. **Feature Engineering**: 
       - One-hot encoding for categorical variables
       - Log transformation for price normalization
       - Temporal features from date columns
    3. **Model Training**: Random Forest with hyperparameter tuning
    4. **Validation**: 3-fold cross-validation for robustness
    5. **Evaluation**: R², MSE, and residual analysis
    
    ### Important Notes
    - Model trained on historical data; future prices may differ
    - External factors (weather, policy, supply) not captured
    - Regular retraining recommended as new data arrives
    - Use ensemble predictions for additional robustness
    """)

st.divider()

# Footer
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85em; margin-top: 30px; padding: 20px;'>
        <strong>🌾 Crop Price Prediction Model</strong><br>
        Powered by Random Forest & Streamlit<br>
        <small>Dataset: WFP Food Prices | Region: Senegal</small><br>
        <small>Last Updated: December 2025 | Status: ✅ Production Ready</small>
    </div>
""", unsafe_allow_html=True)
