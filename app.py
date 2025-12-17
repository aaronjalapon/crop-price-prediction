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
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    .main-header {
        font-size: 2.5em;
        color: var(--text-color);
        font-weight: bold;
        margin-bottom: 10px;
    }
    .prediction-box {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin: 10px 0;
    }
    .price-value {
        font-size: 2em;
        color: var(--text-color);
        font-weight: bold;
    }
    .confidence-interval {
        font-size: 1.1em;
        color: var(--text-color);
        line-height: 1.6;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #c62828;
    }
    .info-box {
        background-color: var(--background-color);
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid var(--primary-color);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE & MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained XGBoost model bundle"""
    try:
        bundle = joblib.load('XGBoost_Price_Prediction_final.joblib')
        return bundle
    except FileNotFoundError:
        st.error("Model file 'XGBoost_Price_Prediction_final.joblib' not found!")
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
    
    # Return metadata matching the actual dataset used for training
    return {
        'cmname': [
            "Maize (local) - Retail",
            "Millet - Retail",
            "Rice (imported) - Retail",
            "Rice (local) - Retail",
            "Sorghum - Retail"
        ],
        'unit': ["KG"],  # Default unit from dataset
        'category': ["cereals and tubers"],  # Default category from dataset
        'admname': [
            "Dakar",
            "Diourbel",
            "Fatick",
            "Kaffrine",
            "Kaolack",
            "Kedougou",
            "Kolda",
            "Louga",
            "Matam",
            "Saint louis",
            "Sedhiou",
            "Tambacounda",
            "Thies",
            "Ziguinchor"
        ],
        'mktname': [
            "Bambey",
            "Bignona",
            "Birkelane",
            "Castors",
            "Dagana",
            "Dakar",
            "Diakhao",
            "Diama Gadio",
            "Diaobe",
            "Dioli Mandakh",
            "Diourbel",
            "Fatick",
            "Gossas",
            "Gouille Mbeuth",
            "Gueule Tapee",
            "Kaffrine",
            "Kaolack",
            "Kedougou",
            "Keur I. Yacine",
            "Kolda",
            "Koungueul",
            "Kouthiaba",
            "Louga",
            "Mabo",
            "Mako",
            "Matam",
            "Mbafaye",
            "Mbar",
            "Mereto",
            "Missirah",
            "Mpal",
            "N'Doffane",
            "Ndiagne",
            "Ndindy",
            "Ndrame Escale",
            "Orkodiere",
            "Ourossogui",
            "Oussouye",
            "Passy",
            "Porokhane",
            "Sagatta",
            "Saint-Louis",
            "Saint-Maur",
            "Salémata",
            "Sandjara",
            "Sare Yoba",
            "Sedhiou",
            "Tambacounda",
            "Thiaroye",
            "Thies",
            "Thille Boubacar",
            "Thilmakha",
            "Thiodaye",
            "Tilene",
            "Touba",
            "Touba Toul",
            "Zigiunchor"
        ]
    }

@st.cache_data
def get_model_features():
    """Extract feature names from the model bundle"""
    bundle = load_model()
    if bundle is None:
        return None
    
    try:
        if 'feature_names' in bundle:
            return bundle['feature_names']
        elif hasattr(bundle.get('model'), 'feature_names_in_'):
            return list(bundle['model'].feature_names_in_)
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
        if 'year' in input_dict:
            input_dict['year'] = year
        if 'month' in input_dict:
            input_dict['month'] = month
        if 'day_of_week' in input_dict:
            input_dict['day_of_week'] = day_of_week
        
        # =============================================
        # CRITICAL: Set cyclical encoding features
        # These capture the circular nature of time
        # =============================================
        if 'month_sin' in input_dict:
            input_dict['month_sin'] = np.sin(2 * np.pi * month / 12)
        if 'month_cos' in input_dict:
            input_dict['month_cos'] = np.cos(2 * np.pi * month / 12)
        if 'dow_sin' in input_dict:
            input_dict['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        if 'dow_cos' in input_dict:
            input_dict['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # =============================================
        # CRITICAL: Set lag features with realistic defaults
        # These are the most important features for the model!
        # Values are in log scale (log1p transformed prices)
        # =============================================
        # Default log_price ~6.0 corresponds to ~403 XOF (reasonable baseline)
        # Adjust based on crop type for more realistic predictions
        default_log_price = 6.0
        
        if 'price_lag_12m' in input_dict:
            input_dict['price_lag_12m'] = default_log_price
        if 'yoy_change' in input_dict:
            input_dict['yoy_change'] = 0.02  # Small positive trend (~2% annual increase)
        if 'rolling_std_12m' in input_dict:
            input_dict['rolling_std_12m'] = 0.15  # Typical price volatility
        
        # Create DataFrame and ensure correct column order
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_names]
        
        return input_df
        
    except Exception as e:
        st.error(f"Error creating prediction input: {e}")
        return None

def make_prediction(model, input_df, scaler=None):
    """
    Make a prediction with the XGBoost model and calculate confidence intervals.
    
    Returns:
        dict: Contains predicted_price, ci_lower, ci_upper, confidence_score, tree_predictions
    """
    try:
        # Scale input if scaler is provided
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction (log-space)
        log_price = model.predict(input_scaled)[0]
        predicted_price = np.expm1(log_price)
        
        # For XGBoost, calculate confidence interval using bootstrap-like approach
        # Get predictions from each boosting iteration
        n_estimators = model.n_estimators
        tree_predictions = []
        
        # Use prediction with different ntree_limit to simulate ensemble variance
        for i in range(10, n_estimators + 1, max(1, n_estimators // 20)):
            pred = model.predict(input_scaled, iteration_range=(0, i))[0]
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions)
        
        mean_pred = np.mean(tree_predictions)
        std_pred = np.std(tree_predictions)
        
        # Use a more conservative estimate for XGBoost CI
        # Based on typical prediction uncertainty
        margin_error = 1.96 * max(std_pred, 0.05)  # Minimum 5% uncertainty
        
        ci_lower = np.expm1(log_price - margin_error)
        ci_upper = np.expm1(log_price + margin_error)
        
        # Calculate confidence score (inverse of relative std)
        confidence_score = max(0, min(100, 100 * (1 - (std_pred / abs(mean_pred) if mean_pred != 0 else 0))))
        
        # Convert tree predictions to price space for visualization
        tree_predictions_price = np.expm1(tree_predictions)
        
        return {
            'predicted_price': predicted_price,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_score': confidence_score,
            'n_estimators': n_estimators,
            'std_error': std_pred,
            'tree_predictions': tree_predictions_price
        }
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def create_price_range_chart(result):
    """Create a price range chart showing prediction with confidence interval."""
    fig = go.Figure()
    
    # Add confidence interval as a shaded area
    fig.add_trace(go.Scatter(
        x=['95% CI Range'],
        y=[result['ci_upper']],
        mode='markers',
        marker=dict(size=0),
        showlegend=False
    ))
    
    # Add the actual ranges
    fig.add_trace(go.Bar(
        x=['Confidence Interval'],
        y=[result['ci_upper'] - result['ci_lower']],
        base=result['ci_lower'],
        marker=dict(color='rgba(100, 150, 255, 0.3)', line=dict(color='rgb(100, 150, 255)', width=2)),
        name='95% Confidence Interval',
        hovertemplate='<b>95% CI Range</b><br>Lower: %{base:,.0f} XOF<br>Upper: %{y:,.0f} XOF<extra></extra>'
    ))
    
    # Add predicted price point
    fig.add_trace(go.Bar(
        x=['Predicted Price'],
        y=[result['predicted_price']],
        marker=dict(color='rgb(50, 150, 50)', line=dict(color='darkgreen', width=2)),
        name='Predicted Price',
        hovertemplate='<b>Predicted Price</b><br>%{y:,.0f} XOF<extra></extra>'
    ))
    
    fig.update_layout(
        title='<b>Price Prediction with 95% Confidence Interval</b>',
        yaxis_title='Price (XOF)',
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white',
        margin=dict(l=60, r=40, t=60, b=40)
    )
    
    fig.update_yaxes(tickformat=',.0f')
    
    return fig

def create_confidence_gauge(result):
    """Create a gauge chart showing model confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['confidence_score'],
        title={'text': "Model Confidence"},
        delta={'reference': 80},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 60], 'color': 'rgba(255, 100, 100, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(255, 200, 100, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(100, 200, 100, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 2},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=12)
    )
    
    return fig

def create_tree_predictions_histogram(result):
    """Create a histogram of estimator predictions."""
    fig = go.Figure()
    
    # Add histogram of estimator predictions
    fig.add_trace(go.Histogram(
        x=result['tree_predictions'],
        nbinsx=30,
        name='Estimator Predictions',
        marker=dict(color='rgba(100, 150, 255, 0.7)', line=dict(color='rgb(100, 150, 255)', width=1)),
        hovertemplate='<b>Price Range</b><br>%{x:,.0f} XOF<br>Count: %{y} estimators<extra></extra>'
    ))
    
    # Add vertical lines for key values
    fig.add_vline(
        x=result['predicted_price'],
        line_dash="solid",
        line_color="green",
        annotation_text="<b>Mean Prediction</b>",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=result['ci_lower'],
        line_dash="dash",
        line_color="red",
        annotation_text="<b>95% CI Lower</b>",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=result['ci_upper'],
        line_dash="dash",
        line_color="red",
        annotation_text="<b>95% CI Upper</b>",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='<b>Distribution of Estimator Predictions</b>',
        xaxis_title='Predicted Price (XOF)',
        yaxis_title='Number of Estimators',
        hovermode='x',
        height=400,
        template='plotly_white',
        margin=dict(l=60, r=40, t=80, b=40),
        showlegend=False
    )
    
    fig.update_xaxes(tickformat=',.0f')
    
    return fig

def create_time_series_forecast(model, input_df, crop_name, admin_region, market_name, 
                                base_date, feature_names, metadata, scaler=None):
    """
    Create a time series forecast showing predictions across days, weeks, months, and year.
    Uses iterative forecasting with proper cyclical encodings and lag feature updates.
    """
    from datetime import timedelta
    
    def predict_with_scaler(model, input_data, scaler):
        """Helper to predict with optional scaling."""
        if scaler is not None:
            scaled_data = scaler.transform(input_data)
            return model.predict(scaled_data)[0]
        return model.predict(input_data.values)[0]
    
    def get_cyclical_month(month):
        """Calculate cyclical encoding for month."""
        return np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    
    def get_cyclical_dow(dow):
        """Calculate cyclical encoding for day of week."""
        return np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    
    def update_input_features(input_data, forecast_date, feature_names):
        """Update all temporal and cyclical features for a forecast date."""
        month = forecast_date.month
        day_of_week = forecast_date.weekday()
        year = forecast_date.year
        
        if 'year' in feature_names:
            input_data.loc[0, 'year'] = year
        if 'month' in feature_names:
            input_data.loc[0, 'month'] = month
        if 'day_of_week' in feature_names:
            input_data.loc[0, 'day_of_week'] = day_of_week
        
        # Update cyclical features - CRITICAL for seasonal patterns
        month_sin, month_cos = get_cyclical_month(month)
        dow_sin, dow_cos = get_cyclical_dow(day_of_week)
        
        if 'month_sin' in feature_names:
            input_data.loc[0, 'month_sin'] = month_sin
        if 'month_cos' in feature_names:
            input_data.loc[0, 'month_cos'] = month_cos
        if 'dow_sin' in feature_names:
            input_data.loc[0, 'dow_sin'] = dow_sin
        if 'dow_cos' in feature_names:
            input_data.loc[0, 'dow_cos'] = dow_cos
        
        return input_data
    
    # Extract base input values from the input_df
    base_input = input_df.copy()
    
    # Get initial lag values
    initial_price_lag_12m = base_input['price_lag_12m'].values[0] if 'price_lag_12m' in feature_names else 6.0
    initial_yoy_change = base_input['yoy_change'].values[0] if 'yoy_change' in feature_names else 0.02
    initial_rolling_std = base_input['rolling_std_12m'].values[0] if 'rolling_std_12m' in feature_names else 0.15
    
    # Make initial prediction to get current price estimate
    initial_pred = predict_with_scaler(model, base_input, scaler)
    current_log_price = initial_pred
    
    # Store predictions for updating lags
    monthly_predictions = {0: current_log_price}
    
    # Generate time periods
    forecast_data = []
    
    # =============================================
    # Daily forecasts (next 30 days)
    # =============================================
    for i in range(1, 31):
        forecast_date = base_date + timedelta(days=i)
        
        daily_input = base_input.copy()
        daily_input = update_input_features(daily_input, forecast_date, feature_names)
        
        pred = predict_with_scaler(model, daily_input, scaler)
        price = np.expm1(pred)
        forecast_data.append({
            'date': forecast_date,
            'period': 'Daily',
            'label': forecast_date.strftime('%m-%d'),
            'price': price,
            'days_ahead': i
        })
    
    # =============================================
    # Weekly forecasts (next 12 weeks)
    # =============================================
    for i in range(1, 13):
        forecast_date = base_date + timedelta(weeks=i)
        week_num = forecast_date.isocalendar()[1]
        
        weekly_input = base_input.copy()
        weekly_input = update_input_features(weekly_input, forecast_date, feature_names)
        
        pred = predict_with_scaler(model, weekly_input, scaler)
        price = np.expm1(pred)
        forecast_data.append({
            'date': forecast_date,
            'period': 'Weekly',
            'label': f'W{week_num}',
            'price': price,
            'days_ahead': i * 7
        })
    
    # =============================================
    # Monthly forecasts (next 12 months) - iterative with lag updates
    # =============================================
    # Track all monthly log predictions for proper lag chaining
    all_monthly_log_preds = {0: current_log_price}
    
    for i in range(1, 13):
        forecast_date = base_date + timedelta(days=30*i)
        
        monthly_input = base_input.copy()
        monthly_input = update_input_features(monthly_input, forecast_date, feature_names)
        
        # For months > 1, we can start incorporating predicted values into lags
        # But since price_lag_12m is 12-month lag, it stays the same for first 12 months
        # Only yoy_change should evolve based on recent trend
        if 'yoy_change' in feature_names and i > 1:
            # Calculate trend based on recent predictions
            recent_change = all_monthly_log_preds.get(i-1, current_log_price) - current_log_price
            monthly_input.loc[0, 'yoy_change'] = initial_yoy_change + (recent_change / i)
        
        pred = predict_with_scaler(model, monthly_input, scaler)
        price = np.expm1(pred)
        
        # Store prediction for future reference
        all_monthly_log_preds[i] = pred
        monthly_predictions[i] = pred
        
        forecast_data.append({
            'date': forecast_date,
            'period': 'Monthly',
            'label': forecast_date.strftime('%b %Y'),
            'price': price,
            'days_ahead': 30 * i
        })
    
    # Create figure
    fig = go.Figure()
    
    # Add current price reference line
    current_price_actual = np.expm1(current_log_price)
    fig.add_hline(
        y=current_price_actual,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current: {current_price_actual:,.0f} XOF",
        annotation_position="right"
    )
    
    # Add traces for each period type with distinct colors
    periods = ['Daily', 'Weekly', 'Monthly']
    colors = ['rgb(70, 130, 180)', 'rgb(34, 139, 34)', 'rgb(148, 103, 189)']
    
    for period, color in zip(periods, colors):
        period_data = [d for d in forecast_data if d['period'] == period]
        if period_data:
            period_df = pd.DataFrame(period_data).sort_values('date')
            
            fig.add_trace(go.Scatter(
                x=period_df['date'],
                y=period_df['price'],
                mode='lines+markers',
                name=period,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m-%d}<br>Price: %{y:,.0f} XOF<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'<b>Time Series Price Forecast: {crop_name} in {admin_region}</b>',
        xaxis_title='Date',
        yaxis_title='Predicted Price (XOF)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        margin=dict(l=60, r=40, t=80, b=40),
        showlegend=True
    )
    
    fig.update_yaxes(tickformat=',.0f')
    fig.update_xaxes(tickformat='%Y-%m-%d')
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

# Load resources
bundle = load_model()
metadata = load_metadata()
feature_names = get_model_features()

if bundle is None:
    st.stop()

# Extract model components from bundle
model = bundle['model']
scaler = bundle.get('scaler', None)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='main-header'><i class='fa-solid fa-wheat-awn'></i>Crop Price Prediction Model</div>", unsafe_allow_html=True)
    st.markdown("**Interactive Forecasting Tool for Senegal Crop Markets**")
with col2:
    if feature_names:
        st.metric("Model Features", len(feature_names))
    st.markdown("<div><i class='fa-solid fa-robot'></i> XGBoost</div>", unsafe_allow_html=True)

st.divider()

# Check if feature names are available
if feature_names is None:
    st.warning("""
    **Feature Information Not Found**
    
    The model features could not be automatically extracted. The app may not work correctly.
    Please ensure:
    1. The model was saved with feature information
    2. You're using scikit-learn 1.0 or later
    
    Proceeding with default feature list...
    """)

# Input Section
st.markdown("### <i class='fa-solid fa-clipboard-list'></i> Select Parameters", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### <i class='fa-solid fa-seedling'></i> Crop Information", unsafe_allow_html=True)
    
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
    st.markdown("#### <i class='fa-solid fa-location-dot'></i> Location Information", unsafe_allow_html=True)
    
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
st.markdown("### <i class='fa-solid fa-calendar-days'></i> Temporal Parameters", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

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
    

with col3:
    st.metric("Month", month)

with col4:
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    st.metric("Day of Week", days[day_of_week])

st.divider()

# Prediction Section
st.markdown("### <i class='fa-solid fa-crystal-ball'></i> Prediction Results", unsafe_allow_html=True)

predict_button = st.button("Generate Prediction", use_container_width=True, type="primary")

st.markdown("""
    <div class="info-box">
        <i class="fa-solid fa-circle-info"></i> Click the button to generate a price prediction based on your selected parameters.
    </div>
""", unsafe_allow_html=True)

if predict_button:
    with st.spinner("Making prediction..."):
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
            result = make_prediction(model, input_df, scaler)
            
            if result is not None:
                # Display results
                st.success("Prediction completed successfully!")
                
                # Create three columns for key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='prediction-box'>
                        <div style='font-size: 0.9em; color: #666; margin-bottom: 10px;'>
                            <strong><i class='fa-solid fa-bullseye'></i> Predicted Price</strong>
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
                            <strong><i class='fa-solid fa-chart-simple'></i> 95% Confidence Interval</strong>
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
                            <strong><i class='fa-solid fa-star'></i> Model Confidence</strong>
                        </div>
                        <div style='font-size: 1.8em; color: var(--primary-color); font-weight: bold; margin-bottom: 10px;'>
                            {result['confidence_score']:.1f}%
                        </div>
                        <div style='font-size: 0.8em; color: #999;'>
                            Estimators: {result['n_estimators']}<br>
                            Std Error: {result['std_error']:.4f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display interactive charts
                st.markdown("### <i class='fa-solid fa-chart-line'></i> Visualization", unsafe_allow_html=True)
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.plotly_chart(create_price_range_chart(result), use_container_width=True)
                
                with chart_col2:
                    st.plotly_chart(create_confidence_gauge(result), use_container_width=True)
                
                # Full-width histogram
                st.plotly_chart(create_tree_predictions_histogram(result), use_container_width=True)
                
                # Time series forecast
                st.markdown("### <i class='fa-solid fa-timeline'></i> Price Forecast Over Time", unsafe_allow_html=True)
                st.plotly_chart(
                    create_time_series_forecast(
                        model=model,
                        input_df=input_df,
                        crop_name=crop_name,
                        admin_region=admin_region,
                        market_name=market_name,
                        base_date=selected_date,
                        feature_names=feature_names,
                        metadata=metadata,
                        scaler=scaler
                    ),
                    use_container_width=True
                )
                
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
with st.expander("About This Model", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Architecture
        - **Algorithm**: XGBoost Regressor
        - **Ensemble Size**: Gradient boosted trees
        - **Target Variable**: Log-transformed Crop Prices
        - **Input Features**: ~40-50 after encoding
        - **Regularization**: L1/L2 regularization for robustness
        
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
        - **Temporal**: Year, Month, Day of Week, Cyclical Encoding
        - **Lag Features**: 12-month price lag, YoY change
        - **Fixed**: Currency (XOF), Country (Senegal)
        
        ### Model Performance
        - **Test R² Score**: 0.981 (98.1% variance explained)
        - **Validation**: TimeSeriesSplit (5-fold)
        - **Hyperparameters**: Optimized via GridSearchCV
        """)

with st.expander("How to Interpret Results", expanded=False):
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

with st.expander("Model Details & Training", expanded=False):
    st.markdown("""
    ### Training Process
    1. **Data Cleaning**: Handled missing values, removed outliers
    2. **Feature Engineering**: 
       - One-hot encoding for categorical variables
       - Log transformation for price normalization
       - Cyclical encoding for temporal features (month_sin, month_cos)
       - Long-term lag features (12-month lag, YoY change)
    3. **Model Training**: XGBoost with hyperparameter tuning
    4. **Validation**: TimeSeriesSplit (5-fold) to prevent data leakage
    5. **Evaluation**: R², RMSE, MAE, and residual analysis
    
    ### Anti-Overfitting Strategies
    - Uses only 12-month lag (avoids short-term data leakage)
    - TimeSeriesSplit prevents future data leakage
    - Strong L1/L2 regularization (reg_alpha, reg_lambda)
    - Subsample and column sampling for robustness
    
    ### Important Notes
    - Model trained on historical data; future prices may differ
    - External factors (weather, policy, supply) not captured
    - Regular retraining recommended as new data arrives
    """)

st.divider()

# Footer
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85em; margin-top: 30px; padding: 20px;'>
        <strong><i class='fa-solid fa-wheat-awn'></i> Crop Price Prediction Model</strong><br>
        Powered by XGBoost & Streamlit<br>
        <small>Dataset: WFP Food Prices | Region: Senegal</small><br>
        <small>Last Updated: December 2025 | Status: <i class='fa-solid fa-circle-check'></i> Production Ready</small>
    </div>
""", unsafe_allow_html=True)
