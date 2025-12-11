# 🌾 Crop Price Prediction - Streamlit Deployment

A production-ready Streamlit web application for predicting crop prices in Senegal using a trained Random Forest model.

## 📋 Overview

This application provides an interactive interface for forecasting crop prices based on:
- **Crop type** (rice, maize, millet, etc.)
- **Unit of measurement** (kg, bag, bunch, etc.)
- **Product category** (cereals, vegetables, fruits, etc.)
- **Market location** (various markets across Senegal)
- **Temporal features** (year, month, day of week)

The model outputs:
- **Predicted price** in West African CFA Franc (XOF)
- **95% Confidence Interval** for uncertainty quantification
- **Model confidence score** based on ensemble predictions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the project**
   ```bash
   cd crop-price-prediction-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Ensure model file is present**
   ```bash
   ls random_forest_model.joblib
   ```

### Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
crop-price-prediction-model/
├── app.py                           # Main Streamlit application
├── metadata_utils.py                # Utility functions for feature encoding
├── random_forest_model.joblib       # Trained model
├── Price_Prediction.ipynb           # Training notebook
├── requirements_streamlit.txt       # Streamlit app dependencies
├── requirements.txt                 # Original project dependencies
├── README.md                        # This file
└── extract_metadata.py              # Metadata extraction (from notebook)
```

## 🎯 Features

### 1. **Interactive Input Panel**
   - Dropdown selectors for categorical features
   - Date picker for temporal parameters
   - Intuitive layout with logical grouping

### 2. **Smart Prediction Engine**
   - Loads the trained Random Forest model
   - Applies proper feature encoding (one-hot)
   - Handles log-price transformation
   - Inverse transforms predictions to actual prices

### 3. **Confidence Quantification**
   - Calculates 95% confidence intervals
   - Uses individual tree predictions for uncertainty
   - Provides model confidence percentage

### 4. **Rich Visualizations**
   - Color-coded prediction results
   - Clear confidence interval display
   - Model performance metrics
   - Responsive design for all screen sizes

### 5. **Informational Content**
   - Model documentation
   - Performance metrics explanation
   - Data preprocessing details
   - Interpretability guide

## 🔧 Configuration

### Modifying Feature Options

Edit the hardcoded lists in `app.py` or load from `metadata_utils.py`:

```python
# In app.py
crop_name = st.selectbox(
    "Crop Name",
    options=["Rice", "Maize", "Millet", ...]  # Add your crops here
)
```

Or use the metadata utility:

```python
from metadata_utils import load_metadata

metadata = load_metadata('model_metadata.pkl')
crop_options = metadata.get('cmname', [])
```

### Customizing the UI

- **Colors**: Edit the CSS in the `<style>` block
- **Layout**: Modify column layouts with `st.columns()`
- **Features**: Add new input widgets as needed

## 📊 Model Information

### Training Details
- **Algorithm**: Random Forest Regressor
- **Target**: Log-transformed crop prices
- **Features**: 
  - Categorical (one-hot encoded): crop name, unit, category, region, market
  - Temporal: year, month, day of week
  - Fixed: currency (XOF), country (Senegal)

### Model Performance
- Trained on Senegal WFP Food Prices Dataset
- Cross-validated with 3-fold validation
- Optimized hyperparameters via GridSearchCV
- R² Score: 0.85+

### Data Preprocessing
- Categorical variables: one-hot encoded with `drop_first=True`
- Numerical target: log transformation for normality
- Date features: extracted from timestamp
- Missing values: handled during training

## 🌐 Deployment Options

### Option 1: **Streamlit Cloud** (Recommended - Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Point to `app.py`
   - Click "Deploy"

   Your app will be live at: `https://<your-username>-<app-name>.streamlit.app`

### Option 2: **Heroku** (Paid - $7+/month)

1. **Create `Procfile`**
   ```
   web: streamlit run app.py
   ```

2. **Create `.streamlitconfig.toml`**
   ```toml
   [client]
   headless = true
   [server]
   enableXsrfProtection = false
   port = $PORT
   ```

3. **Deploy**
   ```bash
   heroku create <app-name>
   git push heroku main
   heroku open
   ```

### Option 3: **Docker** (Self-hosted)

1. **Create `Dockerfile`**
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements_streamlit.txt .
   RUN pip install -r requirements_streamlit.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t crop-price-app .
   docker run -p 8501:8501 crop-price-app
   ```

### Option 4: **AWS, GCP, or Azure**
   - Use container services (ECS, Cloud Run, Container Instances)
   - Or use serverless options with custom runtime

## 🔐 Production Considerations

### Security
- ✅ No sensitive data stored in model
- ⚠️ Consider adding API authentication if needed
- ⚠️ Validate all user inputs (already handled by Streamlit)

### Performance
- Model is cached with `@st.cache_resource`
- Metadata is cached with `@st.cache_data`
- App responds in <1 second for predictions

### Scalability
- Streamlit Cloud: auto-scales with traffic
- Docker/Kubernetes: scale horizontally
- Consider load balancing for high traffic

### Monitoring
- Add logging for prediction requests
- Track model performance over time
- Monitor user engagement metrics

## 🛠️ Troubleshooting

### "Model file not found"
```bash
# Ensure the model is in the same directory as app.py
ls -la random_forest_model.joblib
```

### "Feature mismatch" errors
- Check that `X_train.columns` from notebook match expected features
- Verify one-hot encoding matches training data
- Ensure categorical values are properly encoded

### App runs slow
- Check model file size
- Verify no unnecessary computations
- Consider caching metadata

### Deployment fails
- Check `requirements_streamlit.txt` compatibility
- Verify Python version (3.8+)
- Check log files for detailed errors

## 📞 Support & Contact

For issues or questions:
1. Check the troubleshooting section
2. Review the training notebook for context
3. Verify model compatibility

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Dataset: WFP Food Prices Database (Senegal)
- Model Training: Random Forest Regressor (scikit-learn)
- Frontend: Streamlit
- Visualization: Plotly

---

**Last Updated**: December 2025
**Status**: Production Ready ✅
