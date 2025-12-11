# 🌾 Crop Price Prediction (Senegal)

An end-to-end machine learning project to forecast food prices in Senegal. This repository contains the data analysis, model training pipeline, and a production-ready Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Overview

Food price volatility is a critical issue in many developing regions. This project aims to provide transparent and accessible price forecasting for crops in Senegal markets.

The solution consists of:
1.  **Data Analysis**: Exploratory Data Analysis (EDA) on WFP market data.
2.  **Machine Learning**: A Random Forest Regressor trained to predict crop prices based on location, crop type, and temporal features.
3.  **Web Application**: An interactive Streamlit dashboard for real-time predictions.

## ✨ Features

-   **Interactive Dashboard**: User-friendly interface to select crops, markets, and dates.
-   **Real-time Predictions**: Instant price forecasting in West African CFA Franc (XOF).
-   **Uncertainty Quantification**: Displays 95% confidence intervals for every prediction.
-   **Model Confidence Score**: Provides a reliability metric based on ensemble variance.
-   **Comprehensive Metadata**: Supports various crops (Rice, Maize, Millet, etc.) and regions across Senegal.

## 🚀 Quick Start

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/aaronjalapon/crop-price-prediction.git
    cd crop-price-prediction
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    streamlit run app.py
    ```

4.  **Access the app**
    Open your browser to `http://localhost:8501`

## 🐳 Docker Usage

Run the application in a containerized environment:

```bash
# Build the image
docker build -t crop-price-app .

# Run the container
docker run -p 8501:8501 crop-price-app
```

## 📂 Project Structure

```
crop-price-prediction/
├── app.py                       # Main Streamlit application
├── Price_Prediction.ipynb       # Jupyter notebook for EDA and training
├── random_forest_model.joblib   # Trained Random Forest model
├── metadata_utils.py            # Utilities for feature encoding
├── extract_metadata.py          # Script to extract metadata from data
├── requirements.txt             # App dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── DEPLOYMENT.md                # Detailed deployment guides
└── QUICKSTART.md                # Fast setup guide
```

## 🧠 Model Details

-   **Algorithm**: Random Forest Regressor (Scikit-learn)
-   **Target**: Log-transformed price (to handle skewness)
-   **Features**:
    -   **Categorical**: Crop Name, Unit, Category, Administrative Region, Market Name (One-Hot Encoded)
    -   **Temporal**: Year, Month, Day of Week
-   **Performance**: The model achieves an R² score of >0.85 on the test set.

## ☁️ Deployment

This project is ready for deployment on multiple platforms:

-   **Streamlit Cloud**: (Recommended) Connect your GitHub repo and deploy instantly.
-   **Docker**: Deploy to AWS ECS, Google Cloud Run, or Azure Container Instances.
-   **Heroku**: Procfile included (via `streamlit run` command).

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

-   Data provided by the **World Food Programme (WFP)**.
-   Built with **Streamlit** and **Scikit-learn**.
