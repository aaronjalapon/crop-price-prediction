# 🚀 Quick Start Guide

Get your crop price prediction app running in minutes!

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Run the App
```bash
streamlit run app_v2.py
```

### Step 3: Open in Browser
Navigate to: `http://localhost:8501`

That's it! 🎉

---

## 📋 What You'll See

1. **Crop Information Section**
   - Select crop type, unit, category

2. **Location Information Section**
   - Select administrative region and market

3. **Temporal Parameters**
   - Choose date for prediction

4. **Prediction Results**
   - Predicted price in XOF
   - 95% confidence interval
   - Model confidence score

---

## 🔧 Troubleshooting

### Python not found?
```bash
python3 --version  # Try with python3
pip3 install -r requirements_streamlit.txt
streamlit run app_v2.py
```

### Port 8501 already in use?
```bash
streamlit run app_v2.py --server.port=8502
```

### Model file not found?
```bash
# Check if model exists
ls -la random_forest_model.joblib

# If not, ensure it's in the same directory as app_v2.py
```

---

## 📦 Requirements

- Python 3.8+
- ~200MB disk space (for model + dependencies)
- Internet connection (first run downloads packages)

---

## 🌐 Deploy to Web

### Option 1: Streamlit Cloud (Free, 5 minutes)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect GitHub account
4. Select repo and deploy

### Option 2: Docker (Any server, 10 minutes)
```bash
docker build -t crop-price-app .
docker run -p 8501:8501 crop-price-app
```

### Option 3: More Platforms
See `DEPLOYMENT.md` for detailed guides for Heroku, AWS, Google Cloud, Azure, etc.

---

## 📚 Learn More

- Full deployment guide: `DEPLOYMENT.md`
- Streamlit documentation: `STREAMLIT_README.md`
- Model training: `Price_Prediction.ipynb`
- Metadata utilities: `metadata_utils.py`

---

## 💡 Tips

1. **First time slower**: App caches model on first run
2. **Dark mode**: Streamlit -> Settings -> Theme
3. **Full screen**: Streamlit -> Settings -> Screen width: "Wide"
4. **Development mode**: Streamlit -> Always rerun

---

## ❓ FAQ

**Q: Can I modify the feature options?**
A: Yes, edit the lists in `app_v2.py` (crop names, markets, etc.)

**Q: How accurate is the model?**
A: R² score of 0.85+ on test data. Confidence intervals provided.

**Q: Can I use my own model?**
A: Yes, replace `random_forest_model.joblib` and update feature names.

**Q: Is it production-ready?**
A: Yes! See DEPLOYMENT.md for scaling options.

---

**Need help?** Check the error messages in the terminal or review the troubleshooting section above.

Happy predicting! 🌾
