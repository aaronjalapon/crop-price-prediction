# Crop Price Prediction - Deployment Guide

## 📦 Quick Start

### Local Deployment

```bash
# 1. Local Testing
Ensure the app runs locally before deploying.
```bash
pip install -r requirements.txt
streamlit run app_v2.py
```

# 3. Open browser
# Navigate to: http://localhost:8501
```

## 🚀 Deployment Platforms

### 1. **Streamlit Cloud** (Recommended - Free)

**Pros:**
- Free tier available
- GitHub integration
- Auto-deploys on push
- Custom domain support
- No infrastructure management

**Setup:**

```bash
# 1. Create GitHub repository
git init
git add .
git commit -m "Initial commit: Crop price prediction app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crop-price-prediction.git
git push -u origin main

# 2. Go to https://share.streamlit.io
# 3. Click "New app"
# 4. Select your repository
# 5. Set main file to: app_v2.py
# 6. Deploy

# Your app will be live at: https://YOUR_USERNAME-crop-price-prediction.streamlit.app
```

**Important Files:**
- `app_v2.py` - Main application
- `metadata_utils.py` - Helper utilities
- `random_forest_model.joblib` - Trained model
- `requirements.txt` - Dependencies

---

### 2. **Docker** (Local or Any Server)

**Pros:**
- Portable across environments
- Consistent dependencies
- Easy scaling

**Setup:**

```bash
# 1. Build image
docker build -t crop-price-app .

# 2. Run container
docker run -p 8501:8501 crop-price-app

# 3. Access app
# http://localhost:8501

# 4. Stop container
docker stop <container_id>
```

**Using Docker Compose:**

```bash
# 1. Start all services
docker-compose up -d

# 2. View logs
docker-compose logs -f

# 3. Stop services
docker-compose down
```

---

### 3. **Heroku** (Paid - $7-50/month)

**Pros:**
- Full control
- Can add database
- Good for production

**Setup:**

```bash
# 1. Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# 2. Login to Heroku
heroku login

# 3. Create app
heroku create crop-price-prediction

# 4. Set buildpack (if needed)
heroku buildpacks:set heroku/python

# 5. Deploy
git push heroku main

# 6. View logs
heroku logs --tail

# 7. Open app
heroku open
```

**Procfile Example:**
```
web: streamlit run app_v2.py --server.port=$PORT
```

---

### 4. **AWS** (Pay-per-use)

**Using ECS:**

```bash
# 1. Build and push Docker image to ECR
aws ecr create-repository --repository-name crop-price-app
docker build -t crop-price-app .
docker tag crop-price-app:latest <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/crop-price-app:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker push <AWS_ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/crop-price-app:latest

# 2. Create ECS task definition
# 3. Create ECS service
# 4. Configure load balancer
```

**Using Elastic Beanstalk:**

```bash
# 1. Install EB CLI
pip install awsebcli

# 2. Initialize app
eb init -p "Python 3.10 running on 64bit Amazon Linux 2" crop-price-app

# 3. Create environment
eb create crop-price-production

# 4. Deploy
git add .
git commit -m "Deploy to Elastic Beanstalk"
eb deploy

# 5. Open app
eb open
```

---

### 5. **Google Cloud Run** (Serverless - Free tier available)

```bash
# 1. Set up gcloud CLI
gcloud init

# 2. Build image
gcloud builds submit --tag gcr.io/PROJECT_ID/crop-price-app

# 3. Deploy to Cloud Run
gcloud run deploy crop-price-app \
  --image gcr.io/PROJECT_ID/crop-price-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# 4. View logs
gcloud run logs read crop-price-app
```

---

### 6. **Azure Container Instances** (Per-second billing)

```bash
# 1. Login to Azure
az login

# 2. Create container registry
az acr create --resource-group myResourceGroup \
  --name myacr --sku Basic

# 3. Build image
az acr build --registry myacr --image crop-price-app:latest .

# 4. Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name crop-price-container \
  --image myacr.azurecr.io/crop-price-app:latest \
  --ports 8501 \
  --environment-variables STREAMLIT_SERVER_PORT=8501

# 5. Get IP address
az container show --resource-group myResourceGroup \
  --name crop-price-container --query ipAddress.ip --output tsv
```

---

## 📊 Comparison Table

| Platform | Cost | Setup Time | Maintenance | Scalability | Custom Domain |
|----------|------|-----------|-------------|-------------|---------------|
| Streamlit Cloud | Free | 5 min | None | Auto | Yes |
| Docker Local | Free | 10 min | Medium | Manual | N/A |
| Heroku | $7-50/mo | 10 min | Low | Auto | Yes |
| AWS ECS | Pay-per-use | 30 min | High | Excellent | Yes |
| Google Cloud Run | Free tier | 15 min | Low | Auto | Yes |
| Azure ACI | Pay-per-second | 15 min | Low | Manual | Yes |

---

## ✅ Pre-Deployment Checklist

- [ ] Model file (`random_forest_model.joblib`) exists
- [ ] All dependencies in `requirements.txt`
- [ ] App runs locally: `streamlit run app_v2.py`
- [ ] No hardcoded secrets or API keys
- [ ] `.gitignore` configured properly
- [ ] README updated with instructions
- [ ] Metadata file present (or auto-generated)

---

## 🔐 Security Checklist

- [ ] No model training data exposed
- [ ] No user data stored locally
- [ ] HTTPS enabled (handled by platform)
- [ ] Input validation in place
- [ ] Error messages don't expose system info
- [ ] Rate limiting configured (if applicable)

---

## 📈 Monitoring & Scaling

### Local/Docker
- Monitor CPU and memory usage
- Set up log aggregation
- Configure health checks

### Cloud Platforms
- Enable CloudWatch/Stackdriver/Log Analytics
- Set up alerts for errors
- Monitor response times
- Configure auto-scaling policies

### Best Practices
1. Monitor prediction latency
2. Track model performance over time
3. Alert on error rates
4. Log all predictions for auditing
5. Regularly retrain model

---

## 🛠️ Troubleshooting

### "Port 8501 already in use"
```bash
# Kill existing process
lsof -i :8501
kill -9 <PID>

# Or use different port
streamlit run app_v2.py --server.port=8502
```

### "Model file not found"
```bash
# Check file exists
ls -la random_forest_model.joblib

# In Docker, ensure COPY command includes model
# In Dockerfile: COPY random_forest_model.joblib .
```

### "Out of memory"
```bash
# Docker: Increase memory limit
docker run -m 2g crop-price-app

# Streamlit: Check for memory leaks in prediction logic
# Consider model quantization for smaller size
```

### "Slow predictions"
```bash
# Profile the code
streamlit run app_v2.py --logger.level=debug

# Optimize model loading (use caching)
# Reduce model complexity if needed
```

---

## 📚 Additional Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/deploy-your-app)
- [Docker Documentation](https://docs.docker.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)
- [AWS Container Services](https://aws.amazon.com/containers/)
- [Google Cloud Run Guide](https://cloud.google.com/run/docs)

---

**Created**: December 2025
**Status**: Ready for Production ✅
