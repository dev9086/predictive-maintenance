# Deployment Guide - Render.com

Complete guide to deploy your Predictive Maintenance System on Render.com (FREE).

## 🚀 Quick Start

### Step 1: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub account
3. Authorize Render to access your repositories

### Step 2: Create Database Service
1. Click **New +** → **PostgreSQL**
2. Fill in details:
   - **Name**: `predictive-maintenance-db`
   - **Database**: `maintenance_db`
   - **User**: `postgres`
   - **Region**: Choose closest to you
   - **Plan**: Free
3. Click **Create Database**
4. Copy the **Internal Database URL** (you'll need it)

### Step 3: Create Web Service
1. Click **New +** → **Web Service**
2. Connect your GitHub repository: `dev9086/predictive-maintenance`
3. Fill in details:
   - **Name**: `predictive-maintenance`
   - **Runtime**: Docker
   - **Region**: Same as database
   - **Branch**: `main`
   - **Plan**: Free

### Step 4: Set Environment Variables
In the Web Service settings, click **Environment**:

```
DB_HOST=<internal_db_url_host>
DB_PORT=5432
DB_NAME=maintenance_db
DB_USER=postgres
DB_PASSWORD=<password_from_step_2>
API_PORT=8000
MODEL_VERSION=v1.0
FAILURE_THRESHOLD=0.7
RUL_ALERT_DAYS=7
```

### Step 5: Deploy
- Click **Create Web Service**
- Render automatically deploys from GitHub
- Wait 5-10 minutes for build to complete

### Step 6: Access Your App
Once deployed, you'll get URLs:
- **API**: `https://your-app.onrender.com`
- **Dashboard**: `https://your-app.onrender.com:8501`

---

## 📋 Full Setup Steps

### 1️⃣ GitHub Repository
Ensure your repo has:
- ✅ `Dockerfile`
- ✅ `requirements.txt`
- ✅ `src/` folder with all code
- ✅ `models/` folder with trained models
- ✅ `data/raw/ai4i2020.csv`

### 2️⃣ PostgreSQL Database

**Create Database:**
- Service Type: PostgreSQL
- Name: `predictive-maintenance-db`
- Region: Choose your region
- Plan: Free

**Get Connection Details:**
```
Host: <internal-database-url>
Port: 5432
Database: maintenance_db
User: postgres
Password: <your-password>
```

### 3️⃣ Web Service Configuration

**Create Web Service:**
- Name: `predictive-maintenance`
- Runtime: Docker
- Repository: `dev9086/predictive-maintenance`
- Branch: `main`
- Dockerfile: `./Dockerfile`
- Region: Same as database

### 4️⃣ Environment Variables

```env
DB_HOST=<database_internal_url_host>
DB_PORT=5432
DB_NAME=maintenance_db
DB_USER=postgres
DB_PASSWORD=<database_password>
API_PORT=8000
MODEL_VERSION=v1.0
FAILURE_THRESHOLD=0.7
RUL_ALERT_DAYS=7
```

### 5️⃣ Build & Deploy

Render automatically:
- Pulls code from GitHub
- Builds Docker image
- Initializes database
- Starts services
- Assigns public URLs

---

## 🔗 Access Your Deployment

**After deployment completes:**

| Service | URL |
|---------|-----|
| **API** | `https://your-app.onrender.com` |
| **API Docs** | `https://your-app.onrender.com/docs` |
| **Dashboard** | `https://your-app.onrender.com:8501` |

---

## ⚙️ How It Works

1. **Dockerfile** - Builds container with all dependencies
2. **init_db.py** - Initializes PostgreSQL database
3. **FastAPI Server** - Runs on port 8000
4. **Streamlit App** - Runs on port 8501
5. **PostgreSQL** - Manages all data

---

## 📊 Free Plan Limits

| Resource | Limit |
|----------|-------|
| **Compute** | 0.5 vCPU |
| **Memory** | 512 MB |
| **Database** | 1 GB free PostgreSQL |
| **Sleep** | Sleeps after 15 min inactivity |
| **Cost** | **FREE** |

---

## 🔄 Automatic Updates

Every time you push to GitHub:
1. Render detects changes
2. Automatically rebuilds Docker image
3. Redeploys application
4. No downtime needed

---

## 🚨 Troubleshooting

### App won't start
- Check **Logs** tab for errors
- Ensure all environment variables are set
- Verify database credentials

### Database connection fails
- Check **Internal Database URL** format
- Verify credentials match
- Ensure database is in same region

### Slow response times
- Free tier has limited compute (0.5 vCPU)
- Consider upgrading to paid plan for better performance

### Port conflicts
- FastAPI: 8000
- Streamlit: 8501
- Both should be configured in Dockerfile

---

## 💡 Tips

1. **Monitor Performance**: Check Render Dashboard for metrics
2. **View Logs**: Logs tab shows real-time app output
3. **Manual Deploys**: Click "Deploys" → "New Deploy" to force redeploy
4. **Git Integration**: Push to main branch to auto-deploy
5. **Backup Data**: Regularly backup PostgreSQL database

---

## 📈 Upgrading (Optional)

When free tier is not enough:
- **Starter Plan**: $7/month (1 vCPU, 512 MB RAM)
- **Standard Plan**: $19/month (2 vCPU, 2 GB RAM)
- **Pro Plan**: Custom pricing

---

## 📧 Support

**Issues during deployment?**
- Check Render documentation: https://render.com/docs
- Review app logs in Render Dashboard
- Verify all environment variables are correct

---

## ✅ Deployment Checklist

- [ ] GitHub account connected to Render
- [ ] PostgreSQL database created
- [ ] Database credentials configured
- [ ] Web service created with Docker
- [ ] Environment variables set
- [ ] Deployment completed successfully
- [ ] API endpoints responding
- [ ] Dashboard accessible
- [ ] Database connection working

---

**Your app is now live and free on Render.com!** 🎉
