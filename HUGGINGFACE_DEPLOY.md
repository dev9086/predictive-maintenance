# Hugging Face Spaces Deployment Guide

## 🚀 Deploy Streamlit Dashboard on Hugging Face (FREE)

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co
2. Sign up (free account)
3. Verify your email

### Step 2: Create New Space
1. Click your profile → **"New Space"**
2. Fill details:
   - **Name**: `predictive-maintenance-dashboard`
   - **License**: MIT
   - **Space SDK**: Streamlit
   - **Visibility**: Public (or Private)
3. Click **"Create Space"**

### Step 3: Connect GitHub Repository
1. In your new Space, click **"Settings"**
2. Scroll to **"Repository"**
3. Click **"Connect to GitHub"**
4. Select: `dev9086/predictive-maintenance`
5. Click **"Sync"**

### Step 4: Configure Space
1. In Space settings, go to **"Files and versions"**
2. Click **"Add file"** → **"Create a new file"**
3. Create `README.md` in root with:
```markdown
---
title: Predictive Maintenance Dashboard
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.52.2
app_file: src/streamlit_dashboard.py
pinned: false
---
```

### Step 5: Set Environment Variables (Secrets)
1. Go to **Settings** → **"Variables and secrets"**
2. Click **"New secret"**
3. Add these secrets:

```
DB_HOST = <your_render_postgres_host>
DB_PORT = 5432
DB_NAME = <your_db_name>
DB_USER = <your_user>
DB_PASSWORD = <your_password>
API_PORT = 8000
MODEL_VERSION = v1.0
```

### Step 6: Deploy
1. Space automatically builds
2. Wait 3-5 minutes
3. Your dashboard will be live at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance-dashboard
   ```

---

## 🎯 Your Final Setup

| Service | Platform | URL |
|---------|----------|-----|
| **API** | Render | https://predictive-maintenance-m3v9.onrender.com |
| **Dashboard** | Hugging Face | https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance-dashboard |
| **Database** | Render PostgreSQL | Internal |

---

## ✅ Advantages of Hugging Face

- ✅ **100% FREE** forever
- ✅ No sleep/wake delays
- ✅ Fast loading
- ✅ Auto-deploys from GitHub
- ✅ Built-in secret management
- ✅ Easy to share

---

## 🔧 Alternative: Manual Upload

If GitHub sync doesn't work:

1. Go to your Space
2. Click **"Files"** tab
3. Upload these files:
   - `src/streamlit_dashboard.py`
   - `src/model_inference.py`
   - `src/db_connect.py`
   - `src/config_file.py`
   - `requirements.txt`
   - All files in `models/` folder
4. Create `README.md` with the config above
5. Set secrets in Settings
6. Space auto-deploys

---

## 📝 Quick Commands

**Clone your repo to HF Space:**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/predictive-maintenance-dashboard
cd predictive-maintenance-dashboard
git remote add github https://github.com/dev9086/predictive-maintenance.git
git pull github main
git push origin main
```

---

## 🚨 Troubleshooting

### Dashboard won't load
- Check **"Logs"** tab in Space
- Verify all secrets are set
- Ensure `app_file` path is correct

### Database connection error
- Verify DB credentials in secrets
- Check Render PostgreSQL is running
- Test connection from local machine

### Missing dependencies
- Check `requirements.txt` has all packages
- View build logs in Space

---

## 💡 Tips

1. **Use secrets** for sensitive data (never hardcode)
2. **Monitor logs** to debug issues
3. **Test locally** before deploying
4. **Keep repo updated** - HF syncs automatically

---

**Your dashboard will be live in 5 minutes!** 🎉
