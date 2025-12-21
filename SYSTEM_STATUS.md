# 🏭 Predictive Maintenance System - Status Report
**Date:** December 21, 2025

## ✅ System Status: ALL WORKING

### 1. Database Status ✅
- **Connection:** Working
- **Sensor Readings:** 10,000 records
- **Model Predictions:** 1,050 records
- **Tables:** All created successfully
- **Connection Type:** PostgreSQL via psycopg2

### 2. Streamlit Dashboard ✅
- **Status:** Running on http://localhost:8501
- **Port:** 8501
- **Auto-reload:** Enabled

#### New Features Added:
✨ **3 Modes Available:**

1. **Machine Dashboard** (Original)
   - Select machine from database
   - View latest sensor readings
   - See real-time predictions
   - View prediction history with charts

2. **Manual Prediction** (NEW)
   - Enter sensor values manually
   - Get instant predictions
   - See detailed risk assessment
   - View recommendations
   - Perfect for testing scenarios

3. **Batch Predictions** (NEW)
   - Upload CSV file with multiple rows
   - Or manually enter multiple rows
   - Get predictions for all rows at once
   - Download results as CSV
   - View summary statistics
   - Progress bar for large batches

### 3. FastAPI Server ✅
- **Status:** Running on http://127.0.0.1:8000
- **Port:** 8000
- **Auto-reload:** Enabled
- **API Docs:** http://127.0.0.1:8000/docs
- **Models:** Loaded at startup

#### Available Endpoints:
- `POST /predict` - Single prediction
- `GET /health` - Health check
- API documentation with Swagger UI

### 4. ML Models ✅
- **Location:** `models/` directory
- **Files:**
  - `classifier.pkl` - Failure prediction
  - `regressor.pkl` - RUL estimation
  - `anomaly_detector.pkl` - Anomaly detection
  - `scaler.joblib` - Feature scaling
  - `feature_columns.txt` - Feature names (90 features)

- **Inference Engine:** Hybrid approach
  - Primary: ML models (trained)
  - Fallback: Physics-based simulator
  - Feature engineering: 5 raw → 90 engineered features

### 5. Web Scraper ✅
- **Status:** Available in `src/web_scraper.py`
- **Purpose:** Scrape manufacturer specs and reliability data
- **Features:**
  - Static website scraping (BeautifulSoup)
  - Dynamic website scraping (Selenium)
  - MTBF data extraction
  - Parts lifecycle information
  - Industry benchmarks

**Usage:**
```python
from web_scraper import MaintenanceDataScraper
scraper = MaintenanceDataScraper()
scraper.scrape_manufacturer_specs_static(url, machine_id)
```

### 6. Data Pipeline ✅
**Flow:**
1. Raw sensor data → `data/raw/ai4i2020.csv`
2. ETL process → `src/etl.py`
3. Database storage → PostgreSQL
4. Model inference → `src/model_inference.py`
5. Predictions → Database
6. Visualization → Streamlit Dashboard

## 🎯 How to Use

### Access Streamlit Dashboard:
1. Open browser: http://localhost:8501
2. Select mode from sidebar:
   - **Machine Dashboard:** Monitor existing machines
   - **Manual Prediction:** Test with custom values
   - **Batch Predictions:** Analyze multiple scenarios

### Manual Prediction Example:
1. Select "Manual Prediction" mode
2. Enter values:
   - Air Temperature: 25.0°C
   - Process Temperature: 35.0°C
   - Rotational Speed: 1500 RPM
   - Torque: 40.0 Nm
   - Tool Wear: 100 minutes
3. Click "🔮 Predict"
4. View results:
   - Failure probability
   - Remaining useful life
   - Anomaly status
   - Risk level
   - Recommendations

### Batch Prediction Example:
1. Select "Batch Predictions" mode
2. Choose tab:
   - **Upload CSV:** Upload file with columns
   - **Manual Entry:** Enter multiple rows
3. CSV format:
```csv
air_temperature,process_temperature,rotational_speed,torque,tool_wear
25.0,35.0,1500,40.0,100
30.0,42.0,1600,45.0,150
28.5,38.5,1550,42.5,120
```
4. Click "🔮 Run Batch Predictions"
5. Download results as CSV

### API Usage:
```bash
# Single prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": 1,
    "features": {
      "air_temperature": 25.0,
      "process_temperature": 35.0,
      "rotational_speed": 1500,
      "torque": 40.0,
      "tool_wear": 100
    }
  }'
```

## 📊 System Architecture

```
┌─────────────┐
│  Raw Data   │
│ (CSV/DB)    │
└──────┬──────┘
       │
       v
┌─────────────┐
│  ETL/Data   │
│  Pipeline   │
└──────┬──────┘
       │
       v
┌─────────────┐      ┌──────────────┐
│ PostgreSQL  │◄────►│  Web Scraper │
│  Database   │      │  (External)  │
└──────┬──────┘      └──────────────┘
       │
       │
       ├──────────────────────┬──────────────────┐
       v                      v                  v
┌─────────────┐      ┌─────────────┐    ┌─────────────┐
│   FastAPI   │      │  Streamlit  │    │  ML Models  │
│  (Port 8000)│      │ (Port 8501) │    │   Engine    │
└─────────────┘      └─────────────┘    └─────────────┘
       │                      │                  │
       └──────────────────────┴──────────────────┘
                      │
                      v
              ┌─────────────┐
              │   User      │
              │ Interface   │
              └─────────────┘
```

## 🔧 Maintenance

### Stop Services:
```powershell
# Stop Streamlit (Ctrl+C in terminal)
# Stop FastAPI (Ctrl+C in terminal)
```

### Restart Services:
```powershell
# Terminal 1: FastAPI
.\.venv\Scripts\Activate.ps1
uvicorn src.fastapi_server:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Streamlit
.\.venv\Scripts\Activate.ps1
streamlit run src/streamlit_dashboard.py --server.port 8501
```

### Retrain Models:
```powershell
python src/simple_model_training.py
```

## 📝 Notes

1. **Web Scraper:** Available but requires target URLs to be configured
2. **Models:** Currently using 90-feature models with feature engineering
3. **Fallback:** Physics-based predictor available if ML models fail
4. **Database:** 10,000 sensor readings ready for training/testing

## ✨ Recent Enhancements

- ✅ Added Manual Prediction mode for custom input
- ✅ Added Batch Prediction mode (CSV upload + manual entry)
- ✅ Progress bars for batch processing
- ✅ CSV download for batch results
- ✅ Enhanced risk assessment visualization
- ✅ Summary statistics for batch predictions
- ✅ Interactive data editor for manual batch entry

---

**System Ready for Production! 🚀**
