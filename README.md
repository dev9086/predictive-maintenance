# Predictive Maintenance System

AI-powered predictive maintenance system for industrial machines using machine learning to predict failures, estimate remaining useful life (RUL), and detect anomalies.

## 🎯 What It Does

- **Failure Prediction**: Predicts probability of machine failure using Random Forest classifier
- **RUL Estimation**: Estimates remaining useful life in days using machine learning
- **Anomaly Detection**: Identifies abnormal sensor readings in real-time
- **Real-time Monitoring**: Interactive Streamlit dashboard for live tracking
- **REST API**: Programmatic access via FastAPI with auto-generated documentation
- **Fallback System**: Physics-based predictor when ML models unavailable

## 📊 Model Performance

### Classification (Failure Prediction)
| Metric | Score |
|--------|-------|
| **Accuracy** | 92% |
| **Precision** | 0.30 |
| **Recall** | 0.94 |
| **F1-Score** | 0.45 |

### Regression (RUL Prediction)
| Metric | Score |
|--------|-------|
| **Mean Absolute Error (MAE)** | 5.36 days |
| **Root Mean Squared Error (RMSE)** | 6.29 days |
| **R² Score** | 0.78 |

### Model Generalization
- Train-Test Gap: **<1%** (excellent generalization)
- Overfitting: ✅ Mitigated with careful hyperparameter tuning

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/dev9086/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate
# Or on Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup .env file with your database credentials
# Create .env in project root with:
# DB_HOST=your_host
# DB_PORT=5432
# DB_NAME=your_database
# DB_USER=your_user
# DB_PASSWORD=your_password

# Initialize database
python src/init_db.py
```

### Run Services

**Terminal 1 - FastAPI Server (Port 8000):**
```bash
uvicorn src.fastapi_server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Streamlit Dashboard (Port 8502):**
```bash
streamlit run src/streamlit_dashboard.py --server.port 8502
```

### Access

- **📊 Dashboard**: http://localhost:8502
- **📚 API Docs**: http://127.0.0.1:8000/docs
- **⚙️ OpenAPI**: http://127.0.0.1:8000/openapi.json

## 📈 API Endpoints

### POST /predict
Single machine prediction with sensor data

```bash
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

**Response:**
```json
{
  "success": true,
  "prediction": {
    "failure_probability": 0.15,
    "predicted_rul_days": 20.5,
    "is_anomaly": false,
    "risk_level": "LOW"
  },
  "recommendations": ["Machine operating normally"]
}
```

### GET /health
Server health check

```bash
curl http://127.0.0.1:8000/health
```

## 📂 Project Structure

```
predictive-maintenance/
├── src/
│   ├── config_file.py              # Configuration management
│   ├── db_connect.py               # PostgreSQL connection
│   ├── init_db.py                  # Database initialization
│   ├── etl.py                      # Data pipeline
│   ├── preprocessing.py            # Feature engineering
│   ├── model_training.py           # Model training script
│   ├── model_inference.py          # ML inference engine
│   ├── fastapi_server.py           # REST API
│   ├── streamlit_dashboard.py      # Web UI
│   ├── synthetic_predictor.py      # Physics fallback
│   └── web_scraper.py              # Web scraper utility
├── models/
│   ├── classifier.pkl              # Failure prediction model
│   ├── regressor.pkl               # RUL estimation model
│   ├── anomaly_detector.pkl        # Anomaly detection model
│   ├── scaler.joblib               # Feature scaler
│   └── feature_columns.txt         # Feature list
├── data/
│   └── raw/
│       └── ai4i2020.csv            # Training data (10K rows)
├── sql/
│   ├── schema.sql                  # Database schema
│   └── seed_data.sql               # Sample data
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git rules
├── .env                            # Environment variables (local only)
└── README.md                       # This file
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn, XGBoost |
| **API** | FastAPI, Uvicorn |
| **Web UI** | Streamlit, Plotly |
| **Database** | PostgreSQL |
| **Data Processing** | pandas, numpy |
| **Visualization** | Matplotlib, Seaborn |

## 🎓 Machine Learning Models

### Classifier
- **Algorithm**: Random Forest (50 trees, max_depth=5)
- **Purpose**: Predict machine failure probability
- **Input**: 5 sensor features
- **Output**: Failure probability (0-1)

### Regressor
- **Algorithm**: Random Forest (50 trees, max_depth=5)
- **Purpose**: Estimate remaining useful life
- **Input**: 5 sensor features
- **Output**: RUL in days

### Anomaly Detector
- **Algorithm**: Isolation Forest
- **Purpose**: Detect abnormal sensor readings
- **Contamination**: 5%
- **Output**: Anomaly flag (binary)

## 📋 Dataset

**AI4I 2020 Synthetic Dataset**
- **Size**: 10,000 sensor readings
- **Features**: 5 numeric sensors
  - Air Temperature (K)
  - Process Temperature (K)
  - Rotational Speed (RPM)
  - Torque (Nm)
  - Tool Wear (minutes)
- **Target**: Machine failure (binary)
- **Source**: UCI Machine Learning Repository

## ⚙️ Configuration

Create a `.env` file in the project root (never commit this):

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
API_PORT=8000
MODEL_VERSION=v1.0
FAILURE_THRESHOLD=0.7
RUL_ALERT_DAYS=7
```

## 🔄 Dashboard Features

### Machine Dashboard
- Select machine from database
- View latest sensor readings
- Real-time predictions
- Prediction history with charts

### Manual Prediction
- Enter custom sensor values
- Get instant predictions
- Risk assessment
- Maintenance recommendations

### Batch Predictions
- Upload CSV with multiple rows
- Predict for all rows at once
- Download results
- Summary statistics

## 🚨 Risk Levels

| Level | Failure Probability | Action |
|-------|-------------------|--------|
| 🟢 **LOW** | 0-30% | Monitor normally |
| 🟡 **MEDIUM** | 30-50% | Schedule maintenance |
| 🟠 **HIGH** | 50-70% | Plan downtime soon |
| 🔴 **CRITICAL** | >70% | Immediate action |

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature-name`
5. Open a pull request

## 📧 Support

For issues or questions, open a GitHub issue.

---

**Note**: This project includes pre-trained ML models. Database credentials must be configured in `.env` file locally. Never commit sensitive information.


