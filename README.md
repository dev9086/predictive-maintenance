# Predictive Maintenance System

AI-powered predictive maintenance system for industrial machines using machine learning to predict failures, estimate remaining useful life (RUL), and detect anomalies.

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

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure database (edit .env with your PostgreSQL credentials)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=zepto_sql_project
# DB_USER=postgres
# DB_PASSWORD=your_password

# Initialize database
python src/init_db.py
```

### Run Services

**Terminal 1 - FastAPI Server:**
```bash
uvicorn src.fastapi_server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Streamlit Dashboard:**
```bash
streamlit run src/streamlit_dashboard.py --server.port 8502
```

### Access

- **Streamlit Dashboard**: http://localhost:8502
- **FastAPI Docs**: http://127.0.0.1:8000/docs
- **Prediction API**: POST http://127.0.0.1:8000/predict

## 📊 Project Structure

```
predictive-maintenance/
├── src/
│   ├── config_file.py           # Configuration management
│   ├── db_connect.py            # Database connection
│   ├── init_db.py               # Database initialization
│   ├── etl.py                   # Data pipeline
│   ├── preprocessing.py         # Data preprocessing
│   ├── model_training.py        # Model training
│   ├── model_inference.py       # ML inference engine
│   ├── fastapi_server.py        # REST API server
│   ├── streamlit_dashboard.py   # Web dashboard
│   ├── synthetic_predictor.py   # Physics-based fallback
│   └── web_scraper.py           # Web scraper utility
├── models/
│   ├── classifier.pkl           # Failure prediction model
│   ├── regressor.pkl            # RUL estimation model
│   ├── anomaly_detector.pkl     # Anomaly detection model
│   ├── scaler.joblib            # Feature scaler
│   └── feature_columns.txt      # Feature list
├── data/
│   └── raw/
│       └── ai4i2020.csv         # Training dataset (10K rows)
├── sql/
│   ├── schema.sql               # Database schema
│   └── seed_data.sql            # Sample data
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not in repo)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎯 Key Features

- **Failure Prediction**: Random Forest classifier (92% accuracy)
- **RUL Estimation**: Random Forest regressor (MAE: 5.36 days)
- **Anomaly Detection**: Isolation Forest with 5% contamination
- **Real-time Dashboard**: Interactive Streamlit interface
- **REST API**: FastAPI with auto-generated docs
- **Physics Fallback**: Synthetic predictor when models unavailable
- **Database**: PostgreSQL with 10K sensor readings

## 📈 API Usage

### Single Prediction
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

### Response
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

## 🛠️ Model Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | 92% |
| Precision (Failures) | 0.30 |
| Recall (Failures) | 0.94 |
| RUL MAE | 5.36 days |
| Train-Test Gap | <1% |

## 🔧 Technology Stack

- **ML**: scikit-learn, XGBoost
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit, Plotly
- **Database**: PostgreSQL, psycopg2
- **Data**: pandas, numpy

## 📝 Environment Variables (.env)

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=zepto_sql_project
DB_USER=postgres
DB_PASSWORD=your_password

# API
API_PORT=8000

# Models
MODEL_VERSION=v1.0
FAILURE_THRESHOLD=0.7
RUL_ALERT_DAYS=7
```

## ⚠️ Important Notes

- Models were trained with scikit-learn 1.6.1; compatibility warnings are normal
- Database must be running before starting services
- `.env` file contains sensitive credentials (not in repo)
- Virtual environment `.venv/` is not committed (in .gitignore)

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature-name`
2. Commit changes: `git commit -am "Add feature"`
3. Push to branch: `git push origin feature-name`
4. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

## 📧 Contact

For issues or questions, open a GitHub issue or contact the maintainer.

