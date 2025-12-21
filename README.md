# Predictive Maintenance System

AI-powered predictive maintenance system for industrial machines using machine learning to predict failures, estimate remaining useful life (RUL), and detect anomalies.

## Features

- **Failure Prediction**: ML models predict probability of machine failure
- **RUL Estimation**: Estimates remaining useful life in days
- **Anomaly Detection**: Identifies abnormal sensor readings
- **Real-time Dashboard**: Interactive Streamlit dashboard for monitoring
- **REST API**: FastAPI endpoints for programmatic access
- **PostgreSQL Integration**: Stores sensor data and predictions

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Configure database
# Edit src/config_file.py with your PostgreSQL credentials

# Initialize database
python src/init_db.py

# Train models (optional - physics fallback available)
python src/simple_model_training.py
```

### Run Services

```bash
# Terminal 1: Start FastAPI server
uvicorn src.fastapi_server:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Streamlit dashboard
streamlit run src/streamlit_dashboard.py --server.port 8501
```

### Access

- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI Docs**: http://127.0.0.1:8000/docs
- **API Endpoint**: POST http://127.0.0.1:8000/predict

## Usage

### Send Sensor Reading (API)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": 1,
    "features": {
      "air_temperature": 35.5,
      "process_temperature": 45.2,
      "rotational_speed": 1500,
      "torque": 50.3,
      "tool_wear": 120
    }
  }'
```

### Response

```json
{
  "success": true,
  "prediction": {
    "failure_probability": 0.25,
    "predicted_rul_days": 15.5,
    "is_anomaly": false,
    "risk_level": "LOW"
  },
  "recommendations": ["Machine operating normally"]
}
```

## Project Structure

```
predictive-maintenance/
├── src/
│   ├── config_file.py          # Configuration
│   ├── db_connect.py            # Database utilities
│   ├── model_inference.py       # ML inference engine
│   ├── fastapi_server.py        # REST API server
│   ├── streamlit_dashboard.py   # Web dashboard
│   ├── model_training.py        # Train models
│   └── synthetic_predictor.py   # Physics fallback
├── models/                      # Trained ML models
├── data/
│   └── raw/                     # Raw training data
├── sql/                         # Database schemas
├── requirements.txt             # Dependencies
└── README.md
```

## Models

- **Classifier**: Random Forest for failure prediction
- **Regressor**: Random Forest for RUL estimation
- **Anomaly Detector**: Isolation Forest for anomaly detection

Uses physics-based fallback if ML models unavailable.

## Technologies

- **ML**: scikit-learn, XGBoost
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit, Plotly
- **Database**: PostgreSQL, psycopg2
- **Data**: pandas, numpy

## License

MIT

## Contributing

Pull requests welcome. For major changes, please open an issue first.
