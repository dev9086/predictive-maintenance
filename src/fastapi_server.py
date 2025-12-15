"""
FastAPI REST API for Predictive Maintenance System
Provides endpoints for predictions and machine status
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_file import Config
from db_connect import execute_query, fetch_dataframe, bulk_insert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered predictive maintenance system",
    version=Config.MODEL_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODELS AT STARTUP
# ============================================================================

classifier = None
regressor = None
anomaly_detector = None
scaler = None
feature_columns = []

@app.on_event("startup")
async def load_models():
    """Load models at startup"""
    global classifier, regressor, anomaly_detector, scaler, feature_columns
    
    try:
        model_dir = Config.MODEL_DIR
        
        classifier = joblib.load(os.path.join(model_dir, 'classifier.pkl'))
        regressor = joblib.load(os.path.join(model_dir, 'regressor.pkl'))
        anomaly_detector = joblib.load(os.path.join(model_dir, 'anomaly_detector.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
        # Load feature columns
        with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    machine_id: Optional[int] = None
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    machine_id: Optional[int]
    prediction: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    model_version: str

class MachineInfo(BaseModel):
    """Machine information model"""
    machine_id: int
    machine_name: str
    machine_type: str
    location: str
    manufacturer: Optional[str] = None
    total_readings: Optional[int] = None
    total_failures: Optional[int] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    models_loaded: bool
    timestamp: str

class AlertInfo(BaseModel):
    """Alert information model"""
    machine_id: int
    machine_name: str
    location: str
    failure_probability: float
    predicted_rul: float
    prediction_timestamp: str
    risk_level: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_features(features_dict: Dict[str, float]) -> np.ndarray:
    """Prepare features for prediction"""
    # Create DataFrame with proper feature order
    df = pd.DataFrame([features_dict])
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only required features in correct order
    df = df[feature_columns]
    
    # Scale features
    features_scaled = scaler.transform(df)
    
    return features_scaled

def get_risk_level(failure_prob: float, rul: float) -> str:
    """Determine risk level based on failure probability and RUL"""
    if failure_prob >= 0.7 or rul <= 3:
        return "CRITICAL"
    elif failure_prob >= 0.5 or rul <= 7:
        return "HIGH"
    elif failure_prob >= 0.3 or rul <= 14:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendations(failure_prob: float, rul: float, risk_level: str) -> List[str]:
    """Generate recommendations based on predictions"""
    recommendations = []
    
    if risk_level == "CRITICAL":
        recommendations.append("🚨 IMMEDIATE ACTION REQUIRED")
        recommendations.append("Schedule emergency maintenance within 24 hours")
        recommendations.append("Reduce machine workload immediately")
        recommendations.append("Prepare replacement parts")
    
    elif risk_level == "HIGH":
        recommendations.append("⚠️ Schedule maintenance within 3-5 days")
        recommendations.append("Monitor sensor readings closely")
        recommendations.append("Order replacement parts if needed")
    
    elif risk_level == "MEDIUM":
        recommendations.append("📋 Plan maintenance within 1-2 weeks")
        recommendations.append("Continue routine monitoring")
        recommendations.append("Review maintenance history")
    
    else:
        recommendations.append("✅ Continue normal operations")
        recommendations.append("Maintain regular monitoring schedule")
    
    return recommendations

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API home page"""
    return {
        "service": "Predictive Maintenance API",
        "version": Config.MODEL_VERSION,
        "status": "running",
        "endpoints": {
            "GET /health": "Health check",
            "GET /machines": "List all machines",
            "GET /machine/{id}": "Get machine details",
            "POST /predict": "Get failure prediction",
            "GET /predictions/{machine_id}": "Get prediction history",
            "GET /alerts": "Get active alerts",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([classifier, regressor, anomaly_detector])
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.get("/machines")
async def get_machines():
    """Get list of all machines"""
    try:
        query = """
            SELECT 
                m.machine_id,
                m.machine_name,
                m.machine_type,
                m.location,
                m.manufacturer,
                COUNT(DISTINCT sr.reading_id) as total_readings,
                COUNT(DISTINCT fl.failure_id) as total_failures
            FROM machines m
            LEFT JOIN sensor_readings sr ON m.machine_id = sr.machine_id
            LEFT JOIN failure_logs fl ON m.machine_id = fl.machine_id
            GROUP BY m.machine_id
            ORDER BY m.machine_id
        """
        
        df = fetch_dataframe(query)
        machines = df.to_dict('records')
        
        return {
            "success": True,
            "count": len(machines),
            "machines": machines
        }
    
    except Exception as e:
        logger.error(f"Error fetching machines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/machine/{machine_id}")
async def get_machine_detail(machine_id: int):
    """Get detailed information about a machine"""
    try:
        # Machine info
        machine_query = """
            SELECT * FROM machines WHERE machine_id = %s
        """
        machine_df = fetch_dataframe(machine_query, (machine_id,))
        
        if len(machine_df) == 0:
            raise HTTPException(status_code=404, detail="Machine not found")
        
        # Latest sensor reading
        sensor_query = """
            SELECT * FROM sensor_readings
            WHERE machine_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        sensor_df = fetch_dataframe(sensor_query, (machine_id,))
        
        # Latest prediction
        pred_query = """
            SELECT * FROM model_predictions
            WHERE machine_id = %s
            ORDER BY prediction_timestamp DESC
            LIMIT 1
        """
        pred_df = fetch_dataframe(pred_query, (machine_id,))
        
        # Failure history
        failure_query = """
            SELECT * FROM failure_logs
            WHERE machine_id = %s
            ORDER BY failure_date DESC
            LIMIT 5
        """
        failure_df = fetch_dataframe(failure_query, (machine_id,))
        
        return {
            "success": True,
            "machine": machine_df.to_dict('records')[0],
            "latest_sensor": sensor_df.to_dict('records')[0] if len(sensor_df) > 0 else None,
            "latest_prediction": pred_df.to_dict('records')[0] if len(pred_df) > 0 else None,
            "recent_failures": failure_df.to_dict('records')
        }
    
    except Exception as e:
        logger.error(f"Error fetching machine detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction for a machine
    
    Expected JSON:
    {
        "machine_id": 1,
        "features": {
            "air_temperature": 25.5,
            "process_temperature": 35.2,
            "rotational_speed": 1500,
            "torque": 42.8,
            "tool_wear": 120,
            ... (other features)
        }
    }
    """
    try:
        if not classifier or not regressor:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Prepare features
        features_scaled = prepare_features(request.features)
        
        # Make predictions
        failure_prob = float(classifier.predict_proba(features_scaled)[0][1])
        predicted_rul = float(regressor.predict(features_scaled)[0])
        anomaly_score = float(anomaly_detector.score_samples(features_scaled)[0])
        
        # Determine risk level
        risk_level = get_risk_level(failure_prob, predicted_rul)
        
        # Save prediction to database
        if request.machine_id:
            try:
                bulk_insert(
                    'model_predictions',
                    ['machine_id', 'failure_probability', 'predicted_rul', 
                     'anomaly_score', 'model_version'],
                    [(request.machine_id, failure_prob, predicted_rul, anomaly_score, Config.MODEL_VERSION)]
                )
            except Exception as db_error:
                logger.warning(f"Could not save prediction to DB: {db_error}")
        
        # Prepare response
        response = PredictionResponse(
            success=True,
            machine_id=request.machine_id,
            prediction={
                "failure_probability": round(failure_prob, 4),
                "predicted_rul_days": round(predicted_rul, 2),
                "anomaly_score": round(anomaly_score, 4),
                "risk_level": risk_level
            },
            recommendations=get_recommendations(failure_prob, predicted_rul, risk_level),
            timestamp=datetime.now().isoformat(),
            model_version=Config.MODEL_VERSION
        )
        
        logger.info(f"Prediction made for machine {request.machine_id}: {risk_level} risk")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{machine_id}")
async def get_prediction_history(machine_id: int, limit: int = Query(100, ge=1, le=1000)):
    """Get prediction history for a machine"""
    try:
        query = """
            SELECT 
                pred_id,
                prediction_timestamp,
                failure_probability,
                predicted_rul,
                anomaly_score,
                model_version
            FROM model_predictions
            WHERE machine_id = %s
            ORDER BY prediction_timestamp DESC
            LIMIT %s
        """
        
        df = fetch_dataframe(query, (machine_id, limit))
        predictions = df.to_dict('records')
        
        return {
            "success": True,
            "machine_id": machine_id,
            "count": len(predictions),
            "predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts():
    """Get active alerts"""
    try:
        query = """
            SELECT 
                m.machine_id,
                m.machine_name,
                m.location,
                p.failure_probability,
                p.predicted_rul,
                p.prediction_timestamp
            FROM model_predictions p
            JOIN machines m ON p.machine_id = m.machine_id
            WHERE p.failure_probability >= %s OR p.predicted_rul <= %s
            ORDER BY p.failure_probability DESC, p.predicted_rul ASC
        """
        
        df = fetch_dataframe(query, (Config.FAILURE_THRESHOLD, Config.RUL_ALERT_DAYS))
        alerts = df.to_dict('records')
        
        # Add risk level to each alert
        for alert in alerts:
            alert['risk_level'] = get_risk_level(
                alert['failure_probability'],
                alert['predicted_rul']
            )
        
        return {
            "success": True,
            "count": len(alerts),
            "alerts": alerts,
            "thresholds": {
                "failure_probability": Config.FAILURE_THRESHOLD,
                "rul_days": Config.RUL_ALERT_DAYS
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

logger.info("=" * 60)
logger.info("Predictive Maintenance FastAPI Server")
logger.info(f"Model Version: {Config.MODEL_VERSION}")
logger.info(f"Access documentation at: http://localhost:{Config.API_PORT}/docs")
logger.info("=" * 60)
