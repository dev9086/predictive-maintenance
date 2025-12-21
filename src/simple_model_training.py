"""
Simple Model Training - Uses only 5 raw sensor features
Trains lightweight models without complex feature engineering
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config_file import Config

# Raw features only (no feature engineering)
FEATURE_COLS = [
    'air_temperature',
    'process_temperature',
    'rotational_speed',
    'torque',
    'tool_wear'
]

def load_data():
    """Load raw data"""
    logger.info("Loading data...")
    df = pd.read_csv('data/raw/ai4i2020.csv')
    
    # Rename columns to match our naming
    rename_map = {
        'Air temperature [K]': 'air_temperature',
        'Process temperature [K]': 'process_temperature',
        'Rotational speed [rpm]': 'rotational_speed',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'tool_wear',
        'Target': 'failure',
        'Failure Type': 'failure_type'
    }
    
    df = df.rename(columns=rename_map)
    
    # Convert temperature from Kelvin to Celsius
    if df['air_temperature'].mean() > 200:
        df['air_temperature'] = df['air_temperature'] - 273.15
        df['process_temperature'] = df['process_temperature'] - 273.15
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Failure rate: {df['failure'].mean():.2%}")
    
    return df

def prepare_data(df):
    """Prepare features and targets"""
    X = df[FEATURE_COLS].copy()
    y_class = df['failure'].copy()
    
    # For RUL: simulate days until failure (inverse of failure prob + noise)
    # If failed, RUL=0; if healthy, RUL=random between 10-30 days
    y_rul = np.where(
        y_class == 1,
        np.random.uniform(0, 3, size=len(y_class)),  # Failed: 0-3 days
        np.random.uniform(10, 30, size=len(y_class))  # Healthy: 10-30 days
    )
    
    return X, y_class, y_rul

def train_models(X, y_class, y_rul):
    """Train all models"""
    logger.info("=" * 60)
    logger.info("Training Models")
    logger.info("=" * 60)
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_rul_train, y_rul_test = train_test_split(
        X, y_class, y_rul, test_size=0.2, random_state=42, stratify=y_class
    )
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
    # Scale features
    logger.info("\n1. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Classification Model
    logger.info("\n2. Training failure classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X_train_scaled, y_class_train)
    
    y_pred_class = classifier.predict(X_test_scaled)
    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_class_test, y_pred_class)}")
    
    # 2. Regression Model (RUL)
    logger.info("\n3. Training RUL regressor...")
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    regressor.fit(X_train_scaled, y_rul_train)
    
    y_pred_rul = regressor.predict(X_test_scaled)
    mae = mean_absolute_error(y_rul_test, y_pred_rul)
    rmse = np.sqrt(mean_squared_error(y_rul_test, y_pred_rul))
    logger.info(f"RUL MAE: {mae:.2f} days")
    logger.info(f"RUL RMSE: {rmse:.2f} days")
    
    # 3. Anomaly Detector
    logger.info("\n4. Training anomaly detector...")
    anomaly_detector = IsolationForest(
        contamination=0.1,
        random_state=42
    )
    anomaly_detector.fit(X_train_scaled)
    
    return classifier, regressor, anomaly_detector, scaler

def save_models(classifier, regressor, anomaly_detector, scaler):
    """Save models to disk"""
    logger.info("\n" + "=" * 60)
    logger.info("Saving Models")
    logger.info("=" * 60)
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    joblib.dump(classifier, os.path.join(Config.MODEL_DIR, 'classifier.pkl'))
    logger.info(f"✅ Classifier saved")
    
    joblib.dump(regressor, os.path.join(Config.MODEL_DIR, 'regressor.pkl'))
    logger.info(f"✅ Regressor saved")
    
    joblib.dump(anomaly_detector, os.path.join(Config.MODEL_DIR, 'anomaly_detector.pkl'))
    logger.info(f"✅ Anomaly detector saved")
    
    joblib.dump(scaler, os.path.join(Config.MODEL_DIR, 'scaler.joblib'))
    logger.info(f"✅ Scaler saved")
    
    # Save feature columns
    with open(os.path.join(Config.MODEL_DIR, 'feature_columns.txt'), 'w') as f:
        for col in FEATURE_COLS:
            f.write(f"{col}\n")
    logger.info(f"✅ Feature columns saved ({len(FEATURE_COLS)} features)")

def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("SIMPLE MODEL TRAINING - 5 RAW FEATURES ONLY")
    logger.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y_class, y_rul = prepare_data(df)
    
    # Train models
    classifier, regressor, anomaly_detector, scaler = train_models(X, y_class, y_rul)
    
    # Save models
    save_models(classifier, regressor, anomaly_detector, scaler)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
