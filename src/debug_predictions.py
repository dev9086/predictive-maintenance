"""
Debug why predictions are always 99.99%
This helps identify feature misalignment issues
"""
import pandas as pd
import numpy as np
import joblib
import os
from db_connect import fetch_dataframe
from config_file import Config

def check_model_features():
    """Check what features the model expects"""
    try:
        feature_file = os.path.join(Config.MODEL_DIR, 'feature_columns.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                expected_features = [line.strip() for line in f.readlines()]
            print("✓ Model expects these features:")
            for i, feat in enumerate(expected_features, 1):
                print(f"  {i}. {feat}")
            return expected_features
        else:
            print("❌ feature_columns.txt not found!")
            return None
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return None

def check_sensor_columns():
    """Check what columns exist in sensor_readings"""
    try:
        query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'sensor_readings'
            AND column_name NOT IN ('reading_id', 'machine_id', 'timestamp', 'ts')
            ORDER BY ordinal_position
        """
        df = fetch_dataframe(query)
        sensor_cols = df['column_name'].tolist()
        print("\n✓ Sensor readings columns:")
        for i, col in enumerate(sensor_cols, 1):
            print(f"  {i}. {col}")
        return sensor_cols
    except Exception as e:
        print(f"❌ Error checking sensor columns: {e}")
        return None

def check_sample_sensor_data():
    """Get sample sensor reading"""
    try:
        query = """
            SELECT * FROM sensor_readings 
            WHERE machine_id = 1 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        df = fetch_dataframe(query)
        if df.empty:
            print("❌ No sensor data found")
            return None
        
        print("\n✓ Sample sensor reading:")
        for col, val in df.iloc[0].items():
            print(f"  {col}: {val}")
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"❌ Error getting sensor data: {e}")
        return None

def test_prediction_with_debugging():
    """Test prediction with detailed debugging"""
    try:
        # Load model
        model_path = os.path.join(Config.MODEL_DIR, 'classifier.pkl')
        scaler_path = os.path.join(Config.MODEL_DIR, 'scaler.joblib')
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        classifier = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Get expected features
        expected_features = check_model_features()
        if expected_features is None:
            return
        
        # Get sample sensor data
        sensor_data = check_sample_sensor_data()
        if sensor_data is None:
            return
        
        # Create feature dict (only numeric sensor columns)
        features = {
            'air_temperature': float(sensor_data.get('air_temperature', 0)),
            'process_temperature': float(sensor_data.get('process_temperature', 0)),
            'rotational_speed': float(sensor_data.get('rotational_speed', 0)),
            'torque': float(sensor_data.get('torque', 0)),
            'tool_wear': float(sensor_data.get('tool_wear', 0))
        }
        
        print("\n✓ Features being used for prediction:")
        for k, v in features.items():
            print(f"  {k}: {v}")
        
        # Create DataFrame with all expected features
        df = pd.DataFrame([features])
        
        # Add missing features as 0
        missing_features = []
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
                missing_features.append(col)
        
        if missing_features:
            print(f"\n⚠️  WARNING: {len(missing_features)} missing features filled with 0:")
            for feat in missing_features[:10]:  # Show first 10
                print(f"  - {feat}")
            if len(missing_features) > 10:
                print(f"  ... and {len(missing_features) - 10} more")
        
        # Ensure correct order
        df = df[expected_features]
        
        # Check for NaN or inf
        if df.isnull().any().any():
            print("\n❌ NaN values detected in features!")
            print(df.isnull().sum())
        
        if np.isinf(df.values).any():
            print("\n❌ Inf values detected in features!")
        
        # Scale
        X_scaled = scaler.transform(df)
        
        # Check scaled values
        print("\n✓ Scaled feature statistics:")
        print(f"  Mean: {X_scaled.mean():.4f}")
        print(f"  Std:  {X_scaled.std():.4f}")
        print(f"  Min:  {X_scaled.min():.4f}")
        print(f"  Max:  {X_scaled.max():.4f}")
        
        if np.abs(X_scaled).max() > 100:
            print("  ⚠️  WARNING: Extremely large scaled values detected!")
        
        # Predict
        failure_prob = classifier.predict_proba(X_scaled)[0, 1]
        
        print(f"\n🎯 Prediction Result:")
        print(f"  Failure Probability: {failure_prob:.4f} ({failure_prob*100:.2f}%)")
        
        if failure_prob > 0.99:
            print("\n❌ PROBLEM IDENTIFIED:")
            print("  Prediction is too high (>99%). Possible causes:")
            print("  1. Feature mismatch - model trained on different features")
            print("  2. Missing features filled with 0 confuse the model")
            print("  3. Model needs retraining with current sensor data")
            print("\n💡 SOLUTION:")
            print("  Run: python src/model_training.py")
            print("  This will retrain models with correct features from your database")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("="*70)
    print("🔍 PREDICTIVE MAINTENANCE DEBUG TOOL")
    print("="*70)
    
    print("\n1️⃣ Checking model features...")
    expected_features = check_model_features()
    
    print("\n2️⃣ Checking sensor data columns...")
    sensor_cols = check_sensor_columns()
    
    if expected_features and sensor_cols:
        missing = set(expected_features) - set(sensor_cols)
        if missing:
            print(f"\n⚠️  Model expects {len(missing)} features not in sensor data:")
            for feat in list(missing)[:10]:
                print(f"  - {feat}")
    
    print("\n3️⃣ Testing prediction...")
    test_prediction_with_debugging()
    
    print("\n" + "="*70)
    print("✅ DEBUG COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
