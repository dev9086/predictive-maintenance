"""
Quick Test Script for Predictive Maintenance System
Run this to verify all components are working
"""
import sys
sys.path.insert(0, 'src')

print("=" * 70)
print("🧪 TESTING PREDICTIVE MAINTENANCE SYSTEM")
print("=" * 70)

# Test 1: Database Connection
print("\n1️⃣  Testing Database Connection...")
try:
    from db_connect import connect, close
    conn, cur = connect()
    cur.execute("SELECT COUNT(*) FROM sensor_readings")
    sr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM model_predictions")
    mp = cur.fetchone()[0]
    close(conn, cur)
    print(f"   ✅ Database: OK")
    print(f"   📊 Sensor Readings: {sr:,}")
    print(f"   📊 Predictions: {mp:,}")
except Exception as e:
    print(f"   ❌ Database: FAILED - {e}")

# Test 2: ML Models Loading
print("\n2️⃣  Testing ML Models...")
try:
    from model_inference import get_inference_engine
    engine = get_inference_engine()
    print(f"   ✅ ML Models: OK")
    print(f"   🤖 Fallback Mode: {engine.use_fallback}")
except Exception as e:
    print(f"   ❌ ML Models: FAILED - {e}")

# Test 3: Single Prediction
print("\n3️⃣  Testing Single Prediction...")
try:
    test_features = {
        'air_temperature': 25.0,
        'process_temperature': 35.0,
        'rotational_speed': 1500,
        'torque': 40.0,
        'tool_wear': 100
    }
    result = engine.predict_all(test_features)
    fp = result['failure_prediction']['failure_probability']
    rul = result['rul_prediction']['predicted_rul_days']
    anomaly = result['anomaly_detection']['is_anomaly']
    print(f"   ✅ Prediction: OK")
    print(f"   📈 Failure Probability: {fp:.1%}")
    print(f"   ⏱️  Predicted RUL: {rul:.1f} days")
    print(f"   🔍 Anomaly: {'YES ⚠️' if anomaly else 'NO ✅'}")
except Exception as e:
    print(f"   ❌ Prediction: FAILED - {e}")

# Test 4: Web Scraper Import
print("\n4️⃣  Testing Web Scraper...")
try:
    from web_scraper import MaintenanceDataScraper
    scraper = MaintenanceDataScraper()
    print(f"   ✅ Web Scraper: OK (ready to use)")
except Exception as e:
    print(f"   ❌ Web Scraper: FAILED - {e}")

# Test 5: FastAPI Check
print("\n5️⃣  Testing FastAPI Server...")
try:
    import requests
    response = requests.get("http://127.0.0.1:8000/docs", timeout=2)
    if response.status_code == 200:
        print(f"   ✅ FastAPI: Running on http://127.0.0.1:8000")
        print(f"   📚 Docs: http://127.0.0.1:8000/docs")
    else:
        print(f"   ⚠️  FastAPI: Status {response.status_code}")
except Exception as e:
    print(f"   ❌ FastAPI: Not running - {e}")

# Test 6: Streamlit Check
print("\n6️⃣  Testing Streamlit Dashboard...")
try:
    import requests
    response = requests.get("http://localhost:8501", timeout=2)
    if response.status_code == 200:
        print(f"   ✅ Streamlit: Running on http://localhost:8501")
    else:
        print(f"   ⚠️  Streamlit: Status {response.status_code}")
except Exception as e:
    print(f"   ❌ Streamlit: Not running - {e}")

print("\n" + "=" * 70)
print("✨ TEST COMPLETE")
print("=" * 70)
print("\n📝 Next Steps:")
print("   1. Open Streamlit: http://localhost:8501")
print("   2. Try 'Manual Prediction' mode to test custom inputs")
print("   3. Try 'Batch Predictions' mode to test multiple inputs")
print("   4. Check API docs: http://127.0.0.1:8000/docs")
print("\n🚀 System Ready!")
