"""
Streamlit Dashboard for Predictive Maintenance
Simple, clean, working implementation
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from db_connect import fetch_dataframe
from model_inference import get_inference_engine
from config_file import Config

# Optional Plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🏭 Predictive Maintenance Dashboard")


@st.cache_data(ttl=300)
def load_machines():
    try:
        return fetch_dataframe("SELECT machine_id, machine_name FROM machines ORDER BY machine_id")
    except Exception:
        return pd.DataFrame(columns=["machine_id", "machine_name"])


@st.cache_data(ttl=60)
def get_latest_sensor(machine_id):
    try:
        q = """
            SELECT timestamp, air_temperature, process_temperature, 
                   rotational_speed, torque, tool_wear 
            FROM sensor_readings WHERE machine_id=%s 
            ORDER BY timestamp DESC LIMIT 1
        """
        df = fetch_dataframe(q, (machine_id,))
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_prediction_history(machine_id, limit=30):
    try:
        q = """
            SELECT prediction_timestamp, failure_probability, predicted_rul 
            FROM model_predictions WHERE machine_id=%s 
            ORDER BY prediction_timestamp DESC LIMIT %s
        """
        df = fetch_dataframe(q, (machine_id, limit))
        return df.sort_values('prediction_timestamp') if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# Sidebar: Machine selection
st.sidebar.header("Selection")
machines = load_machines()

if machines.empty:
    st.warning("No machines in database. Run ETL first.")
    st.stop()

machine_map = {row.machine_name: int(row.machine_id) for row in machines.itertuples(index=False)}
selected_name = st.sidebar.selectbox("Machine", list(machine_map.keys()))
machine_id = machine_map[selected_name]

st.header(f"Machine: {selected_name}")

# Load inference engine
try:
    engine = get_inference_engine()
    models_ok = True
except Exception as e:
    st.warning(f"Models unavailable: {e}")
    models_ok = False

# Latest readings
st.subheader("Latest Sensor Reading")
sensor = get_latest_sensor(machine_id)

if sensor is None:
    st.info("No sensor data available.")
else:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Air Temp (°C)", f"{sensor.get('air_temperature', 0):.1f}")
    c2.metric("Process Temp (°C)", f"{sensor.get('process_temperature', 0):.1f}")
    c3.metric("RPM", f"{sensor.get('rotational_speed', 0):.0f}")
    c4.metric("Torque (Nm)", f"{sensor.get('torque', 0):.1f}")
    c5.metric("Tool Wear (min)", f"{sensor.get('tool_wear', 0):.0f}")

    # Run prediction
    if models_ok:
        st.subheader("Predictions")
        try:
            features = {
                'air_temperature': float(sensor['air_temperature']),
                'process_temperature': float(sensor['process_temperature']),
                'rotational_speed': float(sensor['rotational_speed']),
                'torque': float(sensor['torque']),
                'tool_wear': float(sensor['tool_wear'])
            }
            result = engine.predict_all(features)
            
            fp = result['failure_prediction']['failure_probability']
            rul = result['rul_prediction']['predicted_rul_days']
            anomaly = result['anomaly_detection']['is_anomaly']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Failure Probability", f"{fp:.1%}")
            c2.metric("Predicted RUL (days)", f"{rul:.1f}")
            c3.metric("Anomaly", "⚠️ YES" if anomaly else "✅ NO")
            
            if result.get('recommendations'):
                st.subheader("Recommendations")
                for rec in result['recommendations'][:5]:
                    st.write(f"• {rec}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Prediction history
st.subheader("Prediction History")
history = get_prediction_history(machine_id, limit=50)

if not history.empty:
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['prediction_timestamp'],
            y=history['failure_probability'],
            mode='lines+markers',
            name='Failure Probability'
        ))
        fig.update_layout(title="Failure Probability Trend", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_df = history.set_index('prediction_timestamp')
        st.line_chart(chart_df['failure_probability'])
else:
    st.info("No prediction history available.")

st.sidebar.markdown("---")
st.sidebar.info(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
