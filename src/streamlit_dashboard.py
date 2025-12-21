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

# Sidebar: Mode selection
st.sidebar.header("🎯 Mode")
mode = st.sidebar.radio(
    "Select Mode:",
    ["Machine Dashboard", "Manual Prediction", "Batch Predictions"],
    index=0
)

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


# Load inference engine (shared across modes)
try:
    engine = get_inference_engine()
    models_ok = True
except Exception as e:
    st.sidebar.warning(f"⚠️ Models: {e}")
    models_ok = False

# ============================================================================
# MODE 1: MACHINE DASHBOARD (Original)
# ============================================================================
if mode == "Machine Dashboard":
    st.sidebar.header("Selection")
    machines = load_machines()

    if machines.empty:
        st.warning("No machines in database. Run ETL first.")
        st.stop()

    machine_map = {row.machine_name: int(row.machine_id) for row in machines.itertuples(index=False)}
    selected_name = st.sidebar.selectbox("Machine", list(machine_map.keys()))
    machine_id = machine_map[selected_name]

        st.header(f"Machine: {selected_name}")

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

# ============================================================================
# MODE 2: MANUAL PREDICTION
# ============================================================================
elif mode == "Manual Prediction":
    st.header("🎯 Manual Prediction")
    st.write("Enter sensor values manually to get predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Sensor Values")
        air_temp = st.number_input("Air Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        process_temp = st.number_input("Process Temperature (°C)", min_value=0.0, max_value=150.0, value=35.0, step=0.1)
        rpm = st.number_input("Rotational Speed (RPM)", min_value=0, max_value=10000, value=1500, step=10)
    
    with col2:
        st.subheader("More Parameters")
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.1)
        tool_wear = st.number_input("Tool Wear (minutes)", min_value=0, max_value=500, value=100, step=1)
    
    if st.button("🔮 Predict", type="primary"):
        if not models_ok:
            st.error("❌ Models not available. Please check model files.")
        else:
            try:
                features = {
                    'air_temperature': float(air_temp),
                    'process_temperature': float(process_temp),
                    'rotational_speed': float(rpm),
                    'torque': float(torque),
                    'tool_wear': float(tool_wear)
                }
                
                with st.spinner("Running predictions..."):
                    result = engine.predict_all(features)
                
                st.success("✅ Prediction Complete!")
                
                # Display results
                st.subheader("📊 Results")
                c1, c2, c3 = st.columns(3)
                
                fp = result['failure_prediction']['failure_probability']
                rul = result['rul_prediction']['predicted_rul_days']
                anomaly = result['anomaly_detection']['is_anomaly']
                risk = result.get('risk_level', 'UNKNOWN')
                
                c1.metric("Failure Probability", f"{fp:.1%}", 
                         delta="High Risk" if fp > 0.5 else "Low Risk",
                         delta_color="inverse")
                c2.metric("Predicted RUL", f"{rul:.1f} days")
                c3.metric("Status", "⚠️ ANOMALY" if anomaly else "✅ NORMAL")
                
                # Risk indicator
                st.subheader("🎯 Risk Assessment")
                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🔴"}
                st.markdown(f"### {risk_colors.get(risk, '⚪')} Risk Level: **{risk}**")
                
                # Recommendations
                if result.get('recommendations'):
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(result['recommendations'], 1):
                        st.write(f"{i}. {rec}")
                
                # Raw prediction data
                with st.expander("🔍 View Raw Prediction Data"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.exception(e)

# ============================================================================
# MODE 3: BATCH PREDICTIONS
# ============================================================================
elif mode == "Batch Predictions":
    st.header("📊 Batch Predictions")
    st.write("Upload CSV file or enter multiple rows to predict")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Manual Entry"])
    
    with tab1:
        st.subheader("Upload CSV File")
        st.write("CSV should have columns: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head(), use_container_width=True)
                
                # Validate columns
                required_cols = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🔮 Run Batch Predictions", type="primary"):
                        if not models_ok:
                            st.error("❌ Models not available")
                        else:
                            progress_bar = st.progress(0)
                            results = []
                            
                            for idx, row in df.iterrows():
                                try:
                                    features = {
                                        'air_temperature': float(row['air_temperature']),
                                        'process_temperature': float(row['process_temperature']),
                                        'rotational_speed': float(row['rotational_speed']),
                                        'torque': float(row['torque']),
                                        'tool_wear': float(row['tool_wear'])
                                    }
                                    result = engine.predict_all(features)
                                    results.append({
                                        'row': idx + 1,
                                        'failure_probability': result['failure_prediction']['failure_probability'],
                                        'predicted_rul_days': result['rul_prediction']['predicted_rul_days'],
                                        'is_anomaly': result['anomaly_detection']['is_anomaly'],
                                        'risk_level': result.get('risk_level', 'UNKNOWN')
                                    })
                                except Exception as e:
                                    results.append({
                                        'row': idx + 1,
                                        'error': str(e)
                                    })
                                progress_bar.progress((idx + 1) / len(df))
                            
                            st.success(f"✅ Processed {len(results)} predictions")
                            
                            # Display results
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.subheader("📈 Summary Statistics")
                            if 'failure_probability' in results_df.columns:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Average Failure Prob", f"{results_df['failure_probability'].mean():.1%}")
                                c2.metric("Average RUL", f"{results_df['predicted_rul_days'].mean():.1f} days")
                                c3.metric("Anomalies", f"{results_df['is_anomaly'].sum()} / {len(results_df)}")
                            
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with tab2:
        st.subheader("Manual Data Entry")
        st.write("Enter multiple rows of data separated by rows")
        
        num_rows = st.number_input("Number of rows", min_value=1, max_value=100, value=5)
        
        # Create empty dataframe for editing
        default_data = pd.DataFrame({
            'air_temperature': [25.0] * num_rows,
            'process_temperature': [35.0] * num_rows,
            'rotational_speed': [1500] * num_rows,
            'torque': [40.0] * num_rows,
            'tool_wear': [100] * num_rows
        })
        
        edited_df = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
        
        if st.button("🔮 Predict All Rows", type="primary"):
            if not models_ok:
                st.error("❌ Models not available")
            else:
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in edited_df.iterrows():
                    try:
                        features = {
                            'air_temperature': float(row['air_temperature']),
                            'process_temperature': float(row['process_temperature']),
                            'rotational_speed': float(row['rotational_speed']),
                            'torque': float(row['torque']),
                            'tool_wear': float(row['tool_wear'])
                        }
                        result = engine.predict_all(features)
                        results.append({
                            'row': idx + 1,
                            'failure_probability': result['failure_prediction']['failure_probability'],
                            'predicted_rul_days': result['rul_prediction']['predicted_rul_days'],
                            'is_anomaly': result['anomaly_detection']['is_anomaly'],
                            'risk_level': result.get('risk_level', 'UNKNOWN')
                        })
                    except Exception as e:
                        results.append({'row': idx + 1, 'error': str(e)})
                    progress_bar.progress((idx + 1) / len(edited_df))
                
                st.success(f"✅ Completed {len(results)} predictions")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"manual_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


st.sidebar.markdown("---")
st.sidebar.info(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
