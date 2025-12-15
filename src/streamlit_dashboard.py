import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np

from db_connect import fetch_dataframe
from model_inference import get_inference_engine
from config_file import Config

# Optional Plotly support: use if installed, otherwise fall back to streamlit charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def load_machines():
    try:
        df = fetch_dataframe("SELECT machine_id, machine_name FROM machines ORDER BY machine_id")
        return df
    except Exception:
        return pd.DataFrame(columns=["machine_id", "machine_name"])


def get_latest_sensor(machine_id: int):
    q = (
        "SELECT timestamp, air_temperature, process_temperature, rotational_speed, torque, tool_wear "
        "FROM sensor_readings WHERE machine_id=%s ORDER BY timestamp DESC LIMIT 1"
    )
    try:
        df = fetch_dataframe(q, (machine_id,))
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception:
        return None


def main():
    st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
    st.title("Predictive Maintenance — Dashboard")

    # Sidebar: select machine
    st.sidebar.header("Selection")
    machines = load_machines()
    if machines.empty:
        st.sidebar.warning("No machines found in database")
        st.info("Run ETL/init_db first to populate `machines` table.")
        return

    machine_map = {row.machine_name: int(row.machine_id) for row in machines.itertuples(index=False)}
    machine_name = st.sidebar.selectbox("Machine", list(machine_map.keys()))
    machine_id = machine_map[machine_name]

    # Load inference engine
    try:
        engine = get_inference_engine()
    except Exception as e:
        st.error(f"Inference engine unavailable: {e}")
        engine = None

    st.header(f"Machine: {machine_name} (id={machine_id})")

    # Latest sensor readings
    st.subheader("Latest Sensor Reading")
    sensor = get_latest_sensor(machine_id)
    if sensor is None:
        st.info("No sensor readings available for this machine.")
    else:
        sensor_ts = sensor.get('timestamp')
        st.write(f"Timestamp: {sensor_ts}")
        cols = st.columns(5)
        names = ['air_temperature','process_temperature','rotational_speed','torque','tool_wear']
        for c, name in zip(cols, names):
            c.metric(label=name, value=sensor.get(name))

    # Run prediction
    st.subheader("Predictions")
    if engine is None:
        st.info("Models not loaded — prediction disabled.")
    else:
        if sensor is None:
            st.info("Not enough data to predict for this machine.")
        else:
            features = {
                'air_temperature': float(sensor['air_temperature']),
                'process_temperature': float(sensor['process_temperature']),
                'rotational_speed': float(sensor['rotational_speed']),
                'torque': float(sensor['torque']),
                'tool_wear': float(sensor['tool_wear'])
            }
            try:
                result = engine.predict_all(features)
                # Display key metrics
                fp = result['failure_prediction']['failure_probability']
                rul = result['rul_prediction']['predicted_rul_days']
                anomaly = result['anomaly_detection']['is_anomaly']
                risk = result.get('risk_level', 'UNKNOWN')

                st.metric("Failure probability", f"{fp:.2%}")
                st.metric("Predicted RUL (days)", f"{rul:.2f}")
                st.metric("Anomaly", "YES" if anomaly else "NO")
                st.write("**Risk level:**", risk)

                st.subheader("Recommendations")
                for rec in result.get('recommendations', [])[:10]:
                    st.write(f"- {rec}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model version: {Config.MODEL_VERSION}")
    st.sidebar.caption(f"Updated: {datetime.now().isoformat()}")


def run():
    main()


if __name__ == '__main__':
    main()
    
def load_failure_history(machine_id):
    """Load failure history for a machine"""
    query = """
        SELECT 
            failure_date,
            failure_type,
            failure_mode,
            downtime_hours,
            description
        FROM failure_logs
        WHERE machine_id = %s
        ORDER BY failure_date DESC
    """
    return fetch_dataframe(query, (machine_id,))

def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        'CRITICAL': '#ff4444',
        'HIGH': '#ff9800',
        'MEDIUM': '#ffeb3b',
        'LOW': '#4caf50'
    }
    return colors.get(risk_level, '#999999')

def display_risk_badge(risk_level):
    """Display risk level badge"""
    color = get_risk_color(risk_level)
    st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 10px; 
                    border-radius: 5px; text-align: center; font-weight: bold;">
            🚨 RISK LEVEL: {risk_level}
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<p class="main-header">🏭 Smart Predictive Maintenance Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100.png?text=Company+Logo", 
                 use_column_width=True)
        
        st.title("🎛️ Control Panel")
        
        # Machine selection
        machines_df = load_machines()
        
        if len(machines_df) == 0:
            st.error("No machines found in database!")
            return
        
        machine_options = {
            f"{row['machine_name']} ({row['machine_type']})": row['machine_id']
            for _, row in machines_df.iterrows()
        }
        
        selected_machine_name = st.selectbox(
            "Select Machine",
            options=list(machine_options.keys()),
            key="machine_selector"
        )
        
        selected_machine_id = machine_options[selected_machine_name]
        st.session_state.selected_machine = selected_machine_id
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            options=["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=0
        )
        
        hours_map = {
            "Last 24 Hours": 24,
            "Last 7 Days": 168,
            "Last 30 Days": 720
        }
        hours = hours_map[time_range]
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Machine info
        machine_info = machines_df[machines_df['machine_id'] == selected_machine_id].iloc[0]
        st.subheader("📋 Machine Details")
        st.write(f"**Type:** {machine_info['machine_type']}")
        st.write(f"**Location:** {machine_info['location']}")
        st.write(f"**Manufacturer:** {machine_info['manufacturer']}")
        st.write(f"**Installed:** {machine_info['installation_date']}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔮 Live Prediction", 
        "📈 Sensor Monitoring", 
        "📜 History", 
        "⚙️ Settings"
    ])
    
    # TAB 1: Overview
    with tab1:
        show_overview(selected_machine_id, hours)
    
    # TAB 2: Live Prediction
    with tab2:
        show_live_prediction(selected_machine_id)
    
    # TAB 3: Sensor Monitoring
    with tab3:
        show_sensor_monitoring(selected_machine_id, hours)
    
    # TAB 4: History
    with tab4:
        show_history(selected_machine_id)
    
    # TAB 5: Settings
    with tab5:
        show_settings()

# ============================================================================
# TAB CONTENT FUNCTIONS
# ============================================================================

def show_overview(machine_id, hours):
    """Show overview dashboard"""
    st.header("📊 Machine Health Overview")
    
    # Load latest prediction
    predictions_df = load_machine_predictions(machine_id, limit=1)
    
    if len(predictions_df) > 0:
        latest_pred = predictions_df.iloc[0]
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            failure_prob = latest_pred['failure_probability']
            st.metric(
                "Failure Probability",
                f"{failure_prob:.1%}",
                delta=f"{'⚠️ High' if failure_prob > 0.5 else '✅ Normal'}"
            )
        
        with col2:
            rul = latest_pred['predicted_rul']
            st.metric(
                "Remaining Useful Life",
                f"{rul:.1f} days",
                delta=f"{'⚠️ Low' if rul < 7 else '✅ Good'}"
            )
        
        with col3:
            anomaly_score = latest_pred['anomaly_score']
            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.3f}",
                delta=f"{'⚠️ Anomaly' if anomaly_score < -0.3 else '✅ Normal'}"
            )
        
        with col4:
            # Calculate health score
            health_score = (1 - failure_prob) * 100
            st.metric(
                "Health Score",
                f"{health_score:.0f}%",
                delta=f"{'🔴 Poor' if health_score < 50 else '🟢 Good'}"
            )
        
        st.divider()
        
        # Risk Level Display
        risk_level = "CRITICAL" if failure_prob > 0.7 or rul < 3 else \
                     "HIGH" if failure_prob > 0.5 or rul < 7 else \
                     "MEDIUM" if failure_prob > 0.3 or rul < 14 else "LOW"
        
        display_risk_badge(risk_level)
        
        st.divider()
    
    # Recent prediction trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Failure Probability Trend")
        pred_history = load_machine_predictions(machine_id, limit=50)

        if len(pred_history) > 0:
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_history['prediction_timestamp'],
                    y=pred_history['failure_probability'],
                    mode='lines+markers',
                    name='Failure Probability',
                    line=dict(color='red', width=2)
                ))
                # Add threshold line
                fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                             annotation_text="Critical Threshold")
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: simple line chart
                ph = pred_history.copy()
                ph['prediction_timestamp'] = pd.to_datetime(ph['prediction_timestamp'])
                ph = ph.set_index('prediction_timestamp')
                st.line_chart(ph['failure_probability'])
    
    with col2:
        st.subheader("⏰ RUL Trend")

        if len(pred_history) > 0:
            if HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pred_history['prediction_timestamp'],
                    y=pred_history['predicted_rul'],
                    mode='lines+markers',
                    name='RUL',
                    line=dict(color='blue', width=2)
                ))
                # Add warning line
                fig.add_hline(y=7, line_dash="dash", line_color="red",
                             annotation_text="7 Day Warning")
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Days",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                ph = pred_history.copy()
                ph['prediction_timestamp'] = pd.to_datetime(ph['prediction_timestamp'])
                ph = ph.set_index('prediction_timestamp')
                st.line_chart(ph['predicted_rul'])

def show_live_prediction(machine_id):
    """Show live prediction interface"""
    st.header("🔮 Live Failure Prediction")
    
    st.info("💡 Enter current sensor readings to get real-time predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Input Sensor Values")
        
        air_temp = st.slider("Air Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
        process_temp = st.slider("Process Temperature (°C)", 0.0, 60.0, 35.0, 0.1)
        rpm = st.slider("Rotational Speed (RPM)", 1000, 2500, 1500, 10)
    
    with col2:
        st.subheader("📊 Additional Parameters")
        
        torque = st.slider("Torque (Nm)", 0.0, 100.0, 40.0, 0.1)
        tool_wear = st.slider("Tool Wear (minutes)", 0, 300, 100, 1)
    
    st.divider()
    
    # Predict button
    if st.button("🎯 Generate Prediction", use_container_width=True):
        with st.spinner("Running ML models..."):
            try:
                # Prepare features
                features = {
                    'air_temperature': air_temp,
                    'process_temperature': process_temp,
                    'rotational_speed': rpm,
                    'torque': torque,
                    'tool_wear': tool_wear
                }
                
                # Get prediction
                if inference_engine and inference_engine.models_loaded:
                    result = inference_engine.predict_all(features)
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Risk level
                    risk_level = result['risk_level']
                    display_risk_badge(risk_level)
                    
                    st.divider()
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        failure_prob = result['failure_prediction']['failure_probability']
                        st.metric("Failure Probability", f"{failure_prob:.2%}")
                    
                    with col2:
                        rul = result['rul_prediction']['predicted_rul_days']
                        st.metric("Predicted RUL", f"{rul:.1f} days")
                    
                    with col3:
                        is_anomaly = result['anomaly_detection']['is_anomaly']
                        st.metric("Anomaly", "YES ⚠️" if is_anomaly else "NO ✅")
                    
                    st.divider()
                    
                    # Recommendations
                    st.subheader("📋 Recommendations")
                    for rec in result['recommendations']:
                        st.write(f"• {rec}")
                    
                    # Gauges
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Failure probability gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=failure_prob * 100,
                            title={'text': "Failure Risk %"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 60], 'color': "yellow"},
                                    {'range': [60, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # RUL gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=rul,
                            title={'text': "RUL (days)"},
                            gauge={
                                'axis': {'range': [0, 30]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 7], 'color': "red"},
                                    {'range': [7, 14], 'color': "yellow"},
                                    {'range': [14, 30], 'color': "lightgreen"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("❌ Models not loaded. Please check model files.")
            
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")

def show_sensor_monitoring(machine_id, hours):
    """Show sensor monitoring dashboard"""
    st.header("📈 Real-Time Sensor Monitoring")
    
    # Load sensor data
    sensors_df = load_machine_sensors(machine_id, hours)
    
    if len(sensors_df) == 0:
        st.warning("No sensor data available for selected time range")
        return
    
    # Time series plots
    st.subheader("🌡️ Temperature Monitoring")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensors_df['timestamp'],
        y=sensors_df['air_temperature'],
        name='Air Temperature',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=sensors_df['timestamp'],
        y=sensors_df['process_temperature'],
        name='Process Temperature',
        line=dict(color='red')
    ))
    fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)", height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Rotational Speed")
        fig = px.line(sensors_df, x='timestamp', y='rotational_speed')
        fig.update_layout(yaxis_title="RPM", height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔧 Torque")
        fig = px.line(sensors_df, x='timestamp', y='torque', color_discrete_sequence=['orange'])
        fig.update_layout(yaxis_title="Nm", height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tool wear
    st.subheader("🛠️ Tool Wear Progress")
    fig = px.area(sensors_df, x='timestamp', y='tool_wear', color_discrete_sequence=['purple'])
    fig.update_layout(yaxis_title="Minutes", height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Statistical Summary")
    st.dataframe(sensors_df.describe(), use_container_width=True)

def show_history(machine_id):
    """Show historical data and failures"""
    st.header("📜 Machine History")
    
    # Failure history
    st.subheader("🔴 Failure History")
    failures_df = load_failure_history(machine_id)
    
    if len(failures_df) > 0:
        st.dataframe(failures_df, use_container_width=True)
        
        # Failure analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Failure Types Distribution**")
            fig = px.pie(failures_df, names='failure_mode', title="Failure Modes")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Total Downtime**")
            total_downtime = failures_df['downtime_hours'].sum()
            avg_downtime = failures_df['downtime_hours'].mean()
            st.metric("Total Downtime", f"{total_downtime:.1f} hours")
            st.metric("Average Downtime", f"{avg_downtime:.1f} hours")
    else:
        st.info("No failure history recorded for this machine")
    
    # Prediction history
    st.subheader("🔮 Prediction History")
    pred_history = load_machine_predictions(machine_id, limit=100)
    
    if len(pred_history) > 0:
        st.dataframe(pred_history, use_container_width=True)

def show_settings():
    """Show settings and configuration"""
    st.header("⚙️ Dashboard Settings")
    
    st.subheader("🔔 Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        failure_threshold = st.slider(
            "Failure Probability Threshold (%)",
            0, 100, 70, 5
        )
        
        rul_threshold = st.slider(
            "RUL Alert Threshold (days)",
            1, 30, 7, 1
        )
    
    with col2:
        email_alerts = st.checkbox("Enable Email Alerts", value=True)
        telegram_alerts = st.checkbox("Enable Telegram Alerts", value=False)
    
    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")
    
    st.divider()
    
    st.subheader("📊 About This Dashboard")
    st.write("""
    **Smart Predictive Maintenance System v1.0**
    
    This dashboard provides real-time monitoring and predictive analytics for industrial machines.
    
    **Features:**
    - Real-time sensor monitoring
    - ML-powered failure prediction
    - Remaining Useful Life estimation
    - Anomaly detection
    - Automated alerting
    
    **Models:**
    - Classification: Random Forest / XGBoost
    - Regression: Random Forest / XGBoost
    - Anomaly Detection: Isolation Forest
    """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
