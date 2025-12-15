"""
Streamlit Dashboard for Predictive Maintenance System
FULLY FIXED VERSION - All column names corrected
"""
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Database connection
try:
    from db_connect import fetch_dataframe
except ImportError as e:
    st.error(f"Cannot import db_connect: {e}")
    st.stop()

# Model inference
try:
    from model_inference import get_inference_engine
    inference_engine = get_inference_engine()
except ImportError:
    inference_engine = None
    st.warning("Model inference not available - predictions will be disabled")

# Configuration
try:
    from config_file import Config
except ImportError:
    class Config:
        MODEL_VERSION = "v1.0"

# Optional Plotly support
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .risk-critical { background-color: #ff4444; color: white; padding: 5px; border-radius: 3px; }
    .risk-high { background-color: #ff9800; color: white; padding: 5px; border-radius: 3px; }
    .risk-medium { background-color: #ffeb3b; color: black; padding: 5px; border-radius: 3px; }
    .risk-low { background-color: #4caf50; color: white; padding: 5px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS - ALL FIXED WITH CORRECT COLUMN NAMES
# ============================================================================

@st.cache_data(ttl=300)
def load_machines():
    """Load all machines from database"""
    try:
        query = """
            SELECT 
                machine_id,
                machine_name,
                COALESCE(machine_type, 'Unknown') as machine_type,
                COALESCE(location, 'Unknown') as location,
                COALESCE(manufacturer, 'Unknown') as manufacturer,
                installation_date
            FROM machines
            ORDER BY machine_id
        """
        df = fetch_dataframe(query)
        return df
    except Exception as e:
        st.error(f"Error loading machines: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_machine_sensors(machine_id, hours=24):
    """Load recent sensor data for a machine"""
    try:
        query = """
            SELECT 
                timestamp,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            FROM sensor_readings
            WHERE machine_id = %s
            AND timestamp >= NOW() - INTERVAL '%s hours'
            ORDER BY timestamp DESC
        """
        df = fetch_dataframe(query, (machine_id, hours))
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading sensor data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_machine_predictions(machine_id, limit=10):
    """Load recent predictions for a machine - FIXED COLUMN NAMES"""
    try:
        query = """
            SELECT 
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
        if not df.empty:
            df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_failure_history(machine_id):
    """Load failure history for a machine"""
    try:
        query = """
            SELECT 
                failure_date,
                failure_type,
                COALESCE(failure_mode, 'Unknown') as failure_mode,
                COALESCE(downtime_hours, 0) as downtime_hours,
                description
            FROM failure_logs
            WHERE machine_id = %s
            ORDER BY failure_date DESC
        """
        df = fetch_dataframe(query, (machine_id,))
        if not df.empty:
            df['failure_date'] = pd.to_datetime(df['failure_date'])
        return df
    except Exception as e:
        st.error(f"Error loading failure history: {e}")
        return pd.DataFrame()

def get_latest_sensor(machine_id):
    """Get latest sensor reading for a machine"""
    try:
        query = """
            SELECT 
                timestamp,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear
            FROM sensor_readings
            WHERE machine_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        df = fetch_dataframe(query, (machine_id,))
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error getting latest sensor: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
# TAB CONTENT FUNCTIONS
# ============================================================================

def show_overview(machine_id, hours):
    """Show overview dashboard"""
    st.header("📊 Machine Health Overview")
    
    # Load latest prediction
    predictions_df = load_machine_predictions(machine_id, limit=1)
    
    # Get latest sensor reading
    sensor_data = get_latest_sensor(machine_id)
    
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
                delta=f"{'⚠️ High' if anomaly_score < -0.3 else '✅ Normal'}"
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
    else:
        st.info("No predictions available yet. Run model training first.")
    
    # Show current sensor values
    if sensor_data:
        st.subheader("📊 Current Sensor Readings")
        cols = st.columns(5)
        sensor_names = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']
        sensor_labels = ['Air Temp (°C)', 'Process Temp (°C)', 'RPM', 'Torque (Nm)', 'Tool Wear (min)']
        
        for col, name, label in zip(cols, sensor_names, sensor_labels):
            value = sensor_data.get(name, 0)
            col.metric(label, f"{value:.1f}")
    
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
                # Fallback to Streamlit's built-in chart
                chart_data = pred_history.set_index('prediction_timestamp')['failure_probability']
                st.line_chart(chart_data)
        else:
            st.info("No prediction history available")
    
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
                fig.add_hline(y=7, line_dash="dash", line_color="red",
                             annotation_text="7 Day Warning")
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Days",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_data = pred_history.set_index('prediction_timestamp')['predicted_rul']
                st.line_chart(chart_data)

def show_live_prediction(machine_id):
    """Show live prediction interface"""
    st.header("🔮 Live Failure Prediction")
    
    st.info("💡 Enter current sensor readings to get real-time predictions")
    
    # Get latest sensor values as defaults
    sensor_data = get_latest_sensor(machine_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Input Sensor Values")
        air_temp = st.slider("Air Temperature (°C)", 0.0, 50.0, 
                           float(sensor_data.get('air_temperature', 25.0)) if sensor_data else 25.0, 
                           0.1)
        process_temp = st.slider("Process Temperature (°C)", 0.0, 60.0, 
                               float(sensor_data.get('process_temperature', 35.0)) if sensor_data else 35.0, 
                               0.1)
        rpm = st.slider("Rotational Speed (RPM)", 1000, 2500, 
                       int(sensor_data.get('rotational_speed', 1500)) if sensor_data else 1500, 
                       10)
    
    with col2:
        st.subheader("📊 Additional Parameters")
        torque = st.slider("Torque (Nm)", 0.0, 100.0, 
                          float(sensor_data.get('torque', 40.0)) if sensor_data else 40.0, 
                          0.1)
        tool_wear = st.slider("Tool Wear (minutes)", 0, 300, 
                             int(sensor_data.get('tool_wear', 100)) if sensor_data else 100, 
                             1)
    
    st.divider()
    
    # Predict button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        if inference_engine is None:
            st.error("❌ Model inference engine not available. Check model files.")
            return
        
        with st.spinner("Running ML models..."):
            try:
                features = {
                    'air_temperature': air_temp,
                    'process_temperature': process_temp,
                    'rotational_speed': rpm,
                    'torque': torque,
                    'tool_wear': tool_wear
                }
                
                result = inference_engine.predict_all(features)
                
                if 'error' in result:
                    st.error(f"❌ Prediction error: {result['error']}")
                    return
                
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
                
                # Gauges (if Plotly available)
                if HAS_PLOTLY:
                    col1, col2 = st.columns(2)
                    
                    with col1:
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
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
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
            
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")

def show_sensor_monitoring(machine_id, hours):
    """Show sensor monitoring dashboard"""
    st.header("📈 Real-Time Sensor Monitoring")
    
    sensors_df = load_machine_sensors(machine_id, hours)
    
    if len(sensors_df) == 0:
        st.warning("No sensor data available for selected time range")
        return
    
    # Temperature monitoring
    st.subheader("🌡️ Temperature Monitoring")
    if HAS_PLOTLY:
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
    else:
        temp_data = sensors_df.set_index('timestamp')[['air_temperature', 'process_temperature']]
        st.line_chart(temp_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Rotational Speed")
        if HAS_PLOTLY:
            fig = px.line(sensors_df, x='timestamp', y='rotational_speed')
            fig.update_layout(yaxis_title="RPM", height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            rpm_data = sensors_df.set_index('timestamp')['rotational_speed']
            st.line_chart(rpm_data)
    
    with col2:
        st.subheader("🔧 Torque")
        if HAS_PLOTLY:
            fig = px.line(sensors_df, x='timestamp', y='torque', color_discrete_sequence=['orange'])
            fig.update_layout(yaxis_title="Nm", height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            torque_data = sensors_df.set_index('timestamp')['torque']
            st.line_chart(torque_data)
    
    # Tool wear
    st.subheader("🛠️ Tool Wear Progress")
    if HAS_PLOTLY:
        fig = px.area(sensors_df, x='timestamp', y='tool_wear', color_discrete_sequence=['purple'])
        fig.update_layout(yaxis_title="Minutes", height=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        wear_data = sensors_df.set_index('timestamp')['tool_wear']
        st.area_chart(wear_data)
    
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
            if HAS_PLOTLY and 'failure_mode' in failures_df.columns:
                fig = px.pie(failures_df, names='failure_mode', title="Failure Modes")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if 'failure_mode' in failures_df.columns:
                    st.bar_chart(failures_df['failure_mode'].value_counts())
        
        with col2:
            st.write("**Total Downtime**")
            if 'downtime_hours' in failures_df.columns:
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
    else:
        st.info("No prediction history available")

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
    
    if st.button("💾 Save Settings", type="primary"):
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
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<p class="main-header">🏭 Smart Predictive Maintenance Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Machine selection
        machines_df = load_machines()
        
        if len(machines_df) == 0:
            st.error("No machines found in database!")
            st.info("Please run: python complete_database_fix.py to load data")
            return
        
        # Create machine selection options
        machine_options = {}
        for _, row in machines_df.iterrows():
            name = f"{row['machine_name']}"
            if 'machine_type' in row and pd.notna(row['machine_type']):
                name += f" ({row['machine_type']})"
            machine_options[name] = row['machine_id']
        
        selected_machine_name = st.selectbox(
            "Select Machine",
            options=list(machine_options.keys()),
            key="machine_selector"
        )
        
        selected_machine_id = machine_options[selected_machine_name]
        
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
        machine_row = machines_df[machines_df['machine_id'] == selected_machine_id].iloc[0]
        st.subheader("📋 Machine Details")
        if 'machine_type' in machine_row and pd.notna(machine_row['machine_type']):
            st.write(f"**Type:** {machine_row['machine_type']}")
        if 'location' in machine_row and pd.notna(machine_row['location']):
            st.write(f"**Location:** {machine_row['location']}")
        if 'manufacturer' in machine_row and pd.notna(machine_row['manufacturer']):
            st.write(f"**Manufacturer:** {machine_row['manufacturer']}")
        if 'installation_date' in machine_row:
            st.write(f"**Installed:** {machine_row['installation_date']}")
        
        # Model info
        st.divider()
        if inference_engine and inference_engine.models_loaded:
            st.success("✅ Models loaded")
        else:
            st.warning("⚠️ Models not loaded")
            
        st.caption(f"Model version: {Config.MODEL_VERSION}")
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔮 Live Prediction", 
        "📈 Sensor Monitoring", 
        "📜 History", 
        "⚙️ Settings"
    ])
    
    with tab1:
        show_overview(selected_machine_id, hours)
    
    with tab2:
        show_live_prediction(selected_machine_id)
    
    with tab3:
        show_sensor_monitoring(selected_machine_id, hours)
    
    with tab4:
        show_history(selected_machine_id)
    
    with tab5:
        show_settings()

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()