# app.py
# -------------------------------------------------------------
# Crusher Predictive Maintenance Dashboard (Streamlit + Plotly)
# -------------------------------------------------------------
# Features:
# - Loads your CSV (datetime_stamp, Machine_Id, Plant_Id, KPIs, Anomaly_Flag)
# - Sidebar filters (date range, plant, machine, KPI selector)
# - KPI tiles (anomalies today, affected plants/machines, throughput delta)
# - Time trends with anomaly markers + rolling avg
# - Daily anomaly count trend
# - Machine x Day anomaly heatmap
# - Avg KPI per machine
# - Pareto chart (machines by anomaly count)
# - Scatter (KPI vs KPI) & KPI boxplots with anomaly highlighting
# - Alert log table with conditional formatting and CSV download

import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Crusher Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-color: #f9fafb;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--primary-color);
    }
    
    /* KPI tiles styling */
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #4b5563;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Progress bars */
    .progress-bar {
        height: 8px;
        background-color: #374151;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .progress-green { background-color: var(--success-color); }
    .progress-yellow { background-color: var(--warning-color); }
    .progress-red { background-color: var(--danger-color); }
    
    /* Status indicators */
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-green { background-color: var(--success-color); }
    .status-yellow { background-color: var(--warning-color); }
    .status-red { background-color: var(--danger-color); }
    .status-gray { background-color: #6b7280; }
    
    /* Section headers */
    .section-header {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }
    
    /* Alert styling */
    .alert-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid var(--danger-color);
        margin-bottom: 1rem;
    }
    
    /* Equipment cards */
    .equipment-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #4b5563;
        margin-bottom: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .equipment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Helpers
# --------------------------
KPI_COLUMNS = [
    "Vibration_Level",
    "Motor_Temperature",
    "Oil_Pressure",
    "Power_Consumption",
    "Throughput",
    "Bearing_Wear_Index",
]

CATEGORICALS = ["Machine_Id", "Plant_Id"]

def _load_default_sample():
    """
    Fall-back: small synthetic sample if no file provided.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-08-01", "2025-08-31 23:00:00", freq="H")
    df = pd.DataFrame({
        "datetime_stamp": rng.choice(dates, size=1000, replace=True),
        "Machine_Id": rng.choice([f"CR-{i:03d}" for i in range(101, 151)], size=1000),
        "Plant_Id": rng.choice([f"PL-{i:02d}" for i in range(1, 6)], size=1000),
        "Vibration_Level": rng.normal(4, 1, 1000).round(2),
        "Motor_Temperature": rng.normal(70, 5, 1000).round(2),
        "Oil_Pressure": rng.normal(12, 1, 1000).round(2),
        "Power_Consumption": rng.normal(450, 30, 1000).round(2),
        "Throughput": rng.normal(300, 20, 1000).round(2),
        "Bearing_Wear_Index": rng.normal(18, 5, 1000).round(2),
    })
    # 2% anomalies
    n_anom = max(1, int(0.02 * len(df)))
    anom_idx = rng.choice(df.index, size=n_anom, replace=False)
    df.loc[anom_idx, ["Vibration_Level"]] = rng.uniform(10, 20, n_anom).round(2)
    df.loc[anom_idx, ["Motor_Temperature"]] = rng.uniform(90, 120, n_anom).round(2)
    df.loc[anom_idx, ["Oil_Pressure"]] = rng.uniform(2, 6, n_anom).round(2)
    df.loc[anom_idx, ["Power_Consumption"]] = rng.uniform(600, 800, n_anom).round(2)
    df.loc[anom_idx, ["Throughput"]] = rng.uniform(100, 150, n_anom).round(2)
    df.loc[anom_idx, ["Bearing_Wear_Index"]] = rng.uniform(50, 100, n_anom).round(2)
    df["Anomaly_Flag"] = 0
    df.loc[anom_idx, "Anomaly_Flag"] = 1
    return df

@st.cache_data(show_spinner=False)
def load_data(file_buffer=None):
    # Priority 1: user upload
    if file_buffer is not None:
        if file_buffer.name.endswith('.xlsx'):
            df = pd.read_excel(file_buffer)
        else:
            df = pd.read_csv(file_buffer)
    else:
        # Priority 2: local CSV file
        csv_path = "crusher_sensor_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Priority 3: fallback CSV file if user dropped another CSV in the same folder
            default_path = "crusher_sensor_data_with_datetime.csv"
            if os.path.exists(default_path):
                df = pd.read_csv(default_path)
            else:
                df = _load_default_sample()

    # Basic normalization
    # tolerate both 'datetime_stamp' and 'timestamp' names
    if "datetime_stamp" not in df.columns:
        # try common alternatives
        for alt in ["timestamp", "time", "Datetime", "DateTime"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime_stamp"})
                break

    # Ensure required cols
    required = set(["datetime_stamp", "Machine_Id", "Plant_Id", "Anomaly_Flag"] + KPI_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        st.warning(f"Missing columns in data: {', '.join(missing)}")
        # try to continue if possible

    # Parse datetime
    if "datetime_stamp" in df.columns:
        df["datetime_stamp"] = pd.to_datetime(df["datetime_stamp"], errors="coerce")
        df = df.dropna(subset=["datetime_stamp"])

    # Coerce KPIs to numeric
    for k in KPI_COLUMNS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "Anomaly_Flag" in df.columns:
        df["Anomaly_Flag"] = pd.to_numeric(df["Anomaly_Flag"], errors="coerce").fillna(0).astype(int)

    return df.sort_values("datetime_stamp")

def enhanced_kpi_tile(label, value, target=None, delta=None, delta_color="normal", col=None, status="normal"):
    """Enhanced KPI tile with progress bar and status indicator"""
    c = col if col is not None else st
    
    # Status dot
    status_colors = {
        "normal": "status-green",
        "warning": "status-yellow", 
        "critical": "status-red",
        "idle": "status-gray"
    }
    
    # Calculate progress percentage if target is provided
    progress_pct = 0
    progress_color = "progress-green"
    if target is not None and isinstance(value, (int, float)) and isinstance(target, (int, float)):
        progress_pct = min(100, (value / target) * 100)
        if progress_pct >= 90:
            progress_color = "progress-green"
        elif progress_pct >= 70:
            progress_color = "progress-yellow"
        else:
            progress_color = "progress-red"
    
    # Create custom HTML for enhanced tile
    tile_html = f"""
    <div class="metric-container">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span class="status-dot {status_colors.get(status, 'status-gray')}"></span>
            <h4 style="margin: 0; color: #f9fafb; font-size: 0.9rem; font-weight: 500;">{label}</h4>
        </div>
        <div style="display: flex; align-items: baseline; margin-bottom: 0.5rem;">
            <h2 style="margin: 0; color: #f9fafb; font-size: 2rem; font-weight: 700;">{value}</h2>
            {f'<span style="margin-left: 0.5rem; color: #10b981; font-size: 0.9rem;">{delta}</span>' if delta else ''}
        </div>
        {f'<div style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 0.5rem;">Target: {target}</div>' if target else ''}
        {f'<div class="progress-bar"><div class="progress-fill {progress_color}" style="width: {progress_pct}%"></div></div>' if target else ''}
    </div>
    """
    
    c.markdown(tile_html, unsafe_allow_html=True)

def get_status_color(value, thresholds, reverse=False):
    """Get status color based on value thresholds"""
    if reverse:
        if value >= thresholds[1]:
            return "normal"
        elif value >= thresholds[0]:
            return "warning"
        else:
            return "critical"
    else:
        if value <= thresholds[0]:
            return "normal"
        elif value <= thresholds[1]:
            return "warning"
        else:
            return "critical"

def download_csv_button(df, label="Download CSV", filename="filtered_data.csv"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

# --------------------------
# Sidebar: Data & Filters
# --------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #4b5563; margin-bottom: 2rem;">
    <h2 style="color: #f9fafb; margin: 0; font-size: 1.5rem;">⚙️ Crusher AI Dashboard</h2>
    <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Predictive Maintenance</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    "📁 Upload CSV Data",
    type=["csv"],
    help="Upload your sensor data CSV file"
)
df = load_data(uploaded)

# System Status
st.sidebar.markdown("### 🔧 System Status")
if len(df) > 0:
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="status-dot status-green"></span>
        <span style="color: #f9fafb;">All Systems Operational</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="status-dot status-red"></span>
        <span style="color: #f9fafb;">No Data Available</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"**📊 Detected Records:** {len(df)}")
st.sidebar.markdown(f"**🕒 Last Update:** {pd.Timestamp.now().strftime('%H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Controls")

# Date filters
if len(df):
    min_date = df["datetime_stamp"].min()
    max_date = df["datetime_stamp"].max()
else:
    min_date = pd.Timestamp("2025-08-01")
    max_date = pd.Timestamp("2025-08-31 23:00:00")

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

plant_options = sorted(df["Plant_Id"].dropna().unique()) if "Plant_Id" in df.columns else []
plant_sel = st.sidebar.multiselect("Plant_Id filter", plant_options, default=plant_options[:3] if plant_options else [])

machine_options = sorted(df["Machine_Id"].dropna().unique()) if "Machine_Id" in df.columns else []
machine_sel = st.sidebar.multiselect("Machine_Id filter", machine_options, default=machine_options[:10] if machine_options else [])

kpi_default = "Motor_Temperature" if "Motor_Temperature" in df.columns else (KPI_COLUMNS[0] if KPI_COLUMNS else None)
kpi_pick = st.sidebar.selectbox("Primary KPI", [k for k in KPI_COLUMNS if k in df.columns], index=(KPI_COLUMNS.index(kpi_default) if kpi_default in KPI_COLUMNS else 0))
roll_window = st.sidebar.slider("Rolling window (hours) for moving avg", min_value=1, max_value=72, value=24, step=1)

st.sidebar.caption("Expected columns: datetime_stamp, Machine_Id, Plant_Id, "
                   + ", ".join(KPI_COLUMNS) + ", Anomaly_Flag")

# Apply filters
df_f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_f = df_f[(df_f["datetime_stamp"] >= start_dt) & (df_f["datetime_stamp"] <= end_dt)]

if plant_sel and "Plant_Id" in df_f.columns:
    df_f = df_f[df_f["Plant_Id"].isin(plant_sel)]
if machine_sel and "Machine_Id" in df_f.columns:
    df_f = df_f[df_f["Machine_Id"].isin(machine_sel)]

# --------------------------
# Header
# --------------------------
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="color: #f9fafb; margin: 0; font-size: 2.5rem; font-weight: 700;">AI Operations Dashboard</h1>
    <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Time-series monitoring • Anomaly alerts • Machine & plant insights</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Row 1 — Enhanced KPI Tiles
# --------------------------
st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
if len(df_f):
    # Calculate metrics
    latest_day = df_f["datetime_stamp"].dt.date.max()
    anomalies_today = int(df_f[df_f["datetime_stamp"].dt.date == latest_day]["Anomaly_Flag"].sum())
    plants_affected = df_f.loc[df_f["Anomaly_Flag"] == 1, "Plant_Id"].nunique() if "Plant_Id" in df_f.columns else 0
    machines_alert = df_f.loc[df_f["Anomaly_Flag"] == 1, "Machine_Id"].nunique() if "Machine_Id" in df_f.columns else 0
    
    # Calculate anomaly percentage in last 24 hours
    last_24h = df_f["datetime_stamp"].max() - pd.Timedelta(hours=24)
    recent_data = df_f[df_f["datetime_stamp"] >= last_24h]
    anomaly_pct = (recent_data["Anomaly_Flag"].sum() / len(recent_data) * 100) if len(recent_data) > 0 else 0
    
    # Calculate max deviation in each KPI vs baseline (normal operation)
    normal_data = df_f[df_f["Anomaly_Flag"] == 0]  # Baseline = normal operation
    
    # Max deviations from baseline
    max_vibration_dev = 0
    max_temp_dev = 0
    max_pressure_dev = 0
    max_power_dev = 0
    
    if len(normal_data) > 0:
        # Baseline values (normal operation)
        baseline_vibration = normal_data["Vibration_Level"].mean() if "Vibration_Level" in normal_data.columns else 0
        baseline_temp = normal_data["Motor_Temperature"].mean() if "Motor_Temperature" in normal_data.columns else 0
        baseline_pressure = normal_data["Oil_Pressure"].mean() if "Oil_Pressure" in normal_data.columns else 0
        baseline_power = normal_data["Power_Consumption"].mean() if "Power_Consumption" in normal_data.columns else 0
        
        # Calculate max deviations
        if "Vibration_Level" in df_f.columns:
            max_vibration_dev = abs(df_f["Vibration_Level"].max() - baseline_vibration)
        if "Motor_Temperature" in df_f.columns:
            max_temp_dev = abs(df_f["Motor_Temperature"].max() - baseline_temp)
        if "Oil_Pressure" in df_f.columns:
            max_pressure_dev = abs(df_f["Oil_Pressure"].max() - baseline_pressure)
        if "Power_Consumption" in df_f.columns:
            max_power_dev = abs(df_f["Power_Consumption"].max() - baseline_power)
    
    # Status determination based on manager's requirements
    anomaly_status = get_status_color(anomaly_pct, [2, 5])  # <2% green, 2-5% amber, >5% red
    machines_status = get_status_color(machines_alert, [2, 5])  # <2 machines green, 2-5 amber, >5 red
    
    # Max deviation status (higher deviation = worse)
    vibration_dev_status = get_status_color(max_vibration_dev, [2, 5])  # <2 green, 2-5 amber, >5 red
    temp_dev_status = get_status_color(max_temp_dev, [10, 20])  # <10°C green, 10-20°C amber, >20°C red
    pressure_dev_status = get_status_color(max_pressure_dev, [2, 4])  # <2 bar green, 2-4 bar amber, >4 bar red
    power_dev_status = get_status_color(max_power_dev, [50, 100])  # <50kW green, 50-100kW amber, >100kW red
    
    # Manager's Expected KPI Tiles (exactly as requested)
    enhanced_kpi_tile("ANOMALIES LAST 24H", f"{anomaly_pct:.1f}%", target=2.0, delta=f"+{anomaly_pct-2:.1f}%" if anomaly_pct > 2 else f"{anomaly_pct-2:.1f}%", col=col1, status=anomaly_status)
    enhanced_kpi_tile("ACTIVE ALERT MACHINES", machines_alert, target=2, delta=f"+{machines_alert-2}" if machines_alert > 2 else f"{machines_alert-2}", col=col2, status=machines_status)
    enhanced_kpi_tile("MAX VIBRATION DEVIATION", f"{max_vibration_dev:.1f}mm/s", target=2.0, delta=f"+{max_vibration_dev-2:.1f}" if max_vibration_dev > 2 else f"{max_vibration_dev-2:.1f}", col=col3, status=vibration_dev_status)
    enhanced_kpi_tile("MAX TEMP DEVIATION", f"{max_temp_dev:.1f}°C", target=10.0, delta=f"+{max_temp_dev-10:.1f}" if max_temp_dev > 10 else f"{max_temp_dev-10:.1f}", col=col4, status=temp_dev_status)
    enhanced_kpi_tile("MAX PRESSURE DEVIATION", f"{max_pressure_dev:.1f}bar", target=2.0, delta=f"+{max_pressure_dev-2:.1f}" if max_pressure_dev > 2 else f"{max_pressure_dev-2:.1f}", col=col5, status=pressure_dev_status)
    enhanced_kpi_tile("MAX POWER DEVIATION", f"{max_power_dev:.1f}kW", target=50.0, delta=f"+{max_power_dev-50:.1f}" if max_power_dev > 50 else f"{max_power_dev-50:.1f}", col=col6, status=power_dev_status)
else:
    for c in [col1, col2, col3, col4, col5, col6]:
        enhanced_kpi_tile("No Data", "—", col=c, status="idle")

st.markdown("---")

# --------------------------
# Row 2 — Equipment Status
# --------------------------
st.markdown('<h2 class="section-header">🔧 Equipment Status</h2>', unsafe_allow_html=True)

# Equipment tabs
tab1, tab2 = st.tabs(["Crushers", "Sensors"])

with tab1:
    if len(df_f) and "Machine_Id" in df_f.columns:
        # Get unique machines and their status
        machine_status = df_f.groupby("Machine_Id").agg({
            "Anomaly_Flag": "sum",
            "Vibration_Level": "mean",
            "Motor_Temperature": "mean",
            "datetime_stamp": "max"
        }).reset_index()
        
        # Determine status for each machine
        def get_machine_status(row):
            if row["Anomaly_Flag"] > 0:
                return "critical"
            elif row["Vibration_Level"] > 6 or row["Motor_Temperature"] > 80:
                return "warning"
            else:
                return "normal"
        
        machine_status["status"] = machine_status.apply(get_machine_status, axis=1)
        machine_status["efficiency"] = 100 - (machine_status["Vibration_Level"] * 10)
        machine_status["efficiency"] = machine_status["efficiency"].clip(0, 100)
        
        # Display machines in a grid
        cols = st.columns(3)
        for idx, (_, machine) in enumerate(machine_status.iterrows()):
            col_idx = idx % 3
            with cols[col_idx]:
                status_colors = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}
                status_text = {"normal": "Running", "warning": "Warning", "critical": "Error"}
                
                st.markdown(f"""
                <div class="equipment-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_colors[machine['status']]}</span>
                        <h4 style="margin: 0; color: #f9fafb;">{machine['Machine_Id']}</h4>
                    </div>
                    <div style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        Status: {status_text[machine['status']]}
                    </div>
                    <div style="color: #f9fafb; font-size: 1.1rem; font-weight: 600;">
                        Efficiency: {machine['efficiency']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No machine data available.")

with tab2:
    if len(df_f):
        # Sensor status overview
        sensor_cols = st.columns(4)
        
        # Vibration sensor
        avg_vibration = df_f["Vibration_Level"].mean() if "Vibration_Level" in df_f.columns else 0
        vibration_status = "normal" if avg_vibration < 5 else "warning" if avg_vibration < 8 else "critical"
        
        # Temperature sensor
        avg_temp = df_f["Motor_Temperature"].mean() if "Motor_Temperature" in df_f.columns else 0
        temp_status = "normal" if avg_temp < 75 else "warning" if avg_temp < 85 else "critical"
        
        # Pressure sensor
        avg_pressure = df_f["Oil_Pressure"].mean() if "Oil_Pressure" in df_f.columns else 0
        pressure_status = "normal" if 10 < avg_pressure < 15 else "warning" if 8 < avg_pressure < 17 else "critical"
        
        # Power sensor
        avg_power = df_f["Power_Consumption"].mean() if "Power_Consumption" in df_f.columns else 0
        power_status = "normal" if avg_power < 500 else "warning" if avg_power < 600 else "critical"
        
        sensors = [
            ("Vibration Sensor", avg_vibration, vibration_status, "mm/s"),
            ("Temperature Sensor", avg_temp, temp_status, "°C"),
            ("Pressure Sensor", avg_pressure, pressure_status, "bar"),
            ("Power Sensor", avg_power, power_status, "kW")
        ]
        
        for idx, (name, value, status, unit) in enumerate(sensors):
            with sensor_cols[idx]:
                status_colors = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}
                st.markdown(f"""
                <div class="equipment-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_colors[status]}</span>
                        <h4 style="margin: 0; color: #f9fafb;">{name}</h4>
                    </div>
                    <div style="color: #f9fafb; font-size: 1.1rem; font-weight: 600;">
                        {value:.1f} {unit}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No sensor data available.")

st.markdown("---")

# --------------------------
# Row 3 — Active Alerts
# --------------------------
st.markdown('<h2 class="section-header">🚨 Active Alerts</h2>', unsafe_allow_html=True)

if len(df_f):
    # Count alerts by severity
    critical_alerts = len(df_f[df_f["Anomaly_Flag"] == 1])
    warning_alerts = len(df_f[(df_f["Vibration_Level"] > 6) | (df_f["Motor_Temperature"] > 80)])
    info_alerts = len(df_f[(df_f["Oil_Pressure"] < 8) | (df_f["Oil_Pressure"] > 15)])
    
    alert_col1, alert_col2 = st.columns([1, 2])
    
    with alert_col1:
        st.markdown(f"""
        <div class="alert-container">
            <h4 style="color: #f9fafb; margin: 0 0 1rem 0;">Alert Summary</h4>
            <div style="color: #ef4444; font-size: 1.2rem; font-weight: 600;">{critical_alerts} Critical</div>
            <div style="color: #f59e0b; font-size: 1.2rem; font-weight: 600;">{warning_alerts} Warning</div>
            <div style="color: #3b82f6; font-size: 1.2rem; font-weight: 600;">{info_alerts} Info</div>
        </div>
        """, unsafe_allow_html=True)
    
    with alert_col2:
        # Show latest anomaly alert
        latest_anomaly = df_f[df_f["Anomaly_Flag"] == 1].tail(1)
        if len(latest_anomaly) > 0:
            anomaly_row = latest_anomaly.iloc[0]
            st.markdown(f"""
            <div class="alert-container">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span class="status-dot status-red"></span>
                    <h4 style="margin: 0; color: #f9fafb;">Anomaly Alert</h4>
                </div>
                <p style="color: #f9fafb; margin: 0;">
                    System: Anomaly detected in {anomaly_row['Machine_Id']} at {anomaly_row['datetime_stamp'].strftime('%H:%M:%S')} - 
                    High vibration ({anomaly_row['Vibration_Level']:.1f}) and temperature ({anomaly_row['Motor_Temperature']:.1f}°C)
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-container">
                <div style="display: flex; align-items: center;">
                    <span class="status-dot status-green"></span>
                    <span style="color: #f9fafb;">No active alerts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No alert data available.")

st.markdown("---")

# --------------------------
# Row 4 — Time Trend & Alert Monitoring
# --------------------------
st.subheader("📈 Trend Analysis")

# Add KPI selection for heatmap
kpi_heatmap = st.selectbox("Select KPI for time heatmap", [k for k in KPI_COLUMNS if k in df_f.columns], index=0, key="kpi_heatmap")

if len(df_f) and (kpi_pick in df_f.columns):
    # Time series chart
    df_trend = df_f[["datetime_stamp", kpi_pick, "Anomaly_Flag", "Machine_Id", "Plant_Id"]].copy().sort_values("datetime_stamp")
    # Rolling avg (by time; not per machine)
    df_trend["Rolling_Avg"] = df_trend[kpi_pick].rolling(roll_window, min_periods=max(1, roll_window//4)).mean()

    fig = go.Figure()
    # actual
    fig.add_trace(go.Scatter(
        x=df_trend["datetime_stamp"],
        y=df_trend[kpi_pick],
        mode="lines",
        name=f"{kpi_pick} (actual)",
        opacity=0.6
    ))
    # rolling avg
    fig.add_trace(go.Scatter(
        x=df_trend["datetime_stamp"],
        y=df_trend["Rolling_Avg"],
        mode="lines",
        name=f"{kpi_pick} (rolling {roll_window}h)",
        line=dict(width=2)
    ))
    # anomalies as markers
    df_an = df_trend[df_trend["Anomaly_Flag"] == 1]
    if len(df_an):
        fig.add_trace(go.Scatter(
            x=df_an["datetime_stamp"],
            y=df_an[kpi_pick],
            mode="markers",
            name="Anomaly",
            marker=dict(size=8, symbol="x")
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time",
        yaxis_title=kpi_pick
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for selected KPI.")

# Add KPI Heatmap over Time
if len(df_f) and kpi_heatmap in df_f.columns:
    st.subheader("🔥 KPI Heatmap over Time")
    # Create hour vs date heatmap
    df_heatmap = df_f.copy()
    df_heatmap['hour'] = df_heatmap['datetime_stamp'].dt.hour
    df_heatmap['date'] = df_heatmap['datetime_stamp'].dt.date
    
    # Pivot for heatmap
    heatmap_data = df_heatmap.pivot_table(
        index='hour', 
        columns='date', 
        values=kpi_heatmap, 
        aggfunc='mean'
    )
    
    if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
        fig_heatmap = px.imshow(
            heatmap_data, 
            aspect="auto",
            title=f"{kpi_heatmap} Intensity by Hour and Date",
            labels={'x': 'Date', 'y': 'Hour of Day', 'color': kpi_heatmap}
        )
        fig_heatmap.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Not enough data to create heatmap.")
else:
    st.info("No data available for heatmap.")

st.markdown("---")

# --------------------------
# Row 3 — Anomaly Insights (Daily Count + Heatmap)
# --------------------------
c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Daily Anomaly Count")
    if len(df_f):
        daily = (
            df_f.assign(day=df_f["datetime_stamp"].dt.date)
               .groupby("day")["Anomaly_Flag"]
               .sum()
               .reset_index(name="Anomaly_Count")
        )
        fig_daily = px.bar(daily, x="day", y="Anomaly_Count")
        fig_daily.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Day", yaxis_title="Anomalies")
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.info("No daily anomaly data in selection.")

with c2:
    st.subheader("Machine × Day Anomaly Heatmap")
    if len(df_f) and "Machine_Id" in df_f.columns:
        hm = (
            df_f.assign(day=df_f["datetime_stamp"].dt.date)
               .pivot_table(index="Machine_Id", columns="day", values="Anomaly_Flag", aggfunc="sum", fill_value=0)
        )
        if hm.shape[0] and hm.shape[1]:
            fig_hm = px.imshow(hm, aspect="auto")
            fig_hm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Not enough data to build heatmap.")
    else:
        st.info("Machine_Id not available.")

st.markdown("---")

# --------------------------
# Row 4 — Machine & Plant Comparison
# --------------------------
st.subheader("Machine & Plant Comparison")

cc1, cc2 = st.columns([1, 1])
with cc1:
    st.markdown("**Average KPI by Machine**")
    if len(df_f) and "Machine_Id" in df_f.columns:
        kpi_for_bar = st.selectbox("Select KPI for bar chart", [k for k in KPI_COLUMNS if k in df_f.columns], index=0, key="kpi_bar")
        means = df_f.groupby("Machine_Id")[kpi_for_bar].mean().reset_index()
        fig_bar = px.bar(means, x="Machine_Id", y=kpi_for_bar)
        fig_bar.update_layout(height=400, xaxis_title="Machine_Id", yaxis_title=f"Avg {kpi_for_bar}", margin=dict(l=20, r=20, t=30, b=80))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No machine-wise data available.")

with cc2:
    st.markdown("**Pareto — Anomalies by Machine**")
    if len(df_f) and "Machine_Id" in df_f.columns:
        an = (df_f.groupby("Machine_Id")["Anomaly_Flag"].sum().reset_index(name="Anomaly_Count")
                    .sort_values("Anomaly_Count", ascending=False))
        an["Cumulative_%"] = an["Anomaly_Count"].cumsum() / an["Anomaly_Count"].sum() * 100 if an["Anomaly_Count"].sum() > 0 else 0
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p.add_trace(go.Bar(x=an["Machine_Id"], y=an["Anomaly_Count"], name="Anomaly Count"), secondary_y=False)
        fig_p.add_trace(go.Scatter(x=an["Machine_Id"], y=an["Cumulative_%"], name="Cumulative %", mode="lines+markers"), secondary_y=True)
        fig_p.update_yaxes(title_text="Anomaly Count", secondary_y=False)
        fig_p.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
        fig_p.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=80))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("No anomaly data available.")

st.markdown("---")

# --------------------------
# Row 5 — Diagnostics (Scatter + Boxplots)
# --------------------------
st.subheader("Diagnostics")

dd1, dd2 = st.columns([1, 1])
with dd1:
    st.markdown("**Scatter — KPI vs KPI (anomalies highlighted)**")
    kpi_x = st.selectbox("X axis KPI", [k for k in KPI_COLUMNS if k in df_f.columns], index=0, key="kpi_x")
    kpi_y = st.selectbox("Y axis KPI", [k for k in KPI_COLUMNS if k in df_f.columns and k != kpi_x], index=0, key="kpi_y")
    if len(df_f):
        fig_sc = px.scatter(
            df_f, x=kpi_x, y=kpi_y,
            color=df_f["Anomaly_Flag"].map({0: "Normal", 1: "Anomaly"}) if "Anomaly_Flag" in df_f.columns else None,
            hover_data=["datetime_stamp", "Machine_Id", "Plant_Id"] if "Plant_Id" in df_f.columns else ["datetime_stamp", "Machine_Id"],
            opacity=0.8
        )
        fig_sc.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("No data for scatter plot.")

with dd2:
    st.markdown("**Boxplot — KPI Distribution**")
    kpi_box = st.selectbox("Boxplot KPI", [k for k in KPI_COLUMNS if k in df_f.columns], index=0, key="kpi_box")
    if len(df_f):
        fig_box = px.box(df_f, y=kpi_box, points="outliers", color=df_f["Anomaly_Flag"].map({0: "Normal", 1: "Anomaly"}) if "Anomaly_Flag" in df_f.columns else None)
        fig_box.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), yaxis_title=kpi_box)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No data for boxplot.")

st.markdown("---")

# --------------------------
# Row 6 — Alert Log (Enhanced)
# --------------------------
st.subheader("🚨 Anomaly Table (Filtered View)")

# Add filter options for the alert table
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    show_only_anomalies = st.checkbox("Show only anomalies (Anomaly_Flag=1)", value=False)
with col_filter2:
    max_rows = st.slider("Max rows to display", min_value=50, max_value=1000, value=200, step=50)

if len(df_f):
    # Filter data based on options
    df_show = df_f.copy()
    if show_only_anomalies:
        df_show = df_show[df_show["Anomaly_Flag"] == 1]
    
    # Limit rows
    df_show = df_show.head(max_rows)
    
    # Order columns nicely if present
    ordered_cols = ["datetime_stamp", "Machine_Id", "Plant_Id"] + [c for c in KPI_COLUMNS if c in df_show.columns] + ["Anomaly_Flag"]
    ordered_cols = [c for c in ordered_cols if c in df_show.columns]
    df_show = df_show[ordered_cols]
    
    # Add flag icons for anomalies
    def highlight_anomalies(row):
        if row["Anomaly_Flag"] == 1:
            return ["background-color: #ffebee; border-left: 4px solid #f44336"] * len(row)
        return [""] * len(row)
    
    # Create styled dataframe
    styled_df = df_show.style.apply(highlight_anomalies, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )
    
    # Add anomaly summary
    anomaly_count = len(df_show[df_show["Anomaly_Flag"] == 1])
    total_count = len(df_show)
    st.caption(f"📊 Showing {anomaly_count} anomalies out of {total_count} total records")
    
    download_csv_button(df_show, "📥 Download Alert Table as CSV", "alerts_filtered.csv")
else:
    st.info("No data to display in alert log.")

st.markdown("---")

# --------------------------
# Row 7 — Machine & Plant Drilldowns
# --------------------------
st.subheader("📊 Dashboard Drilldowns")

# Machine Drilldown
st.markdown("#### 🔧 Machine Drilldown")
if len(df_f) and "Machine_Id" in df_f.columns:
    machine_options = sorted(df_f["Machine_Id"].unique())
    selected_machine = st.selectbox("Select Machine for detailed view", machine_options, key="machine_drilldown")
    
    if selected_machine:
        machine_data = df_f[df_f["Machine_Id"] == selected_machine].copy()
        
        col_drill1, col_drill2 = st.columns(2)
        
        with col_drill1:
            st.markdown(f"**📈 {selected_machine} KPI Timeline**")
            if len(machine_data) > 0:
                fig_machine = go.Figure()
                for kpi in KPI_COLUMNS:
                    if kpi in machine_data.columns:
                        fig_machine.add_trace(go.Scatter(
                            x=machine_data["datetime_stamp"],
                            y=machine_data[kpi],
                            mode="lines+markers",
                            name=kpi,
                            opacity=0.7
                        ))
                
                fig_machine.update_layout(
                    height=300,
                    title=f"KPIs for {selected_machine}",
                    xaxis_title="Time",
                    yaxis_title="KPI Value"
                )
                st.plotly_chart(fig_machine, use_container_width=True)
        
        with col_drill2:
            st.markdown(f"**⚠️ {selected_machine} Anomaly Timeline**")
            if len(machine_data) > 0:
                anomaly_timeline = machine_data.groupby(machine_data["datetime_stamp"].dt.date)["Anomaly_Flag"].sum().reset_index()
                fig_anomaly = px.bar(anomaly_timeline, x="datetime_stamp", y="Anomaly_Flag", title=f"Daily Anomalies for {selected_machine}")
                fig_anomaly.update_layout(height=300)
                st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Machine summary stats
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Total Anomalies", int(machine_data["Anomaly_Flag"].sum()))
        with col_stats2:
            anomaly_rate = (machine_data["Anomaly_Flag"].sum() / len(machine_data) * 100) if len(machine_data) > 0 else 0
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        with col_stats3:
            avg_temp = machine_data["Motor_Temperature"].mean() if "Motor_Temperature" in machine_data.columns else 0
            st.metric("Avg Motor Temp", f"{avg_temp:.1f}°C")

# Plant Comparison
st.markdown("#### 🏭 Plant Comparison")
if len(df_f) and "Plant_Id" in df_f.columns:
    plant_options = sorted(df_f["Plant_Id"].unique())
    selected_plants = st.multiselect("Select Plants to Compare", plant_options, default=plant_options[:2], key="plant_comparison")
    
    if selected_plants:
        plant_data = df_f[df_f["Plant_Id"].isin(selected_plants)]
        
        col_plant1, col_plant2 = st.columns(2)
        
        with col_plant1:
            st.markdown("**📊 Plant Anomaly Heatmap**")
            plant_anomaly = plant_data.groupby(["Plant_Id", plant_data["datetime_stamp"].dt.date])["Anomaly_Flag"].sum().reset_index()
            plant_pivot = plant_anomaly.pivot(index="Plant_Id", columns="datetime_stamp", values="Anomaly_Flag").fillna(0)
            
            if plant_pivot.shape[0] > 0 and plant_pivot.shape[1] > 0:
                fig_plant_heatmap = px.imshow(plant_pivot, aspect="auto", title="Anomalies by Plant and Date")
                fig_plant_heatmap.update_layout(height=300)
                st.plotly_chart(fig_plant_heatmap, use_container_width=True)
        
        with col_plant2:
            st.markdown("**⚡ Cross-Plant Throughput Efficiency**")
            if "Throughput" in plant_data.columns:
                plant_throughput = plant_data.groupby("Plant_Id")["Throughput"].agg(['mean', 'std']).reset_index()
                plant_throughput['efficiency'] = plant_throughput['mean'] / plant_throughput['std']  # Higher mean, lower std = more efficient
                
                fig_efficiency = px.bar(plant_throughput, x="Plant_Id", y="efficiency", title="Plant Efficiency (Mean/Std)")
                fig_efficiency.update_layout(height=300)
                st.plotly_chart(fig_efficiency, use_container_width=True)

st.markdown("---")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption(
    "Tip: Use the sidebar to refine plants, machines, KPIs, and date range. "
    "Upload your CSV with columns: datetime_stamp, Machine_Id, Plant_Id, "
    + ", ".join(KPI_COLUMNS) + ", Anomaly_Flag."
)
