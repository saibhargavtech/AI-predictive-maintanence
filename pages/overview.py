# Overview Page - Leadership Snapshot
# ===================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our modular components
from config.styles import apply_custom_styles
from config.constants import KPI_COLUMNS
from utils.data_loader import load_data
from utils.calculations import calculate_anomaly_metrics, calculate_max_deviations, calculate_equipment_status, calculate_sensor_status
from components.kpi_tiles import render_kpi_tiles
from components.equipment_status import render_equipment_status
from components.alerts import render_active_alerts

def render_overview_page():
    """Render the Overview page - Leadership Snapshot"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # --------------------------
    # Row 1 — Leadership KPI Cards
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Leadership Snapshot</h2>', unsafe_allow_html=True)
    
    # Calculate leadership KPIs
    avg_throughput = df_f["Throughput"].mean() if "Throughput" in df_f.columns else 0
    avg_temp = df_f["Motor_Temperature"].mean() if "Motor_Temperature" in df_f.columns else 0
    anomaly_metrics = calculate_anomaly_metrics(df_f)
    
    # Create leadership KPI tiles
    col1, col2, col3 = st.columns(3)
    
    from components.kpi_tiles import enhanced_kpi_tile
    from utils.calculations import get_status_color
    from config.constants import THRESHOLDS, TARGETS
    
    # Average Crushing Rate
    crushing_status = get_status_color(avg_throughput, [100, 150], reverse=True)
    enhanced_kpi_tile(
        "AVERAGE CRUSHING RATE", 
        f"{avg_throughput:.0f} tonnes/h", 
        target=TARGETS["crushing_rate"], 
        delta=f"+{avg_throughput-TARGETS['crushing_rate']:.0f}" if avg_throughput > TARGETS["crushing_rate"] else f"{avg_throughput-TARGETS['crushing_rate']:.0f}", 
        col=col1, 
        status=crushing_status
    )
    
    # Average Motor Temperature
    temp_status = get_status_color(avg_temp, [70, 80])
    enhanced_kpi_tile(
        "AVERAGE MOTOR TEMP", 
        f"{avg_temp:.1f}°C", 
        target=70, 
        delta=f"+{avg_temp-70:.1f}" if avg_temp > 70 else f"{avg_temp-70:.1f}", 
        col=col2, 
        status=temp_status
    )
    
    # Crusher Efficiency
    efficiency = (avg_throughput / TARGETS["crushing_rate"]) * 100 if avg_throughput > 0 else 0
    efficiency_status = get_status_color(efficiency, [80, 90], reverse=True)
    enhanced_kpi_tile(
        "CRUSHER EFFICIENCY", 
        f"{efficiency:.1f}%", 
        target=TARGETS["efficiency"], 
        delta=f"+{efficiency-TARGETS['efficiency']:.1f}%" if efficiency > TARGETS["efficiency"] else f"{efficiency-TARGETS['efficiency']:.1f}%", 
        col=col3, 
        status=efficiency_status
    )
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Anomaly Trend Line
    # --------------------------
    st.markdown('<h2 class="section-header">📈 Daily Anomaly Trend</h2>', unsafe_allow_html=True)
    
    if len(df_f):
        # Calculate daily anomaly counts
        daily_anomalies = (
            df_f.assign(day=df_f["datetime_stamp"].dt.date)
               .groupby("day")["Anomaly_Flag"]
               .sum()
               .reset_index(name="Anomaly_Count")
        )
        
        # Create trend line chart
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_anomalies["day"],
            y=daily_anomalies["Anomaly_Count"],
            mode="lines+markers",
            name="Daily Anomalies",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date",
            yaxis_title="Anomaly Count",
            title="Daily Anomaly Trend",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No data available for anomaly trend.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Equipment Status Summary
    # --------------------------
    st.markdown('<h2 class="section-header">🔧 Equipment Status Summary</h2>', unsafe_allow_html=True)
    
    if len(df_f):
        machine_status = calculate_equipment_status(df_f)
        sensor_status = calculate_sensor_status(df_f)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_machines = len(machine_status) if len(machine_status) > 0 else 0
            st.metric("Total Machines", total_machines)
        
        with col2:
            running_machines = len(machine_status[machine_status["status"] == "normal"]) if len(machine_status) > 0 else 0
            st.metric("Running Machines", running_machines)
        
        with col3:
            warning_machines = len(machine_status[machine_status["status"] == "warning"]) if len(machine_status) > 0 else 0
            st.metric("Warning Machines", warning_machines)
        
        with col4:
            critical_machines = len(machine_status[machine_status["status"] == "critical"]) if len(machine_status) > 0 else 0
            st.metric("Critical Machines", critical_machines)
        
        # Render equipment status
        render_equipment_status(machine_status, sensor_status)
    else:
        st.info("No equipment data available.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 4 — Active Alerts Summary
    # --------------------------
    render_active_alerts(df_f)
