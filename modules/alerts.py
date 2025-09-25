# Alerts & Monitoring Page
# ========================

import streamlit as st
import pandas as pd

# Import our modular components
from config.constants import KPI_COLUMNS
from utils.calculations import calculate_anomaly_metrics, calculate_max_deviations
from components.kpi_tiles import enhanced_kpi_tile
from components.alerts import render_alert_log, render_if_score_critical_alerts
from utils.calculations import get_status_color
from config.constants import THRESHOLDS, TARGETS

def render_alerts_page():
    """Render the Alerts & Monitoring page"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # --------------------------
    # Row 1 — IF Score Critical Alerts (MOVED TO TOP)
    # --------------------------
    render_if_score_critical_alerts(df_f)
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Alert KPI Cards
    # --------------------------
    st.markdown('<h2 class="section-header">🚨 Alert KPI Cards</h2>', unsafe_allow_html=True)
    
    # Calculate alert metrics
    anomaly_metrics = calculate_anomaly_metrics(df_f)
    deviations = calculate_max_deviations(df_f)
    
    # Create alert KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    
    # Anomaly Percentage
    anomaly_status = get_status_color(anomaly_metrics["anomaly_pct"], THRESHOLDS["anomaly_rate"])
    enhanced_kpi_tile(
        "ANOMALY PERCENTAGE", 
        f"{anomaly_metrics['anomaly_pct']:.1f}%", 
        target=TARGETS["anomaly_rate"], 
        delta=f"+{anomaly_metrics['anomaly_pct']-TARGETS['anomaly_rate']:.1f}%" if anomaly_metrics['anomaly_pct'] > TARGETS['anomaly_rate'] else f"{anomaly_metrics['anomaly_pct']-TARGETS['anomaly_rate']:.1f}%", 
        col=col1, 
        status=anomaly_status
    )
    
    # Active Alert Machines
    machines_status = get_status_color(anomaly_metrics["machines_alert"], THRESHOLDS["active_machines"])
    enhanced_kpi_tile(
        "ACTIVE ALERT MACHINES", 
        anomaly_metrics["machines_alert"], 
        target=TARGETS["active_machines"], 
        delta=f"+{anomaly_metrics['machines_alert']-TARGETS['active_machines']}" if anomaly_metrics['machines_alert'] > TARGETS['active_machines'] else f"{anomaly_metrics['machines_alert']-TARGETS['active_machines']}", 
        col=col2, 
        status=machines_status
    )
    
    # Plants Affected
    plants_status = get_status_color(anomaly_metrics["plants_affected"], [1, 3])
    enhanced_kpi_tile(
        "PLANTS AFFECTED", 
        anomaly_metrics["plants_affected"], 
        target=1, 
        delta=f"+{anomaly_metrics['plants_affected']-1}" if anomaly_metrics['plants_affected'] > 1 else f"{anomaly_metrics['plants_affected']-1}", 
        col=col3, 
        status=plants_status
    )
    
    # Total Anomalies
    total_anomalies = df_f["Anomaly_Flag"].sum()
    enhanced_kpi_tile(
        "TOTAL ANOMALIES", 
        int(total_anomalies), 
        target=20, 
        delta=f"+{total_anomalies-20}" if total_anomalies > 20 else f"{total_anomalies-20}", 
        col=col4, 
        status="critical" if total_anomalies > 50 else "warning" if total_anomalies > 20 else "normal"
    )
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Anomaly Table with Filters
    # --------------------------
    st.markdown('<h2 class="section-header">📋 Anomaly Table (Filtered View)</h2>', unsafe_allow_html=True)
    
    # Advanced filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Plant filter
        plant_options = ["All"] + sorted(df_f["Plant_Id"].dropna().unique().tolist()) if "Plant_Id" in df_f.columns else ["All"]
        selected_plant = st.selectbox("Filter by Plant", plant_options)
    
    with col2:
        # Machine filter
        machine_options = ["All"] + sorted(df_f["Machine_Id"].dropna().unique().tolist()) if "Machine_Id" in df_f.columns else ["All"]
        selected_machine = st.selectbox("Filter by Machine", machine_options)
    
    with col3:
        # Anomaly type filter
        anomaly_type = st.selectbox("Anomaly Type", ["All", "Anomalies Only", "Normal Only"])
    
    # Apply filters
    filtered_df = df_f.copy()
    
    if selected_plant != "All" and "Plant_Id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Plant_Id"] == selected_plant]
    
    if selected_machine != "All" and "Machine_Id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Machine_Id"] == selected_machine]
    
    if anomaly_type == "Anomalies Only":
        filtered_df = filtered_df[filtered_df["Anomaly_Flag"] == 1]
    elif anomaly_type == "Normal Only":
        filtered_df = filtered_df[filtered_df["Anomaly_Flag"] == 0]
    
    # Render filtered alert log
    render_alert_log(filtered_df, KPI_COLUMNS)
    
    st.markdown("---")
    
    # --------------------------
    # Row 4 — Comprehensive Alert Filtering Table
    # --------------------------
    st.markdown('<h2 class="section-header">🔍 Comprehensive Alert Filtering</h2>', unsafe_allow_html=True)
    
    # Add IF score-based filtering
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # IF Score Alert Level Filter
        if_score_filter = st.selectbox(
            "Alert Level (IF Score)", 
            ["All", "Critical (≥0.7)", "High (0.6-0.7)", "Medium (0.5-0.6)", "Low (<0.5)"]
        )
    
    with col2:
        # Priority Filter
        priority_filter = st.selectbox(
            "Priority Level", 
            ["All", "Critical", "High", "Medium", "Low"]
        )
    
    with col3:
        # Machine Status Filter
        machine_status_filter = st.selectbox(
            "Machine Status", 
            ["All", "Running", "Warning", "Critical"]
        )
    
    with col4:
        # Time Range Filter
        time_filter = st.selectbox(
            "Time Range", 
            ["All", "Last 24h", "Last 7 days", "Last 30 days"]
        )
    
    # Apply comprehensive filters
    comprehensive_filtered_df = df_f.copy()
    
    # Apply IF score filter
    if if_score_filter != "All":
        if if_score_filter == "Critical (≥0.7)":
            comprehensive_filtered_df = comprehensive_filtered_df[comprehensive_filtered_df["if_score"] >= 0.7]
        elif if_score_filter == "High (0.6-0.7)":
            comprehensive_filtered_df = comprehensive_filtered_df[(comprehensive_filtered_df["if_score"] >= 0.6) & (comprehensive_filtered_df["if_score"] < 0.7)]
        elif if_score_filter == "Medium (0.5-0.6)":
            comprehensive_filtered_df = comprehensive_filtered_df[(comprehensive_filtered_df["if_score"] >= 0.5) & (comprehensive_filtered_df["if_score"] < 0.6)]
        elif if_score_filter == "Low (<0.5)":
            comprehensive_filtered_df = comprehensive_filtered_df[comprehensive_filtered_df["if_score"] < 0.5]
    
    # Apply time filter
    if time_filter != "All":
        from datetime import datetime, timedelta
        now = datetime.now()
        if time_filter == "Last 24h":
            cutoff = now - timedelta(hours=24)
        elif time_filter == "Last 7 days":
            cutoff = now - timedelta(days=7)
        elif time_filter == "Last 30 days":
            cutoff = now - timedelta(days=30)
        
        # Convert datetime_stamp to datetime if it's not already
        comprehensive_filtered_df["datetime_stamp"] = pd.to_datetime(comprehensive_filtered_df["datetime_stamp"])
        comprehensive_filtered_df = comprehensive_filtered_df[comprehensive_filtered_df["datetime_stamp"] >= cutoff]
    
    # Display comprehensive filtered table
    if len(comprehensive_filtered_df) > 0:
        st.markdown("### 📋 Filtered Alert Details")
        
        # Select columns to display
        display_cols = [
            "datetime_stamp", "Machine_Id", "Plant_Id", "if_score", 
            "Vibration_Level", "Motor_Temperature", "Oil_Pressure", 
            "Power_Consumption", "Anomaly_Flag"
        ]
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in comprehensive_filtered_df.columns]
        display_df = comprehensive_filtered_df[available_cols]
        
        # Sort by IF score (highest first)
        display_df = display_df.sort_values("if_score", ascending=False)
        
        # Style the dataframe based on IF score
        def highlight_by_if_score(row):
            if row["if_score"] >= 0.7:
                return ["background-color: #2d1b1b; color: #ffffff; border-left: 4px solid #dc2626"] * len(row)
            elif row["if_score"] >= 0.6:
                return ["background-color: #2d2419; color: #ffffff; border-left: 4px solid #f59e0b"] * len(row)
            elif row["if_score"] >= 0.5:
                return ["background-color: #1a2d2d; color: #ffffff; border-left: 4px solid #17a2b8"] * len(row)
            else:
                return ["background-color: #1a2d1a; color: #ffffff; border-left: 4px solid #28a745"] * len(row)
        
        styled_df = display_df.style.apply(highlight_by_if_score, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Show filter summary
        st.caption(f"📊 Showing {len(display_df)} alerts with filters: {if_score_filter}, {priority_filter}, {machine_status_filter}, {time_filter}")
        
        # Download button for filtered alerts
        from utils.data_loader import download_csv_button
        download_csv_button(
            display_df, 
            "📥 Download Filtered Alerts as CSV", 
            "filtered_alerts_comprehensive.csv"
        )
        
    else:
        st.info("No alerts match the selected filters.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 5 — Alert Summary Statistics
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Alert Summary Statistics</h2>', unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            anomaly_count = len(filtered_df[filtered_df["Anomaly_Flag"] == 1])
            st.metric("Anomaly Records", anomaly_count)
        
        with col2:
            normal_count = len(filtered_df[filtered_df["Anomaly_Flag"] == 0])
            st.metric("Normal Records", normal_count)
        
        with col3:
            if len(filtered_df) > 0:
                anomaly_rate = (anomaly_count / len(filtered_df)) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        
        with col4:
            unique_machines = filtered_df["Machine_Id"].nunique() if "Machine_Id" in filtered_df.columns else 0
            st.metric("Unique Machines", unique_machines)
    else:
        st.info("No data available for selected filters.")
