# Alert Components
# ================

import streamlit as st
import pandas as pd
from utils.calculations import calculate_if_score_alerts, get_if_score_alert_level
from config.constants import ALERT_COLORS, IF_SCORE_THRESHOLDS

def render_active_alerts(df_f):
    """
    Render active alerts section
    
    Args:
        df_f: Filtered DataFrame
    """
    st.markdown('<h2 class="section-header">🚨 Active Alerts</h2>', unsafe_allow_html=True)
    
    if len(df_f):
        # Count alerts by severity
        critical_alerts = len(df_f[df_f["Anomaly_Flag"] == 1])
        warning_alerts = len(df_f[(df_f["Vibration_Level"] > 6) | (df_f["Motor_Temperature"] > 80)])
        info_alerts = len(df_f[(df_f["Oil_Pressure"] < 8) | (df_f["Oil_Pressure"] > 15)])
        
        alert_col1, alert_col2 = st.columns([1, 2])
        
        with alert_col1:
            render_alert_summary(critical_alerts, warning_alerts, info_alerts)
        
        with alert_col2:
            render_latest_alert(df_f)
    else:
        st.info("No alert data available.")

def render_alert_summary(critical_alerts, warning_alerts, info_alerts):
    """Render alert summary with counts"""
    st.markdown(f"""
    <div class="alert-container">
        <h4 style="color: #f9fafb; margin: 0 0 1rem 0;">Alert Summary</h4>
        <div style="color: #ef4444; font-size: 1.2rem; font-weight: 600;">{critical_alerts} Critical</div>
        <div style="color: #f59e0b; font-size: 1.2rem; font-weight: 600;">{warning_alerts} Warning</div>
        <div style="color: #3b82f6; font-size: 1.2rem; font-weight: 600;">{info_alerts} Info</div>
    </div>
    """, unsafe_allow_html=True)

def render_latest_alert(df_f):
    """Render latest anomaly alert"""
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

def render_alert_log(df_f, KPI_COLUMNS):
    """
    Render enhanced alert log table
    
    Args:
        df_f: Filtered DataFrame
        KPI_COLUMNS: List of KPI column names
    """
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
                return ["background-color: #2d1b1b; color: #ffffff; border-left: 4px solid #f44336"] * len(row)
            return ["background-color: #1a2d1a; color: #ffffff"] * len(row)
        
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
        
        # Download button
        from utils.data_loader import download_csv_button
        download_csv_button(df_show, "📥 Download Alert Table as CSV", "alerts_filtered.csv")
    else:
        st.info("No data to display in alert log.")

def render_if_score_critical_alerts(df_f):
    """
    Render IF score-based critical alerts as color-coded cards
    
    Args:
        df_f: Filtered DataFrame with if_score column
    """
    st.markdown('<h2 class="section-header">🚨 Critical Alerts (IF Score Based)</h2>', unsafe_allow_html=True)
    
    if "if_score" not in df_f.columns:
        st.warning("IF score column not found in data.")
        return
    
    # Calculate IF score alerts
    alert_data = calculate_if_score_alerts(df_f)
    
    if "error" in alert_data:
        st.error(alert_data["error"])
        return
    
    # Display alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Critical Alerts", 
            alert_data["total_critical"],
            delta=f"IF Score ≥ {IF_SCORE_THRESHOLDS['critical']}"
        )
    
    with col2:
        st.metric(
            "Machines Affected", 
            alert_data["machines_critical"]
        )
    
    with col3:
        st.metric(
            "Plants Affected", 
            alert_data["plants_critical"]
        )
    
    with col4:
        st.metric(
            "Max IF Score", 
            f"{alert_data['max_if_score']:.3f}"
        )
    
    # Display critical records as cards
    if alert_data["total_critical"] > 0:
        st.markdown("### 🔴 Critical Alert Cards")
        
        # Show critical records with details
        critical_df = alert_data["critical_records"].copy()
        
        # Sort by IF score (highest first)
        critical_df = critical_df.sort_values("if_score", ascending=False)
        
        # Limit to top 12 records for better display
        critical_df = critical_df.head(12)
        
        # Create cards in rows of 3 using only Streamlit native components
        for i in range(0, len(critical_df), 3):
            cols = st.columns(3)
            
            for j, col in enumerate(cols):
                if i + j < len(critical_df):
                    row = critical_df.iloc[i + j]
                    
                    # Format timestamp
                    timestamp = row["datetime_stamp"].strftime("%m/%d %H:%M") if pd.notna(row["datetime_stamp"]) else "N/A"
                    
                    # Format critical parameters
                    critical_params = row["critical_params"] if isinstance(row["critical_params"], list) else []
                    params_text = ", ".join(critical_params) if critical_params else "Multiple parameters"
                    
                    # Create critical alert card using only Streamlit components
                    with col:
                        # Use st.error for critical alerts (red background)
                        st.error(f"""
                        **🚨 {row['Machine_Id']}** (IF: {row['if_score']:.3f})
                        
                        **Plant:** {row['Plant_Id']}  
                        **Time:** {timestamp}  
                        **Critical Parameters:** {params_text}  
                        **Vibration:** {row['Vibration_Level']:.1f} mm/s  
                        **Motor Temp:** {row['Motor_Temperature']:.1f}°C  
                        **Crushing Rate:** {row['Throughput']:.1f} tonnes/h
                        """)
        
        # Show parameter breakdown
        st.markdown("### 📊 Critical Parameter Analysis")
        
        # Count critical parameters
        all_critical_params = []
        for params_list in critical_df["critical_params"]:
            all_critical_params.extend(params_list)
        
        if all_critical_params:
            param_counts = pd.Series(all_critical_params).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Common Critical Parameters:**")
                for param, count in param_counts.head(5).items():
                    st.write(f"• {param}: {count} occurrences")
            
            with col2:
                st.markdown("**IF Score Distribution:**")
                st.write(f"• Average IF Score: {alert_data['avg_if_score']:.3f}")
                st.write(f"• Maximum IF Score: {alert_data['max_if_score']:.3f}")
                st.write(f"• Critical Threshold: {IF_SCORE_THRESHOLDS['critical']}")
        
        # Download button for critical alerts
        from utils.data_loader import download_csv_button
        download_csv_button(
            critical_df, 
            "📥 Download Critical Alerts as CSV", 
            "critical_alerts.csv"
        )
        
    else:
        st.success("✅ No critical alerts detected! All IF scores are below the critical threshold.")
        
        # Show IF score distribution
        st.markdown("### 📊 IF Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average IF Score", f"{alert_data['avg_if_score']:.3f}")
        
        with col2:
            st.metric("Maximum IF Score", f"{alert_data['max_if_score']:.3f}")
        
        # Show threshold info
        st.info(f"🔍 **Alert Thresholds:**\n- Low: < {IF_SCORE_THRESHOLDS['low']}\n- Medium: {IF_SCORE_THRESHOLDS['low']}-{IF_SCORE_THRESHOLDS['medium']}\n- High: {IF_SCORE_THRESHOLDS['medium']}-{IF_SCORE_THRESHOLDS['high']}\n- Critical: ≥ {IF_SCORE_THRESHOLDS['critical']}")
