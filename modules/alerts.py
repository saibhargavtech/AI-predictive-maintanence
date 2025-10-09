# Alerts & Monitoring Page
# ========================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    st.markdown("---")
    
    # --------------------------
    # Row 6 — ROI Analysis with Uptime/Downtime Data
    # --------------------------
    st.markdown('<h2 class="section-header">💰 ROI Analysis - Uptime & Downtime</h2>', unsafe_allow_html=True)
    
    # Load uptime/downtime data
    try:
        uptime_df = pd.read_csv('uptime_downtime_data.csv')
        uptime_df['datetime_stamp'] = pd.to_datetime(uptime_df['datetime_stamp'])
        
        # Filter uptime data based on current filters
        if len(df_f) > 0:
            # Get current date range from filtered data
            min_date = df_f['datetime_stamp'].min()
            max_date = df_f['datetime_stamp'].max()
            
            # Filter uptime data to match current date range
            uptime_filtered = uptime_df[
                (uptime_df['datetime_stamp'] >= min_date) & 
                (uptime_df['datetime_stamp'] <= max_date)
            ]
            
            # Apply plant filter if selected
            if 'Plant_Id' in df_f.columns and len(df_f['Plant_Id'].unique()) < len(uptime_df['Plant_Id'].unique()):
                selected_plants = df_f['Plant_Id'].unique()
                uptime_filtered = uptime_filtered[uptime_filtered['Plant_Id'].isin(selected_plants)]
            
            # Apply machine filter if selected
            if 'Machine_Id' in df_f.columns and len(df_f['Machine_Id'].unique()) < len(uptime_df['Machine_Id'].unique()):
                selected_machines = df_f['Machine_Id'].unique()
                uptime_filtered = uptime_filtered[uptime_filtered['Machine_Id'].isin(selected_machines)]
        else:
            uptime_filtered = uptime_df
        
        if len(uptime_filtered) > 0:
            # Calculate ROI metrics
            total_maintenance_cost = uptime_filtered['Maintenance_Cost_USD'].sum()
            total_productivity_loss = uptime_filtered['Productivity_Loss_USD'].sum()
            total_roi_impact = uptime_filtered['ROI_Impact_USD'].sum()
            total_downtime_hours = uptime_filtered[uptime_filtered['Status'] == 'Downtime']['Duration_Hours'].sum()
            total_running_hours = uptime_filtered[uptime_filtered['Status'] == 'Running']['Duration_Hours'].sum()
            total_hours = total_downtime_hours + total_running_hours
            uptime_percentage = (total_running_hours / total_hours * 100) if total_hours > 0 else 0
            avg_efficiency = uptime_filtered[uptime_filtered['Status'] == 'Running']['Efficiency_Percent'].mean()
            
            # ROI KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            
            # Uptime Percentage
            uptime_status = "normal" if uptime_percentage >= 90 else "warning" if uptime_percentage >= 80 else "critical"
            enhanced_kpi_tile(
                "UPTIME PERCENTAGE", 
                f"{uptime_percentage:.1f}%", 
                target=90.0, 
                delta=f"{uptime_percentage-90:.1f}%" if uptime_percentage >= 90 else f"{uptime_percentage-90:.1f}%", 
                col=col1, 
                status=uptime_status
            )
            
            # Average Efficiency
            efficiency_status = "normal" if avg_efficiency >= 90 else "warning" if avg_efficiency >= 85 else "critical"
            enhanced_kpi_tile(
                "AVERAGE EFFICIENCY", 
                f"{avg_efficiency:.1f}%", 
                target=90.0, 
                delta=f"{avg_efficiency-90:.1f}%" if avg_efficiency >= 90 else f"{avg_efficiency-90:.1f}%", 
                col=col2, 
                status=efficiency_status
            )
            
            # Total Maintenance Cost
            cost_status = "critical" if total_maintenance_cost > 100000 else "warning" if total_maintenance_cost > 50000 else "normal"
            enhanced_kpi_tile(
                "MAINTENANCE COST", 
                f"${total_maintenance_cost:,.0f}", 
                target=50000, 
                delta=f"+${total_maintenance_cost-50000:,.0f}" if total_maintenance_cost > 50000 else f"${total_maintenance_cost-50000:,.0f}", 
                col=col3, 
                status=cost_status
            )
            
            # ROI Impact
            roi_status = "critical" if total_roi_impact < -100000 else "warning" if total_roi_impact < -50000 else "normal"
            enhanced_kpi_tile(
                "ROI IMPACT", 
                f"${total_roi_impact:,.0f}", 
                target=0, 
                delta=f"{total_roi_impact:,.0f}", 
                col=col4, 
                status=roi_status
            )
            
            st.markdown("---")
            
            # Charts Section
            col1, col2 = st.columns(2)
            
            with col1:
                # Downtime Reasons Pie Chart
                downtime_reasons = uptime_filtered[uptime_filtered['Status'] == 'Downtime']['Downtime_Reason'].value_counts()
                if len(downtime_reasons) > 0:
                    fig_pie = px.pie(
                        values=downtime_reasons.values,
                        names=downtime_reasons.index,
                        title="Downtime Reasons Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Daily ROI Impact Trend
                daily_roi = uptime_filtered.groupby(uptime_filtered['datetime_stamp'].dt.date)['ROI_Impact_USD'].sum().reset_index()
                daily_roi.columns = ['Date', 'ROI_Impact']
                
                fig_line = px.line(
                    daily_roi, 
                    x='Date', 
                    y='ROI_Impact',
                    title="Daily ROI Impact Trend",
                    color_discrete_sequence=['#ef4444']
                )
                fig_line.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white',
                    xaxis_title="Date",
                    yaxis_title="ROI Impact (USD)"
                )
                fig_line.update_traces(line=dict(width=3))
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Plant-wise Analysis
            st.markdown("### 🏭 Plant-wise ROI Analysis")
            plant_analysis = uptime_filtered.groupby('Plant_Id').agg({
                'ROI_Impact_USD': 'sum',
                'Maintenance_Cost_USD': 'sum',
                'Productivity_Loss_USD': 'sum',
                'Duration_Hours': 'sum',
                'Status': lambda x: (x == 'Running').sum() / len(x) * 100  # Uptime percentage
            }).round(2)
            plant_analysis.columns = ['ROI Impact', 'Maintenance Cost', 'Productivity Loss', 'Total Hours', 'Uptime %']
            plant_analysis = plant_analysis.sort_values('ROI Impact')
            
            # Style the dataframe
            def highlight_roi(row):
                if row['ROI Impact'] < -100000:
                    return ['background-color: #2d1b1b; color: #ffffff'] * len(row)
                elif row['ROI Impact'] < -50000:
                    return ['background-color: #2d2419; color: #ffffff'] * len(row)
                else:
                    return ['background-color: #1a2d1a; color: #ffffff'] * len(row)
            
            styled_plant_df = plant_analysis.style.apply(highlight_roi, axis=1)
            st.dataframe(styled_plant_df, use_container_width=True)
            
            # Machine Performance Ranking
            st.markdown("### 🏆 Machine Performance Ranking")
            machine_analysis = uptime_filtered.groupby('Machine_Id').agg({
                'ROI_Impact_USD': 'sum',
                'Efficiency_Percent': 'mean',
                'Status': lambda x: (x == 'Running').sum() / len(x) * 100
            }).round(2)
            machine_analysis.columns = ['ROI Impact', 'Avg Efficiency %', 'Uptime %']
            
            # Sort by ROI Impact (ascending - least negative first = best performers)
            machine_analysis_sorted = machine_analysis.sort_values('ROI Impact', ascending=True)
            
            # Show top and bottom machines
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Performing Machines** (Best ROI)")
                # Top performers = least negative ROI values (closest to 0)
                top_performers = machine_analysis_sorted.head(10)
                st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.markdown("**Bottom Performing Machines** (Worst ROI)")
                # Bottom performers = most negative ROI values (worst performers)
                bottom_performers = machine_analysis_sorted.tail(10)
                st.dataframe(bottom_performers, use_container_width=True)
            
            # Show summary if filtered data has few machines
            unique_machines = len(machine_analysis)
            if unique_machines < 20:
                st.info(f"📊 Showing performance for {unique_machines} machines based on current filters. Use 'All' filters to see complete ranking of all 50 machines.")
            
            # Download button for ROI data
            from utils.data_loader import download_csv_button
            download_csv_button(
                uptime_filtered, 
                "📥 Download ROI Analysis Data as CSV", 
                "roi_analysis_data.csv"
            )
            
        else:
            st.info("No uptime/downtime data available for the selected filters.")
            
    except FileNotFoundError:
        st.warning("Uptime/downtime data file not found. Please ensure 'uptime_downtime_data.csv' is available.")
    except Exception as e:
        st.error(f"Error loading uptime/downtime data: {str(e)}")
