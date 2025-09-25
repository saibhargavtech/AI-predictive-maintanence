# Anomaly Visualization Page
# ==========================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modular components
from config.constants import KPI_COLUMNS

def render_anomaly_viz_page():
    """Render the Anomaly Visualization page"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # --------------------------
    # Row 1 — Scatter Plot (KPI vs KPI with Anomalies Highlighted)
    # --------------------------
    st.markdown('<h2 class="section-header">🔍 Scatter Plot Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        available_kpis = [k for k in KPI_COLUMNS if k in df_f.columns]
        x_kpi = st.selectbox("X-axis KPI", available_kpis, key="scatter_x")
    
    with col2:
        y_kpi = st.selectbox("Y-axis KPI", [k for k in available_kpis if k != x_kpi], key="scatter_y")
    
    with col3:
        if x_kpi and y_kpi and x_kpi != y_kpi:
            # Create scatter plot
            fig_scatter = px.scatter(
                df_f, 
                x=x_kpi, 
                y=y_kpi,
                color=df_f["Anomaly_Flag"].map({0: "Normal", 1: "Anomaly"}) if "Anomaly_Flag" in df_f.columns else None,
                hover_data=["datetime_stamp", "Machine_Id", "Plant_Id"] if "Plant_Id" in df_f.columns else ["datetime_stamp", "Machine_Id"],
                opacity=0.7,
                title=f"{x_kpi} vs {y_kpi} (Anomalies Highlighted)",
                color_discrete_map={"Normal": "#3b82f6", "Anomaly": "#ef4444"}
            )
            
            fig_scatter.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Please select different KPIs for X and Y axes.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Boxplots of Each KPI with Anomalies Overlayed
    # --------------------------
    st.markdown('<h2 class="section-header">📦 KPI Distribution Analysis</h2>', unsafe_allow_html=True)
    
    # Select KPIs for boxplot
    selected_kpis_box = st.multiselect("Select KPIs for Boxplot Analysis", available_kpis, default=available_kpis[:4])
    
    if selected_kpis_box:
        # Create subplot for multiple boxplots
        fig_box = make_subplots(
            rows=1, 
            cols=len(selected_kpis_box),
            subplot_titles=selected_kpis_box
        )
        
        for i, kpi in enumerate(selected_kpis_box, 1):
            # Normal data
            normal_data = df_f[df_f["Anomaly_Flag"] == 0][kpi].dropna()
            if len(normal_data) > 0:
                fig_box.add_trace(
                    go.Box(
                        y=normal_data,
                        name="Normal",
                        marker_color="#3b82f6",
                        opacity=0.7
                    ),
                    row=1, col=i
                )
            
            # Anomaly data
            anomaly_data = df_f[df_f["Anomaly_Flag"] == 1][kpi].dropna()
            if len(anomaly_data) > 0:
                fig_box.add_trace(
                    go.Box(
                        y=anomaly_data,
                        name="Anomaly",
                        marker_color="#ef4444",
                        opacity=0.7
                    ),
                    row=1, col=i
                )
        
        fig_box.update_layout(
            height=500,
            title="KPI Distributions: Normal vs Anomaly",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Anomaly Frequency Trend (Per Day)
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Daily Anomaly Frequency</h2>', unsafe_allow_html=True)
    
    if len(df_f) > 0:
        # Calculate daily anomaly counts
        daily_anomalies = (
            df_f.assign(day=df_f["datetime_stamp"].dt.date)
               .groupby("day")["Anomaly_Flag"]
               .agg(['sum', 'count'])
               .reset_index()
        )
        daily_anomalies.columns = ['day', 'anomaly_count', 'total_count']
        daily_anomalies['anomaly_rate'] = (daily_anomalies['anomaly_count'] / daily_anomalies['total_count'] * 100).round(2)
        
        # Create dual-axis chart
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Anomaly count (bars)
        fig_dual.add_trace(
            go.Bar(
                x=daily_anomalies['day'],
                y=daily_anomalies['anomaly_count'],
                name="Anomaly Count",
                marker_color="#ef4444",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Anomaly rate (line)
        fig_dual.add_trace(
            go.Scatter(
                x=daily_anomalies['day'],
                y=daily_anomalies['anomaly_rate'],
                mode="lines+markers",
                name="Anomaly Rate %",
                line=dict(color="#f59e0b", width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig_dual.update_xaxes(title_text="Date")
        fig_dual.update_yaxes(title_text="Anomaly Count", secondary_y=False)
        fig_dual.update_yaxes(title_text="Anomaly Rate (%)", secondary_y=True)
        
        fig_dual.update_layout(
            height=500,
            title="Daily Anomaly Frequency and Rate",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_dual, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_daily_anomalies = daily_anomalies['anomaly_count'].mean()
            st.metric("Avg Daily Anomalies", f"{avg_daily_anomalies:.1f}")
        
        with col2:
            max_daily_anomalies = daily_anomalies['anomaly_count'].max()
            st.metric("Max Daily Anomalies", int(max_daily_anomalies))
        
        with col3:
            avg_anomaly_rate = daily_anomalies['anomaly_rate'].mean()
            st.metric("Avg Anomaly Rate", f"{avg_anomaly_rate:.1f}%")
        
        with col4:
            max_anomaly_rate = daily_anomalies['anomaly_rate'].max()
            st.metric("Max Anomaly Rate", f"{max_anomaly_rate:.1f}%")
    
    st.markdown("---")
    
    # --------------------------
    # Row 4 — Anomaly Pattern Analysis
    # --------------------------
    st.markdown('<h2 class="section-header">🔍 Anomaly Pattern Analysis</h2>', unsafe_allow_html=True)
    
    if len(df_f) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomalies by hour of day
            hourly_anomalies = (
                df_f[df_f["Anomaly_Flag"] == 1]
                .assign(hour=df_f["datetime_stamp"].dt.hour)
                .groupby("hour")["Anomaly_Flag"]
                .count()
                .reset_index(name="anomaly_count")
            )
            
            fig_hourly = px.bar(
                hourly_anomalies,
                x="hour",
                y="anomaly_count",
                title="Anomalies by Hour of Day",
                color="anomaly_count",
                color_continuous_scale="Reds"
            )
            fig_hourly.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Anomalies by day of week
            daily_anomalies = (
                df_f[df_f["Anomaly_Flag"] == 1]
                .assign(day_of_week=df_f["datetime_stamp"].dt.day_name())
                .groupby("day_of_week")["Anomaly_Flag"]
                .count()
                .reset_index(name="anomaly_count")
            )
            
            # Order days of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_anomalies['day_of_week'] = pd.Categorical(daily_anomalies['day_of_week'], categories=day_order, ordered=True)
            daily_anomalies = daily_anomalies.sort_values('day_of_week')
            
            fig_daily = px.bar(
                daily_anomalies,
                x="day_of_week",
                y="anomaly_count",
                title="Anomalies by Day of Week",
                color="anomaly_count",
                color_continuous_scale="Oranges"
            )
            fig_daily.update_layout(
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
