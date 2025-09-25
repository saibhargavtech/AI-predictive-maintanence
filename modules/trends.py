# Trend Analysis Page
# ===================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modular components
from config.constants import KPI_COLUMNS

def render_trends_page():
    """Render the Trend Analysis page"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # --------------------------
    # Row 1 — Line Charts (datetime_stamp vs KPIs)
    # --------------------------
    st.markdown('<h2 class="section-header">📈 Time Series Line Charts</h2>', unsafe_allow_html=True)
    
    # KPI selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_kpi = st.selectbox("Select KPI for Analysis", KPI_COLUMNS, key="trend_kpi")
        roll_window = st.slider("Rolling Window (hours)", min_value=1, max_value=72, value=24, step=1)
    
    with col2:
        if selected_kpi in df_f.columns:
            # Prepare data
            df_trend = df_f[["datetime_stamp", selected_kpi, "Anomaly_Flag", "Machine_Id", "Plant_Id"]].copy().sort_values("datetime_stamp")
            
            # Calculate rolling average
            df_trend["Rolling_Avg"] = df_trend[selected_kpi].rolling(roll_window, min_periods=max(1, roll_window//4)).mean()
            
            # Create line chart
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=df_trend["datetime_stamp"],
                y=df_trend[selected_kpi],
                mode="lines",
                name=f"{selected_kpi} (actual)",
                opacity=0.6,
                line=dict(color="#3b82f6")
            ))
            
            # Rolling average
            fig.add_trace(go.Scatter(
                x=df_trend["datetime_stamp"],
                y=df_trend["Rolling_Avg"],
                mode="lines",
                name=f"{selected_kpi} (rolling {roll_window}h avg)",
                line=dict(color="#ef4444", width=3)
            ))
            
            # Anomalies as markers
            df_anomalies = df_trend[df_trend["Anomaly_Flag"] == 1]
            if len(df_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=df_anomalies["datetime_stamp"],
                    y=df_anomalies[selected_kpi],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(size=10, symbol="x", color="#f59e0b")
                ))
            
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Time",
                yaxis_title=selected_kpi,
                title=f"{selected_kpi} Trend Analysis",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for {selected_kpi}")
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Multiple KPI Comparison
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Multiple KPI Comparison</h2>', unsafe_allow_html=True)
    
    # Select multiple KPIs
    available_kpis = [k for k in KPI_COLUMNS if k in df_f.columns]
    selected_kpis = st.multiselect("Select KPIs to Compare", available_kpis, default=available_kpis[:3])
    
    if selected_kpis:
        # Create subplot with multiple KPIs
        fig_sub = make_subplots(
            rows=len(selected_kpis), 
            cols=1,
            subplot_titles=selected_kpis,
            vertical_spacing=0.1
        )
        
        for i, kpi in enumerate(selected_kpis, 1):
            df_kpi = df_f[["datetime_stamp", kpi, "Anomaly_Flag"]].copy().sort_values("datetime_stamp")
            
            # Add actual values
            fig_sub.add_trace(
                go.Scatter(
                    x=df_kpi["datetime_stamp"],
                    y=df_kpi[kpi],
                    mode="lines",
                    name=f"{kpi} (actual)",
                    opacity=0.7,
                    line=dict(color=f"hsl({i*60}, 70%, 50%)")
                ),
                row=i, col=1
            )
            
            # Add anomalies
            df_anom = df_kpi[df_kpi["Anomaly_Flag"] == 1]
            if len(df_anom) > 0:
                fig_sub.add_trace(
                    go.Scatter(
                        x=df_anom["datetime_stamp"],
                        y=df_anom[kpi],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(size=8, symbol="x", color="#ef4444"),
                        showlegend=(i == 1)  # Only show legend for first subplot
                    ),
                    row=i, col=1
                )
        
        fig_sub.update_layout(
            height=400 * len(selected_kpis),
            title="Multi-KPI Trend Comparison",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_sub, use_container_width=True)
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Heatmap of KPI by Date/Hour
    # --------------------------
    st.markdown('<h2 class="section-header">🔥 KPI Heatmap by Date/Hour</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        heatmap_kpi = st.selectbox("Select KPI for Heatmap", available_kpis, key="heatmap_kpi")
    
    with col2:
        if heatmap_kpi in df_f.columns:
            # Create hour vs date heatmap
            df_heatmap = df_f.copy()
            df_heatmap['hour'] = df_heatmap['datetime_stamp'].dt.hour
            df_heatmap['date'] = df_heatmap['datetime_stamp'].dt.date
            
            # Pivot for heatmap
            heatmap_data = df_heatmap.pivot_table(
                index='hour', 
                columns='date', 
                values=heatmap_kpi, 
                aggfunc='mean'
            )
            
            if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
                fig_heatmap = px.imshow(
                    heatmap_data, 
                    aspect="auto",
                    title=f"{heatmap_kpi} Intensity by Hour and Date",
                    labels={'x': 'Date', 'y': 'Hour of Day', 'color': heatmap_kpi},
                    color_continuous_scale="RdYlBu_r"
                )
                fig_heatmap.update_layout(
                    height=500, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Not enough data to create heatmap.")
        else:
            st.info(f"No data available for {heatmap_kpi}")
