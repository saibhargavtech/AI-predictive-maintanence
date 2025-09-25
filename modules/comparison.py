# Plant / Machine Comparison Page
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modular components
from config.constants import KPI_COLUMNS

def render_comparison_page():
    """Render the Plant / Machine Comparison page"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # --------------------------
    # Row 1 — Bar Chart: Average KPIs per Machine
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Average KPIs per Machine</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        available_kpis = [k for k in KPI_COLUMNS if k in df_f.columns]
        selected_kpi_bar = st.selectbox("Select KPI for Bar Chart", available_kpis, key="bar_kpi")
        chart_type = st.radio("Chart Type", ["Bar Chart", "Horizontal Bar Chart"])
    
    with col2:
        if selected_kpi_bar in df_f.columns and "Machine_Id" in df_f.columns:
            # Calculate average KPI per machine
            machine_avg = df_f.groupby("Machine_Id")[selected_kpi_bar].mean().reset_index()
            machine_avg = machine_avg.sort_values(selected_kpi_bar, ascending=False)
            
            if chart_type == "Bar Chart":
                fig_bar = px.bar(
                    machine_avg,
                    x="Machine_Id",
                    y=selected_kpi_bar,
                    title=f"Average {selected_kpi_bar} by Machine",
                    color=selected_kpi_bar,
                    color_continuous_scale="Viridis"
                )
            else:
                fig_bar = px.bar(
                    machine_avg,
                    x=selected_kpi_bar,
                    y="Machine_Id",
                    orientation='h',
                    title=f"Average {selected_kpi_bar} by Machine",
                    color=selected_kpi_bar,
                    color_continuous_scale="Viridis"
                )
            
            fig_bar.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(f"No data available for {selected_kpi_bar}")
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Pareto Chart: Anomaly Counts by Machine
    # --------------------------
    st.markdown('<h2 class="section-header">📈 Pareto Analysis - Anomalies by Machine</h2>', unsafe_allow_html=True)
    
    if "Machine_Id" in df_f.columns:
        # Calculate anomaly counts per machine
        machine_anomalies = (
            df_f.groupby("Machine_Id")["Anomaly_Flag"]
            .sum()
            .reset_index(name="Anomaly_Count")
            .sort_values("Anomaly_Count", ascending=False)
        )
        
        # Calculate cumulative percentage
        machine_anomalies["Cumulative_Count"] = machine_anomalies["Anomaly_Count"].cumsum()
        machine_anomalies["Cumulative_Percent"] = (machine_anomalies["Cumulative_Count"] / machine_anomalies["Anomaly_Count"].sum() * 100).round(1)
        
        # Create Pareto chart
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for anomaly counts
        fig_pareto.add_trace(
            go.Bar(
                x=machine_anomalies["Machine_Id"],
                y=machine_anomalies["Anomaly_Count"],
                name="Anomaly Count",
                marker_color="#3b82f6",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Line chart for cumulative percentage
        fig_pareto.add_trace(
            go.Scatter(
                x=machine_anomalies["Machine_Id"],
                y=machine_anomalies["Cumulative_Percent"],
                mode="lines+markers",
                name="Cumulative %",
                line=dict(color="#ef4444", width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        fig_pareto.update_xaxes(title_text="Machine ID")
        fig_pareto.update_yaxes(title_text="Anomaly Count", secondary_y=False)
        fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 100])
        
        fig_pareto.update_layout(
            height=500,
            title="Pareto Analysis: Machines by Anomaly Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Pareto insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_anomalies = machine_anomalies["Anomaly_Count"].sum()
            st.metric("Total Anomalies", int(total_anomalies))
        
        with col2:
            top_machine = machine_anomalies.iloc[0]["Machine_Id"]
            top_count = machine_anomalies.iloc[0]["Anomaly_Count"]
            st.metric("Top Problem Machine", f"{top_machine} ({top_count})")
        
        with col3:
            # Find 80% threshold
            eighty_percent_machines = len(machine_anomalies[machine_anomalies["Cumulative_Percent"] <= 80])
            st.metric("Machines Causing 80% Issues", eighty_percent_machines)
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Heatmap: Plant vs Anomaly Count
    # --------------------------
    st.markdown('<h2 class="section-header">🔥 Plant vs Anomaly Heatmap</h2>', unsafe_allow_html=True)
    
    if "Plant_Id" in df_f.columns and "Machine_Id" in df_f.columns:
        # Create plant vs machine anomaly heatmap
        plant_machine_anomalies = (
            df_f.groupby(["Plant_Id", "Machine_Id"])["Anomaly_Flag"]
            .sum()
            .reset_index(name="Anomaly_Count")
        )
        
        # Pivot for heatmap
        heatmap_data = plant_machine_anomalies.pivot(
            index="Plant_Id", 
            columns="Machine_Id", 
            values="Anomaly_Count"
        ).fillna(0)
        
        if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
            fig_heatmap = px.imshow(
                heatmap_data,
                aspect="auto",
                title="Anomaly Count by Plant and Machine",
                labels={'x': 'Machine ID', 'y': 'Plant ID', 'color': 'Anomaly Count'},
                color_continuous_scale="Reds"
            )
            fig_heatmap.update_layout(
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Not enough data to create plant-machine heatmap.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 4 — Plant Performance Comparison
    # --------------------------
    st.markdown('<h2 class="section-header">🏭 Plant Performance Comparison</h2>', unsafe_allow_html=True)
    
    if "Plant_Id" in df_f.columns:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            comparison_kpi = st.selectbox("Select KPI for Plant Comparison", available_kpis, key="plant_kpi")
            comparison_metric = st.radio("Comparison Metric", ["Mean", "Median", "Max", "Min"])
        
        with col2:
            if comparison_kpi in df_f.columns:
                # Calculate plant statistics
                if comparison_metric == "Mean":
                    plant_stats = df_f.groupby("Plant_Id")[comparison_kpi].mean().reset_index()
                elif comparison_metric == "Median":
                    plant_stats = df_f.groupby("Plant_Id")[comparison_kpi].median().reset_index()
                elif comparison_metric == "Max":
                    plant_stats = df_f.groupby("Plant_Id")[comparison_kpi].max().reset_index()
                else:  # Min
                    plant_stats = df_f.groupby("Plant_Id")[comparison_kpi].min().reset_index()
                
                plant_stats = plant_stats.sort_values(comparison_kpi, ascending=False)
                
                fig_plant = px.bar(
                    plant_stats,
                    x="Plant_Id",
                    y=comparison_kpi,
                    title=f"Plant Comparison - {comparison_metric} {comparison_kpi}",
                    color=comparison_kpi,
                    color_continuous_scale="Blues"
                )
                fig_plant.update_layout(
                    height=500,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_plant, use_container_width=True)
            else:
                st.info(f"No data available for {comparison_kpi}")
    
    st.markdown("---")
    
    # --------------------------
    # Row 5 — Machine Efficiency Ranking
    # --------------------------
    st.markdown('<h2 class="section-header">⚡ Machine Efficiency Ranking</h2>', unsafe_allow_html=True)
    
    if "Machine_Id" in df_f.columns:
        # Calculate efficiency score (inverse of anomaly rate)
        machine_efficiency = (
            df_f.groupby("Machine_Id")
            .agg({
                "Anomaly_Flag": ["sum", "count"],
                "Vibration_Level": "mean",
                "Motor_Temperature": "mean"
            })
            .reset_index()
        )
        
        # Flatten column names
        machine_efficiency.columns = ["Machine_Id", "Anomaly_Count", "Total_Records", "Avg_Vibration", "Avg_Temperature"]
        
        # Calculate efficiency metrics
        machine_efficiency["Anomaly_Rate"] = (machine_efficiency["Anomaly_Count"] / machine_efficiency["Total_Records"] * 100).round(2)
        machine_efficiency["Efficiency_Score"] = (100 - machine_efficiency["Anomaly_Rate"]).round(1)
        machine_efficiency["Efficiency_Score"] = machine_efficiency["Efficiency_Score"].clip(0, 100)
        
        # Sort by efficiency
        machine_efficiency = machine_efficiency.sort_values("Efficiency_Score", ascending=False)
        
        # Create efficiency ranking chart
        fig_efficiency = px.bar(
            machine_efficiency,
            x="Machine_Id",
            y="Efficiency_Score",
            title="Machine Efficiency Ranking (Based on Anomaly Rate)",
            color="Efficiency_Score",
            color_continuous_scale="RdYlGn"
        )
        fig_efficiency.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Top and bottom performers
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_machine = machine_efficiency.iloc[0]["Machine_Id"]
            best_score = machine_efficiency.iloc[0]["Efficiency_Score"]
            st.metric("Best Performer", f"{best_machine} ({best_score}%)")
        
        with col2:
            worst_machine = machine_efficiency.iloc[-1]["Machine_Id"]
            worst_score = machine_efficiency.iloc[-1]["Efficiency_Score"]
            st.metric("Needs Attention", f"{worst_machine} ({worst_score}%)")
        
        with col3:
            avg_efficiency = machine_efficiency["Efficiency_Score"].mean()
            st.metric("Average Efficiency", f"{avg_efficiency:.1f}%")
