# Equipment Status Components
# ============================

import streamlit as st
import pandas as pd

def render_equipment_status(machine_status, sensor_status):
    """
    Render equipment status section with tabs for Crushers and Sensors
    
    Args:
        machine_status: DataFrame with machine status data
        sensor_status: Dict with sensor status data
    """
    st.markdown('<h2 class="section-header">🔧 Equipment Status</h2>', unsafe_allow_html=True)
    
    # Equipment tabs
    tab1, tab2 = st.tabs(["Crushers", "Sensors"])
    
    with tab1:
        render_crusher_status(machine_status)
    
    with tab2:
        render_sensor_status(sensor_status)

def render_crusher_status(machine_status):
    """Render crusher/machine status cards"""
    if len(machine_status) > 0:
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

def render_sensor_status(sensor_status):
    """Render sensor status cards"""
    if sensor_status:
        # Sensor status overview
        sensor_cols = st.columns(4)
        
        sensors = list(sensor_status.values())
        for idx, sensor in enumerate(sensors):
            with sensor_cols[idx]:
                status_colors = {"normal": "🟢", "warning": "🟡", "critical": "🔴"}
                st.markdown(f"""
                <div class="equipment-card">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{status_colors[sensor['status']]}</span>
                        <h4 style="margin: 0; color: #f9fafb;">{sensor['name']}</h4>
                    </div>
                    <div style="color: #f9fafb; font-size: 1.1rem; font-weight: 600;">
                        {sensor['value']:.1f} {sensor['unit']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No sensor data available.")







