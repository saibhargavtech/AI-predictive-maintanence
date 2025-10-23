# KPI Tile Components
# ===================

import streamlit as st
from config.constants import THRESHOLDS, TARGETS
from utils.calculations import get_status_color

def enhanced_kpi_tile(label, value, target=None, delta=None, delta_color="normal", col=None, status="normal"):
    """
    Enhanced KPI tile with progress bar and status indicator
    
    Args:
        label: KPI label
        value: Current value
        target: Target value (optional)
        delta: Change from previous period (optional)
        delta_color: Color for delta indicator
        col: Streamlit column to display in
        status: Status indicator ("normal", "warning", "critical", "idle")
    """
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

def render_kpi_tiles(anomaly_metrics, deviations):
    """
    Render all KPI tiles based on calculated metrics
    
    Args:
        anomaly_metrics: Dict with anomaly_pct, machines_alert, plants_affected
        deviations: Dict with max deviations for each KPI
    """
    from utils.calculations import get_status_color
    
    # Status determination based on manager's requirements
    anomaly_status = get_status_color(anomaly_metrics["anomaly_pct"], THRESHOLDS["anomaly_rate"])
    machines_status = get_status_color(anomaly_metrics["machines_alert"], THRESHOLDS["active_machines"])
    
    # Max deviation status (higher deviation = worse)
    vibration_dev_status = get_status_color(deviations["max_vibration_dev"], THRESHOLDS["vibration_deviation"])
    temp_dev_status = get_status_color(deviations["max_temp_dev"], THRESHOLDS["temp_deviation"])
    pressure_dev_status = get_status_color(deviations["max_pressure_dev"], THRESHOLDS["pressure_deviation"])
    power_dev_status = get_status_color(deviations["max_power_dev"], THRESHOLDS["power_deviation"])
    
    # Create columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Manager's Expected KPI Tiles (exactly as requested)
    enhanced_kpi_tile(
        "ANOMALIES LAST 24H", 
        f"{anomaly_metrics['anomaly_pct']:.1f}%", 
        target=TARGETS["anomaly_rate"], 
        delta=f"+{anomaly_metrics['anomaly_pct']-TARGETS['anomaly_rate']:.1f}%" if anomaly_metrics['anomaly_pct'] > TARGETS['anomaly_rate'] else f"{anomaly_metrics['anomaly_pct']-TARGETS['anomaly_rate']:.1f}%", 
        col=col1, 
        status=anomaly_status
    )
    
    enhanced_kpi_tile(
        "ACTIVE ALERT MACHINES", 
        anomaly_metrics["machines_alert"], 
        target=TARGETS["active_machines"], 
        delta=f"+{anomaly_metrics['machines_alert']-TARGETS['active_machines']}" if anomaly_metrics['machines_alert'] > TARGETS['active_machines'] else f"{anomaly_metrics['machines_alert']-TARGETS['active_machines']}", 
        col=col2, 
        status=machines_status
    )
    
    enhanced_kpi_tile(
        "MAX VIBRATION DEVIATION", 
        f"{deviations['max_vibration_dev']:.1f}mm/s", 
        target=TARGETS["vibration_deviation"], 
        delta=f"+{deviations['max_vibration_dev']-TARGETS['vibration_deviation']:.1f}" if deviations['max_vibration_dev'] > TARGETS['vibration_deviation'] else f"{deviations['max_vibration_dev']-TARGETS['vibration_deviation']:.1f}", 
        col=col3, 
        status=vibration_dev_status
    )
    
    enhanced_kpi_tile(
        "MAX TEMP DEVIATION", 
        f"{deviations['max_temp_dev']:.1f}°C", 
        target=TARGETS["temp_deviation"], 
        delta=f"+{deviations['max_temp_dev']-TARGETS['temp_deviation']:.1f}" if deviations['max_temp_dev'] > TARGETS['temp_deviation'] else f"{deviations['max_temp_dev']-TARGETS['temp_deviation']:.1f}", 
        col=col4, 
        status=temp_dev_status
    )
    
    enhanced_kpi_tile(
        "MAX PRESSURE DEVIATION", 
        f"{deviations['max_pressure_dev']:.1f}bar", 
        target=TARGETS["pressure_deviation"], 
        delta=f"+{deviations['max_pressure_dev']-TARGETS['pressure_deviation']:.1f}" if deviations['max_pressure_dev'] > TARGETS['pressure_deviation'] else f"{deviations['max_pressure_dev']-TARGETS['pressure_deviation']:.1f}", 
        col=col5, 
        status=pressure_dev_status
    )
    
    enhanced_kpi_tile(
        "MAX POWER DEVIATION", 
        f"{deviations['max_power_dev']:.1f}kW", 
        target=TARGETS["power_deviation"], 
        delta=f"+{deviations['max_power_dev']-TARGETS['power_deviation']:.1f}" if deviations['max_power_dev'] > TARGETS['power_deviation'] else f"{deviations['max_power_dev']-TARGETS['power_deviation']:.1f}", 
        col=col6, 
        status=power_dev_status
    )







