# KPI Calculation Functions
# =========================

import pandas as pd
import streamlit as st
from config.constants import THRESHOLDS, TARGETS, EQUIPMENT_THRESHOLDS, IF_SCORE_THRESHOLDS, ALERT_COLORS

def get_status_color(value, thresholds, reverse=False):
    """
    Get status color based on value thresholds
    
    Args:
        value: Current value
        thresholds: [low_threshold, high_threshold]
        reverse: If True, higher values are better (e.g., efficiency)
    
    Returns:
        "normal", "warning", or "critical"
    """
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

def calculate_anomaly_metrics(df_f):
    """
    Calculate anomaly-related metrics
    
    Returns:
        dict: Contains anomaly_pct, machines_alert, plants_affected
    """
    # Calculate anomaly percentage from entire dataset (more meaningful for maintenance)
    anomaly_pct = (df_f["Anomaly_Flag"].sum() / len(df_f) * 100) if len(df_f) > 0 else 0
    
    # Count active machines under alert
    machines_alert = df_f.loc[df_f["Anomaly_Flag"] == 1, "Machine_Id"].nunique() if "Machine_Id" in df_f.columns else 0
    
    # Count plants affected
    plants_affected = df_f.loc[df_f["Anomaly_Flag"] == 1, "Plant_Id"].nunique() if "Plant_Id" in df_f.columns else 0
    
    return {
        "anomaly_pct": anomaly_pct,
        "machines_alert": machines_alert,
        "plants_affected": plants_affected
    }

def calculate_max_deviations(df_f):
    """
    Calculate max deviation in each KPI vs baseline (normal operation)
    
    Returns:
        dict: Contains max deviations for each KPI
    """
    # Baseline = normal operation (Anomaly_Flag = 0)
    normal_data = df_f[df_f["Anomaly_Flag"] == 0]
    
    deviations = {
        "max_vibration_dev": 0,
        "max_temp_dev": 0,
        "max_pressure_dev": 0,
        "max_power_dev": 0
    }
    
    if len(normal_data) > 0:
        # Calculate baseline values
        baseline_vibration = normal_data["Vibration_Level"].mean() if "Vibration_Level" in normal_data.columns else 0
        baseline_temp = normal_data["Motor_Temperature"].mean() if "Motor_Temperature" in normal_data.columns else 0
        baseline_pressure = normal_data["Oil_Pressure"].mean() if "Oil_Pressure" in normal_data.columns else 0
        baseline_power = normal_data["Power_Consumption"].mean() if "Power_Consumption" in normal_data.columns else 0
        
        # Calculate max deviations
        if "Vibration_Level" in df_f.columns:
            deviations["max_vibration_dev"] = abs(df_f["Vibration_Level"].max() - baseline_vibration)
        if "Motor_Temperature" in df_f.columns:
            deviations["max_temp_dev"] = abs(df_f["Motor_Temperature"].max() - baseline_temp)
        if "Oil_Pressure" in df_f.columns:
            deviations["max_pressure_dev"] = abs(df_f["Oil_Pressure"].max() - baseline_pressure)
        if "Power_Consumption" in df_f.columns:
            deviations["max_power_dev"] = abs(df_f["Power_Consumption"].max() - baseline_power)
    
    return deviations

def calculate_equipment_status(df_f):
    """
    Calculate equipment status for each machine
    
    Returns:
        DataFrame: Machine status with efficiency and status indicators
    """
    if "Machine_Id" not in df_f.columns:
        return pd.DataFrame()
    
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
        elif (row["Vibration_Level"] > EQUIPMENT_THRESHOLDS["vibration_warning"] or 
              row["Motor_Temperature"] > EQUIPMENT_THRESHOLDS["temp_warning"]):
            return "warning"
        else:
            return "normal"
    
    machine_status["status"] = machine_status.apply(get_machine_status, axis=1)
    machine_status["efficiency"] = 100 - (machine_status["Vibration_Level"] * 10)
    machine_status["efficiency"] = machine_status["efficiency"].clip(0, 100)
    
    return machine_status

def calculate_sensor_status(df_f):
    """
    Calculate sensor status overview
    
    Returns:
        dict: Sensor status for each sensor type
    """
    sensors = {}
    
    # Vibration sensor
    if "Vibration_Level" in df_f.columns:
        avg_vibration = df_f["Vibration_Level"].mean()
        vibration_status = "normal" if avg_vibration < 5 else "warning" if avg_vibration < 8 else "critical"
        sensors["vibration"] = {
            "name": "Vibration Sensor",
            "value": avg_vibration,
            "status": vibration_status,
            "unit": "mm/s"
        }
    
    # Temperature sensor
    if "Motor_Temperature" in df_f.columns:
        avg_temp = df_f["Motor_Temperature"].mean()
        temp_status = "normal" if avg_temp < 75 else "warning" if avg_temp < 85 else "critical"
        sensors["temperature"] = {
            "name": "Temperature Sensor",
            "value": avg_temp,
            "status": temp_status,
            "unit": "°C"
        }
    
    # Pressure sensor
    if "Oil_Pressure" in df_f.columns:
        avg_pressure = df_f["Oil_Pressure"].mean()
        pressure_status = "normal" if 10 < avg_pressure < 15 else "warning" if 8 < avg_pressure < 17 else "critical"
        sensors["pressure"] = {
            "name": "Pressure Sensor",
            "value": avg_pressure,
            "status": pressure_status,
            "unit": "bar"
        }
    
    # Power sensor
    if "Power_Consumption" in df_f.columns:
        avg_power = df_f["Power_Consumption"].mean()
        power_status = "normal" if avg_power < 500 else "warning" if avg_power < 600 else "critical"
        sensors["power"] = {
            "name": "Power Sensor",
            "value": avg_power,
            "status": power_status,
            "unit": "kW"
        }
    
    return sensors

def get_if_score_alert_level(if_score):
    """
    Get alert level based on IF score
    
    Args:
        if_score: Isolation Forest score (0-1)
    
    Returns:
        str: "low", "medium", "high", or "critical"
    """
    if if_score >= IF_SCORE_THRESHOLDS["critical"]:
        return "critical"
    elif if_score >= IF_SCORE_THRESHOLDS["high"]:
        return "high"
    elif if_score >= IF_SCORE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"

def get_critical_parameters(row):
    """
    Determine which parameters made the record critical
    
    Args:
        row: DataFrame row with sensor values
    
    Returns:
        list: List of critical parameters
    """
    critical_params = []
    
    # Check each parameter against thresholds
    if row.get("Vibration_Level", 0) > EQUIPMENT_THRESHOLDS["vibration_critical"]:
        critical_params.append(f"Vibration ({row['Vibration_Level']:.1f} mm/s)")
    
    if row.get("Motor_Temperature", 0) > EQUIPMENT_THRESHOLDS["temp_critical"]:
        critical_params.append(f"Temperature ({row['Motor_Temperature']:.1f}°C)")
    
    if row.get("Oil_Pressure", 0) < EQUIPMENT_THRESHOLDS["pressure_min"] or row.get("Oil_Pressure", 0) > EQUIPMENT_THRESHOLDS["pressure_max"]:
        critical_params.append(f"Pressure ({row['Oil_Pressure']:.1f} bar)")
    
    if row.get("Power_Consumption", 0) > EQUIPMENT_THRESHOLDS["power_critical"]:
        critical_params.append(f"Power ({row['Power_Consumption']:.1f} kW)")
    
    return critical_params

def calculate_if_score_alerts(df_f):
    """
    Calculate IF score-based alerts
    
    Args:
        df_f: Filtered DataFrame with if_score column
    
    Returns:
        dict: Alert statistics and critical records
    """
    if "if_score" not in df_f.columns:
        return {"error": "IF score column not found"}
    
    # Get critical alerts (IF score >= 0.7)
    critical_alerts = df_f[df_f["if_score"] >= IF_SCORE_THRESHOLDS["critical"]].copy()
    
    # Add alert level and critical parameters
    critical_alerts["alert_level"] = critical_alerts["if_score"].apply(get_if_score_alert_level)
    critical_alerts["critical_params"] = critical_alerts.apply(get_critical_parameters, axis=1)
    
    # Calculate statistics
    total_critical = len(critical_alerts)
    machines_critical = critical_alerts["Machine_Id"].nunique() if len(critical_alerts) > 0 else 0
    plants_critical = critical_alerts["Plant_Id"].nunique() if len(critical_alerts) > 0 else 0
    
    return {
        "total_critical": total_critical,
        "machines_critical": machines_critical,
        "plants_critical": plants_critical,
        "critical_records": critical_alerts,
        "avg_if_score": df_f["if_score"].mean(),
        "max_if_score": df_f["if_score"].max()
    }
