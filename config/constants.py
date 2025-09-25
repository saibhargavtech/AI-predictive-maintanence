# Constants and Configuration
# ================================

# KPI Column Names
KPI_COLUMNS = [
    "Vibration_Level",
    "Motor_Temperature", 
    "Oil_Pressure",
    "Power_Consumption",
    "Throughput",
    "Bearing_Wear_Index",
    "if_score",
]

# Crusher-specific KPI Names and Units
CRUSHER_KPI_INFO = {
    "Vibration_Level": {"name": "Vibration", "unit": "mm/s", "description": "Crusher vibration level"},
    "Motor_Temperature": {"name": "Motor Temp", "unit": "°C", "description": "Motor operating temperature"},
    "Oil_Pressure": {"name": "Oil Pressure", "unit": "bar", "description": "Hydraulic oil pressure"},
    "Power_Consumption": {"name": "Power", "unit": "kW", "description": "Electrical power consumption"},
    "Throughput": {"name": "Crushing Rate", "unit": "tonnes/h", "description": "Material throughput rate"},
    "Bearing_Wear_Index": {"name": "Bearing Wear", "unit": "%", "description": "Bearing wear percentage"},
    "if_score": {"name": "IF Score", "unit": "", "description": "Isolation Forest anomaly score"},
}

# Categorical Columns
CATEGORICALS = ["Machine_Id", "Plant_Id"]

# KPI Thresholds for Color Coding
THRESHOLDS = {
    "anomaly_rate": [2, 5],           # <2% green, 2-5% amber, >5% red
    "active_machines": [2, 5],        # <2 green, 2-5 amber, >5 red
    "vibration_deviation": [2, 5],    # <2mm/s green, 2-5mm/s amber, >5mm/s red
    "temp_deviation": [10, 20],       # <10°C green, 10-20°C amber, >20°C red
    "pressure_deviation": [2, 4],     # <2bar green, 2-4bar amber, >4bar red
    "power_deviation": [50, 100],     # <50kW green, 50-100kW amber, >100kW red
}

# KPI Targets (Crusher-specific)
TARGETS = {
    "anomaly_rate": 2.0,                    # % anomalies
    "active_machines": 2,                    # machines under alert
    "vibration_deviation": 2.0,              # mm/s deviation
    "temp_deviation": 10.0,                   # °C deviation
    "pressure_deviation": 2.0,               # bar deviation
    "power_deviation": 50.0,                 # kW deviation
    "crushing_rate": 150.0,                  # tonnes/h target
    "efficiency": 85.0,                      # % efficiency target
}

# Equipment Status Thresholds (Crusher-specific)
EQUIPMENT_THRESHOLDS = {
    "vibration_warning": 6,      # mm/s - Crusher vibration warning
    "vibration_critical": 10,    # mm/s - Crusher vibration critical
    "temp_warning": 80,          # °C - Motor temperature warning
    "temp_critical": 90,         # °C - Motor temperature critical
    "pressure_min": 8,           # bar - Minimum hydraulic pressure
    "pressure_max": 17,          # bar - Maximum hydraulic pressure
    "power_warning": 500,        # kW - Power consumption warning
    "power_critical": 600,       # kW - Power consumption critical
    "crushing_rate_min": 100,    # tonnes/h - Minimum crushing rate
    "crushing_rate_max": 200,    # tonnes/h - Maximum crushing rate
    "efficiency_min": 70,        # % - Minimum efficiency
    "efficiency_warning": 80,   # % - Efficiency warning
}

# Isolation Forest Score Thresholds
IF_SCORE_THRESHOLDS = {
    "low": 0.5,                  # 0.5-0.6: Low risk
    "medium": 0.6,               # 0.6-0.7: Medium risk  
    "high": 0.7,                 # 0.7-0.8: High risk
    "critical": 0.7,             # 0.7+: Critical (red alerts)
}

# Alert Level Colors
ALERT_COLORS = {
    "low": "#10b981",            # Green
    "medium": "#f59e0b",         # Amber
    "high": "#ef4444",           # Red
    "critical": "#dc2626",       # Dark Red
}

# File Paths
DEFAULT_DATA_PATH = "data_with_anomaly.csv"
FALLBACK_DATA_PATH = "crusher_sensor_data.csv"
