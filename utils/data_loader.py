# Data Loading Functions
# ======================

import os
import numpy as np
import pandas as pd
import streamlit as st
from config.constants import KPI_COLUMNS, DEFAULT_DATA_PATH, FALLBACK_DATA_PATH

def _load_default_sample():
    """
    Fall-back: small synthetic sample if no file provided.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-08-01", "2025-08-31 23:00:00", freq="H")
    df = pd.DataFrame({
        "datetime_stamp": rng.choice(dates, size=1000, replace=True),
        "Machine_Id": rng.choice([f"CR-{i:03d}" for i in range(101, 151)], size=1000),
        "Plant_Id": rng.choice([f"PL-{i:02d}" for i in range(1, 6)], size=1000),
        "Vibration_Level": rng.normal(4, 1, 1000).round(2),
        "Motor_Temperature": rng.normal(70, 5, 1000).round(2),
        "Oil_Pressure": rng.normal(12, 1, 1000).round(2),
        "Power_Consumption": rng.normal(450, 30, 1000).round(2),
        "Throughput": rng.normal(300, 20, 1000).round(2),
        "Bearing_Wear_Index": rng.normal(18, 5, 1000).round(2),
    })
    # 2% anomalies
    n_anom = max(1, int(0.02 * len(df)))
    anom_idx = rng.choice(df.index, size=n_anom, replace=False)
    df.loc[anom_idx, ["Vibration_Level"]] = rng.uniform(10, 20, n_anom).round(2)
    df.loc[anom_idx, ["Motor_Temperature"]] = rng.uniform(90, 120, n_anom).round(2)
    df.loc[anom_idx, ["Oil_Pressure"]] = rng.uniform(2, 6, n_anom).round(2)
    df.loc[anom_idx, ["Power_Consumption"]] = rng.uniform(600, 800, n_anom).round(2)
    df.loc[anom_idx, ["Throughput"]] = rng.uniform(100, 150, n_anom).round(2)
    df.loc[anom_idx, ["Bearing_Wear_Index"]] = rng.uniform(50, 100, n_anom).round(2)
    df["Anomaly_Flag"] = 0
    df.loc[anom_idx, "Anomaly_Flag"] = 1
    return df

@st.cache_data(show_spinner=False)
def load_data(file_buffer=None):
    """
    Load data from uploaded file, local CSV, or generate sample data.
    
    Priority:
    1. User uploaded file
    2. Local CSV file (crusher_sensor_data.csv)
    3. Fallback CSV file
    4. Synthetic sample data
    """
    # Priority 1: user upload
    if file_buffer is not None:
        if file_buffer.name.endswith('.xlsx'):
            df = pd.read_excel(file_buffer)
        else:
            df = pd.read_csv(file_buffer)
    else:
        # Priority 2: local CSV file
        if os.path.exists(DEFAULT_DATA_PATH):
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            # Priority 3: fallback CSV file if user dropped another CSV in the same folder
            if os.path.exists(FALLBACK_DATA_PATH):
                df = pd.read_csv(FALLBACK_DATA_PATH)
            else:
                df = _load_default_sample()

    # Basic normalization
    # tolerate both 'datetime_stamp' and 'timestamp' names
    if "datetime_stamp" not in df.columns:
        # try common alternatives
        for alt in ["timestamp", "time", "Datetime", "DateTime"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime_stamp"})
                break

    # Ensure required cols
    required = set(["datetime_stamp", "Machine_Id", "Plant_Id", "Anomaly_Flag"] + KPI_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        st.warning(f"Missing columns in data: {', '.join(missing)}")
        # try to continue if possible

    # Parse datetime
    if "datetime_stamp" in df.columns:
        df["datetime_stamp"] = pd.to_datetime(df["datetime_stamp"], errors="coerce")
        df = df.dropna(subset=["datetime_stamp"])

    # Coerce KPIs to numeric
    for k in KPI_COLUMNS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "Anomaly_Flag" in df.columns:
        df["Anomaly_Flag"] = pd.to_numeric(df["Anomaly_Flag"], errors="coerce").fillna(0).astype(int)
    
    # Ensure if_score is properly handled
    if "if_score" in df.columns:
        df["if_score"] = pd.to_numeric(df["if_score"], errors="coerce")
        # Fill any missing IF scores with median value
        if df["if_score"].isna().any():
            median_if_score = df["if_score"].median()
            df["if_score"] = df["if_score"].fillna(median_if_score)

    return df.sort_values("datetime_stamp")

def download_csv_button(df, label="Download CSV", filename="filtered_data.csv"):
    """Create a download button for CSV data"""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")
