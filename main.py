# Main Dashboard Application with Navigation
# ==========================================

import streamlit as st
import pandas as pd

# Import our modular components
from config.styles import apply_custom_styles
from config.constants import KPI_COLUMNS
from config.navigation import render_sidebar_navigation, PAGES
from utils.data_loader import load_data

# Import page modules
# from pages.overview import render_overview_page
# from pages.alerts import render_alerts_page
# from pages.trends import render_trends_page
# from pages.anomaly_viz import render_anomaly_viz_page
# from pages.comparison import render_comparison_page

from modules.overview import render_overview_page
from modules.alerts import render_alerts_page
from modules.trends import render_trends_page
from modules.anomaly_viz import render_anomaly_viz_page
from modules.comparison import render_comparison_page
from modules.root_cause_analysis import render_root_cause_analysis_page

def render_model_development_page():
    """Render the Model Development Centre page"""
    # Import ML backend
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Backend'))
    
    try:
        from streamlit_ml_backend import render_ml_backend
        # Directly render the ML backend
        render_ml_backend()
    except Exception as e:
        st.error(f"Error loading ML backend: {str(e)}")
        st.info("Please ensure the ML backend files are available in the Backend directory.")
        
        # Fallback to simple page
        st.markdown("## 🤖 Model Development Centre")
        st.markdown("**Anomaly & Novelty Detection Pipeline**")
        
        st.info("""
        🚀 **Welcome to the Model Development Centre!**
        
        This section provides advanced machine learning capabilities for predictive maintenance.
        """)
        
        st.markdown("### 📋 Available ML Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔍 Anomaly Detection")
            st.markdown("- Isolation Forest Algorithm")
            st.markdown("- PCA Visualization")
            st.markdown("- Anomaly Scoring")
            st.markdown("- CSV Export")
        
        with col2:
            st.markdown("#### 🆕 Novelty Detection")
            st.markdown("- Autoencoder Training")
            st.markdown("- PyTorch Integration")
            st.markdown("- Reconstruction Errors")
            st.markdown("- Threshold Analysis")
        
        with col3:
            st.markdown("#### 📊 Full Pipeline")
            st.markdown("- Complete Workflow")
            st.markdown("- All Algorithms")
            st.markdown("- Interactive Training")
            st.markdown("- Results Comparison")

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Apply custom styling first (includes aggressive CSS to hide unwanted elements)
apply_custom_styles()

# --------------------------
# Sidebar: Navigation & Data
# --------------------------
render_sidebar_navigation()

# Load default data (no upload needed)
df = load_data(None)

# System Status
st.sidebar.markdown("### 🔧 System Status")
if len(df) > 0:
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="status-dot status-green"></span>
        <span style="color: #f9fafb;">All Systems Operational</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="status-dot status-red"></span>
        <span style="color: #f9fafb;">No Data Available</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"**📊 Detected Records:** {len(df)}")
st.sidebar.markdown(f"**🕒 Last Update:** {pd.Timestamp.now().strftime('%H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Data Filters")

# Date filters
if len(df):
    min_date = df["datetime_stamp"].min()
    max_date = df["datetime_stamp"].max()
else:
    min_date = pd.Timestamp("2025-08-01")
    max_date = pd.Timestamp("2025-08-31 23:00:00")

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

plant_options = sorted(df["Plant_Id"].dropna().unique()) if "Plant_Id" in df.columns else []
plant_sel = st.sidebar.multiselect("Plant_Id filter", plant_options, default=plant_options[:3] if plant_options else [])

machine_options = sorted(df["Machine_Id"].dropna().unique()) if "Machine_Id" in df.columns else []
machine_sel = st.sidebar.multiselect("Machine_Id filter", machine_options, default=machine_options[:10] if machine_options else [])

st.sidebar.caption("Expected columns: datetime_stamp, Machine_Id, Plant_Id, "
                   + ", ".join(KPI_COLUMNS) + ", Anomaly_Flag, if_score")

# Apply filters
df_f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_f = df_f[(df_f["datetime_stamp"] >= start_dt) & (df_f["datetime_stamp"] <= end_dt)]

if plant_sel and "Plant_Id" in df_f.columns:
    df_f = df_f[df_f["Plant_Id"].isin(plant_sel)]
if machine_sel and "Machine_Id" in df_f.columns:
    df_f = df_f[df_f["Machine_Id"].isin(machine_sel)]

# Store filtered data in session state for pages to access
st.session_state['filtered_data'] = df_f

# --------------------------
# Main Content Area
# --------------------------

# Get current page
current_page = st.session_state.get('current_page', 'Overview')

# Render the appropriate page
if current_page == 'Overview':
    render_overview_page()
elif current_page == 'Alerts & Monitoring':
    render_alerts_page()
elif current_page == 'Trend Analysis':
    render_trends_page()
elif current_page == 'Anomaly Visualization':
    render_anomaly_viz_page()
elif current_page == 'Plant / Machine Comparison':
    render_comparison_page()
elif current_page == 'Root Cause Analysis and Predictive Indicators':
    render_root_cause_analysis_page()
elif current_page == 'Model Development Centre':
    render_model_development_page()
else:
    # Default to Overview
    st.session_state['current_page'] = 'Overview'
    render_overview_page()

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption(
    "💡 **Tip:** Use the sidebar navigation to switch between different analysis views. "
    "Each page provides specialized insights for different aspects of your predictive maintenance data."
)