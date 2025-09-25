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
from pages.overview import render_overview_page
from pages.alerts import render_alerts_page
from pages.trends import render_trends_page
from pages.anomaly_viz import render_anomaly_viz_page
from pages.comparison import render_comparison_page

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

# Hide Streamlit's default elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stToolbar"] {display:none;}
    .stApp > header {display:none;}
    .stApp > div[data-testid="stHeader"] {display:none;}
    .stApp > div[data-testid="stToolbar"] {display:none;}
    .stApp > div[data-testid="stDecoration"] {display:none;}
    .stApp > div[data-testid="stStatusWidget"] {display:none;}
    .stApp > div[data-testid="stAppViewContainer"] > div[data-testid="stHeader"] {display:none;}
    .stApp > div[data-testid="stAppViewContainer"] > div[data-testid="stToolbar"] {display:none;}
    .stApp > div[data-testid="stAppViewContainer"] > div[data-testid="stDecoration"] {display:none;}
    .stApp > div[data-testid="stAppViewContainer"] > div[data-testid="stStatusWidget"] {display:none;}
    
    /* Hide additional unwanted elements */
    .stApp > div[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] > div:first-child {display:none;}
    .stApp > div[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] > div:nth-child(2) {display:none;}
    div[data-testid="stSidebar"] > div:first-child {display:none;}
    div[data-testid="stSidebar"] > div:nth-child(2) {display:none;}
    
    /* Hide search and navigation elements */
    div[data-testid="stSidebar"] input {display:none;}
    div[data-testid="stSidebar"] button[title="Search"] {display:none;}
    div[data-testid="stSidebar"] div[role="search"] {display:none;}
    
    /* Hide any remaining unwanted elements */
    .stApp > div[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] > div[style*="search"] {display:none;}
    .stApp > div[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] > div[style*="main"] {display:none;}
    
    /* Hide all possible navigation/search elements */
    div[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebar"] > div:nth-child(2),
    div[data-testid="stSidebar"] > div:nth-child(3),
    div[data-testid="stSidebar"] > div:nth-child(4) {display:none !important;}
    
    /* Hide any text containing "main", "alerts", "anomaly", etc. */
    div[data-testid="stSidebar"] *:contains("main"),
    div[data-testid="stSidebar"] *:contains("alerts"),
    div[data-testid="stSidebar"] *:contains("anomaly"),
    div[data-testid="stSidebar"] *:contains("comparison"),
    div[data-testid="stSidebar"] *:contains("overview"),
    div[data-testid="stSidebar"] *:contains("trends") {display:none !important;}
    
    /* Force hide any input fields or search boxes */
    input[type="text"],
    input[type="search"],
    input[placeholder*="search"],
    input[placeholder*="Search"],
    button[aria-label*="search"],
    button[aria-label*="Search"] {display:none !important;}
</style>
""", unsafe_allow_html=True)

# Apply custom styling
apply_custom_styles()

# --------------------------
# Sidebar: Navigation & Data
# --------------------------
render_sidebar_navigation()

# Data upload and filtering
uploaded = st.sidebar.file_uploader(
    "📁 Upload Model Output File CSV",
    type=["csv"],
    help="Upload your model output CSV file with IF scores"
)
df = load_data(uploaded)

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