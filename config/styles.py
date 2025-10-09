# CSS Styling for Dashboard
# =========================

# CSS Styling for Dashboard
# =========================

CUSTOM_CSS = """
<style>
    /* Hide Streamlit default navigation elements but keep sidebar functional */
    div[data-testid="stSidebar"] > div:first-child > div:first-child,
    div[data-testid="stSidebar"] > div:first-child > div:nth-child(2),
    div[data-testid="stSidebar"] > div:first-child > div:nth-child(3) {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    
    /* Hide any elements containing unwanted text */
    div[data-testid="stSidebar"] *:contains("main"),
    div[data-testid="stSidebar"] *:contains("alerts"),
    div[data-testid="stSidebar"] *:contains("anomaly"),
    div[data-testid="stSidebar"] *:contains("comparison"),
    div[data-testid="stSidebar"] *:contains("overview"),
    div[data-testid="stSidebar"] *:contains("trends") {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    
    /* Hide any input fields or search elements */
    div[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] button[title="Search"],
    div[data-testid="stSidebar"] div[role="search"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
    }
    
    /* Keep sidebar content visible and functional */
    div[data-testid="stSidebarContent"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {display:none !important;}
    div[data-testid="stToolbar"] {display:none !important;}
    .stApp > header {display:none !important;}
    .stApp > div[data-testid="stHeader"] {display:none !important;}
    .stApp > div[data-testid="stToolbar"] {display:none !important;}
    .stApp > div[data-testid="stDecoration"] {display:none !important;}
    .stApp > div[data-testid="stStatusWidget"] {display:none !important;}
    
    /* Main theme colors */
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --text-color: #f9fafb;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--primary-color);
    }
    
    /* KPI tiles styling */
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #4b5563;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Progress bars */
    .progress-bar {
        height: 8px;
        background-color: #374151;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .progress-green { background-color: var(--success-color); }
    .progress-yellow { background-color: var(--warning-color); }
    .progress-red { background-color: var(--danger-color); }
    
    /* Status indicators */
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-green { background-color: var(--success-color); }
    .status-yellow { background-color: var(--warning-color); }
    .status-red { background-color: var(--danger-color); }
    .status-gray { background-color: #6b7280; }
    
    /* Section headers */
    .section-header {
        color: var(--text-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }
    
    /* Alert styling */
    .alert-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid var(--danger-color);
        margin-bottom: 1rem;
    }
    
    /* Equipment cards */
    .equipment-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #4b5563;
        margin-bottom: 0.5rem;
        transition: transform 0.2s ease;
    }
    
    .equipment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
"""

def apply_custom_styles():
    """Apply custom CSS styles to the dashboard"""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
