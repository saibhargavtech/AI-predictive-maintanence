# Navigation Configuration
# =======================

PAGES = {
    "Overview": {
        "icon": "📊",
        "description": "Leadership Snapshot",
        "file": "pages/overview.py"
    },
    "Alerts & Monitoring": {
        "icon": "🚨", 
        "description": "Anomaly Table & KPI Cards",
        "file": "pages/alerts.py"
    },
    "Trend Analysis": {
        "icon": "📈",
        "description": "Line Charts & Heatmaps", 
        "file": "pages/trends.py"
    },
    "Anomaly Visualization": {
        "icon": "⚠️",
        "description": "Scatter Plots & Boxplots",
        "file": "pages/anomaly_viz.py"
    },
    "Plant / Machine Comparison": {
        "icon": "🏭",
        "description": "Bar Charts & Pareto Analysis",
        "file": "pages/comparison.py"
    },
    "Root Cause Analysis and Predictive Indicators": {
        "icon": "🔍",
        "description": "Failure Pattern & Predictive Analysis",
        "file": "pages/root_cause_analysis.py"
    }
}

def render_sidebar_navigation():
    """Render the sidebar navigation menu"""
    import streamlit as st
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #4b5563; margin-bottom: 2rem;">
        <h2 style="color: #f9fafb; margin: 0; font-size: 1.5rem;">⚙️ Predictive Maintenance Dashboard</h2>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Time-series monitoring • Anomaly alerts • Machine & plant insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Get current page from URL or default to Overview
    current_page = st.session_state.get('current_page', 'Overview')
    
    for page_name, page_info in PAGES.items():
        is_selected = (current_page == page_name)
        
        if is_selected:
            st.sidebar.markdown(f"""
            <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                        border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem; 
                        border-left: 4px solid #60a5fa;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem;">{page_info['icon']}</span>
                    <span style="color: #f9fafb; font-weight: 600;">{page_name}</span>
                </div>
                <div style="color: #dbeafe; font-size: 0.8rem; margin-top: 0.25rem;">{page_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.sidebar.button(f"{page_info['icon']} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state['current_page'] = page_name
                st.rerun()
    
    st.sidebar.markdown("---")






