# Root Cause Analysis Page
# ========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Import our modular components
from config.constants import KPI_COLUMNS, CRUSHER_KPI_INFO, THRESHOLDS, TARGETS
from utils.calculations import get_status_color
from components.kpi_tiles import enhanced_kpi_tile

def render_root_cause_analysis_page():
    """Render the Root Cause Analysis page"""
    
    # Get data from session state
    df_f = st.session_state.get('filtered_data', pd.DataFrame())
    
    if len(df_f) == 0:
        st.warning("No data available. Please upload a CSV file in the sidebar.")
        return
    
    # Check if we have anomaly data
    if 'Anomaly_Flag' not in df_f.columns or df_f['Anomaly_Flag'].sum() == 0:
        st.warning("No anomaly data found. Root Cause Analysis requires anomaly labels.")
        return
    
    # --------------------------
    # Page Header
    # --------------------------
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="color: #f9fafb; margin: 0; font-size: 2.5rem; font-weight: 700;">🔍 Root Cause Analysis and Predictive Indicators</h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Advanced failure pattern analysis • Correlation insights • Machine-specific predictive indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --------------------------
    # Row 1 — RCA Summary KPIs
    # --------------------------
    st.markdown('<h2 class="section-header">📊 Root Cause Analysis Summary</h2>', unsafe_allow_html=True)
    
    # Calculate RCA metrics
    total_anomalies = df_f['Anomaly_Flag'].sum()
    anomaly_rate = (total_anomalies / len(df_f)) * 100
    affected_machines = df_f[df_f['Anomaly_Flag'] == 1]['Machine_Id'].nunique()
    affected_plants = df_f[df_f['Anomaly_Flag'] == 1]['Plant_Id'].nunique()
    
    # Calculate correlation strength with anomalies
    kpi_correlations = {}
    for kpi in KPI_COLUMNS:
        if kpi in df_f.columns and kpi != 'if_score':
            corr = abs(df_f[kpi].corr(df_f['Anomaly_Flag']))
            kpi_correlations[kpi] = corr
    
    # Find strongest predictor
    strongest_predictor = max(kpi_correlations.items(), key=lambda x: x[1]) if kpi_correlations else ('N/A', 0)
    
    # Create RCA KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    
    enhanced_kpi_tile(
        "TOTAL ANOMALIES", 
        total_anomalies, 
        target=10,
        delta=f"+{total_anomalies-10}" if total_anomalies > 10 else f"{total_anomalies-10}",
        col=col1, 
        status=get_status_color(total_anomalies, [5, 15])
    )
    
    enhanced_kpi_tile(
        "ANOMALY RATE", 
        f"{anomaly_rate:.1f}%", 
        target=2.0,
        delta=f"+{anomaly_rate-2:.1f}%" if anomaly_rate > 2 else f"{anomaly_rate-2:.1f}%",
        col=col2, 
        status=get_status_color(anomaly_rate, [1, 3])
    )
    
    enhanced_kpi_tile(
        "AFFECTED MACHINES", 
        affected_machines, 
        target=3,
        delta=f"+{affected_machines-3}" if affected_machines > 3 else f"{affected_machines-3}",
        col=col3, 
        status=get_status_color(affected_machines, [2, 5])
    )
    
    enhanced_kpi_tile(
        "STRONGEST PREDICTOR", 
        f"{strongest_predictor[0].replace('_', ' ')}", 
        target=0.7,
        delta=f"{strongest_predictor[1]:.3f}",
        col=col4, 
        status=get_status_color(strongest_predictor[1], [0.5, 0.7])
    )
    
    st.markdown("---")
    
    # --------------------------
    # Row 2 — Correlation Analysis
    # --------------------------
    st.markdown('<h2 class="section-header">🔗 KPI Correlation Analysis</h2>', unsafe_allow_html=True)
    
    col_corr1, col_corr2 = st.columns([1, 1])
    
    with col_corr1:
        st.markdown("**📊 Correlation Matrix with Anomalies**")
        
        # Create correlation matrix
        kpi_cols_for_corr = [k for k in KPI_COLUMNS if k in df_f.columns and k != 'if_score']
        corr_data = df_f[kpi_cols_for_corr + ['Anomaly_Flag']].corr()
        
        # Create heatmap
        fig_corr = px.imshow(
            corr_data,
            text_auto=".3f",
            aspect="auto",
            title="KPI Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col_corr2:
        st.markdown("**📈 Feature Importance Ranking**")
        
        # Calculate feature importance using correlation
        feature_importance = []
        for kpi in kpi_cols_for_corr:
            corr = abs(df_f[kpi].corr(df_f['Anomaly_Flag']))
            feature_importance.append({
                'KPI': kpi.replace('_', ' '),
                'Correlation': corr,
                'Impact': 'High' if corr > 0.7 else 'Medium' if corr > 0.4 else 'Low'
            })
        
        importance_df = pd.DataFrame(feature_importance).sort_values('Correlation', ascending=True)
        
        # Create horizontal bar chart
        fig_importance = px.bar(
            importance_df, 
            x='Correlation', 
            y='KPI',
            color='Impact',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
            title="Feature Importance (Correlation with Anomalies)",
            orientation='h'
        )
        fig_importance.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # --------------------------
    # Row 3 — Failure Pattern Analysis
    # --------------------------
    st.markdown('<h2 class="section-header">🔍 Failure Pattern Analysis</h2>', unsafe_allow_html=True)
    
    # Get anomaly data
    anomaly_data = df_f[df_f['Anomaly_Flag'] == 1].copy()
    normal_data = df_f[df_f['Anomaly_Flag'] == 0].copy()
    
    col_pattern1, col_pattern2 = st.columns([1, 1])
    
    with col_pattern1:
        st.markdown("**📊 Normal vs Anomaly KPI Comparison**")
        
        # Create comparison chart
        comparison_data = []
        for kpi in kpi_cols_for_corr:
            normal_mean = normal_data[kpi].mean()
            anomaly_mean = anomaly_data[kpi].mean()
            comparison_data.append({
                'KPI': kpi.replace('_', ' '),
                'Normal': normal_mean,
                'Anomaly': anomaly_mean,
                'Difference': anomaly_mean - normal_mean
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Normal Operation',
            x=comp_df['KPI'],
            y=comp_df['Normal'],
            marker_color='#10b981'
        ))
        fig_comp.add_trace(go.Bar(
            name='During Anomaly',
            x=comp_df['KPI'],
            y=comp_df['Anomaly'],
            marker_color='#ef4444'
        ))
        
        fig_comp.update_layout(
            title="KPI Values: Normal vs Anomaly Conditions",
            barmode='group',
            height=400,
            margin=dict(l=20, r=20, t=50, b=80)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with col_pattern2:
        st.markdown("**🎯 Anomaly Signature Analysis**")
        
        # Calculate anomaly signature (average values during anomalies)
        signature_data = []
        for kpi in kpi_cols_for_corr:
            normal_std = normal_data[kpi].std()
            anomaly_mean = anomaly_data[kpi].mean()
            normal_mean = normal_data[kpi].mean()
            
            # Calculate how many standard deviations away from normal
            z_score = (anomaly_mean - normal_mean) / normal_std if normal_std > 0 else 0
            
            signature_data.append({
                'KPI': kpi.replace('_', ' '),
                'Z-Score': z_score,
                'Deviation': 'High' if abs(z_score) > 2 else 'Medium' if abs(z_score) > 1 else 'Low'
            })
        
        sig_df = pd.DataFrame(signature_data).sort_values('Z-Score', key=abs, ascending=False)
        
        # Create signature chart
        fig_sig = px.bar(
            sig_df,
            x='KPI',
            y='Z-Score',
            color='Deviation',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
            title="Anomaly Signature (Standard Deviations from Normal)",
            text='Z-Score'
        )
        fig_sig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_sig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=80))
        st.plotly_chart(fig_sig, use_container_width=True)
    
    st.markdown("---")
    
    # --------------------------
    # Row 4 — Temporal Pattern Analysis
    # --------------------------
    st.markdown('<h2 class="section-header">⏰ Temporal Pattern Analysis</h2>', unsafe_allow_html=True)
    
    col_temp1, col_temp2 = st.columns([1, 1])
    
    with col_temp1:
        st.markdown("**📅 Anomaly Distribution by Day of Week**")
        
        # Add day of week
        df_f['day_of_week'] = pd.to_datetime(df_f['datetime_stamp']).dt.day_name()
        anomaly_by_day = df_f[df_f['Anomaly_Flag'] == 1]['day_of_week'].value_counts()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        anomaly_by_day = anomaly_by_day.reindex([day for day in day_order if day in anomaly_by_day.index])
        
        fig_day = px.bar(
            x=anomaly_by_day.index,
            y=anomaly_by_day.values,
            title="Anomalies by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Number of Anomalies'}
        )
        fig_day.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_day, use_container_width=True)
    
    with col_temp2:
        st.markdown("**🕐 Anomaly Distribution by Hour**")
        
        # Add hour
        df_f['hour'] = pd.to_datetime(df_f['datetime_stamp']).dt.hour
        anomaly_by_hour = df_f[df_f['Anomaly_Flag'] == 1]['hour'].value_counts().sort_index()
        
        fig_hour = px.bar(
            x=anomaly_by_hour.index,
            y=anomaly_by_hour.values,
            title="Anomalies by Hour of Day",
            labels={'x': 'Hour of Day', 'y': 'Number of Anomalies'}
        )
        fig_hour.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_hour, use_container_width=True)
    
    st.markdown("---")
    
    # --------------------------
    # Row 5 — Predictive Indicators
    # --------------------------
    st.markdown('<h2 class="section-header">🔮 Predictive Indicators</h2>', unsafe_allow_html=True)
    
    st.markdown("**📊 Early Warning Signs Analysis**")
    
    # Calculate predictive indicators based on actual parameter thresholds
    # Define business-friendly parameter thresholds
    parameter_thresholds = {
        'Vibration_Level': {'warning': 6, 'critical': 8},
        'Motor_Temperature': {'warning': 80, 'critical': 85},
        'Oil_Pressure': {'warning_low': 8, 'warning_high': 15, 'critical_low': 6, 'critical_high': 17},
        'Power_Consumption': {'warning': 500, 'critical': 600},
        'Bearing_Wear_Index': {'warning': 30, 'critical': 50},
        'Throughput': {'warning': 200, 'critical': 150}
    }
    
    # Calculate risk level based on parameter violations
    def calculate_risk_level(row):
        critical_violations = 0
        warning_violations = 0
        
        for param, thresholds in parameter_thresholds.items():
            if param in row and pd.notna(row[param]):
                value = row[param]
                
                if param == 'Oil_Pressure':
                    if value <= thresholds['critical_low'] or value >= thresholds['critical_high']:
                        critical_violations += 1
                    elif value <= thresholds['warning_low'] or value >= thresholds['warning_high']:
                        warning_violations += 1
                elif param == 'Throughput':
                    if value <= thresholds['critical']:
                        critical_violations += 1
                    elif value <= thresholds['warning']:
                        warning_violations += 1
                else:
                    if value >= thresholds['critical']:
                        critical_violations += 1
                    elif value >= thresholds['warning']:
                        warning_violations += 1
        
        if critical_violations > 0:
            return 'Critical'
        elif warning_violations > 0:
            return 'Warning'
        else:
            return 'Low'
    
    # Apply risk calculation to all rows
    df_f['risk_level'] = df_f.apply(calculate_risk_level, axis=1)
    
    # Get latest data for each machine with risk levels
    latest_machine_data = df_f.sort_values('datetime_stamp').groupby('Machine_Id').tail(1)
    
    # Filter machines with warning or critical risk
    high_risk_machines = latest_machine_data[latest_machine_data['risk_level'].isin(['Warning', 'Critical'])]
    
    if len(high_risk_machines) > 0:
        st.markdown("**⚠️ Machines Requiring Attention**")
        
        # Create detailed machine table with all parameters
        machine_table_data = []
        for _, machine in high_risk_machines.iterrows():
            # Determine status based on parameters
            status_issues = []
            if machine['Vibration_Level'] > 8:
                status_issues.append("High Vibration")
            if machine['Motor_Temperature'] > 85:
                status_issues.append("High Temperature")
            if machine['Oil_Pressure'] < 8 or machine['Oil_Pressure'] > 15:
                status_issues.append("Oil Pressure Issue")
            if machine['Power_Consumption'] > 550:
                status_issues.append("High Power")
            if machine['Bearing_Wear_Index'] > 40:
                status_issues.append("Bearing Wear")
            if machine['Throughput'] < 200:
                status_issues.append("Low Throughput")
            
            status_text = ", ".join(status_issues) if status_issues else "Multiple Parameters"
            
            machine_table_data.append({
                'Machine ID': machine['Machine_Id'],
                'Plant': machine['Plant_Id'],
                'Risk Level': machine['risk_level'],
                'Vibration (mm/s)': f"{machine['Vibration_Level']:.1f}",
                'Temperature (°C)': f"{machine['Motor_Temperature']:.1f}",
                'Oil Pressure (bar)': f"{machine['Oil_Pressure']:.1f}",
                'Power (kW)': f"{machine['Power_Consumption']:.0f}",
                'Throughput (tonnes/h)': f"{machine['Throughput']:.0f}",
                'Bearing Wear (%)': f"{machine['Bearing_Wear_Index']:.1f}",
                'Main Issues': status_text,
                'Last Update': machine['datetime_stamp'].strftime('%m-%d %H:%M')
            })
        
        machine_df = pd.DataFrame(machine_table_data)
        
        # Display as expandable sections for each risk level
        for risk_level in ['Critical', 'Warning']:
            risk_machines = machine_df[machine_df['Risk Level'] == risk_level]
            if len(risk_machines) > 0:
                risk_color = "🔴" if risk_level == "Critical" else "🟡"
                
                with st.expander(f"{risk_color} {risk_level} Risk Machines ({len(risk_machines)} machines)", expanded=(risk_level == "Critical")):
                    st.dataframe(
                        risk_machines,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add detailed explanation for each machine
                    st.markdown("**📋 Flagging Explanation:**")
                    for _, machine in risk_machines.iterrows():
                        machine_id = machine['Machine ID']
                        explanations = []
                        
                        # Check each parameter against thresholds
                        vibration = float(machine['Vibration (mm/s)'])
                        temperature = float(machine['Temperature (°C)'])
                        oil_pressure = float(machine['Oil Pressure (bar)'])
                        power = float(machine['Power (kW)'])
                        throughput = float(machine['Throughput (tonnes/h)'])
                        bearing_wear = float(machine['Bearing Wear (%)'])
                        
                        if vibration >= 8:
                            explanations.append(f"🔴 **Critical Vibration**: {vibration:.1f} mm/s exceeds critical threshold (8 mm/s)")
                        elif vibration >= 6:
                            explanations.append(f"🟡 **Warning Vibration**: {vibration:.1f} mm/s exceeds warning threshold (6 mm/s)")
                            
                        if temperature >= 85:
                            explanations.append(f"🔴 **Critical Temperature**: {temperature:.1f}°C exceeds critical threshold (85°C)")
                        elif temperature >= 80:
                            explanations.append(f"🟡 **Warning Temperature**: {temperature:.1f}°C exceeds warning threshold (80°C)")
                            
                        if oil_pressure <= 6 or oil_pressure >= 17:
                            explanations.append(f"🔴 **Critical Oil Pressure**: {oil_pressure:.1f} bar outside critical range (6-17 bar)")
                        elif oil_pressure <= 8 or oil_pressure >= 15:
                            explanations.append(f"🟡 **Warning Oil Pressure**: {oil_pressure:.1f} bar outside warning range (8-15 bar)")
                            
                        if power >= 600:
                            explanations.append(f"🔴 **Critical Power**: {power:.0f} kW exceeds critical threshold (600 kW)")
                        elif power >= 500:
                            explanations.append(f"🟡 **Warning Power**: {power:.0f} kW exceeds warning threshold (500 kW)")
                            
                        if bearing_wear >= 50:
                            explanations.append(f"🔴 **Critical Bearing Wear**: {bearing_wear:.1f}% exceeds critical threshold (50%)")
                        elif bearing_wear >= 30:
                            explanations.append(f"🟡 **Warning Bearing Wear**: {bearing_wear:.1f}% exceeds warning threshold (30%)")
                            
                        if throughput <= 150:
                            explanations.append(f"🔴 **Critical Throughput**: {throughput:.0f} tonnes/h below critical threshold (150 tonnes/h)")
                        elif throughput <= 200:
                            explanations.append(f"🟡 **Warning Throughput**: {throughput:.0f} tonnes/h below warning threshold (200 tonnes/h)")
                        
                        if explanations:
                            st.markdown(f"**{machine_id}** (Plant: {machine['Plant']}):")
                            for explanation in explanations:
                                st.markdown(f"- {explanation}")
                            st.markdown("")
                    
                    # Add specific recommendations for this risk level
                    if risk_level == "Critical":
                        st.markdown("**🎯 Immediate Actions Required:**")
                        st.markdown("1. **Stop operations** and inspect immediately")
                        st.markdown("2. **Check all parameters** manually")
                        st.markdown("3. **Notify maintenance team**")
                        st.markdown("4. **Document findings** for root cause analysis")
                    else:
                        st.markdown("**🎯 Preventive Actions:**")
                        st.markdown("1. **Monitor closely** within 2 hours")
                        st.markdown("2. **Schedule maintenance** within 24 hours")
                        st.markdown("3. **Track parameter trends**")
                        st.markdown("4. **Review operating conditions**")
    else:
        st.success("✅ **All machines operating within normal parameters**")
        st.markdown("No immediate attention required. Continue regular monitoring.")
        
        st.markdown("---")
        
        # Predictive trend analysis
        st.markdown("**📈 Predictive Trend Analysis**")
        
        # Show machines with deteriorating parameters over time
        col_trend1, col_trend2 = st.columns([1, 1])
        
        with col_trend1:
            st.markdown("**📊 Parameter Deterioration Trends**")
            
            # Calculate parameter trends for each machine
            machine_trends = []
            for machine_id in df_f['Machine_Id'].unique():
                machine_data = df_f[df_f['Machine_Id'] == machine_id].sort_values('datetime_stamp')
                if len(machine_data) > 5:
                    # Compare recent vs earlier averages
                    recent_data = machine_data.tail(5)
                    earlier_data = machine_data.head(5)
                    
                    # Check for deteriorating trends in key parameters
                    deteriorating_params = []
                    
                    # Vibration trend
                    if recent_data['Vibration_Level'].mean() > earlier_data['Vibration_Level'].mean() + 1:
                        deteriorating_params.append("Vibration")
                    
                    # Temperature trend
                    if recent_data['Motor_Temperature'].mean() > earlier_data['Motor_Temperature'].mean() + 5:
                        deteriorating_params.append("Temperature")
                    
                    # Oil pressure trend (decreasing is bad)
                    if recent_data['Oil_Pressure'].mean() < earlier_data['Oil_Pressure'].mean() - 1:
                        deteriorating_params.append("Oil Pressure")
                    
                    # Throughput trend (decreasing is bad)
                    if recent_data['Throughput'].mean() < earlier_data['Throughput'].mean() - 20:
                        deteriorating_params.append("Throughput")
                    
                    if deteriorating_params:
                        machine_trends.append({
                            'Machine': machine_id,
                            'Plant': machine_data['Plant_Id'].iloc[0],
                            'Deteriorating Parameters': ", ".join(deteriorating_params),
                            'Recent Vibration': recent_data['Vibration_Level'].mean(),
                            'Recent Temperature': recent_data['Motor_Temperature'].mean(),
                            'Recent Oil Pressure': recent_data['Oil_Pressure'].mean()
                        })
            
            if machine_trends:
                st.markdown("**⚠️ Machines with Deteriorating Parameters:**")
                for machine in machine_trends[:5]:
                    st.markdown(f"- **{machine['Machine']}** (Plant: {machine['Plant']}): {machine['Deteriorating Parameters']}")
            else:
                st.markdown("✅ No machines showing deteriorating parameter trends")
        
        with col_trend2:
            st.markdown("**🎯 Predictive Maintenance Recommendations**")
            
            # Generate recommendations based on current risk levels
            recommendations = []
            
            if len(high_risk_machines) > 0:
                critical_count = len(high_risk_machines[high_risk_machines['risk_level'] == 'Critical'])
                warning_count = len(high_risk_machines[high_risk_machines['risk_level'] == 'Warning'])
                
                if critical_count > 0:
                    recommendations.append(f"**🚨 URGENT**: {critical_count} machines need immediate attention")
                if warning_count > 0:
                    recommendations.append(f"**⚠️ MONITOR**: {warning_count} machines need preventive maintenance")
            
            # Add trend-based recommendations
            if machine_trends:
                deteriorating_count = len(machine_trends)
                if deteriorating_count > 0:
                    recommendations.append(f"**📈 TREND**: {deteriorating_count} machines showing deteriorating parameters")
            
            if not recommendations:
                recommendations.append("✅ **All systems normal** - continue regular monitoring")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
    
    st.markdown("---")
    
    # --------------------------
    # Row 6 — Actionable Machine Alerts
    # --------------------------
    st.markdown('<h2 class="section-header">🚨 Actionable Machine Alerts</h2>', unsafe_allow_html=True)
    
    # Get current data for actionable insights
    current_data = df_f.copy()
    current_data['datetime_stamp'] = pd.to_datetime(current_data['datetime_stamp'])
    
    # Get the most recent data for each machine
    latest_data = current_data.sort_values('datetime_stamp').groupby('Machine_Id').tail(1)
    
    # IF Score thresholds (primary indicator)
    if_score_thresholds = {
        'low_risk': 0.5,
        'medium_risk': 0.6, 
        'high_risk': 0.7,
        'critical_risk': 0.75
    }
    
    # Secondary parameter thresholds for context
    parameter_thresholds = {
        'Vibration_Level': {'warning': 6, 'critical': 10},
        'Motor_Temperature': {'warning': 80, 'critical': 90},
        'Oil_Pressure': {'warning_low': 8, 'warning_high': 15, 'critical_low': 6, 'critical_high': 17},
        'Power_Consumption': {'warning': 500, 'critical': 600},
        'Bearing_Wear_Index': {'warning': 30, 'critical': 50}
    }
    
    # Check each machine for issues
    machine_alerts = []
    
    for _, machine in latest_data.iterrows():
        machine_id = machine['Machine_Id']
        alerts = []
        risk_level = "Low"
        
        # PRIMARY: Check IF Score first
        if 'if_score' in machine and pd.notna(machine['if_score']):
            if_score = machine['if_score']
            
            if if_score >= if_score_thresholds['critical_risk']:
                risk_level = "Critical"
                alerts.append(f"🔴 **CRITICAL ANOMALY**: IF Score = {if_score:.3f} (Critical: >{if_score_thresholds['critical_risk']})")
            elif if_score >= if_score_thresholds['high_risk']:
                risk_level = "High"
                alerts.append(f"🟠 **HIGH RISK**: IF Score = {if_score:.3f} (High: >{if_score_thresholds['high_risk']})")
            elif if_score >= if_score_thresholds['medium_risk']:
                risk_level = "Medium"
                alerts.append(f"🟡 **MEDIUM RISK**: IF Score = {if_score:.3f} (Medium: >{if_score_thresholds['medium_risk']})")
            else:
                risk_level = "Low"
        
        # SECONDARY: Check specific parameters for context (only if IF score indicates risk)
        if risk_level != "Low":
            for param, thresholds in parameter_thresholds.items():
                if param in machine and pd.notna(machine[param]):
                    value = machine[param]
                    
                    if param == 'Oil_Pressure':
                        # Special handling for oil pressure (both low and high are bad)
                        if value <= thresholds['critical_low'] or value >= thresholds['critical_high']:
                            alerts.append(f"🔴 **CRITICAL**: {param.replace('_', ' ')} = {value:.1f} (Normal: 8-15)")
                        elif value <= thresholds['warning_low'] or value >= thresholds['warning_high']:
                            alerts.append(f"🟡 **WARNING**: {param.replace('_', ' ')} = {value:.1f} (Normal: 8-15)")
                    else:
                        # For other parameters, higher is worse (except throughput)
                        if param == 'Throughput':
                            if value <= 100:  # Very low throughput
                                alerts.append(f"🔴 **CRITICAL**: {param.replace('_', ' ')} = {value:.1f} (Normal: >200)")
                            elif value <= 150:
                                alerts.append(f"🟡 **WARNING**: {param.replace('_', ' ')} = {value:.1f} (Normal: >200)")
                        else:
                            if value >= thresholds['critical']:
                                alerts.append(f"🔴 **CRITICAL**: {param.replace('_', ' ')} = {value:.1f} (Normal: <{thresholds['critical']})")
                            elif value >= thresholds['warning']:
                                alerts.append(f"🟡 **WARNING**: {param.replace('_', ' ')} = {value:.1f} (Normal: <{thresholds['warning']})")
        
        # Only add machines with IF score risk or parameter issues
        if risk_level != "Low" or alerts:
            critical_count = len([a for a in alerts if '🔴' in a or 'CRITICAL' in a])
            high_count = len([a for a in alerts if '🟠' in a or 'HIGH' in a])
            
            machine_alerts.append({
                'Machine': machine_id,
                'Plant': machine['Plant_Id'],
                'Alerts': alerts,
                'Alert_Count': len(alerts),
                'Risk_Level': risk_level,
                'IF_Score': machine.get('if_score', 0),
                'Critical_Count': critical_count,
                'High_Count': high_count,
                'Last_Update': machine['datetime_stamp'].strftime('%Y-%m-%d %H:%M')
            })
    
    # Sort by IF score (highest risk first), then by critical alerts
    machine_alerts.sort(key=lambda x: (x['IF_Score'], x['Critical_Count'], x['High_Count']), reverse=True)
    
    if machine_alerts:
        st.markdown("**⚠️ Machines Requiring Immediate Attention (Based on IF Score)**")
        
        # Show top 10 machines with issues
        for i, machine_alert in enumerate(machine_alerts[:10]):
            risk_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            risk_icon = risk_color.get(machine_alert['Risk_Level'], "⚪")
            
            with st.expander(f"{risk_icon} {machine_alert['Machine']} (Plant: {machine_alert['Plant']}) - IF Score: {machine_alert['IF_Score']:.3f} - {machine_alert['Risk_Level']} Risk", expanded=(i < 3)):
                st.markdown(f"**Last Update:** {machine_alert['Last_Update']}")
                st.markdown(f"**🎯 Primary Risk Indicator:** IF Score = **{machine_alert['IF_Score']:.3f}** ({machine_alert['Risk_Level']} Risk)")
                
                if machine_alert['Alerts']:
                    st.markdown("**Issues Found:**")
                    for alert in machine_alert['Alerts']:
                        st.markdown(f"- {alert}")
                
                # Add specific action recommendations based on IF score
                st.markdown("**🎯 Recommended Actions:**")
                if machine_alert['Risk_Level'] == "Critical":
                    st.markdown("1. **🚨 IMMEDIATE**: Stop machine and inspect")
                    st.markdown("2. **🔍 Check**: All parameters manually - high anomaly probability")
                    st.markdown("3. **📞 Notify**: Maintenance team immediately")
                    st.markdown("4. **📊 Analyze**: Root cause of high IF score")
                elif machine_alert['Risk_Level'] == "High":
                    st.markdown("1. **⚠️ URGENT**: Monitor closely within 1 hour")
                    st.markdown("2. **🔧 Inspect**: Key parameters (vibration, temperature, pressure)")
                    st.markdown("3. **📅 Schedule**: Preventive maintenance within 24 hours")
                    st.markdown("4. **📈 Track**: IF score trends")
                elif machine_alert['Risk_Level'] == "Medium":
                    st.markdown("1. **👀 Monitor**: Check parameters within 2 hours")
                    st.markdown("2. **📋 Document**: Parameter trends for analysis")
                    st.markdown("3. **🔧 Schedule**: Preventive maintenance soon")
                    st.markdown("4. **📊 Review**: IF score patterns")
                else:
                    st.markdown("1. **✅ Continue**: Regular monitoring")
                    st.markdown("2. **📊 Track**: IF score trends")
                    st.markdown("3. **📋 Document**: Baseline parameters")
        
        # Summary statistics based on IF score risk levels
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        
        with col_alert1:
            critical_machines = len([m for m in machine_alerts if m['Risk_Level'] == 'Critical'])
            st.metric("Critical Risk Machines", critical_machines, delta=f"IF Score >0.75 - {critical_machines} need immediate attention")
        
        with col_alert2:
            high_risk_machines = len([m for m in machine_alerts if m['Risk_Level'] == 'High'])
            st.metric("High Risk Machines", high_risk_machines, delta=f"IF Score 0.7-0.75 - {high_risk_machines} need urgent monitoring")
        
        with col_alert3:
            medium_risk_machines = len([m for m in machine_alerts if m['Risk_Level'] == 'Medium'])
            st.metric("Medium Risk Machines", medium_risk_machines, delta=f"IF Score 0.6-0.7 - {medium_risk_machines} need monitoring")
    
    else:
        st.success("✅ **All machines operating within normal parameters**")
        st.markdown("No immediate action required. Continue regular monitoring.")
    
    st.markdown("---")
    
    # --------------------------
    # Row 7 — Root Cause Summary
    # --------------------------
    st.markdown('<h2 class="section-header">📋 Root Cause Summary</h2>', unsafe_allow_html=True)
    
    # Generate root cause insights
    col_summary1, col_summary2 = st.columns([1, 1])
    
    with col_summary1:
        st.markdown("**🎯 Key Findings**")
        
        # Top correlations
        top_correlations = sorted(kpi_correlations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        st.markdown("**Primary Contributing Factors:**")
        for i, (kpi, corr) in enumerate(top_correlations, 1):
            kpi_name = kpi.replace('_', ' ').title()
            st.markdown(f"{i}. **{kpi_name}** (Correlation: {corr:.3f})")
        
        # Temporal patterns
        if len(anomaly_data) > 0:
            most_common_day = df_f[df_f['Anomaly_Flag'] == 1]['day_of_week'].mode().iloc[0] if 'day_of_week' in df_f.columns else 'N/A'
            most_common_hour = df_f[df_f['Anomaly_Flag'] == 1]['hour'].mode().iloc[0] if 'hour' in df_f.columns else 'N/A'
            
            st.markdown(f"**Temporal Patterns:**")
            st.markdown(f"- Most common day: **{most_common_day}**")
            st.markdown(f"- Most common hour: **{most_common_hour}:00**")
    
    with col_summary2:
        st.markdown("**💡 Recommendations**")
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if strongest_predictor[1] > 0.7:
            kpi_name = strongest_predictor[0].replace('_', ' ').title()
            recommendations.append(f"**Monitor {kpi_name} closely** - strongest predictor of failures")
        
        if affected_machines > 3:
            recommendations.append(f"**Focus on {affected_machines} high-risk machines** for preventive maintenance")
        
        if anomaly_rate > 2:
            recommendations.append("**Review maintenance schedules** - anomaly rate above target")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        if not recommendations:
            st.markdown("✅ **System operating within normal parameters**")
    
    st.markdown("---")
    
    # --------------------------
    # Footer
    # --------------------------
    st.caption(
        "💡 **Root Cause Analysis Insights:** This analysis identifies the primary factors contributing to equipment failures, "
        "enabling proactive maintenance and improved operational efficiency."
    )
