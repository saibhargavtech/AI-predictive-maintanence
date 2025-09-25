# ⚙️ Predictive Maintenance Dashboard

A comprehensive Streamlit dashboard for predictive maintenance analysis, featuring anomaly detection, trend analysis, and equipment monitoring.

## 🚀 Features

- **📊 Overview Dashboard**: KPI tiles and equipment status
- **🚨 Alerts & Monitoring**: Real-time anomaly detection and alerts
- **📈 Trend Analysis**: Time series analysis with rolling averages and heatmaps
- **⚠️ Anomaly Visualization**: Scatter plots, boxplots, and pattern analysis
- **🏭 Plant/Machine Comparison**: Pareto analysis and efficiency ranking

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 1.5.0+
- Plotly 5.15.0+

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd MaintanenceDashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## 📊 Data Format

The dashboard expects CSV files with the following columns:
- `datetime_stamp`: Timestamp of the reading
- `Machine_Id`: Machine identifier
- `Plant_Id`: Plant identifier
- `Vibration_Level`: Vibration level (mm/s)
- `Motor_Temperature`: Motor temperature (°C)
- `Oil_Pressure`: Oil pressure (bar)
- `Power_Consumption`: Power consumption (kW)
- `Throughput`: Material throughput (tonnes/h)
- `Bearing_Wear_Index`: Bearing wear percentage (%)
- `Anomaly_Flag`: Binary anomaly flag (0/1)
- `if_score`: Isolation Forest anomaly score (0-1)

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `main.py` as the main file
5. Deploy!

### Local Deployment

```bash
streamlit run main.py --server.port 8501
```

## 📱 Usage

1. **Upload Data**: Use the sidebar to upload your CSV file
2. **Filter Data**: Apply date range and plant/machine filters
3. **Navigate**: Use the sidebar navigation to explore different analysis views
4. **Analyze**: Each page provides specialized insights for different aspects of your data

## 🔧 Configuration

- Modify `config/constants.py` to adjust thresholds and targets
- Update `config/styles.py` for custom styling
- Edit `config/navigation.py` to customize page navigation

## 📈 Key Metrics

- **Anomaly Rate**: Percentage of anomalous readings
- **Equipment Efficiency**: Based on vibration and temperature
- **IF Score**: Isolation Forest anomaly detection score
- **Pareto Analysis**: 80/20 rule for problem identification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions, please open an issue in the GitHub repository.
