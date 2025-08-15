# 🚗 NYC Motor Vehicle Collisions AI Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **AI-powered analysis and prediction of NYC traffic collisions using advanced machine learning**

## 🎯 What This Does

This project analyzes NYC motor vehicle collision data to:
- **Predict accident risk** using machine learning models
- **Identify high-risk zones** with geospatial analysis
- **Analyze temporal patterns** (time, day, season)
- **Provide actionable insights** for traffic safety improvements

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for large datasets)
- NYC Collisions dataset

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd Vehicular-collisions-NYC

# Install dependencies
pip install -r requirements.txt

# Download data (if not included)
# Place Motor_Vehicle_Collisions_-_Crashes.csv in project root
```

### Run the Dashboard
```bash
# Start AI Dashboard
streamlit run streamlit_app.py

# Open Jupyter Notebook for analysis
jupyter notebook nyc_collisions_advanced_analysis.ipynb
```

## 📊 Features

### 🤖 AI-Powered Dashboard
- **Real-time risk prediction** for any location/time
- **Interactive visualizations** with Plotly
- **Machine learning models** (Random Forest, XGBoost, Deep Learning)
- **Geographic heatmaps** showing collision hotspots

### 📈 Advanced Analytics
- **Time series analysis** of accident patterns
- **Feature engineering** for ML models
- **Model performance comparison** with AUC scores
- **Data filtering and export** capabilities

### 🗺️ Geographic Intelligence
- **Interactive NYC maps** with collision overlays
- **Risk zone identification** using clustering
- **Safety recommendations** for high-risk areas

## 🏗️ Project Structure

```
Vehicular-collisions-NYC/
├── streamlit_app.py                    # Main AI Dashboard (Clean & Optimized)
├── nyc_collisions_advanced_analysis.ipynb  # Comprehensive ML Analysis Notebook  
├── requirements.txt                     # Essential Dependencies Only
├── Motor_Vehicle_Collisions_-_Crashes.csv  # Data File (Download Required)
├── .gitignore                          # Git Ignore Rules
└── README.md                           # Documentation
```

## 🔧 Technical Details

### Machine Learning Models
- **Random Forest**: Baseline classification
- **Gradient Boosting**: Ensemble learning
- **XGBoost**: Optimized gradient boosting
- **LightGBM**: Light gradient boosting
- **Deep Learning**: Neural network with TensorFlow

### Features Used
- **Temporal**: Hour, day, month, weekend, rush hour, night
- **Geographic**: Latitude, longitude
- **Derived**: Risk score, severity classification

### Performance Metrics
- **AUC Score**: Model discrimination ability
- **Cross-validation**: Robust performance estimation
- **Feature importance**: Model interpretability

## 📱 Using the Dashboard

### 1. **AI Prediction Tab**
- Set time, date, and location parameters
- Get real-time risk predictions
- View confidence scores and model outputs

### 2. **Time Analysis Tab**
- Hourly accident distribution with ML predictions
- Day-of-week risk analysis
- Seasonal pattern identification

### 3. **Geography Tab**
- Interactive NYC collision heatmap
- High-risk zone identification
- Safety improvement recommendations

### 4. **Data Explorer Tab**
- Filter data by risk level, time, severity
- Download filtered datasets
- Custom analysis capabilities

## 🧪 Jupyter Notebook

The `nyc_collisions_advanced_analysis.ipynb` provides:
- **Complete ML pipeline** from data loading to deployment
- **Model training and comparison**
- **Feature engineering examples**
- **Deep learning implementation**
- **Geospatial analysis**

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Docker (Optional)
```bash
docker build -t nyc-collisions-ai .
docker run -p 8501:8501 nyc-collisions-ai
```

## 📊 Data Source

**NYC Open Data**: [Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

**Data Schema**:
- Crash date/time, location (lat/long)
- Casualty counts (injured/killed)
- Contributing factors
- Vehicle types involved

**Data Quality**:
- Real-time updates from NYC agencies
- GPS coordinates for mapping
- Comprehensive coverage of NYC boroughs

## 🤝 Contributing

### Bug Reports
- Use GitHub Issues
- Include error messages and steps to reproduce
- Specify your environment (OS, Python version)

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Suggest implementation approach

### Code Contributions
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 🐛 Troubleshooting

### Common Issues

**Data Loading Errors**
- Ensure CSV file is in project root
- Check file permissions
- Verify CSV format matches expected schema

**ML Model Issues**
- Install scikit-learn: `pip install scikit-learn`
- Check Python version compatibility
- Ensure sufficient RAM for large datasets

**Map Display Problems**
- Install folium: `pip install folium`
- Check internet connection for map tiles
- Verify coordinate data quality

**Performance Issues**
- Reduce max_rows in sidebar
- Use smaller ML sample sizes
- Enable caching for repeated operations

### Getting Help
1. Check this README first
2. Review error messages carefully
3. Search existing GitHub issues
4. Create new issue with details

## 📈 Performance Tips

- **Data Loading**: Use appropriate max_rows for your system
- **ML Training**: Start with smaller sample sizes
- **Caching**: Streamlit caches data and models automatically
- **Maps**: Sample data for large datasets to improve performance

## 🔒 Privacy & Security

- **No personal data** is collected or stored
- **Public dataset** from NYC Open Data
- **Local processing** - data stays on your machine
- **No external API calls** except for map tiles

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **NYC Open Data** for providing collision data
- **Streamlit** for the web framework
- **Scikit-learn** for machine learning tools
- **Plotly** for interactive visualizations
- **Folium** for geographic mapping

## 📞 Support

- **GitHub Issues**: [Create Issue](https://github.com/your-repo/issues)
- **Documentation**: This README and inline code comments
- **Community**: Check discussions and existing solutions

---

**Made with ❤️ and 🤖 AI for NYC Traffic Safety**

*Last updated: 2024*
