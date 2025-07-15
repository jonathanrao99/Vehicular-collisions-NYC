# 🚗 NYC Motor Vehicle Collisions Analysis Dashboard

A comprehensive, interactive web application for analyzing and visualizing motor vehicle collision data in New York City. Built with Streamlit, this dashboard provides powerful insights into traffic safety patterns, dangerous locations, and temporal trends.

![NYC Collisions Dashboard](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Features](#-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Source](#-data-source)
- [Technical Architecture](#-technical-architecture)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 📊 **Comprehensive Analytics**
- **Summary Metrics**: Total collisions, injuries, fatalities, and date ranges
- **Time-based Analysis**: Hourly, daily, and monthly collision patterns
- **Geographic Visualization**: Interactive 3D maps with injury severity filtering
- **Dangerous Locations**: Top hazardous streets and borough analysis
- **Victim Analysis**: Breakdown by pedestrian, cyclist, and motorist injuries

### 🗺️ **Interactive Visualizations**
- **3D Hexagon Maps**: Spatial clustering of collision hotspots
- **Time Series Charts**: Trend analysis over time periods
- **Bar and Pie Charts**: Comparative analysis across categories
- **Real-time Filtering**: Dynamic data exploration capabilities

### 🔧 **Advanced Functionality**
- **Data Export**: Download filtered datasets as CSV
- **Configurable Settings**: Adjustable data loading and display parameters
- **Error Handling**: Robust error management and user feedback
- **Responsive Design**: Optimized for various screen sizes

## 📸 Screenshots

### Dashboard Overview
![Dashboard Overview](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Dashboard+Overview)

### Geographic Analysis
![Geographic Analysis](https://via.placeholder.com/800x400/ff7f0e/ffffff?text=Geographic+Analysis)

### Time-based Trends
![Time Analysis](https://via.placeholder.com/800x400/2ca02c/ffffff?text=Time-based+Analysis)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/nyc-collisions-analysis.git
   cd nyc-collisions-analysis
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Data**
   - Extract the `Motor_Vehicle_Collisions_-_Crashes.rar` file
   - Ensure the CSV file is in the same directory as `app.py`
   - Alternatively, download the latest data from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

## 📖 Usage

### Running the Application

1. **Start the Streamlit Server**
   ```bash
   streamlit run app.py
   ```

2. **Access the Dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The dashboard will load automatically

### Using the Dashboard

#### **Main Dashboard**
- View summary metrics at the top
- Navigate between analysis tabs
- Use sidebar settings to configure data loading

#### **Time Analysis Tab**
- Explore collision patterns by hour, day, and month
- Identify peak collision times
- Analyze seasonal trends

#### **Geographic Analysis Tab**
- Adjust injury severity threshold
- Explore 3D collision hotspots
- Zoom and pan the interactive map

#### **Dangerous Locations Tab**
- View top hazardous streets
- Analyze borough-level statistics
- Examine contributing factors

#### **Victim Analysis Tab**
- Compare injury types (pedestrians, cyclists, motorists)
- Analyze fatality patterns
- Understand demographic impacts

#### **Raw Data Tab**
- Filter data by date range and injury count
- Download filtered datasets
- Export analysis results

## 📊 Data Source

This application uses the **NYC Motor Vehicle Collisions - Crashes** dataset from NYC Open Data:

- **Source**: [NYC Open Data Portal](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
- **Update Frequency**: Daily
- **Data Fields**: 29 columns including location, time, injuries, fatalities, and contributing factors
- **Coverage**: All five NYC boroughs
- **Time Period**: Historical data from 2012 to present

### Key Data Fields

| Field | Description | Type |
|-------|-------------|------|
| `CRASH_DATE` | Date of collision | Date |
| `CRASH_TIME` | Time of collision | Time |
| `LATITUDE` | Geographic latitude | Float |
| `LONGITUDE` | Geographic longitude | Float |
| `BOROUGH` | NYC borough | String |
| `ON_STREET_NAME` | Street name | String |
| `INJURED_PERSONS` | Total injured | Integer |
| `KILLED_PERSONS` | Total fatalities | Integer |

## 🏗️ Technical Architecture

### **Frontend**
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **PyDeck**: 3D mapping and geospatial visualization
- **Custom CSS**: Enhanced styling and user experience

### **Backend**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Python**: Core programming language

### **Data Processing Pipeline**
```
Raw CSV → Data Validation → Cleaning → Feature Engineering → Visualization
```

### **Performance Optimizations**
- **Caching**: Streamlit caching for improved performance
- **Data Filtering**: Configurable row limits
- **Memory Management**: Efficient data handling
- **Error Handling**: Graceful failure management

## 📚 API Documentation

### Core Functions

#### `load_data(file_path, max_rows=None)`
Loads and preprocesses collision data.

**Parameters:**
- `file_path` (str): Path to CSV file
- `max_rows` (int): Maximum rows to load

**Returns:**
- `pd.DataFrame`: Processed collision data

#### `create_summary_metrics(data)`
Generates dashboard summary metrics.

**Parameters:**
- `data` (pd.DataFrame): Collision dataset

#### `create_time_analysis(data)`
Creates time-based visualizations.

**Parameters:**
- `data` (pd.DataFrame): Collision dataset

#### `create_geographic_analysis(data)`
Generates geographic visualizations.

**Parameters:**
- `data` (pd.DataFrame): Collision dataset

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
   ```bash
   git fork https://github.com/yourusername/nyc-collisions-analysis.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add appropriate documentation
   - Include error handling

4. **Test Your Changes**
   ```bash
   streamlit run app.py
   ```

5. **Submit Pull Request**
   - Provide clear description of changes
   - Include screenshots if UI changes
   - Reference any related issues

### Development Guidelines

- **Code Style**: Follow PEP 8 conventions
- **Documentation**: Add docstrings to all functions
- **Testing**: Test with different data sizes
- **Error Handling**: Implement robust error management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NYC Open Data** for providing the collision dataset
- **Streamlit** team for the excellent web framework
- **Plotly** for interactive visualization capabilities
- **PyDeck** for 3D mapping functionality

## 📞 Support

For questions, issues, or feature requests:

- **Issues**: [GitHub Issues](https://github.com/yourusername/nyc-collisions-analysis/issues)
- **Email**: your.email@example.com
- **Documentation**: [Wiki](https://github.com/yourusername/nyc-collisions-analysis/wiki)

---

**Made with ❤️ for NYC traffic safety**
