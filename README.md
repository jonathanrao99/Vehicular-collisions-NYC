# 🚗 NYC Motor Vehicle Collisions Analysis Dashboard

**A simple, powerful tool to understand traffic safety in New York City**

## 🤔 What is this project?

This is a **web application** that helps you explore and understand car accidents in New York City. Think of it as a smart dashboard that turns boring data into easy-to-understand charts and maps.

### 🎯 Why was this built?

**The Problem:** Traffic accidents are a big issue in NYC, but the data is hard to understand. Police reports, news articles, and raw data files are confusing and don't show the full picture.

**The Solution:** This dashboard makes accident data **visual, interactive, and easy to understand**. Anyone can use it to:
- See where accidents happen most often
- Understand when accidents are most likely to occur
- Identify dangerous streets and intersections
- Track trends over time

### 💡 How is this beneficial?

**For Everyone:**
- **Safer driving:** Know which areas and times are most dangerous
- **Better planning:** Choose safer routes and travel times
- **Awareness:** Understand traffic safety patterns in your neighborhood

**For City Officials:**
- **Data-driven decisions:** Identify where to improve roads and traffic signals
- **Resource allocation:** Focus safety efforts where they're needed most
- **Progress tracking:** See if safety improvements are working

**For Researchers:**
- **Easy data access:** No need to download and process large files
- **Visual insights:** See patterns that might be hidden in raw data
- **Export capabilities:** Download filtered data for further analysis

## 🚀 Quick Start Guide

### Step 1: Get the Project
```bash
# Download the project
git clone https://github.com/yourusername/nyc-collisions-analysis.git

# Go into the project folder
cd nyc-collisions-analysis
```

### Step 2: Set Up Your Computer
You need Python installed on your computer. If you don't have it:
- **Windows/Mac:** Download from [python.org](https://python.org)
- **Linux:** Usually comes pre-installed

### Step 3: Install Dependencies
```bash
# Install the required packages
pip install -r requirements.txt
```

### Step 4: Get the Data
You need the NYC collision data file:
1. Download from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
2. Save the CSV file as `Motor_Vehicle_Collisions_-_Crashes.csv` in the project folder
3. Or extract the `Motor_Vehicle_Collisions_-_Crashes.rar` file if you have it

### Step 5: Run the Application
```bash
# Start the dashboard
streamlit run app.py
```

### Step 6: Open Your Browser
- Go to `http://localhost:8501`
- The dashboard will open automatically!

## 📊 What Can You Do With This Dashboard?

### 🕐 Time Analysis
- **See peak accident hours:** Find out when accidents happen most
- **Day of week patterns:** Discover which days are safest/riskiest
- **Monthly trends:** Track how accident rates change over time

### 🗺️ Geographic Analysis
- **Interactive 3D map:** See accident hotspots across NYC
- **Injury severity filtering:** Focus on serious accidents
- **Street-level insights:** Identify dangerous intersections

### ⚠️ Dangerous Locations
- **Top hazardous streets:** See which roads have the most accidents
- **Borough comparisons:** Compare safety across NYC areas
- **Contributing factors:** Understand what causes accidents

### 👥 Victim Analysis
- **Victim types:** See how pedestrians, cyclists, and drivers are affected
- **Injury vs. fatality breakdown:** Understand the severity of accidents
- **Demographic insights:** Track different types of victims

### 📋 Data Export
- **Filter data:** Select specific time periods or injury levels
- **Download results:** Export filtered data as CSV files
- **Custom analysis:** Use the data in other tools

## 🛠️ Technical Details (For Developers)

### What Technologies Are Used?
- **Streamlit:** Creates the web interface
- **Plotly:** Makes interactive charts
- **PyDeck:** Creates 3D maps
- **Pandas:** Processes the data
- **Python:** The programming language

### Project Structure
```
nyc-collisions-analysis/
├── app.py              # Main application
├── config.py           # Settings and configuration
├── utils.py            # Helper functions
├── requirements.txt    # Python packages needed
├── README.md          # This file
├── Dockerfile         # For container deployment
└── docker-compose.yml # For easy deployment
```

### Advanced Setup Options

#### Using Docker (Recommended for Production)
```bash
# Build and run with Docker
docker-compose up -d

# Access at http://localhost:8501
```

#### Using Make Commands
```bash
# Install dependencies
make install

# Run the application
make run

# Build Docker image
make docker-build
```

## 🔧 Troubleshooting

### Common Issues

**"Data file not found"**
- Make sure the CSV file is in the same folder as `app.py`
- Check that the filename is exactly `Motor_Vehicle_Collisions_-_Crashes.csv`

**"Module not found"**
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8 or higher

**"Port already in use"**
- Close other applications using port 8501
- Or change the port in the command: `streamlit run app.py --server.port 8502`

**"Application is slow"**
- Reduce the "Maximum Rows to Load" in the sidebar
- Close other applications to free up memory

### Getting Help
- Check the logs for error messages
- Make sure all files are in the correct locations
- Try running the tests: `python test_app.py`

## 📈 Data Source

This dashboard uses official NYC collision data from:
- **Source:** [NYC Open Data Portal](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
- **Updated:** Daily
- **Coverage:** All five NYC boroughs
- **Time Period:** Historical data from 2012 to present

The data includes:
- Location (latitude/longitude, street names)
- Time and date of accidents
- Number of injuries and fatalities
- Contributing factors
- Vehicle and victim types

## 🤝 Contributing

Want to help improve this dashboard?

1. **Report bugs:** Create an issue on GitHub
2. **Suggest features:** Let us know what you'd like to see
3. **Share data insights:** Tell us what you discover
4. **Improve documentation:** Help make it clearer for others

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **NYC Open Data** for providing the collision dataset
- **Streamlit** team for the amazing web framework
- **Plotly** for beautiful interactive visualizations
- **PyDeck** for 3D mapping capabilities

---

**Made with ❤️ for NYC traffic safety**

*This dashboard helps make NYC streets safer by making accident data accessible and understandable to everyone.*
