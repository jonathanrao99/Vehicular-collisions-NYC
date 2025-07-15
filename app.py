import pandas as pd
import streamlit as st
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NYC Motor Vehicle Collisions Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
@st.cache_resource
def get_config():
    """Get application configuration"""
    config = {
        'data_file': 'Motor_Vehicle_Collisions_-_Crashes.csv',
        'max_rows': 100000,
        'map_style': "mapbox://styles/mapbox/light-v9",
        'default_lat': 40.7128,
        'default_lon': -74.0060,
        'default_zoom': 10
    }
    return config

@st.cache_data
def load_data(file_path, max_rows=None):
    """
    Load and preprocess collision data with error handling
    
    Args:
        file_path (str): Path to the CSV file
        max_rows (int): Maximum number of rows to load
    
    Returns:
        pd.DataFrame: Processed collision data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            st.info("Please ensure the Motor_Vehicle_Collisions_-_Crashes.csv file is in the same directory as this application.")
            return None
        
        # Load data with progress indicator
        with st.spinner("Loading collision data..."):
            data = pd.read_csv(
                file_path, 
                nrows=max_rows, 
                parse_dates=[['CRASH_DATE', 'CRASH_TIME']],
                low_memory=False
            )
        
        # Data preprocessing
        st.info("Preprocessing data...")
        
        # Remove rows with missing coordinates
        initial_count = len(data)
        data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
        final_count = len(data)
        
        if final_count < initial_count:
            st.warning(f"Removed {initial_count - final_count} rows with missing coordinates")
        
        # Clean and standardize column names
        data.columns = data.columns.str.lower()
        data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
        
        # Convert date/time to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data['date/time']):
            data['date/time'] = pd.to_datetime(data['date/time'], errors='coerce')
        
        # Remove rows with invalid dates
        data.dropna(subset=['date/time'], inplace=True)
        
        # Add derived columns
        data['year'] = data['date/time'].dt.year
        data['month'] = data['date/time'].dt.month
        data['day_of_week'] = data['date/time'].dt.day_name()
        data['hour'] = data['date/time'].dt.hour
        data['total_injured'] = (
            data['injured_persons'].fillna(0) + 
            data['injured_pedestrians'].fillna(0) + 
            data['injured_cyclists'].fillna(0) + 
            data['injured_motorists'].fillna(0)
        )
        data['total_killed'] = (
            data['killed_persons'].fillna(0) + 
            data['killed_pedestrians'].fillna(0) + 
            data['killed_cyclists'].fillna(0) + 
            data['killed_motorists'].fillna(0)
        )
        
        st.success(f"Successfully loaded {len(data):,} collision records")
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")
        return None

def create_summary_metrics(data):
    """Create summary metrics for the dashboard"""
    if data is None or len(data) == 0:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Collisions",
            value=f"{len(data):,}",
            delta=None
        )
    
    with col2:
        total_injured = data['total_injured'].sum()
        st.metric(
            label="Total Injured",
            value=f"{total_injured:,}",
            delta=None
        )
    
    with col3:
        total_killed = data['total_killed'].sum()
        st.metric(
            label="Total Fatalities",
            value=f"{total_killed:,}",
            delta=None
        )
    
    with col4:
        date_range = f"{data['date/time'].min().strftime('%Y-%m-%d')} to {data['date/time'].max().strftime('%Y-%m-%d')}"
        st.metric(
            label="Date Range",
            value=date_range,
            delta=None
        )

def create_time_analysis(data):
    """Create time-based analysis visualizations"""
    if data is None or len(data) == 0:
        return
    
    st.header("📊 Time-based Analysis")
    
    # Hourly distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Collisions by Hour of Day")
        hourly_counts = data['hour'].value_counts().sort_index()
        fig_hourly = px.bar(
            x=hourly_counts.index, 
            y=hourly_counts.values,
            title="Collision Frequency by Hour",
            labels={'x': 'Hour of Day', 'y': 'Number of Collisions'}
        )
        fig_hourly.update_layout(showlegend=False)
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("Collisions by Day of Week")
        daily_counts = data['day_of_week'].value_counts()
        fig_daily = px.bar(
            x=daily_counts.index, 
            y=daily_counts.values,
            title="Collision Frequency by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Number of Collisions'}
        )
        fig_daily.update_layout(showlegend=False)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Monthly trend
    st.subheader("Monthly Trend")
    monthly_counts = data.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
    
    fig_monthly = px.line(
        monthly_counts, 
        x='date', 
        y='count',
        title="Monthly Collision Trend",
        labels={'date': 'Date', 'count': 'Number of Collisions'}
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

def create_geographic_analysis(data):
    """Create geographic analysis visualizations"""
    if data is None or len(data) == 0:
        return
    
    st.header("🗺️ Geographic Analysis")
    
    # Injury severity map
    st.subheader("Collision Map by Injury Severity")
    
    injury_threshold = st.slider(
        "Minimum number of injured persons to display",
        min_value=0,
        max_value=int(data['total_injured'].max()),
        value=1,
        help="Adjust this slider to filter collisions by injury severity"
    )
    
    filtered_data = data[data['total_injured'] >= injury_threshold]
    
    if len(filtered_data) > 0:
        # Calculate map center
        midpoint = (
            filtered_data['latitude'].mean(),
            filtered_data['longitude'].mean()
        )
        
        # Create 3D hexagon layer
        st.pydeck_chart(pdk.Deck(
            map_style=get_config()['map_style'],
            initial_view_state={
                "latitude": midpoint[0],
                "longitude": midpoint[1],
                "zoom": get_config()['default_zoom'],
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=filtered_data[['date/time', 'latitude', 'longitude', 'total_injured']],
                    get_position=['longitude', 'latitude'],
                    radius=100,
                    extruded=True,
                    pickable=True,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    get_fill_color=[255, 0, 0, 180],
                    get_line_color=[255, 255, 255],
                    tooltip={
                        "html": "<b>Injured:</b> {total_injured}<br/><b>Date:</b> {date/time}",
                        "style": {"backgroundColor": "steelblue", "color": "white"}
                    }
                ),
            ],
        ))
    else:
        st.info("No collisions found with the selected injury threshold")

def create_dangerous_locations_analysis(data):
    """Analyze and display dangerous locations"""
    if data is None or len(data) == 0:
        return
    
    st.header("⚠️ Dangerous Locations Analysis")
    
    # Street analysis
    st.subheader("Most Dangerous Streets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Streets with most injuries
        street_injuries = data.groupby('on_street_name')['total_injured'].sum().sort_values(ascending=False).head(10)
        fig_streets = px.bar(
            x=street_injuries.values,
            y=street_injuries.index,
            orientation='h',
            title="Top 10 Streets by Total Injuries",
            labels={'x': 'Total Injuries', 'y': 'Street Name'}
        )
        st.plotly_chart(fig_streets, use_container_width=True)
    
    with col2:
        # Borough analysis
        if 'borough' in data.columns:
            borough_injuries = data.groupby('borough')['total_injured'].sum().sort_values(ascending=False)
            fig_borough = px.pie(
                values=borough_injuries.values,
                names=borough_injuries.index,
                title="Injuries by Borough"
            )
            st.plotly_chart(fig_borough, use_container_width=True)
    
    # Contributing factors
    st.subheader("Contributing Factors")
    
    # Check for contributing factor columns
    factor_columns = [col for col in data.columns if 'contributing_factor' in col.lower()]
    
    if factor_columns:
        # Analyze top contributing factors
        all_factors = []
        for col in factor_columns:
            factors = data[col].dropna()
            all_factors.extend(factors)
        
        factor_counts = pd.Series(all_factors).value_counts().head(10)
        
        fig_factors = px.bar(
            x=factor_counts.values,
            y=factor_counts.index,
            orientation='h',
            title="Top 10 Contributing Factors",
            labels={'x': 'Count', 'y': 'Factor'}
        )
        st.plotly_chart(fig_factors, use_container_width=True)
    else:
        st.info("Contributing factor data not available in this dataset")

def create_victim_analysis(data):
    """Analyze victim types and demographics"""
    if data is None or len(data) == 0:
        return
    
    st.header("👥 Victim Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Victim type breakdown
        victim_types = {
            'Pedestrians': data['injured_pedestrians'].sum(),
            'Cyclists': data['injured_cyclists'].sum(),
            'Motorists': data['injured_motorists'].sum(),
            'Others': data['injured_persons'].sum() - (
                data['injured_pedestrians'].sum() + 
                data['injured_cyclists'].sum() + 
                data['injured_motorists'].sum()
            )
        }
        
        fig_victims = px.pie(
            values=list(victim_types.values()),
            names=list(victim_types.keys()),
            title="Injuries by Victim Type"
        )
        st.plotly_chart(fig_victims, use_container_width=True)
    
    with col2:
        # Fatalities breakdown
        fatality_types = {
            'Pedestrians': data['killed_pedestrians'].sum(),
            'Cyclists': data['killed_cyclists'].sum(),
            'Motorists': data['killed_motorists'].sum(),
            'Others': data['killed_persons'].sum() - (
                data['killed_pedestrians'].sum() + 
                data['killed_cyclists'].sum() + 
                data['killed_motorists'].sum()
            )
        }
        
        fig_fatalities = px.pie(
            values=list(fatality_types.values()),
            names=list(fatality_types.keys()),
            title="Fatalities by Victim Type"
        )
        st.plotly_chart(fig_fatalities, use_container_width=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🚗 NYC Motor Vehicle Collisions Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive analysis and visualization of motor vehicle collisions in New York City")
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    
    config = get_config()
    
    # Data loading
    data_file = st.sidebar.text_input(
        "Data File Path",
        value=config['data_file'],
        help="Path to the Motor Vehicle Collisions CSV file"
    )
    
    max_rows = st.sidebar.number_input(
        "Maximum Rows to Load",
        min_value=1000,
        max_value=500000,
        value=config['max_rows'],
        step=10000,
        help="Limit the number of rows loaded to improve performance"
    )
    
    # Load data
    data = load_data(data_file, max_rows)
    
    if data is None:
        st.error("Failed to load data. Please check the file path and try again.")
        return
    
    # Main dashboard
    create_summary_metrics(data)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Time Analysis", 
        "🗺️ Geographic Analysis", 
        "⚠️ Dangerous Locations", 
        "👥 Victim Analysis",
        "📋 Raw Data"
    ])
    
    with tab1:
        create_time_analysis(data)
    
    with tab2:
        create_geographic_analysis(data)
    
    with tab3:
        create_dangerous_locations_analysis(data)
    
    with tab4:
        create_victim_analysis(data)
    
    with tab5:
        st.header("📋 Raw Data")
        
        # Data filters
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "Filter by Date Range",
                value=(data['date/time'].min().date(), data['date/time'].max().date()),
                min_value=data['date/time'].min().date(),
                max_value=data['date/time'].max().date()
            )
        
        with col2:
            min_injuries = st.number_input(
                "Minimum Injuries",
                min_value=0,
                max_value=int(data['total_injured'].max()),
                value=0
            )
        
        # Apply filters
        filtered_data = data[
            (data['date/time'].dt.date >= date_range[0]) &
            (data['date/time'].dt.date <= date_range[1]) &
            (data['total_injured'] >= min_injuries)
        ]
        
        st.write(f"Showing {len(filtered_data):,} records")
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"nyc_collisions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
