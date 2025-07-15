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

# --- Modern UI Customization ---
MODERN_CSS = """
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        background-color: #f7f9fb;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a2639;
        text-align: left;
        margin-bottom: 1.5rem;
        letter-spacing: -1px;
    }
    .metric-card {
        background: #fff;
        box-shadow: 0 2px 12px 0 rgba(31,39,79,0.07);
        border-radius: 1rem;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #1f77b4;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 24px 0 rgba(31,39,79,0.13);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a2639;
        border-radius: 0.5rem 0.5rem 0 0;
        background: #f0f2f6;
        margin-bottom: -2px;
        transition: background 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #fff;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
    }
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
        background: #1f77b4;
        color: #fff;
        border: none;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #174e7c;
    }
    .footer {
        margin-top: 2.5rem;
        padding: 1.2rem 0 0.5rem 0;
        text-align: center;
        color: #7a869a;
        font-size: 1rem;
        border-top: 1px solid #e6e8ec;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 1.5rem;
    }
    .sidebar-logo img {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        box-shadow: 0 2px 8px 0 rgba(31,39,79,0.10);
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f77b4;
        letter-spacing: -0.5px;
    }
    .about-section {
        font-size: 1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# --- Sidebar Modernization ---
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '<img src="https://img.icons8.com/color/96/000000/car-crash.png" alt="Logo">'
        '<span class="sidebar-title">NYC Collisions</span>'
        '</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="about-section">'
        'A modern dashboard for analyzing and visualizing motor vehicle collisions in New York City. <br>'
        '<b>Built with Streamlit, Plotly, and PyDeck.</b>'
        '</div>', unsafe_allow_html=True
    )

# --- Logging and Page Config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="NYC Motor Vehicle Collisions Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration and Data Loading (unchanged) ---
@st.cache_resource
def get_config():
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
    try:
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            st.info("Please ensure the Motor_Vehicle_Collisions_-_Crashes.csv file is in the same directory as this application.")
            return None
        with st.spinner("Loading collision data..."):
            data = pd.read_csv(
                file_path, 
                nrows=max_rows, 
                parse_dates=[['CRASH_DATE', 'CRASH_TIME']],
                low_memory=False
            )
        st.info("Preprocessing data...")
        initial_count = len(data)
        data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
        final_count = len(data)
        if final_count < initial_count:
            st.warning(f"Removed {initial_count - final_count} rows with missing coordinates")
        data.columns = data.columns.str.lower()
        data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
        if not pd.api.types.is_datetime64_any_dtype(data['date/time']):
            data['date/time'] = pd.to_datetime(data['date/time'], errors='coerce')
        data.dropna(subset=['date/time'], inplace=True)
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

# --- Modern Metric Cards ---
def create_summary_metrics(data):
    if data is None or len(data) == 0:
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">🚗<br><span style="font-size:1.5rem;font-weight:700">{}</span><br><span style="color:#7a869a">Total Collisions</span></div>'.format(f"{len(data):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🩹<br><span style="font-size:1.5rem;font-weight:700">{}</span><br><span style="color:#7a869a">Total Injured</span></div>'.format(f"{data['total_injured'].sum():,}"), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">☠️<br><span style="font-size:1.5rem;font-weight:700">{}</span><br><span style="color:#7a869a">Total Fatalities</span></div>'.format(f"{data['total_killed'].sum():,}"), unsafe_allow_html=True)
    with col4:
        date_range = f"{data['date/time'].min().strftime('%Y-%m-%d')} to {data['date/time'].max().strftime('%Y-%m-%d')}"
        st.markdown('<div class="metric-card">📅<br><span style="font-size:1.1rem;font-weight:700">{}</span><br><span style="color:#7a869a">Date Range</span></div>'.format(date_range), unsafe_allow_html=True)

# --- Chart Styling Helper ---
def modernize_plotly(fig):
    fig.update_layout(
        font_family="Inter, Segoe UI, Arial, sans-serif",
        font_color="#1a2639",
        plot_bgcolor="#f7f9fb",
        paper_bgcolor="#fff",
        margin=dict(l=30, r=30, t=60, b=30),
        title_font=dict(size=20, color="#1f77b4", family="Inter, Segoe UI, Arial, sans-serif"),
        legend=dict(bgcolor="#fff", bordercolor="#e6e8ec", borderwidth=1, font=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor="#e6e8ec"),
        yaxis=dict(showgrid=True, gridcolor="#e6e8ec"),
        hoverlabel=dict(bgcolor="#1f77b4", font_size=13, font_family="Inter, Segoe UI, Arial, sans-serif"),
    )
    return fig

# --- Time Analysis ---
def create_time_analysis(data):
    if data is None or len(data) == 0:
        return
    st.header("📊 Time-based Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Collisions by Hour of Day")
        hourly_counts = data['hour'].value_counts().sort_index()
        fig_hourly = px.bar(
            x=hourly_counts.index, 
            y=hourly_counts.values,
            title="Collision Frequency by Hour",
            labels={'x': 'Hour of Day', 'y': 'Number of Collisions'},
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(modernize_plotly(fig_hourly), use_container_width=True)
    with col2:
        st.subheader("Collisions by Day of Week")
        daily_counts = data['day_of_week'].value_counts()
        fig_daily = px.bar(
            x=daily_counts.index, 
            y=daily_counts.values,
            title="Collision Frequency by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Number of Collisions'},
            color_discrete_sequence=["#ff7f0e"]
        )
        st.plotly_chart(modernize_plotly(fig_daily), use_container_width=True)
    st.subheader("Monthly Trend")
    monthly_counts = data.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
    fig_monthly = px.line(
        monthly_counts, 
        x='date', 
        y='count',
        title="Monthly Collision Trend",
        labels={'date': 'Date', 'count': 'Number of Collisions'},
        markers=True,
        color_discrete_sequence=["#2ca02c"]
    )
    st.plotly_chart(modernize_plotly(fig_monthly), use_container_width=True)

# --- Geographic Analysis ---
def create_geographic_analysis(data):
    if data is None or len(data) == 0:
        return
    st.header("🗺️ Geographic Analysis")
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
        midpoint = (
            filtered_data['latitude'].mean(),
            filtered_data['longitude'].mean()
        )
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

# --- Dangerous Locations ---
def create_dangerous_locations_analysis(data):
    if data is None or len(data) == 0:
        return
    st.header("⚠️ Dangerous Locations Analysis")
    st.subheader("Most Dangerous Streets")
    col1, col2 = st.columns(2)
    with col1:
        street_injuries = data.groupby('on_street_name')['total_injured'].sum().sort_values(ascending=False).head(10)
        fig_streets = px.bar(
            x=street_injuries.values,
            y=street_injuries.index,
            orientation='h',
            title="Top 10 Streets by Total Injuries",
            labels={'x': 'Total Injuries', 'y': 'Street Name'},
            color_discrete_sequence=["#d62728"]
        )
        st.plotly_chart(modernize_plotly(fig_streets), use_container_width=True)
    with col2:
        if 'borough' in data.columns:
            borough_injuries = data.groupby('borough')['total_injured'].sum().sort_values(ascending=False)
            fig_borough = px.pie(
                values=borough_injuries.values,
                names=borough_injuries.index,
                title="Injuries by Borough",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig_borough.update_traces(textinfo='percent+label', pull=[0.05]*len(borough_injuries))
            st.plotly_chart(modernize_plotly(fig_borough), use_container_width=True)
    st.subheader("Contributing Factors")
    factor_columns = [col for col in data.columns if 'contributing_factor' in col.lower()]
    if factor_columns:
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
            labels={'x': 'Count', 'y': 'Factor'},
            color_discrete_sequence=["#9467bd"]
        )
        st.plotly_chart(modernize_plotly(fig_factors), use_container_width=True)
    else:
        st.info("Contributing factor data not available in this dataset")

# --- Victim Analysis ---
def create_victim_analysis(data):
    if data is None or len(data) == 0:
        return
    st.header("👥 Victim Analysis")
    col1, col2 = st.columns(2)
    with col1:
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
            title="Injuries by Victim Type",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_victims.update_traces(textinfo='percent+label', pull=[0.05]*len(victim_types))
        st.plotly_chart(modernize_plotly(fig_victims), use_container_width=True)
    with col2:
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
            title="Fatalities by Victim Type",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_fatalities.update_traces(textinfo='percent+label', pull=[0.05]*len(fatality_types))
        st.plotly_chart(modernize_plotly(fig_fatalities), use_container_width=True)

# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🚗 NYC Motor Vehicle Collisions Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### <span style='color:#4a5568'>Comprehensive, modern analysis and visualization of motor vehicle collisions in New York City</span>", unsafe_allow_html=True)
    config = get_config()
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
    data = load_data(data_file, max_rows)
    if data is None:
        st.error("Failed to load data. Please check the file path and try again.")
        return
    create_summary_metrics(data)
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
        filtered_data = data[
            (data['date/time'].dt.date >= date_range[0]) &
            (data['date/time'].dt.date <= date_range[1]) &
            (data['total_injured'] >= min_injuries)
        ]
        st.write(f"Showing {len(filtered_data):,} records")
        st.dataframe(filtered_data, use_container_width=True)
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"nyc_collisions_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    # --- Footer ---
    st.markdown(
        '<div class="footer">'
        'Made with ❤️ by <a href="https://github.com/yourusername/nyc-collisions-analysis" style="color:#1f77b4;text-decoration:none;font-weight:600;">NYC Collisions Team</a> | '
        '<a href="https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95" style="color:#7a869a;text-decoration:none;">NYC Open Data</a>'
        '</div>', unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
