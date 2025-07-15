"""
Configuration settings for NYC Motor Vehicle Collisions Analysis Dashboard.

This module contains all configuration parameters for the application,
including data sources, visualization settings, and performance options.
"""

import os
from pathlib import Path

# Application Configuration
APP_CONFIG = {
    'name': 'NYC Motor Vehicle Collisions Analysis',
    'version': '2.0.0',
    'description': 'Comprehensive analysis and visualization of motor vehicle collisions in New York City',
    'author': 'Your Name',
    'email': 'your.email@example.com',
    'github': 'https://github.com/yourusername/nyc-collisions-analysis'
}

# Data Configuration
DATA_CONFIG = {
    'default_file': 'Motor_Vehicle_Collisions_-_Crashes.csv',
    'max_rows': 100000,
    'min_rows': 1000,
    'date_columns': ['CRASH_DATE', 'CRASH_TIME'],
    'required_columns': ['LATITUDE', 'LONGITUDE', 'CRASH_DATE', 'CRASH_TIME'],
    'injury_columns': [
        'injured_persons', 'injured_pedestrians', 'injured_cyclists', 'injured_motorists'
    ],
    'fatality_columns': [
        'killed_persons', 'killed_pedestrians', 'killed_cyclists', 'killed_motorists'
    ]
}

# Map Configuration
MAP_CONFIG = {
    'style': "mapbox://styles/mapbox/light-v9",
    'default_lat': 40.7128,
    'default_lon': -74.0060,
    'default_zoom': 10,
    'pitch': 50,
    'hexagon_radius': 100,
    'elevation_scale': 4,
    'elevation_range': [0, 1000]
}

# Visualization Configuration
VIZ_CONFIG = {
    'chart_height': 400,
    'color_scheme': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#9467bd'
    },
    'max_display_items': 10,
    'animation_duration': 500
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'cache_ttl': 3600,  # 1 hour
    'max_memory_usage': '2GB',
    'chunk_size': 10000,
    'enable_profiling': False
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'NYC Motor Vehicle Collisions Analysis',
    'page_icon': '🚗',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'primaryColor': '#1f77b4',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730'
    }
}

# Error Messages
ERROR_MESSAGES = {
    'file_not_found': 'Data file not found. Please ensure the CSV file is in the correct location.',
    'invalid_data': 'Invalid data format. Please check the CSV file structure.',
    'loading_error': 'Error loading data. Please try again or check the file path.',
    'processing_error': 'Error processing data. Please contact support if the issue persists.',
    'no_data': 'No data available for the selected criteria.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Data loaded successfully',
    'export_complete': 'Data exported successfully',
    'analysis_complete': 'Analysis completed successfully'
}

# File Paths
def get_data_path():
    """Get the path to the data file."""
    current_dir = Path(__file__).parent
    return current_dir / DATA_CONFIG['default_file']

def get_log_path():
    """Get the path for log files."""
    current_dir = Path(__file__).parent
    logs_dir = current_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    return logs_dir / 'app.log'

# Environment Configuration
def get_environment_config():
    """Get environment-specific configuration."""
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return {
            'debug': False,
            'log_level': 'WARNING',
            'cache_ttl': 7200,  # 2 hours
            'max_rows': 500000
        }
    elif env == 'staging':
        return {
            'debug': True,
            'log_level': 'INFO',
            'cache_ttl': 1800,  # 30 minutes
            'max_rows': 200000
        }
    else:  # development
        return {
            'debug': True,
            'log_level': 'DEBUG',
            'cache_ttl': 300,  # 5 minutes
            'max_rows': 50000
        }

# Validation Configuration
VALIDATION_CONFIG = {
    'latitude_range': (-90, 90),
    'longitude_range': (-180, 180),
    'nyc_bounds': {
        'lat_min': 40.4774,
        'lat_max': 40.9176,
        'lon_min': -74.2591,
        'lon_max': -73.7004
    },
    'date_range': {
        'min_year': 2012,
        'max_year': 2030
    }
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_export_rows': 100000,
    'filename_template': 'nyc_collisions_{timestamp}.csv'
}