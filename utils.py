"""
Utility functions for NYC Motor Vehicle Collisions Analysis Dashboard.

This module contains helper functions for data processing, validation,
and common operations used throughout the application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any, List
import re

from config import (
    DATA_CONFIG, VALIDATION_CONFIG, ERROR_MESSAGES, 
    SUCCESS_MESSAGES, EXPORT_CONFIG
)

logger = logging.getLogger(__name__)

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat (float): Latitude value
        lon (float): Longitude value
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    try:
        lat_valid = VALIDATION_CONFIG['latitude_range'][0] <= lat <= VALIDATION_CONFIG['latitude_range'][1]
        lon_valid = VALIDATION_CONFIG['longitude_range'][0] <= lon <= VALIDATION_CONFIG['longitude_range'][1]
        
        # Check if coordinates are within NYC bounds
        nyc_bounds = VALIDATION_CONFIG['nyc_bounds']
        nyc_valid = (
            nyc_bounds['lat_min'] <= lat <= nyc_bounds['lat_max'] and
            nyc_bounds['lon_min'] <= lon <= nyc_bounds['lon_max']
        )
        
        return lat_valid and lon_valid and nyc_valid
    except (TypeError, ValueError):
        return False

def validate_date(date_value: Any) -> bool:
    """
    Validate date values.
    
    Args:
        date_value: Date value to validate
        
    Returns:
        bool: True if date is valid, False otherwise
    """
    try:
        if pd.isna(date_value):
            return False
        
        if isinstance(date_value, str):
            date_obj = pd.to_datetime(date_value)
        elif isinstance(date_value, (datetime, pd.Timestamp)):
            date_obj = date_value
        else:
            return False
        
        year = date_obj.year
        return VALIDATION_CONFIG['date_range']['min_year'] <= year <= VALIDATION_CONFIG['date_range']['max_year']
    except (ValueError, TypeError):
        return False

def clean_street_name(street_name: str) -> str:
    """
    Clean and standardize street names.
    
    Args:
        street_name (str): Raw street name
        
    Returns:
        str: Cleaned street name
    """
    if pd.isna(street_name) or not isinstance(street_name, str):
        return "Unknown"
    
    # Remove extra whitespace and convert to title case
    cleaned = re.sub(r'\s+', ' ', street_name.strip()).title()
    
    # Handle common abbreviations
    abbreviations = {
        'St': 'Street',
        'Ave': 'Avenue',
        'Blvd': 'Boulevard',
        'Dr': 'Drive',
        'Ln': 'Lane',
        'Rd': 'Road',
        'Pl': 'Place',
        'Ct': 'Court',
        'Way': 'Way',
        'Hwy': 'Highway',
        'Pkwy': 'Parkway'
    }
    
    for abbr, full in abbreviations.items():
        cleaned = re.sub(rf'\b{abbr}\b', full, cleaned, flags=re.IGNORECASE)
    
    return cleaned

def calculate_severity_score(row: pd.Series) -> float:
    """
    Calculate a severity score for a collision based on injuries and fatalities.
    
    Args:
        row (pd.Series): Row from collision dataset
        
    Returns:
        float: Severity score (higher = more severe)
    """
    try:
        # Weight fatalities more heavily than injuries
        fatalities = sum([
            row.get('killed_persons', 0) or 0,
            row.get('killed_pedestrians', 0) or 0,
            row.get('killed_cyclists', 0) or 0,
            row.get('killed_motorists', 0) or 0
        ])
        
        injuries = sum([
            row.get('injured_persons', 0) or 0,
            row.get('injured_pedestrians', 0) or 0,
            row.get('injured_cyclists', 0) or 0,
            row.get('injured_motorists', 0) or 0
        ])
        
        # Severity score: fatalities * 10 + injuries
        return fatalities * 10 + injuries
    except Exception as e:
        logger.warning(f"Error calculating severity score: {e}")
        return 0.0

def get_time_period_label(hour: int) -> str:
    """
    Get human-readable time period label for a given hour.
    
    Args:
        hour (int): Hour of day (0-23)
        
    Returns:
        str: Time period label
    """
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def get_season(date: datetime) -> str:
    """
    Get season for a given date.
    
    Args:
        date (datetime): Date to determine season for
        
    Returns:
        str: Season name
    """
    month = date.month
    
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def format_number(value: float, decimals: int = 0) -> str:
    """
    Format numbers with appropriate suffixes (K, M, etc.).
    
    Args:
        value (float): Number to format
        decimals (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if value >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        return f"{value/1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def get_data_summary_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for the dataset.
    
    Args:
        data (pd.DataFrame): Collision dataset
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    if data is None or len(data) == 0:
        return {}
    
    try:
        stats = {
            'total_collisions': len(data),
            'date_range': {
                'start': data['date/time'].min().strftime('%Y-%m-%d'),
                'end': data['date/time'].max().strftime('%Y-%m-%d')
            },
            'total_injured': data['total_injured'].sum(),
            'total_fatalities': data['total_killed'].sum(),
            'avg_injuries_per_collision': data['total_injured'].mean(),
            'collisions_with_injuries': len(data[data['total_injured'] > 0]),
            'collisions_with_fatalities': len(data[data['total_killed'] > 0]),
            'unique_streets': data['on_street_name'].nunique(),
            'unique_boroughs': data.get('borough', pd.Series()).nunique() if 'borough' in data.columns else 0,
            'most_common_hour': data['hour'].mode().iloc[0] if len(data['hour'].mode()) > 0 else None,
            'most_common_day': data['day_of_week'].mode().iloc[0] if len(data['day_of_week'].mode()) > 0 else None
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating summary stats: {e}")
        return {}

def filter_data_by_criteria(
    data: pd.DataFrame,
    date_range: Optional[Tuple[datetime, datetime]] = None,
    borough: Optional[str] = None,
    min_injuries: int = 0,
    max_injuries: Optional[int] = None,
    time_period: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter dataset based on multiple criteria.
    
    Args:
        data (pd.DataFrame): Original dataset
        date_range (Tuple[datetime, datetime], optional): Date range filter
        borough (str, optional): Borough filter
        min_injuries (int): Minimum injuries filter
        max_injuries (int, optional): Maximum injuries filter
        time_period (str, optional): Time period filter (Morning, Afternoon, Evening, Night)
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    filtered = data.copy()
    
    try:
        # Date range filter
        if date_range:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered['date/time'].dt.date >= start_date.date()) &
                (filtered['date/time'].dt.date <= end_date.date())
            ]
        
        # Borough filter
        if borough and 'borough' in filtered.columns:
            filtered = filtered[filtered['borough'].str.contains(borough, case=False, na=False)]
        
        # Injuries filter
        filtered = filtered[filtered['total_injured'] >= min_injuries]
        
        if max_injuries is not None:
            filtered = filtered[filtered['total_injured'] <= max_injuries]
        
        # Time period filter
        if time_period:
            time_periods = {
                'Morning': (5, 12),
                'Afternoon': (12, 17),
                'Evening': (17, 21),
                'Night': (21, 5)
            }
            
            if time_period in time_periods:
                start_hour, end_hour = time_periods[time_period]
                if start_hour < end_hour:
                    filtered = filtered[
                        (filtered['hour'] >= start_hour) & (filtered['hour'] < end_hour)
                    ]
                else:  # Night period spans midnight
                    filtered = filtered[
                        (filtered['hour'] >= start_hour) | (filtered['hour'] < end_hour)
                    ]
        
        return filtered
        
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        return data

def export_data_to_csv(data: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Export filtered data to CSV format.
    
    Args:
        data (pd.DataFrame): Data to export
        filename (str, optional): Custom filename
        
    Returns:
        str: Generated filename
    """
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = EXPORT_CONFIG['filename_template'].format(timestamp=timestamp)
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Export data
        data.to_csv(
            filename,
            index=False,
            encoding=EXPORT_CONFIG['csv_encoding'],
            date_format=EXPORT_CONFIG['date_format']
        )
        
        logger.info(f"Data exported successfully to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise

def get_top_locations(data: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Get top locations by a specific metric.
    
    Args:
        data (pd.DataFrame): Dataset
        column (str): Column to aggregate by
        top_n (int): Number of top results to return
        
    Returns:
        pd.DataFrame: Top locations
    """
    try:
        if column not in data.columns:
            return pd.DataFrame()
        
        # Group by location and sum injuries
        grouped = data.groupby(column)['total_injured'].agg(['sum', 'count']).reset_index()
        grouped.columns = [column, 'total_injuries', 'collision_count']
        
        # Sort by total injuries and get top N
        top_locations = grouped.sort_values('total_injuries', ascending=False).head(top_n)
        
        return top_locations
        
    except Exception as e:
        logger.error(f"Error getting top locations: {e}")
        return pd.DataFrame()

def calculate_trends(data: pd.DataFrame, time_column: str = 'date/time') -> Dict[str, float]:
    """
    Calculate trend indicators for the dataset.
    
    Args:
        data (pd.DataFrame): Dataset
        time_column (str): Column containing time information
        
    Returns:
        Dict[str, float]: Trend indicators
    """
    try:
        # Group by month and calculate monthly totals
        monthly_data = data.groupby(data[time_column].dt.to_period('M')).agg({
            'total_injured': 'sum',
            'total_killed': 'sum'
        }).reset_index()
        
        if len(monthly_data) < 2:
            return {}
        
        # Calculate trends using linear regression
        x = np.arange(len(monthly_data))
        y_injuries = monthly_data['total_injured'].values
        y_fatalities = monthly_data['total_killed'].values
        
        # Linear regression for injuries
        slope_injuries = np.polyfit(x, y_injuries, 1)[0]
        
        # Linear regression for fatalities
        slope_fatalities = np.polyfit(x, y_fatalities, 1)[0]
        
        return {
            'injury_trend': slope_injuries,
            'fatality_trend': slope_fatalities,
            'trend_periods': len(monthly_data)
        }
        
    except Exception as e:
        logger.error(f"Error calculating trends: {e}")
        return {}