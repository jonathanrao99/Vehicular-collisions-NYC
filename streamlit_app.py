"""
NYC Motor Vehicle Collisions AI Dashboard
Advanced analytics and machine learning for traffic safety analysis
"""

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NYC Collisions AI Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "NYC Motor Vehicle Collisions AI Dashboard - Advanced analytics powered by machine learning"
    }
)

# Import dependencies with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap, Fullscreen, MousePosition
    import streamlit.components.v1 as components
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

# --- Performance Settings ---
@st.cache_resource
def get_app_config():
    """Cache application configuration for better performance"""
    return {
        'data_sample_size': 100000,  # Increased for better analysis
        'cache_ttl': 3600,  # 1 hour cache
        'map_sample_size': 10000,  # Increased for better visualization
        'chart_height': 500,
        'animation_enabled': True
    }

# --- Enhanced Modern CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
        max-width: 1400px;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.3;
    }
    
    .dashboard-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        line-height: 1.1;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        border-color: #667eea;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0;
        line-height: 1;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        margin-top: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.875rem 1.75rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
        font-size: 0.95rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -5px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f7fafc;
        padding: 8px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 14px;
        color: #718096;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        border: none;
        padding: 14px 24px;
        transition: all 0.2s ease;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
    }
    
    .chart-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
        border: 1px solid #3b82f6;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3cd 0%, #fffbeb 100%);
        border: 1px solid #f59e0b;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(34, 197, 94, 0.1);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    @media (max-width: 768px) {
        .dashboard-title {
            font-size: 2.5rem;
        }
        .metric-container {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
NYC_BOROUGHS = {
    'Manhattan': [40.7589, -73.9851, '#FF6B6B'],
    'Brooklyn': [40.6782, -73.9442, '#4ECDC4'],
    'Queens': [40.7282, -73.7949, '#45B7D1'],
    'Bronx': [40.8448, -73.8648, '#96CEB4'],
    'Staten Island': [40.5795, -74.1502, '#FFEAA7']
}

# Initialize session state
if 'selected_lat' not in st.session_state:
    st.session_state.selected_lat = 40.7128
if 'selected_lon' not in st.session_state:
    st.session_state.selected_lon = -74.0060

# --- Enhanced Data Loading with Caching ---
@st.cache_data(ttl=get_app_config()['cache_ttl'])
def load_data(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load and preprocess collision data with enhanced caching"""
    try:
        # Try to load the data
        if os.path.exists('Motor_Vehicle_Collisions_-_Crashes.csv'):
            df = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes.csv')
        else:
            st.error("Data file not found. Please upload 'Motor_Vehicle_Collisions_-_Crashes.csv'")
            return pd.DataFrame()
        
        # Sample data for performance if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Enhanced preprocessing
        df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], errors='coerce')
        df['CRASH_TIME'] = pd.to_datetime(df['CRASH_TIME'], format='%H:%M', errors='coerce')
        
        # Extract enhanced time features
        df['year'] = df['CRASH_DATE'].dt.year
        df['month'] = df['CRASH_DATE'].dt.month
        df['day_of_week'] = df['CRASH_DATE'].dt.dayofweek
        df['hour'] = df['CRASH_TIME'].dt.hour
        df['day_name'] = df['CRASH_DATE'].dt.day_name()
        df['month_name'] = df['CRASH_DATE'].dt.month_name()
        df['quarter'] = df['CRASH_DATE'].dt.quarter
        
        # Enhanced binary features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
        
        # Calculate casualties with enhanced logic
        injury_cols = [col for col in df.columns if 'INJURED' in col]
        killed_cols = [col for col in df.columns if 'KILLED' in col]
        
        df['total_injured'] = df[injury_cols].sum(axis=1, skipna=True)
        df['total_killed'] = df[killed_cols].sum(axis=1, skipna=True)
        df['total_casualties'] = df['total_injured'] + df['total_killed']
        
        # Enhanced severity classification
        df['is_serious'] = ((df['total_killed'] > 0) | (df['total_injured'] >= 2)).astype(int)
        df['severity_level'] = pd.cut(
            df['total_casualties'], 
            bins=[-1, 0, 1, 3, float('inf')], 
            labels=['No Injury', 'Minor', 'Moderate', 'Severe']
        )
        
        # Enhanced risk scoring
        df['risk_score'] = (
            df['total_killed'] * 10 + 
            df['total_injured'] * 3 + 
            df['is_rush_hour'] * 2 + 
            df['is_night'] * 1.5 + 
            df['is_weekend'] * 1.2 +
            df['is_holiday_season'] * 1.3
        )
        
        # Location processing
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            
            # Enhanced borough classification
            def get_borough_enhanced(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return 'Unknown'
                # More precise borough boundaries
                if 40.7 <= lat <= 40.83 and -74.02 <= lon <= -73.91:
                    return 'Manhattan'
                elif 40.57 <= lat <= 40.74 and -74.05 <= lon <= -73.83:
                    return 'Brooklyn'
                elif 40.67 <= lat <= 40.81 and -73.96 <= lon <= -73.70:
                    return 'Queens'
                elif 40.79 <= lat <= 40.92 and -73.93 <= lon <= -73.77:
                    return 'Bronx'
                elif 40.49 <= lat <= 40.65 and -74.26 <= lon <= -74.05:
                    return 'Staten Island'
                else:
                    return 'Other NYC Area'
            
            df['borough'] = df.apply(lambda x: get_borough_enhanced(x['LATITUDE'], x['LONGITUDE']), axis=1)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['CRASH_DATE', 'hour'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# --- Enhanced ML Model Training ---
@st.cache_resource
def train_enhanced_models(data: pd.DataFrame) -> Dict[str, Any]:
    """Train multiple ML models with enhanced performance"""
    if not ML_AVAILABLE or data.empty:
        return {}
    
    try:
        # Prepare features
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_night', 'is_holiday_season']
        if 'LATITUDE' in data.columns and 'LONGITUDE' in data.columns:
            feature_cols.extend(['LATITUDE', 'LONGITUDE'])
        
        # Clean data
        ml_data = data[feature_cols + ['is_serious']].dropna()
        if len(ml_data) < 1000:
            return {}
        
        X = ml_data[feature_cols].astype(np.float32)
        y = ml_data['is_serious'].astype(np.int8)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {}
        
        # Random Forest with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        rf_score = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
        models['Random Forest'] = {'model': rf_model, 'score': rf_score}
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_score = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])
        models['Gradient Boosting'] = {'model': gb_model, 'score': gb_score}
        
        # XGBoost if available
        if XGB_AVAILABLE:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            xgb_score = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
            models['XGBoost'] = {'model': xgb_model, 'score': xgb_score}
        
        # LightGBM if available
        if LGB_AVAILABLE:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_score = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
            models['LightGBM'] = {'model': lgb_model, 'score': lgb_score}
        
        # Select best model
        best_model_name = max(models.keys(), key=lambda k: models[k]['score'])
        best_model = models[best_model_name]['model']
        
        return {
            'models': models,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'feature_names': feature_cols,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return {}

# --- Enhanced Prediction Interface ---
def create_enhanced_prediction_interface(model_info: Dict[str, Any]) -> None:
    """Create an enhanced prediction interface with interactive features"""
    st.subheader("🎯 AI-Powered Risk Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Location Selection")
        
        if MAP_AVAILABLE:
            # Interactive map
            m = folium.Map(
                location=[st.session_state.selected_lat, st.session_state.selected_lon],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add borough boundaries and markers
            for name, (lat, lon, color) in NYC_BOROUGHS.items():
                folium.CircleMarker(
                    [lat, lon],
                    radius=8,
                    popup=f"{name}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
            
            # Add selected location marker
            folium.Marker(
                [st.session_state.selected_lat, st.session_state.selected_lon],
                popup="Selected Location",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add plugins
            Fullscreen().add_to(m)
            MousePosition().add_to(m)
            
            # Display map
            map_html = m._repr_html_()
            components.html(map_html, height=400)
            
            # Location inputs
            col_lat, col_lon = st.columns(2)
            with col_lat:
                new_lat = st.number_input(
                    "Latitude", 
                    value=st.session_state.selected_lat, 
                    format="%.4f",
                    step=0.0001
                )
            with col_lon:
                new_lon = st.number_input(
                    "Longitude", 
                    value=st.session_state.selected_lon, 
                    format="%.4f",
                    step=0.0001
                )
            
            if new_lat != st.session_state.selected_lat or new_lon != st.session_state.selected_lon:
                st.session_state.selected_lat = new_lat
                st.session_state.selected_lon = new_lon
                st.rerun()
            
            # Quick presets
            st.markdown("**Quick Location Presets:**")
            preset_cols = st.columns(3)
            for i, (name, coords) in enumerate(list(NYC_BOROUGHS.items())[:3]):
                with preset_cols[i]:
                    if st.button(name, key=f"preset_{i}", use_container_width=True):
                        st.session_state.selected_lat = coords[0]
                        st.session_state.selected_lon = coords[1]
                        st.rerun()
        
        else:
            st.warning("Map visualization not available. Using coordinate inputs.")
            st.session_state.selected_lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            st.session_state.selected_lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
    
    with col2:
        st.markdown("### ⏰ Time & Conditions")
        
        # Time inputs
        selected_hour = st.slider("Hour of Day", 0, 23, 12)
        selected_day = st.selectbox(
            "Day of Week",
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        selected_month = st.selectbox(
            "Month",
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December']
        )
        
        # Convert to numeric
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        day_num = day_mapping[selected_day]
        month_num = month_mapping[selected_month]
        
        # Calculate derived features
        is_weekend = 1 if day_num in [5, 6] else 0
        is_rush_hour = 1 if (7 <= selected_hour <= 9) or (17 <= selected_hour <= 19) else 0
        is_night = 1 if selected_hour >= 22 or selected_hour <= 5 else 0
        is_holiday_season = 1 if month_num in [11, 12, 1] else 0
        
        # Display condition indicators
        st.markdown("**Conditions:**")
        conditions = []
        if is_weekend:
            conditions.append("🏠 Weekend")
        if is_rush_hour:
            conditions.append("🚗 Rush Hour")
        if is_night:
            conditions.append("🌙 Night Time")
        if is_holiday_season:
            conditions.append("🎄 Holiday Season")
        
        if conditions:
            for condition in conditions:
                st.markdown(f"- {condition}")
        else:
            st.markdown("- ✅ Normal Conditions")
    
    # Make prediction if model is available
    if model_info and 'best_model' in model_info:
        st.markdown("---")
        
        # Prepare features
        features = [
            selected_hour, day_num, month_num, is_weekend, 
            is_rush_hour, is_night, is_holiday_season
        ]
        
        # Add coordinates if available in model
        if 'LATITUDE' in model_info['feature_names']:
            features.extend([st.session_state.selected_lat, st.session_state.selected_lon])
        
        # Make prediction
        try:
            model = model_info['best_model']
            prediction_proba = model.predict_proba([features])[0]
            risk_probability = prediction_proba[1]
            
            # Display results
            st.markdown("### 🎯 Risk Assessment Results")
            
            # Risk level
            if risk_probability >= 0.7:
                risk_level = "HIGH RISK"
                risk_color = "🔴"
                advice = "Exercise extreme caution. Consider avoiding this time/location if possible."
            elif risk_probability >= 0.4:
                risk_level = "MODERATE RISK"
                risk_color = "🟡"
                advice = "Be extra cautious and follow all traffic safety measures."
            else:
                risk_level = "LOW RISK"
                risk_color = "🟢"
                advice = "Normal safety precautions recommended."
            
            # Results display
            col_prob, col_level = st.columns(2)
            with col_prob:
                st.metric(
                    "Risk Probability",
                    f"{risk_probability:.1%}",
                    delta=f"{risk_probability - 0.5:.1%} vs baseline"
                )
            with col_level:
                st.metric("Risk Level", f"{risk_color} {risk_level}")
            
            # Progress bar
            st.progress(risk_probability)
            
            # Advice
            if risk_probability >= 0.4:
                st.warning(f"⚠️ {advice}")
            else:
                st.success(f"✅ {advice}")
            
            # Model information
            with st.expander("🤖 Model Details"):
                st.write(f"**Model Used:** {model_info['best_model_name']}")
                if 'models' in model_info:
                    st.write("**Model Performance (AUC Scores):**")
                    for name, info in model_info['models'].items():
                        st.write(f"- {name}: {info['score']:.3f}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    else:
        st.info("🤖 ML models not available. Please check data and dependencies.")

# --- Enhanced Visualizations ---
def create_enhanced_time_analysis(data: pd.DataFrame) -> None:
    """Create enhanced time-based analysis with multiple charts"""
    st.subheader("📊 Temporal Analysis")
    
    config = get_app_config()
    
    # Hourly pattern
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accidents by Hour")
        hourly_data = data.groupby('hour').agg({
            'is_serious': ['count', 'mean']
        }).round(3)
        hourly_data.columns = ['Total', 'Serious_Rate']
        hourly_data = hourly_data.reset_index()
        
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=("Accident Volume vs Severity Rate",)
        )
        
        fig.add_trace(
            go.Bar(x=hourly_data['hour'], y=hourly_data['Total'], 
                   name="Total Accidents", opacity=0.7, marker_color='#667eea'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=hourly_data['hour'], y=hourly_data['Serious_Rate'], 
                      mode='lines+markers', name="Serious Rate", 
                      line=dict(color='#764ba2', width=3)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Number of Accidents", secondary_y=False)
        fig.update_yaxes(title_text="Serious Accident Rate", secondary_y=True)
        fig.update_layout(height=config['chart_height'], showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Weekly Patterns")
        daily_data = data.groupby(['day_name', 'day_of_week']).size().reset_index(name='count')
        daily_data = daily_data.sort_values('day_of_week')
        
        fig = px.bar(
            daily_data, x='day_name', y='count',
            title="Accidents by Day of Week",
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=config['chart_height'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    st.markdown("#### Monthly and Seasonal Trends")
    col3, col4 = st.columns(2)
    
    with col3:
        monthly_data = data.groupby(['month', 'month_name']).size().reset_index(name='count')
        monthly_data = monthly_data.sort_values('month')
        
        fig = px.line(
            monthly_data, x='month_name', y='count',
            title="Monthly Accident Trends",
            markers=True
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=config['chart_height'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Severity by conditions
        condition_data = data.groupby(['is_weekend', 'is_rush_hour', 'is_night'])['is_serious'].mean().reset_index()
        condition_data['condition'] = condition_data.apply(lambda x: 
            f"{'Weekend' if x['is_weekend'] else 'Weekday'}, " +
            f"{'Rush' if x['is_rush_hour'] else 'Non-Rush'}, " +
            f"{'Night' if x['is_night'] else 'Day'}", axis=1)
        
        fig = px.bar(
            condition_data, x='condition', y='is_serious',
            title="Serious Accident Rate by Conditions",
            color='is_serious',
            color_continuous_scale='Reds'
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=config['chart_height'])
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_geographic_analysis(data: pd.DataFrame) -> None:
    """Create enhanced geographic analysis with heatmaps and statistics"""
    st.subheader("🗺️ Geographic Analysis")
    
    config = get_app_config()
    
    if not MAP_AVAILABLE:
        st.warning("Map visualization not available")
        return
    
    # Borough analysis
    if 'borough' in data.columns:
        st.markdown("#### Borough Comparison")
        borough_stats = data.groupby('borough').agg({
            'is_serious': ['count', 'mean', 'sum'],
            'risk_score': 'mean'
        }).round(3)
        
        borough_stats.columns = ['Total_Accidents', 'Serious_Rate', 'Serious_Count', 'Avg_Risk_Score']
        borough_stats = borough_stats.reset_index().sort_values('Total_Accidents', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                borough_stats, x='borough', y='Total_Accidents',
                title="Total Accidents by Borough",
                color='Total_Accidents',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=config['chart_height'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                borough_stats, x='borough', y='Serious_Rate',
                title="Serious Accident Rate by Borough",
                color='Serious_Rate',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=config['chart_height'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Interactive heatmap
    st.markdown("#### Risk Heatmap")
    
    # Sample data for heatmap performance
    map_data = data.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if len(map_data) > config['map_sample_size']:
        map_data = map_data.sample(n=config['map_sample_size'], random_state=42)
    
    if len(map_data) > 0:
        # Create heatmap
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles='CartoDB positron')
        
        # Add heatmap layer
        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['risk_score']] 
                     for idx, row in map_data.iterrows() 
                     if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE'])]
        
        if heat_data:
            HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
        
        # Add borough markers
        for name, (lat, lon, color) in NYC_BOROUGHS.items():
            folium.CircleMarker(
                [lat, lon],
                radius=10,
                popup=f"{name}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=3
            ).add_to(m)
        
        # Add plugins
        Fullscreen().add_to(m)
        
        # Display map
        map_html = m._repr_html_()
        components.html(map_html, height=500)
        
        st.info("🔥 Heatmap shows risk intensity across NYC. Darker areas indicate higher risk.")

def create_enhanced_data_explorer(data: pd.DataFrame) -> None:
    """Create an enhanced data exploration interface"""
    st.subheader("🔍 Data Explorer")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Date Range", f"{(data['CRASH_DATE'].max() - data['CRASH_DATE'].min()).days} days")
    with col3:
        st.metric("Unique Locations", f"{data[['LATITUDE', 'LONGITUDE']].dropna().drop_duplicates().shape[0]:,}")
    
    # Interactive filters
    st.markdown("#### Interactive Filters")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        year_range = st.slider(
            "Year Range",
            min_value=int(data['year'].min()),
            max_value=int(data['year'].max()),
            value=(int(data['year'].min()), int(data['year'].max()))
        )
    
    with col_filter2:
        selected_boroughs = st.multiselect(
            "Boroughs",
            options=data['borough'].unique() if 'borough' in data.columns else [],
            default=data['borough'].unique() if 'borough' in data.columns else []
        )
    
    with col_filter3:
        severity_filter = st.selectbox(
            "Severity Level",
            options=['All', 'Serious Only', 'Minor Only'],
            index=0
        )
    
    # Apply filters
    filtered_data = data[
        (data['year'] >= year_range[0]) & 
        (data['year'] <= year_range[1])
    ]
    
    if selected_boroughs and 'borough' in data.columns:
        filtered_data = filtered_data[filtered_data['borough'].isin(selected_boroughs)]
    
    if severity_filter == 'Serious Only':
        filtered_data = filtered_data[filtered_data['is_serious'] == 1]
    elif severity_filter == 'Minor Only':
        filtered_data = filtered_data[filtered_data['is_serious'] == 0]
    
    # Display filtered statistics
    st.markdown(f"#### Filtered Data ({len(filtered_data):,} records)")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Injuries", f"{filtered_data['total_injured'].sum():,}")
    with col_stat2:
        st.metric("Fatalities", f"{filtered_data['total_killed'].sum():,}")
    with col_stat3:
        st.metric("Serious Rate", f"{filtered_data['is_serious'].mean():.1%}")
    with col_stat4:
        st.metric("Avg Risk Score", f"{filtered_data['risk_score'].mean():.1f}")
    
    # Sample data table
    if st.checkbox("Show Sample Data"):
        st.dataframe(
            filtered_data.head(1000)[['CRASH_DATE', 'CRASH_TIME', 'borough', 'total_injured', 
                                     'total_killed', 'is_serious', 'risk_score']],
            height=300
        )

# --- Main Application ---
def main():
    """Main application function with enhanced features"""
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">NYC Collisions AI Dashboard</h1>
        <p class="dashboard-subtitle">Advanced Machine Learning Analytics for Traffic Safety</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    config = get_app_config()
    
    with st.spinner("🔄 Loading and processing data..."):
        data = load_data(sample_size=config['data_sample_size'])
    
    if data.empty:
        st.error("❌ No data available. Please check the data file.")
        return
    
    # Display key metrics
    st.markdown("### 📈 Key Metrics")
    
    metric_cols = st.columns(5)
    metrics = [
        ("Total Accidents", f"{len(data):,}"),
        ("Total Injuries", f"{data['total_injured'].sum():,}"),
        ("Total Fatalities", f"{data['total_killed'].sum():,}"),
        ("Serious Rate", f"{data['is_serious'].mean():.1%}"),
        ("Avg Risk Score", f"{data['risk_score'].mean():.1f}")
    ]
    
    for i, (label, value) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Train ML models
    model_info = {}
    if ML_AVAILABLE:
        with st.spinner("🤖 Training AI models..."):
            model_info = train_enhanced_models(data)
        
        if model_info:
            st.success(f"✅ AI models trained successfully! Best model: {model_info.get('best_model_name', 'Unknown')}")
        else:
            st.warning("⚠️ Could not train ML models with current data")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 AI Prediction", 
        "📊 Time Analysis", 
        "🗺️ Geography", 
        "🔍 Data Explorer"
    ])
    
    with tab1:
        create_enhanced_prediction_interface(model_info)
    
    with tab2:
        create_enhanced_time_analysis(data)
    
    with tab3:
        create_enhanced_geographic_analysis(data)
    
    with tab4:
        create_enhanced_data_explorer(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #718096;">
        🚗 NYC Collisions AI Dashboard | Powered by Advanced Machine Learning<br>
        Data source: <a href="https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95" 
        style="color: #667eea;">NYC Open Data</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
