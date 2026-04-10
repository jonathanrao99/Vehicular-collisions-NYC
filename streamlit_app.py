"""
NYC motor vehicle collisions — Streamlit explorer and risk model UI.

Theme tokens live in .streamlit/config.toml (see README → UI theme).
Custom CSS below is scoped to classes prefixed with nyc- for maintainability.
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths & data source (methodology / freshness)
# -----------------------------------------------------------------------------
DATA_FILE = "Motor_Vehicle_Collisions_-_Crashes.csv"
NYC_OPEN_DATA_URL = (
    "https://data.cityofnewyork.us/Public-Safety/"
    "Motor-Vehicle-Collisions-Crashes/h9gi-nx95"
)

# Plotly semantic colors — align with [theme] primaryColor in config.toml (#0f766e)
CHART = {
    "accent": "#0f766e",
    "accent_dark": "#115e59",
    "neutral": "#64748b",
    "volume": "#475569",
    "serious_line": "#b45309",
    "grid": "#e2e8f0",
    "bar_volume": "#cbd5e1",
}

# Folium borough markers — distinct hues + labels in popup (not color-only)
BOROUGH_MAP_STYLE: List[Tuple[str, float, float, str]] = [
    ("Manhattan", 40.7589, -73.9851, "#0072B2"),
    ("Brooklyn", 40.6782, -73.9442, "#E69F00"),
    ("Queens", 40.7282, -73.7949, "#009E73"),
    ("Bronx", 40.8448, -73.8648, "#CC79A7"),
    ("Staten Island", 40.5795, -74.1502, "#D55E00"),
]
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DAY_TO_NUM = {d: i for i, d in enumerate(WEEKDAYS)}
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

st.set_page_config(
    page_title="NYC motor vehicle crashes",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"Data source: NYC Open Data ({NYC_OPEN_DATA_URL})",
    },
)

# Scoped layout CSS — see README UI theme; avoid global Streamlit overrides where possible.
st.markdown(
    """
<style>
  /* nyc- prefix: layout + typography helpers only; colors come from config.toml theme */
  .nyc-page-title { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em;
    color: var(--text-color); margin: 0 0 0.25rem 0; }
  .nyc-lead { font-size: 1rem; color: #64748b; line-height: 1.5; max-width: 52rem; margin: 0 0 1.25rem 0; }
  .nyc-section-kicker { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748b; margin: 2rem 0 0.5rem 0; }
  .nyc-method p, .nyc-method li { font-size: 0.9rem; color: #475569; line-height: 1.55; }
  .main .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }
  .nyc-kicker-tight { margin-top: 0.5rem !important; }
  .nyc-control-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #94a3b8; margin: 0 0 0.35rem 0; }
  /* Risk tab: highlight estimate block (Streamlit widgets follow this heading in-document order) */
  .nyc-estimate-anchor { border-left: 4px solid #0f766e; padding: 0.35rem 0 0 0.85rem; margin: 0.75rem 0 0.5rem 0; }
  .nyc-estimate-anchor h3 { font-size: 1.05rem; font-weight: 600; margin: 0; color: #0f172a; }
</style>
""",
    unsafe_allow_html=True,
)

# Optional imports
try:
    import plotly.express as px
except ImportError:
    px = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    xgb = None  # type: ignore

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    lgb = None  # type: ignore

try:
    import folium
    from folium.plugins import Fullscreen, HeatMap, MarkerCluster, MousePosition
    import streamlit.components.v1 as components

    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


@st.cache_resource
def get_app_config() -> Dict[str, Any]:
    return {
        "data_sample_default": 100_000,
        "data_sample_max": 250_000,
        "data_sample_min": 5_000,
        "cache_ttl": 3600,
        "map_sample_size": 10_000,
        "risk_map_crash_max": 5_000,
        "chart_height": 420,
    }


def _plotly_layout_base(title: str, y_primary: str, y_secondary: Optional[str] = None) -> Dict[str, Any]:
    layout: Dict[str, Any] = {
        "title": {"text": title, "font": {"size": 16}},
        "font": {"family": "sans-serif", "color": "#334155"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "height": get_app_config()["chart_height"],
        "margin": {"t": 48, "b": 48, "l": 56, "r": 56},
        "xaxis": {"showgrid": True, "gridcolor": CHART["grid"], "zeroline": False},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    }
    layout["yaxis"] = {
        "title": y_primary,
        "showgrid": True,
        "gridcolor": CHART["grid"],
        "zeroline": False,
        "tickformat": ",",
    }
    if y_secondary:
        layout["yaxis2"] = {
            "title": y_secondary,
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "tickformat": ".0%",
            "range": [0, 1],
        }
    return layout


def _normalize_collision_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    renames = {}
    if "CRASH DATE" in df.columns and "CRASH_DATE" not in df.columns:
        renames["CRASH DATE"] = "CRASH_DATE"
    if "CRASH TIME" in df.columns and "CRASH_TIME" not in df.columns:
        renames["CRASH TIME"] = "CRASH_TIME"
    if renames:
        df = df.rename(columns=renames)
    return df


def _preprocess_collision_df(df: pd.DataFrame) -> pd.DataFrame:
    df["CRASH_DATE"] = pd.to_datetime(df["CRASH_DATE"], errors="coerce")
    time_str = df["CRASH_TIME"].astype(str).str.strip()
    crash_dt = pd.to_datetime(
        df["CRASH_DATE"].dt.strftime("%Y-%m-%d") + " " + time_str,
        errors="coerce",
    )
    df["hour"] = crash_dt.dt.hour
    df["year"] = df["CRASH_DATE"].dt.year
    df["month"] = df["CRASH_DATE"].dt.month
    df["day_of_week"] = df["CRASH_DATE"].dt.dayofweek
    df["day_name"] = df["CRASH_DATE"].dt.day_name()
    df["month_name"] = df["CRASH_DATE"].dt.month_name()
    df["quarter"] = df["CRASH_DATE"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = (
        (df["hour"].between(7, 9)) | (df["hour"].between(17, 19))
    ).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12, 1]).astype(int)

    injury_cols = [c for c in df.columns if "INJURED" in c]
    killed_cols = [c for c in df.columns if "KILLED" in c]
    df["total_injured"] = df[injury_cols].sum(axis=1, skipna=True)
    df["total_killed"] = df[killed_cols].sum(axis=1, skipna=True)
    df["total_casualties"] = df["total_injured"] + df["total_killed"]
    df["is_serious"] = ((df["total_killed"] > 0) | (df["total_injured"] >= 2)).astype(int)
    df["severity_level"] = pd.cut(
        df["total_casualties"],
        bins=[-1, 0, 1, 3, float("inf")],
        labels=["No Injury", "Minor", "Moderate", "Severe"],
    )
    df["risk_score"] = (
        df["total_killed"] * 10
        + df["total_injured"] * 3
        + df["is_rush_hour"] * 2
        + df["is_night"] * 1.5
        + df["is_weekend"] * 1.2
        + df["is_holiday_season"] * 1.3
    )

    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

        def borough(lat: float, lon: float) -> str:
            if pd.isna(lat) or pd.isna(lon):
                return "Unknown"
            if 40.7 <= lat <= 40.83 and -74.02 <= lon <= -73.91:
                return "Manhattan"
            if 40.57 <= lat <= 40.74 and -74.05 <= lon <= -73.83:
                return "Brooklyn"
            if 40.67 <= lat <= 40.81 and -73.96 <= lon <= -73.70:
                return "Queens"
            if 40.79 <= lat <= 40.92 and -73.93 <= lon <= -73.77:
                return "Bronx"
            if 40.49 <= lat <= 40.65 and -74.26 <= lon <= -74.05:
                return "Staten Island"
            return "Other NYC Area"

        df["borough"] = df.apply(lambda r: borough(r["LATITUDE"], r["LONGITUDE"]), axis=1)

    return df.dropna(subset=["CRASH_DATE", "hour"])


@st.cache_data(ttl=get_app_config()["cache_ttl"], show_spinner=False)
def load_collision_data(sample_size: Optional[int], path: str, file_mtime: float) -> Dict[str, Any]:
    """
    Load CSV; cache invalidates when file_mtime changes (replace CSV → auto-refresh).
    No Streamlit UI calls inside (cache-safe).
    """
    out: Dict[str, Any] = {
        "df": pd.DataFrame(),
        "error": None,
        "rows_read": 0,
        "rows_after_clean": 0,
    }
    try:
        if not os.path.isfile(path):
            out["error"] = "missing_file"
            return out
        df = pd.read_csv(path)
        out["rows_read"] = len(df)
        df = _normalize_collision_df_columns(df)
        if "CRASH_DATE" not in df.columns or "CRASH_TIME" not in df.columns:
            out["error"] = "schema"
            return out
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        df = _preprocess_collision_df(df)
        out["df"] = df
        out["rows_after_clean"] = len(df)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"load_failed:{exc}"
    return out


def apply_view_filters(
    data: pd.DataFrame,
    year_lo: int,
    year_hi: int,
    boroughs: List[str],
    severity: str,
) -> pd.DataFrame:
    v = data[(data["year"] >= year_lo) & (data["year"] <= year_hi)]
    if boroughs and "borough" in v.columns:
        v = v[v["borough"].isin(boroughs)]
    if severity == "Serious only":
        v = v[v["is_serious"] == 1]
    elif severity == "Non-serious only":
        v = v[v["is_serious"] == 0]
    return v


@st.cache_resource
def train_models(data: pd.DataFrame) -> Dict[str, Any]:
    if not ML_AVAILABLE or data.empty:
        return {}
    try:
        feature_cols = [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_rush_hour",
            "is_night",
            "is_holiday_season",
        ]
        if "LATITUDE" in data.columns and "LONGITUDE" in data.columns:
            feature_cols.extend(["LATITUDE", "LONGITUDE"])
        ml_data = data[feature_cols + ["is_serious"]].dropna()
        if len(ml_data) < 1000:
            return {}
        X = ml_data[feature_cols].astype(np.float32)
        y = ml_data["is_serious"].astype(np.int8)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        models: Dict[str, Any] = {}
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        models["Random Forest"] = {"model": rf, "score": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])}
        gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42
        )
        gb.fit(X_train, y_train)
        models["Gradient Boosting"] = {
            "model": gb,
            "score": roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]),
        }
        if XGB_AVAILABLE and xgb is not None:
            xm = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            xm.fit(X_train, y_train)
            models["XGBoost"] = {"model": xm, "score": roc_auc_score(y_test, xm.predict_proba(X_test)[:, 1])}
        if LGB_AVAILABLE and lgb is not None:
            lm = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            lm.fit(X_train, y_train)
            models["LightGBM"] = {"model": lm, "score": roc_auc_score(y_test, lm.predict_proba(X_test)[:, 1])}
        best = max(models.keys(), key=lambda k: models[k]["score"])
        return {
            "models": models,
            "best_model": models[best]["model"],
            "best_model_name": best,
            "feature_names": feature_cols,
        }
    except Exception as exc:  # noqa: BLE001
        return {"_train_error": str(exc)}


def render_methodology_block(path: str, pack: Dict[str, Any]) -> None:
    stat = os.stat(path) if os.path.isfile(path) else None
    modified = (
        datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M local")
        if stat
        else "—"
    )
    size_mb = f"{stat.st_size / (1024**2):.1f} MB" if stat else "—"

    st.markdown('<p class="nyc-section-kicker">Notes</p>', unsafe_allow_html=True)
    with st.expander("Where the data comes from and what we changed", expanded=False):
        st.markdown(
            f"""
<div class="nyc-method">
<p>This app reads NYC’s published motor-vehicle crash file and builds charts and a simple classifier on top.
Use the filters above, then open each tab for the risk model, time charts, a heat map, or a CSV download.</p>
<ul>
<li><strong>Dataset:</strong> <a href="{NYC_OPEN_DATA_URL}" target="_blank" rel="noopener">Motor Vehicle Collisions — Crashes</a> on NYC Open Data.</li>
<li><strong>Local file:</strong> <code>{path}</code> · modified <strong>{modified}</strong> · size about <strong>{size_mb}</strong>.</li>
<li><strong>Rows:</strong> {pack.get("rows_read", 0):,} read from the file; {pack.get("rows_after_clean", 0):,} kept after dropping rows without a valid date or hour.</li>
<li><strong>Row cap:</strong> If you set a max below the file size, we take a random sample (always seed 42) before cleaning.</li>
<li><strong>Serious crashes:</strong> We label a row serious if someone died or at least two people were injured — that’s what the model predicts.</li>
<li><strong>Risk score (heat map):</strong> A homemade index from injuries, deaths, and time of day. It isn’t from the city.</li>
<li><strong>Map tab:</strong> At most {get_app_config()["map_sample_size"]:,} points with coordinates, for speed.</li>
<li><strong>Risk tab map:</strong> Up to {get_app_config()["risk_map_crash_max"]:,} points from your current filters, clustered. Amber = serious label, gray = not.</li>
<li><strong>Model:</strong> Trained on the loaded sample; we report ROC AUC on a 20% holdout set. Useful for exploration, not for policy or engineering decisions.</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )


def _prepare_collisions_for_risk_map(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Rows with coordinates; random sample if over max_points (seed 42)."""
    if df.empty or "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        return pd.DataFrame()
    pts = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    if pts.empty:
        return pts
    if len(pts) > max_points:
        pts = pts.sample(n=max_points, random_state=42)
    return pts


def _folium_risk_map(
    lat: float,
    lon: float,
    height: int = 300,
    collisions: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Folium map: borough refs, clustered crash points (filtered sample), then your input pin.
    Returns optional markdown caption about the crash sample (caller should render with st.caption).
    """
    if not MAP_AVAILABLE:
        return None
    cfg = get_app_config()
    max_c = cfg["risk_map_crash_max"]

    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")
    for name, plat, plon, color in BOROUGH_MAP_STYLE:
        folium.CircleMarker(
            [plat, plon],
            radius=5,
            popup=name,
            tooltip=name,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.55,
            weight=2,
        ).add_to(m)

    total_geo = 0
    if collisions is not None and not collisions.empty:
        total_geo = int(collisions.dropna(subset=["LATITUDE", "LONGITUDE"]).shape[0])
        pts = _prepare_collisions_for_risk_map(collisions, max_c)
        if not pts.empty:
            crash_group = folium.FeatureGroup(name="Crashes (your filters)", show=True)
            cluster = MarkerCluster(max_cluster_radius=50).add_to(crash_group)
            for _, r in pts.iterrows():
                serious = int(r.get("is_serious", 0) or 0) == 1
                col = CHART["serious_line"] if serious else CHART["neutral"]
                pop_parts = [
                    "<b>Crash</b>",
                    f"Serious: {'yes' if serious else 'no'}",
                ]
                if "CRASH_DATE" in r.index and pd.notna(r["CRASH_DATE"]):
                    d = r["CRASH_DATE"]
                    pop_parts.append(
                        f"Date: {d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d}"
                    )
                if "borough" in r.index and pd.notna(r["borough"]) and str(r["borough"]):
                    pop_parts.append(f"Borough: {r['borough']}")
                folium.CircleMarker(
                    location=[float(r["LATITUDE"]), float(r["LONGITUDE"])],
                    radius=5,
                    color=col,
                    fill=True,
                    fillColor=col,
                    fillOpacity=0.5,
                    weight=1,
                    popup=folium.Popup("<br>".join(pop_parts), max_width=220),
                ).add_to(cluster)
            crash_group.add_to(m)
            folium.LayerControl(position="topright", collapsed=True).add_to(m)

    folium.Marker(
        [lat, lon],
        popup="Lat/lon from the form",
        tooltip="Selected coordinates",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    Fullscreen().add_to(m)
    MousePosition().add_to(m)
    components.html(m._repr_html_(), height=height)

    if total_geo > 0:
        shown = min(total_geo, max_c)
        return (
            f"Showing {shown:,} crashes on the map"
            + (
                f" (out of {total_geo:,} with coordinates in your current filters)"
                if shown < total_geo
                else " — that’s everything geocoded in your filters"
            )
            + " Amber = serious; gray = not serious."
        )
    return None


def render_prediction_ui(model_info: Dict[str, Any], collisions_view: pd.DataFrame) -> None:
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Risk model</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pick a time and place. The model estimates how often a crash like that would be labeled serious "
        "(someone killed, or two or more injured). It’s only as good as the data you loaded — don’t use it for insurance or legal decisions."
    )

    if "risk_lat" not in st.session_state:
        st.session_state.risk_lat = 40.7128
    if "risk_lon" not in st.session_state:
        st.session_state.risk_lon = -74.0060

    st.markdown('<p class="nyc-control-label">When</p>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns([1.35, 1.0, 1.1])
    with g1:
        hour = st.slider(
            "Hour",
            0,
            23,
            12,
            help="Hour on the clock, same way NYC recorded it in the file.",
        )
    with g2:
        day = st.selectbox("Weekday", WEEKDAYS, index=0)
    with g3:
        month = st.selectbox("Month", MONTHS, index=0)

    st.caption("Or jump to a rough borough center (fills latitude and longitude below)")
    bp = st.columns(5)
    for i, (full, plat, plon, _) in enumerate(BOROUGH_MAP_STYLE):
        with bp[i]:
            if st.button(
                full,
                key=f"risk_bp_{i}",
                help=f"Approximate center of {full} ({plat:.4f}, {plon:.4f})",
                use_container_width=True,
            ):
                st.session_state.risk_lat = plat
                st.session_state.risk_lon = plon
                st.rerun()

    st.markdown('<p class="nyc-control-label">Where</p>', unsafe_allow_html=True)
    g4, g5 = st.columns(2)
    with g4:
        st.number_input("Latitude", format="%.4f", step=0.0001, key="risk_lat")
    with g5:
        st.number_input("Longitude", format="%.4f", step=0.0001, key="risk_lon")

    nlat = float(st.session_state.risk_lat)
    nlon = float(st.session_state.risk_lon)

    day_num = DAY_TO_NUM[day]
    month_num = MONTH_TO_NUM[month]
    is_weekend = 1 if day_num in (5, 6) else 0
    is_rush = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    is_holiday = 1 if month_num in (11, 12, 1) else 0
    flags = []
    if is_weekend:
        flags.append("Weekend")
    if is_rush:
        flags.append("Rush hour")
    if is_night:
        flags.append("Night")
    if is_holiday:
        flags.append("Holiday-season stretch (Nov–Jan)")
    st.caption(
        "The model also sees: "
        + (", ".join(flags) if flags else "weekday, off-peak, daytime")
    )

    if not model_info or "best_model" not in model_info:
        st.warning(
            "No model right now. You need scikit-learn, at least 1,000 rows after cleaning, and a successful training run."
        )
        if MAP_AVAILABLE:
            nocol, mapcol = st.columns([1, 1])
            with nocol:
                st.caption(
                    "You can still move the map and coordinates above. The number next to the map appears when training works."
                )
            with mapcol:
                st.caption("Crashes use your filters (clustered). Blue marker = your lat/lon.")
                crash_caption = _folium_risk_map(
                    nlat, nlon, height=320, collisions=collisions_view
                )
                if crash_caption:
                    st.caption(crash_caption)
        return

    feats = [hour, day_num, month_num, is_weekend, is_rush, is_night, is_holiday]
    if "LATITUDE" in model_info["feature_names"]:
        feats.extend([nlat, nlon])
    try:
        model = model_info["best_model"]
        proba = float(model.predict_proba([feats])[0][1])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction error: {exc}")
        return

    if proba >= 0.7:
        label, hint = "High", "Above 70% on this model’s scale."
    elif proba >= 0.4:
        label, hint = "Medium", "Between 40% and 70%."
    else:
        label, hint = "Low", "Below 40%."

    st.markdown(
        '<div class="nyc-estimate-anchor"><h3>Result</h3></div>',
        unsafe_allow_html=True,
    )
    est_col, map_col = st.columns([1.05, 1])
    with est_col:
        m1, m2, m3 = st.columns([1.15, 1.0, 1.25])
        with m1:
            st.metric(
                "P(serious)",
                f"{proba:.1%}",
                help="Estimated probability of the serious label. Model chosen by ROC AUC on a held-out test set.",
            )
        with m2:
            st.metric("Bucket", label, help=hint)
        with m3:
            st.caption("Same value as a 0–100% bar")
            st.progress(min(max(proba, 0.0), 1.0))
        st.caption(hint)
    with map_col:
        if MAP_AVAILABLE:
            st.caption(
                "Dots are crashes that match your filters (clustered when zoomed out). "
                "Small circles mark borough centers; the blue marker is your latitude and longitude."
            )
            crash_caption = _folium_risk_map(
                nlat, nlon, height=360, collisions=collisions_view
            )
            if crash_caption:
                st.caption(crash_caption)
        else:
            st.caption("Install the `folium` package to show the map.")

    with st.expander("Models and test scores", expanded=False):
        st.write(f"Using: {model_info['best_model_name']}")
        st.caption("ROC AUC on 20% of the data held out after training. Higher means better ranking, not perfect predictions.")
        for name, info in model_info.get("models", {}).items():
            st.write(f"- {name}: {info['score']:.3f}")


def render_time_charts(view: pd.DataFrame) -> None:
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Over time</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "First chart: crash count by hour (bars) and share that were serious (line). Left axis is count, right axis is share."
    )
    h = view.groupby("hour").agg({"is_serious": ["count", "mean"]}).round(4)
    h.columns = ["total", "serious_rate"]
    h = h.reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=h["hour"],
            y=h["total"],
            name="Crash count",
            marker_color=CHART["bar_volume"],
            hovertemplate="Hour %{x}<br>Count %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=h["hour"],
            y=h["serious_rate"],
            name="Serious share",
            mode="lines+markers",
            line=dict(color=CHART["serious_line"], width=2),
            hovertemplate="Hour %{x}<br>Serious share %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_plotly_layout_base(
            "Crashes by hour: volume and serious share",
            "Crash count (n)",
            "Serious share (proportion)",
        )
    )
    fig.update_xaxes(title_text="Hour of day (0–23)", dtick=2)
    st.plotly_chart(fig, use_container_width=True)

    if px is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        daily = view.groupby(["day_name", "day_of_week"]).size().reset_index(name="count")
        daily = daily.sort_values("day_of_week")
        fig2 = px.bar(
            daily,
            x="day_name",
            y="count",
            title="Crashes by weekday",
            labels={"day_name": "Weekday", "count": "Count (n)"},
            color_discrete_sequence=[CHART["accent"]],
        )
        fig2.update_layout(**{k: v for k, v in _plotly_layout_base("", "Count", None).items() if k != "title"})
        fig2.update_layout(title={"text": "Crashes by weekday", "font": {"size": 16}})
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        monthly = view.groupby(["month", "month_name"]).size().reset_index(name="count")
        monthly = monthly.sort_values("month")
        fig3 = px.line(
            monthly,
            x="month_name",
            y="count",
            title="Crashes by month",
            markers=True,
            labels={"month_name": "Month", "count": "Count (n)"},
        )
        fig3.update_traces(line=dict(color=CHART["accent"]))
        fig3.update_layout(**{k: v for k, v in _plotly_layout_base("", "Count", None).items() if k != "title"})
        fig3.update_layout(title={"text": "Crashes by month", "font": {"size": 16}})
        fig3.update_xaxes(tickangle=35)
        st.plotly_chart(fig3, use_container_width=True)

    cond = view.groupby(["is_weekend", "is_rush_hour", "is_night"])["is_serious"].mean().reset_index()
    cond["label"] = cond.apply(
        lambda r: f"{'Wknd' if r['is_weekend'] else 'Wkdy'} / "
        f"{'Rush' if r['is_rush_hour'] else 'Off'} / "
        f"{'Night' if r['is_night'] else 'Day'}",
        axis=1,
    )
    fig4 = px.bar(
        cond,
        x="label",
        y="is_serious",
        title="Serious share: weekend, rush hour, night",
        labels={"label": "Context (weekday · rush · night)", "is_serious": "Serious share"},
        color="is_serious",
        color_continuous_scale=[[0, CHART["bar_volume"]], [1, CHART["serious_line"]]],
    )
    fig4.update_layout(**{k: v for k, v in _plotly_layout_base("", "Serious share", None).items() if k != "title"})
    fig4.update_layout(title={"text": "Serious share by weekend, rush, night", "font": {"size": 16}})
    fig4.update_yaxes(tickformat=".0%")
    fig4.update_xaxes(tickangle=25)
    st.plotly_chart(fig4, use_container_width=True)


def render_geo(view: pd.DataFrame) -> None:
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Map</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Borough comes from latitude and longitude using simple bounding boxes. "
        "Heat intensity follows the app’s risk score (casualties and time), not anything published by the city."
    )

    if "borough" in view.columns and len(view) > 0:
        bs = (
            view.groupby("borough")
            .agg({"is_serious": ["count", "mean", "sum"], "risk_score": "mean"})
            .round(4)
        )
        bs.columns = ["total", "serious_rate", "serious_n", "avg_risk"]
        bs = bs.reset_index().sort_values("total", ascending=False)
        g1, g2 = st.columns(2)
        with g1:
            f1 = go.Figure(
                data=[
                    go.Bar(
                        x=bs["borough"],
                        y=bs["total"],
                        name="Count",
                        marker_color=CHART["accent"],
                        hovertemplate="%{x}<br>Count %{y:,}<extra></extra>",
                    )
                ]
            )
            f1.update_layout(
                **_plotly_layout_base("Crashes by borough", "Count (n)", None),
            )
            st.plotly_chart(f1, use_container_width=True)
        with g2:
            f2 = go.Figure(
                data=[
                    go.Bar(
                        x=bs["borough"],
                        y=bs["serious_rate"],
                        name="Serious share",
                        marker_color=CHART["serious_line"],
                        hovertemplate="%{x}<br>Serious share %{y:.1%}<extra></extra>",
                    )
                ]
            )
            f2.update_layout(
                **_plotly_layout_base("Serious share by borough", "Serious share", None),
            )
            f2.update_yaxes(tickformat=".0%")
            st.plotly_chart(f2, use_container_width=True)

    if not MAP_AVAILABLE:
        st.info("Install `folium` to show the map.")
        return

    cfg = get_app_config()
    md = view.dropna(subset=["LATITUDE", "LONGITUDE"])
    if len(md) > cfg["map_sample_size"]:
        md = md.sample(n=cfg["map_sample_size"], random_state=42)
        st.caption(
            f"Showing a random {cfg['map_sample_size']:,} crashes with coordinates so the page stays responsive."
        )
    if md.empty:
        st.warning("No latitude or longitude in this filter. Try widening years or boroughs.")
        return
    heat = [
        [r["LATITUDE"], r["LONGITUDE"], r["risk_score"]]
        for _, r in md.iterrows()
        if pd.notna(r["LATITUDE"]) and pd.notna(r["LONGITUDE"])
    ]
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="CartoDB positron")
    if heat:
        HeatMap(heat, radius=12, max_zoom=12).add_to(m)
    for name, lat, lon, color in BOROUGH_MAP_STYLE:
        folium.CircleMarker(
            [lat, lon],
            radius=8,
            popup=name,
            tooltip=name,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.5,
            weight=2,
        ).add_to(m)
    Fullscreen().add_to(m)
    components.html(m._repr_html_(), height=480)


def render_data_export(view: pd.DataFrame, full: pd.DataFrame) -> None:
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Export</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(view):,} rows with your current filters (out of {len(full):,} after cleaning). "
        "The CSV matches what the charts use."
    )
    u1, u2, u3 = st.columns(3)
    with u1:
        st.metric("Injuries (sum)", f"{view['total_injured'].sum():,.0f}")
    with u2:
        st.metric("Fatalities (sum)", f"{view['total_killed'].sum():,.0f}")
    with u3:
        st.metric("Serious share", f"{view['is_serious'].mean():.1%}")

    cols = [
        c
        for c in [
            "CRASH_DATE",
            "CRASH_TIME",
            "borough",
            "LATITUDE",
            "LONGITUDE",
            "total_injured",
            "total_killed",
            "is_serious",
            "risk_score",
            "hour",
            "year",
        ]
        if c in view.columns
    ]
    show = st.toggle("Show first 500 rows", value=False)
    if show:
        st.dataframe(view[cols].head(500), use_container_width=True, height=320)

    csv_buf = io.StringIO()
    view[cols].to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name="nyc_collisions_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    cfg = get_app_config()

    st.markdown('<p class="nyc-page-title">NYC motor vehicle crashes</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="nyc-lead">Browse the city’s crash export: filter by year and borough, look at timing and maps, '
        "try a simple model that guesses how often a crash counts as serious, and download a CSV if you need it. "
        "Public data only; this isn’t official analysis from DOT or NYPD.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### How many rows to load")
        st.caption("Smaller loads are faster; raise the cap if you have RAM and want more history.")
        cap = st.slider(
            "Max rows from file",
            min_value=cfg["data_sample_min"],
            max_value=cfg["data_sample_max"],
            value=min(cfg["data_sample_default"], cfg["data_sample_max"]),
            step=5_000,
            help="Random sample before cleaning (seed 42) when the file is bigger than this.",
        )
        if st.button("Reload file (clear cache)", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if not os.path.isfile(DATA_FILE):
        st.error(f"Can’t find `{DATA_FILE}` in this folder.")
        st.markdown(
            f"- Download from [NYC Open Data]({NYC_OPEN_DATA_URL}).\n"
            "- Save it here with that name, or rename your file to match."
        )
        return

    mtime = os.path.getmtime(DATA_FILE)
    with st.spinner("Loading CSV and cleaning dates…"):
        pack = load_collision_data(cap, DATA_FILE, mtime)

    if pack.get("error") == "missing_file":
        st.error("The file was there and then it wasn’t. Check the path and try again.")
        return
    if pack.get("error") == "schema":
        st.error(
            "This file doesn’t look like the NYC crashes export. We need crash date and time columns "
            "(spaces or underscores both work — e.g. `CRASH DATE` or `CRASH_DATE`)."
        )
        return
    if pack.get("error") and str(pack["error"]).startswith("load_failed"):
        st.error("Couldn’t read the file. Close it in other apps if it’s open, or check that it’s a valid CSV.")
        st.code(str(pack["error"]))
        return

    data = pack["df"]
    if data.empty:
        st.warning(
            "No rows left after cleaning. The date or time columns may not parse, or the file might be empty."
        )
        return

    y_min, y_max = int(data["year"].min()), int(data["year"].max())
    b_opts = sorted(data["borough"].dropna().unique().tolist()) if "borough" in data.columns else []

    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Filters</p>',
        unsafe_allow_html=True,
    )
    f1, f2, f3 = st.columns((1, 1, 1))
    with f1:
        yr = st.slider("Years", y_min, y_max, (y_min, y_max))
    with f2:
        bor = st.multiselect(
            "Borough",
            options=b_opts,
            default=b_opts,
            help="Assigned from coordinates. Uncheck to exclude Unknown or other areas.",
        )
    with f3:
        sev = st.selectbox(
            "Severity",
            ["All", "Serious only", "Non-serious only"],
            help="Serious: at least one death, or two or more injuries.",
        )

    view = apply_view_filters(data, yr[0], yr[1], bor, sev)
    if view.empty:
        st.warning(
            "No rows match. Try a wider year range, select more boroughs, or set severity to All."
        )
        render_methodology_block(DATA_FILE, pack)
        return

    st.markdown('<p class="nyc-section-kicker">Summary</p>', unsafe_allow_html=True)
    mcols = st.columns(5)
    metrics = [
        ("Crashes in view", f"{len(view):,}"),
        ("People hurt (total)", f"{view['total_injured'].sum():,.0f}"),
        ("Lives lost (total)", f"{view['total_killed'].sum():,.0f}"),
        ("Serious share", f"{view['is_serious'].mean():.1%}"),
        ("Days covered", f"{(view['CRASH_DATE'].max() - view['CRASH_DATE'].min()).days:,}"),
    ]
    for i, (lab, val) in enumerate(metrics):
        with mcols[i]:
            st.metric(lab, val)

    st.divider()
    render_methodology_block(DATA_FILE, pack)

    model_info: Dict[str, Any] = {}
    if ML_AVAILABLE:
        with st.spinner("Training models…"):
            raw_models = train_models(data)
        # Do not mutate @st.cache_resource return value
        te = raw_models.get("_train_error") if isinstance(raw_models, dict) else None
        model_info = {
            k: v
            for k, v in (raw_models or {}).items()
            if k != "_train_error"
        }
        if te:
            st.warning("Model training failed. Charts, map, and export still work; the risk tab won’t show a score.")
            st.caption(te)
            model_info = {}
        elif model_info:
            st.caption(
                f"Fit on {len(data):,} cleaned rows; 20% held out for scoring. "
                f"Best ROC AUC: {model_info.get('best_model_name', '')}."
            )

    tab_risk, tab_time, tab_map, tab_data = st.tabs(
        ["Risk", "Time", "Map", "Export"]
    )
    with tab_risk:
        render_prediction_ui(model_info, view)
    with tab_time:
        if len(view) < 10:
            st.info("Need at least 10 rows for these charts. Relax the filters.")
        else:
            render_time_charts(view)
    with tab_map:
        render_geo(view)
    with tab_data:
        render_data_export(view, data)

    st.divider()
    st.caption(
        f"Data: [NYC Open Data]({NYC_OPEN_DATA_URL}). "
        "For exploration only, not official safety planning."
    )


if __name__ == "__main__":
    main()
