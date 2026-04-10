"""
NYC motor vehicle collisions — Streamlit explorer and risk model UI.

Theme tokens live in .streamlit/config.toml (see README → UI theme).
Custom CSS below is scoped to classes prefixed with nyc- for maintainability.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote
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
# Community mirror of the same NYC export (handy for a single downloadable file).
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/tush32/motor-vehicle-collisions-crashes"
# Socrata API (same dataset) — used when ``Motor_Vehicle_Collisions_-_Crashes.csv`` is not in the app (e.g. Streamlit Cloud).
SOCRATA_CRASHES_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
SOCRATA_PAGE_SIZE = 50_000

# API field names (lowercase) → same column labels as the published CSV export (for existing preprocessing).
SOCRATA_TO_CSV_COLUMNS: Dict[str, str] = {
    "crash_date": "CRASH_DATE",
    "crash_time": "CRASH_TIME",
    "borough": "BOROUGH",
    "zip_code": "ZIP CODE",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "location": "LOCATION",
    "on_street_name": "ON STREET NAME",
    "cross_street_name": "CROSS STREET NAME",
    "off_street_name": "OFF STREET NAME",
    "number_of_persons_injured": "NUMBER OF PERSONS INJURED",
    "number_of_persons_killed": "NUMBER OF PERSONS KILLED",
    "number_of_pedestrians_injured": "NUMBER OF PEDESTRIANS INJURED",
    "number_of_pedestrians_killed": "NUMBER OF PEDESTRIANS KILLED",
    "number_of_cyclist_injured": "NUMBER OF CYCLIST INJURED",
    "number_of_cyclist_killed": "NUMBER OF CYCLIST KILLED",
    "number_of_motorist_injured": "NUMBER OF MOTORIST INJURED",
    "number_of_motorist_killed": "NUMBER OF MOTORIST KILLED",
    "contributing_factor_vehicle_1": "CONTRIBUTING FACTOR VEHICLE 1",
    "contributing_factor_vehicle_2": "CONTRIBUTING FACTOR VEHICLE 2",
    "contributing_factor_vehicle_3": "CONTRIBUTING FACTOR VEHICLE 3",
    "contributing_factor_vehicle_4": "CONTRIBUTING FACTOR VEHICLE 4",
    "contributing_factor_vehicle_5": "CONTRIBUTING FACTOR VEHICLE 5",
    "collision_id": "COLLISION_ID",
    "vehicle_type_code1": "VEHICLE TYPE CODE 1",
    "vehicle_type_code2": "VEHICLE TYPE CODE 2",
    "vehicle_type_code3": "VEHICLE TYPE CODE 3",
    "vehicle_type_code4": "VEHICLE TYPE CODE 4",
    "vehicle_type_code5": "VEHICLE TYPE CODE 5",
    "vehicle_type_code_1": "VEHICLE TYPE CODE 1",
    "vehicle_type_code_2": "VEHICLE TYPE CODE 2",
    "vehicle_type_code_3": "VEHICLE TYPE CODE 3",
    "vehicle_type_code_4": "VEHICLE TYPE CODE 4",
    "vehicle_type_code_5": "VEHICLE TYPE CODE 5",
}

# Plotly semantic colors — align with [theme] primaryColor in config.toml (#0f766e)
CHART = {
    "accent": "#0f766e",
    "accent_dark": "#115e59",
    "neutral": "#64748b",
    "volume": "#475569",
    "serious_line": "#b45309",
    "grid": "#e2e8f0",
    "bar_volume": "#cbd5e1",
    "roc_palette": ["#0f766e", "#b45309", "#6366f1", "#db2777", "#64748b"],
}

# Default ML training — serialized to JSON for @st.cache_resource keying
DEFAULT_ML_CONFIG: Dict[str, Any] = {
    "use_coords": True,
    "test_size": 0.2,
    "train_rf": True,
    "train_gb": True,
    "train_xgb": True,
    "train_lgb": True,
    "rf_n_estimators": 150,
    "rf_max_depth": 15,
    "rf_min_samples_leaf": 5,
    "gb_n_estimators": 100,
    "gb_learning_rate": 0.1,
    "gb_max_depth": 8,
    "xgb_n_estimators": 100,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.1,
    "lgb_n_estimators": 100,
    "lgb_max_depth": 8,
    "lgb_learning_rate": 0.1,
}
DEFAULT_ML_CONFIG_JSON = json.dumps(DEFAULT_ML_CONFIG, sort_keys=True)

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
        "About": (
            f"Data: NYC Open Data ({NYC_OPEN_DATA_URL}) · "
            f"CSV mirror on Kaggle ({KAGGLE_DATASET_URL})"
        ),
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
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    roc_auc_score = None  # type: ignore[misc, assignment]
    accuracy_score = None  # type: ignore[misc, assignment]
    confusion_matrix = None  # type: ignore[misc, assignment]
    f1_score = None  # type: ignore[misc, assignment]
    precision_score = None  # type: ignore[misc, assignment]
    recall_score = None  # type: ignore[misc, assignment]
    roc_curve = None  # type: ignore[misc, assignment]

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
        "model_export_max_rows": 50_000,
    }


def _plotly_layout_base(
    title: str,
    y_primary: str,
    y_secondary: Optional[str] = None,
    *,
    legend_below: bool = True,
) -> Dict[str, Any]:
    """Plotly layout shared across charts. Legend defaults below the plot so it never covers the title."""
    h = get_app_config()["chart_height"]
    margin = {"t": 52, "l": 56, "r": 56, "b": 48}
    if legend_below:
        margin["b"] = 96
        legend: Dict[str, Any] = {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
        }
    else:
        legend = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        }
        margin["t"] = 64
    layout: Dict[str, Any] = {
        "title": {
            "text": title,
            "font": {"size": 16},
            "x": 0.5,
            "xanchor": "center",
            "pad": {"t": 4, "b": 10},
        },
        "font": {"family": "sans-serif", "color": "#334155"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "height": h,
        "margin": margin,
        "xaxis": {"showgrid": True, "gridcolor": CHART["grid"], "zeroline": False},
        "legend": legend,
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
        "source": "file",
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


def _http_get_socrata_json(url: str, app_token: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NYC-motor-vehicle-crashes-streamlit/1.0",
            "Accept": "application/json",
        },
    )
    tok = (app_token or "").strip()
    if tok:
        req.add_header("X-App-Token", tok)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def _prepare_socrata_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={k: v for k, v in SOCRATA_TO_CSV_COLUMNS.items() if k in df.columns})
    for c in df.columns:
        if "NUMBER OF" in c or c in ("LATITUDE", "LONGITUDE", "COLLISION_ID"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=get_app_config()["cache_ttl"], show_spinner=False)
def load_collision_data_from_api(max_rows: int, app_token: str) -> Dict[str, Any]:
    """Fetch crash rows from NYC Open Data (Socrata), newest crash_date first, up to max_rows."""
    out: Dict[str, Any] = {
        "df": pd.DataFrame(),
        "error": None,
        "rows_read": 0,
        "rows_after_clean": 0,
        "source": "api",
    }
    try:
        want = max(1, int(max_rows))
        rows_accum: List[Dict[str, Any]] = []
        offset = 0
        while len(rows_accum) < want:
            chunk_sz = min(SOCRATA_PAGE_SIZE, want - len(rows_accum))
            # Space in SoQL order must be percent-encoded (raw space breaks urllib on some runtimes).
            order_param = quote("crash_date DESC", safe="")
            url = (
                f"{SOCRATA_CRASHES_URL}?$limit={chunk_sz}&$offset={offset}"
                f"&$order={order_param}"
            )
            chunk = _http_get_socrata_json(url, app_token)
            if not isinstance(chunk, list) or not chunk:
                break
            rows_accum.extend(chunk)
            offset += len(chunk)
            if len(chunk) < chunk_sz:
                break
        df = pd.DataFrame(rows_accum)
        if df.empty:
            out["error"] = "api_empty"
            return out
        df = _prepare_socrata_dataframe(df)
        df = _normalize_collision_df_columns(df)
        if "CRASH_DATE" not in df.columns or "CRASH_TIME" not in df.columns:
            out["error"] = "schema"
            return out
        out["rows_read"] = len(df)
        df = _preprocess_collision_df(df)
        out["df"] = df
        out["rows_after_clean"] = len(df)
    except urllib.error.HTTPError as exc:
        out["error"] = f"load_failed:HTTP {exc.code} — {exc.reason}"
    except urllib.error.URLError as exc:
        out["error"] = f"load_failed:network {exc.reason!s}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"load_failed:{exc}"
    return out


def _socrata_app_token() -> str:
    env = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "").strip()
    if env:
        return env
    try:
        return str(st.secrets["NYC_OPEN_DATA_APP_TOKEN"]).strip()
    except Exception:
        return ""


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


def _binary_metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    pred = (proba >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


@st.cache_resource(show_spinner=False)
def train_models(data: pd.DataFrame, config_json: str) -> Dict[str, Any]:
    """Train selected estimators; ``config_json`` keys the cache so UI changes retrain."""
    if not ML_AVAILABLE or data.empty or roc_auc_score is None:
        return {}
    try:
        cfg = json.loads(config_json)
    except json.JSONDecodeError:
        cfg = json.loads(DEFAULT_ML_CONFIG_JSON)

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
        use_coords = bool(cfg.get("use_coords", True))
        if use_coords and "LATITUDE" in data.columns and "LONGITUDE" in data.columns:
            feature_cols.extend(["LATITUDE", "LONGITUDE"])

        ml_data = data[feature_cols + ["is_serious"]].dropna()
        if len(ml_data) < 1000:
            return {}

        X = ml_data[feature_cols].astype(np.float32)
        y = ml_data["is_serious"].astype(np.int8)
        ts = float(cfg.get("test_size", 0.2))
        ts = min(0.45, max(0.1, ts))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=42, stratify=y
        )
        y_test_np = y_test.to_numpy(dtype=np.int8)

        models: Dict[str, Any] = {}
        if cfg.get("train_rf", True):
            rf = RandomForestClassifier(
                n_estimators=int(cfg.get("rf_n_estimators", 150)),
                max_depth=int(cfg.get("rf_max_depth", 15)),
                min_samples_split=10,
                min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 5)),
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            rf.fit(X_train, y_train)
            p = rf.predict_proba(X_test)[:, 1]
            models["Random Forest"] = {
                "model": rf,
                "score": float(roc_auc_score(y_test, p)),
                "metrics_05": _binary_metrics_at_threshold(y_test_np, p),
            }
        if cfg.get("train_gb", True):
            gb = GradientBoostingClassifier(
                n_estimators=int(cfg.get("gb_n_estimators", 100)),
                learning_rate=float(cfg.get("gb_learning_rate", 0.1)),
                max_depth=int(cfg.get("gb_max_depth", 8)),
                random_state=42,
            )
            gb.fit(X_train, y_train)
            p = gb.predict_proba(X_test)[:, 1]
            models["Gradient Boosting"] = {
                "model": gb,
                "score": float(roc_auc_score(y_test, p)),
                "metrics_05": _binary_metrics_at_threshold(y_test_np, p),
            }
        if cfg.get("train_xgb", True) and XGB_AVAILABLE and xgb is not None:
            xm = xgb.XGBClassifier(
                n_estimators=int(cfg.get("xgb_n_estimators", 100)),
                max_depth=int(cfg.get("xgb_max_depth", 8)),
                learning_rate=float(cfg.get("xgb_learning_rate", 0.1)),
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            xm.fit(X_train, y_train)
            p = xm.predict_proba(X_test)[:, 1]
            models["XGBoost"] = {
                "model": xm,
                "score": float(roc_auc_score(y_test, p)),
                "metrics_05": _binary_metrics_at_threshold(y_test_np, p),
            }
        if cfg.get("train_lgb", True) and LGB_AVAILABLE and lgb is not None:
            lm = lgb.LGBMClassifier(
                n_estimators=int(cfg.get("lgb_n_estimators", 100)),
                max_depth=int(cfg.get("lgb_max_depth", 8)),
                learning_rate=float(cfg.get("lgb_learning_rate", 0.1)),
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            lm.fit(X_train, y_train)
            p = lm.predict_proba(X_test)[:, 1]
            models["LightGBM"] = {
                "model": lm,
                "score": float(roc_auc_score(y_test, p)),
                "metrics_05": _binary_metrics_at_threshold(y_test_np, p),
            }

        if not models:
            return {"_train_error": "No models selected (or XGBoost/LightGBM unavailable)."}

        probas_test: Dict[str, np.ndarray] = {}
        metrics_table: Dict[str, Any] = {}
        for name, info in models.items():
            m = info["model"]
            pr = m.predict_proba(X_test)[:, 1]
            probas_test[name] = np.asarray(pr, dtype=np.float64)
            row = {"roc_auc": info["score"], **info["metrics_05"]}
            metrics_table[name] = row

        best = max(models.keys(), key=lambda k: models[k]["score"])
        return {
            "models": models,
            "best_model": models[best]["model"],
            "best_model_name": best,
            "feature_names": feature_cols,
            "y_test": y_test_np,
            "probas_test": probas_test,
            "metrics_table": metrics_table,
            "train_config": cfg,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"_train_error": str(exc)}


def _score_view_with_best_model(
    view: pd.DataFrame,
    model_info: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Subset of rows with complete model features plus ``p_serious`` from the best estimator."""
    if not model_info or "best_model" not in model_info:
        return None
    fn: List[str] = model_info["feature_names"]
    if any(c not in view.columns for c in fn):
        return None
    need = fn + (["is_serious"] if "is_serious" in view.columns else [])
    sub = view[need].dropna()
    if sub.empty:
        return None
    X = sub[fn].astype(np.float32)
    model = model_info["best_model"]
    proba = model.predict_proba(X)[:, 1]
    out = sub.copy()
    out["p_serious"] = proba.astype(np.float64)
    return out


def _parse_ml_config_json(s: str) -> Dict[str, Any]:
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else json.loads(DEFAULT_ML_CONFIG_JSON)
    except (json.JSONDecodeError, TypeError):
        return json.loads(DEFAULT_ML_CONFIG_JSON)


def _plot_metrics_comparison(metrics_table: Dict[str, Any]) -> go.Figure:
    names = list(metrics_table.keys())
    fig = go.Figure()
    for metric, label in [
        ("roc_auc", "ROC AUC"),
        ("accuracy", "Accuracy @0.5"),
        ("f1", "F1 @0.5"),
    ]:
        fig.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=[float(metrics_table[n].get(metric, 0)) for n in names],
                hovertemplate="%{x}<br>" + label + ": %{y:.4f}<extra></extra>",
            )
        )
    base_layout = _plotly_layout_base("Holdout metrics by model", "Score", None)
    base_layout.pop("legend", None)
    fig.update_layout(
        **base_layout,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.5, xanchor="center"),
    )
    fig.update_layout(margin=dict(t=52, b=120, l=56, r=56))
    return fig


def _plot_roc_curves(y_test: np.ndarray, probas_test: Dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    palette = CHART["roc_palette"]
    for i, (name, proba) in enumerate(probas_test.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=name,
                line=dict(width=2, color=color),
                hovertemplate=name + "<br>FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash", color=CHART["neutral"], width=1),
            hovertemplate="Baseline<extra></extra>",
        )
    )
    fig.update_layout(
        **_plotly_layout_base("ROC curves (holdout set)", "True positive rate", None),
    )
    fig.update_xaxes(title_text="False positive rate", constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _plot_confusion_matrix_fig(y_true: np.ndarray, proba: np.ndarray, title: str) -> go.Figure:
    pred = (proba >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true, pred)
    labels = ["Not serious", "Serious"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {x}" for x in labels],
            y=[f"Actual {x}" for x in labels],
            colorscale=[[0, "#f1f5f9"], [1, CHART["accent"]]],
            text=cm,
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        **_plotly_layout_base(title, "", None),
    )
    fig.update_xaxes(side="bottom")
    return fig


def _model_card_payload(
    model_info: Dict[str, Any],
    data: pd.DataFrame,
    pack: Dict[str, Any],
) -> Dict[str, Any]:
    scores = {
        name: round(float(info["score"]), 5)
        for name, info in (model_info.get("models") or {}).items()
        if isinstance(info, dict) and "score" in info
    }
    metrics_full: Dict[str, Any] = {}
    for n, d in (model_info.get("metrics_table") or {}).items():
        if isinstance(d, dict):
            metrics_full[n] = {
                k: round(float(v), 5) if isinstance(v, (float, int, np.floating, np.integer)) else v
                for k, v in d.items()
            }
    if not metrics_full:
        metrics_full = {n: {"roc_auc": scores.get(n)} for n in scores}
    return {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "label": (
            "Serious crash: at least one fatality or two or more people injured "
            "(same rule as the Risk tab)."
        ),
        "training_sample_rows_after_clean": int(len(data)),
        "rows_read_from_csv": int(pack.get("rows_read") or 0),
        "feature_columns_in_order": list(model_info.get("feature_names") or []),
        "best_model_name_by_roc_auc": model_info.get("best_model_name"),
        "scoring_model_in_ui": model_info.get("scoring_model_label"),
        "holdout_metric": "ROC AUC on stratified test split (random_state=42); test_size from Model tab",
        "train_config": model_info.get("train_config"),
        "n_train": model_info.get("n_train"),
        "n_test": model_info.get("n_test"),
        "model_metrics_holdout": metrics_full,
        "model_test_roc_auc": scores,
    }


def render_methodology_block(path: str, pack: Dict[str, Any]) -> None:
    stat = os.stat(path) if os.path.isfile(path) else None
    modified = (
        datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M local")
        if stat
        else "—"
    )
    size_mb = f"{stat.st_size / (1024**2):.1f} MB" if stat else "—"

    src = pack.get("source", "file")
    if src == "api":
        origin_li = (
            f'<li><strong>Live data:</strong> Loaded from the '
            f'<a href="{SOCRATA_CRASHES_URL}" target="_blank" rel="noopener">Socrata API</a> '
            f"(newest crashes first), up to your sidebar row cap. No CSV is bundled with this deployment.</li>"
        )
        row_cap_li = (
            "<li><strong>Row cap:</strong> The API returns at most that many recent records (no extra random sample). "
            "If requests are throttled, add <code>NYC_OPEN_DATA_APP_TOKEN</code> under Streamlit "
            "<strong>Secrets</strong> (free from NYC Open Data).</li>"
        )
        rows_read_note = "pulled from the API"
    else:
        origin_li = (
            f'<li><strong>Local file:</strong> <code>{path}</code> · modified <strong>{modified}</strong> · '
            f"size about <strong>{size_mb}</strong>.</li>"
        )
        row_cap_li = (
            "<li><strong>Row cap:</strong> If you set a max below the file size, we take a random sample "
            "(always seed 42) before cleaning.</li>"
        )
        rows_read_note = "read from the file"

    st.markdown('<p class="nyc-section-kicker">Notes</p>', unsafe_allow_html=True)
    with st.expander("Where the data comes from and what we changed", expanded=False):
        st.markdown(
            f"""
<div class="nyc-method">
<p>This app reads NYC’s published motor-vehicle crash file and builds charts and a simple classifier on top.
Use the filters above, then open each tab for the risk model, time charts, a heat map, or batch model scores and a small model report.</p>
<ul>
<li><strong>Dataset:</strong> <a href="{NYC_OPEN_DATA_URL}" target="_blank" rel="noopener">Motor Vehicle Collisions — Crashes</a> on NYC Open Data. For a local CSV you can also use the community mirror on <a href="{KAGGLE_DATASET_URL}" target="_blank" rel="noopener">Kaggle</a> — download and save as <code>{DATA_FILE}</code> in this folder.</li>
{origin_li}
<li><strong>Rows:</strong> {pack.get("rows_read", 0):,} {rows_read_note}; {pack.get("rows_after_clean", 0):,} kept after dropping rows without a valid date or hour.</li>
{row_cap_li}
<li><strong>Serious crashes:</strong> We label a row serious if someone died or at least two people were injured — that’s what the model predicts.</li>
<li><strong>Risk score (heat map):</strong> A homemade index from injuries, deaths, and time of day. It isn’t from the city.</li>
<li><strong>Map tab:</strong> At most {get_app_config()["map_sample_size"]:,} points with coordinates, for speed.</li>
<li><strong>Risk tab map:</strong> Up to {get_app_config()["risk_map_crash_max"]:,} points from your current filters, clustered. Amber = serious label, gray = not.</li>
<li><strong>Model:</strong> Trained on the full cleaned sample from the sidebar cap; holdout fraction and hyperparameters are configurable on the Model tab (cached per setting). The Model tab re-scores filtered rows that have every feature filled, plots ROC and confusion on the holdout set, shows importances when the estimator supports them, and offers a scored CSV plus a JSON report (row cap {get_app_config()["model_export_max_rows"]:,}). For exploration only.</li>
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
                width="stretch",
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

    scorer = model_info.get("scoring_model_label") or model_info.get("best_model_name", "")
    st.caption(f"Estimator for this number: {scorer} (picker is above the tabs).")

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
                help="Estimated probability of the serious label for the estimator you selected above the tabs.",
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
        leader = model_info.get("best_model_name", "")
        scorer = model_info.get("scoring_model_label", leader)
        st.write(f"Scoring with: {scorer}")
        if scorer != leader:
            st.caption(f"Highest ROC AUC on the holdout set this run: {leader}.")
        tc = model_info.get("train_config") or {}
        ts = float(tc.get("test_size", 0.2))
        st.caption(
            f"Holdout fraction {ts:.0%} (stratified, seed 42). ROC AUC measures ranking, not “accuracy.” "
            "Accuracy / F1 / precision / recall below use a 0.5 probability cutoff."
        )
        mt = model_info.get("metrics_table")
        if mt:
            st.dataframe(pd.DataFrame(mt).T.round(4), width="stretch")
        else:
            for name, info in model_info.get("models", {}).items():
                st.write(f"- {name}: {info['score']:.3f}")


def render_time_charts(view: pd.DataFrame) -> None:
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Over time</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Bars are how many crashes; the line is the share we labeled serious (death or 2+ hurt). "
        "Left axis count, right axis share."
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
            name="How ugly it got",
            mode="lines+markers",
            line=dict(color=CHART["serious_line"], width=2),
            hovertemplate="Hour %{x}<br>Serious share %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_plotly_layout_base(
            "Crashes by hour — volume vs. how ugly it got",
            "Crash count (n)",
            "Serious share (proportion)",
        )
    )
    fig.update_xaxes(title_text="Hour of day (0–23)", dtick=2)
    st.plotly_chart(fig, width="stretch")

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
        fig2.update_layout(
            title={"text": "Crashes by weekday", "font": {"size": 16}, "x": 0.5, "xanchor": "center"}
        )
        st.plotly_chart(fig2, width="stretch")
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
        fig3.update_layout(
            title={"text": "Crashes by month", "font": {"size": 16}, "x": 0.5, "xanchor": "center"}
        )
        fig3.update_xaxes(tickangle=35)
        st.plotly_chart(fig3, width="stretch")

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
    fig4.update_layout(
        title={
            "text": "How ugly it got: weekend, rush, night",
            "font": {"size": 16},
            "x": 0.5,
            "xanchor": "center",
        }
    )
    fig4.update_yaxes(tickformat=".0%")
    fig4.update_xaxes(tickangle=25)
    st.plotly_chart(fig4, width="stretch")


def _factor_text_usable(s: str) -> bool:
    if not isinstance(s, str):
        return False
    x = s.strip()
    if len(x) < 2:
        return False
    xl = x.lower()
    if xl in ("nan", "none", "null"):
        return False
    return "unspecified" not in xl


def render_pattern_charts(view: pd.DataFrame) -> None:
    """Extra breakdowns: year trend, contributing factors, vehicle types, vulnerable road users."""
    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">More patterns</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Factors and vehicle types come straight from the file (vehicle 1 fields). "
        "Counts are descriptive—not causes proved in a crash."
    )

    ped_c = "NUMBER OF PEDESTRIANS INJURED"
    cyc_c = "NUMBER OF CYCLIST INJURED"
    mot_c = "NUMBER OF MOTORIST INJURED"
    vm = st.columns(3)
    with vm[0]:
        if ped_c in view.columns:
            n = (pd.to_numeric(view[ped_c], errors="coerce").fillna(0) > 0).sum()
            st.metric("Crashes with a pedestrian hurt", f"{n:,}")
        else:
            st.metric("Crashes with a pedestrian hurt", "—")
    with vm[1]:
        if cyc_c in view.columns:
            n = (pd.to_numeric(view[cyc_c], errors="coerce").fillna(0) > 0).sum()
            st.metric("Crashes with a cyclist hurt", f"{n:,}")
        else:
            st.metric("Crashes with a cyclist hurt", "—")
    with vm[2]:
        if mot_c in view.columns:
            n = (pd.to_numeric(view[mot_c], errors="coerce").fillna(0) > 0).sum()
            st.metric("Crashes with a driver hurt", f"{n:,}")
        else:
            st.metric("Crashes with a driver hurt", "—")

    yr_counts = view.groupby("year").size().reset_index(name="count")
    if len(yr_counts) > 1:
        yr_ser = view.groupby("year")["is_serious"].mean().reset_index()
        yr_m = yr_counts.merge(yr_ser, on="year")
        fig_y = make_subplots(specs=[[{"secondary_y": True}]])
        fig_y.add_trace(
            go.Bar(
                x=yr_m["year"],
                y=yr_m["count"],
                name="Crashes",
                marker_color=CHART["bar_volume"],
                hovertemplate="Year %{x}<br>Count %{y:,}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_y.add_trace(
            go.Scatter(
                x=yr_m["year"],
                y=yr_m["is_serious"],
                name="How ugly it got",
                mode="lines+markers",
                line=dict(color=CHART["serious_line"], width=2),
                hovertemplate="Year %{x}<br>Serious share %{y:.1%}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_y.update_layout(
            **_plotly_layout_base(
                "Crashes by year — volume vs. how ugly it got",
                "Crash count (n)",
                "Serious share (proportion)",
            )
        )
        fig_y.update_xaxes(title_text="Year", dtick=1)
        st.plotly_chart(fig_y, width="stretch")

    fac_col = "CONTRIBUTING FACTOR VEHICLE 1"
    veh_col = "VEHICLE TYPE CODE 1"
    pc1, pc2 = st.columns(2)

    with pc1:
        if fac_col in view.columns and px is not None:
            sub = view.copy()
            sub["_fac"] = sub[fac_col].astype(str).str.strip()
            sub = sub[sub["_fac"].map(_factor_text_usable)]
            min_n = max(30, min(200, len(sub) // 200))
            if len(sub) >= min_n:
                g = sub.groupby("_fac", as_index=False).agg(
                    n=("is_serious", "size"),
                    serious=("is_serious", "mean"),
                )
                g = g[g["n"] >= min_n].sort_values("n", ascending=False).head(14)
                if g.empty:
                    st.info("No single factor met the minimum count in this slice.")
                else:
                    g["label"] = g["_fac"].str.slice(0, 44)
                    fig_f = px.bar(
                        g.sort_values("n"),
                        x="n",
                        y="label",
                        orientation="h",
                        title="Top reported factors (vehicle 1)",
                        labels={"n": "Crashes (n)", "label": "Factor"},
                        color="serious",
                        color_continuous_scale=[[0, CHART["bar_volume"]], [1, CHART["serious_line"]]],
                    )
                    fig_f.update_layout(
                        **{
                            k: v
                            for k, v in _plotly_layout_base("", "Crashes (n)", None).items()
                            if k not in ("title", "yaxis", "yaxis2")
                        }
                    )
                    fig_f.update_layout(
                        title={
                            "text": "What people wrote down as a factor (vehicle 1)",
                            "font": {"size": 15},
                            "x": 0.5,
                            "xanchor": "center",
                        },
                        yaxis={"title": ""},
                        coloraxis_colorbar={"title": "Serious share"},
                    )
                    st.plotly_chart(fig_f, width="stretch")
            else:
                st.info("Not enough labeled factors in this slice to chart.")
        elif px is None:
            st.caption("Install plotly express for factor charts.")

    with pc2:
        if veh_col in view.columns and px is not None:
            sub = view.copy()
            sub["_veh"] = sub[veh_col].astype(str).str.strip()
            sub = sub[sub["_veh"].map(_factor_text_usable)]
            min_n = max(30, min(200, len(sub) // 200))
            if len(sub) >= min_n:
                g = sub.groupby("_veh", as_index=False).agg(
                    n=("is_serious", "size"),
                    serious=("is_serious", "mean"),
                )
                g = g[g["n"] >= min_n].sort_values("n", ascending=False).head(14)
                if g.empty:
                    st.info("No vehicle type met the minimum count in this slice.")
                else:
                    g["label"] = g["_veh"].str.slice(0, 44)
                    fig_v = px.bar(
                        g.sort_values("n"),
                        x="n",
                        y="label",
                        orientation="h",
                        title="Top vehicle types (vehicle 1)",
                        labels={"n": "Crashes (n)", "label": "Vehicle type"},
                        color="serious",
                        color_continuous_scale=[[0, CHART["bar_volume"]], [1, CHART["serious_line"]]],
                    )
                    fig_v.update_layout(
                        **{
                            k: v
                            for k, v in _plotly_layout_base("", "Crashes (n)", None).items()
                            if k not in ("title", "yaxis", "yaxis2")
                        }
                    )
                    fig_v.update_layout(
                        title={
                            "text": "Vehicle types showing up most (vehicle 1)",
                            "font": {"size": 15},
                            "x": 0.5,
                            "xanchor": "center",
                        },
                        yaxis={"title": ""},
                        coloraxis_colorbar={"title": "Serious share"},
                    )
                    st.plotly_chart(fig_v, width="stretch")
            else:
                st.info("Not enough vehicle-type rows in this slice to chart.")
        elif px is None:
            pass

    on_street = "ON STREET NAME"
    if on_street in view.columns and px is not None:
        sub = view.copy()
        sub["_st"] = sub[on_street].astype(str).str.strip()
        sub = sub[sub["_st"].map(_factor_text_usable)]
        min_n = max(20, min(150, len(sub) // 300))
        if len(sub) >= min_n:
            g = sub.groupby("_st", as_index=False).agg(
                n=("is_serious", "size"),
                serious=("is_serious", "mean"),
            )
            g = g[g["n"] >= min_n].sort_values("n", ascending=False).head(12)
            if g.empty:
                st.info("No street name met the minimum count in this slice.")
            else:
                g["label"] = g["_st"].str.slice(0, 40)
                fig_s = px.bar(
                    g.sort_values("n"),
                    x="n",
                    y="label",
                    orientation="h",
                    labels={"n": "Crashes (n)", "label": "Street"},
                    color="serious",
                    color_continuous_scale=[[0, CHART["bar_volume"]], [1, CHART["serious_line"]]],
                )
                fig_s.update_layout(
                    **{
                        k: v
                        for k, v in _plotly_layout_base("", "Crashes (n)", None).items()
                        if k not in ("title", "yaxis", "yaxis2")
                    }
                )
                fig_s.update_layout(
                    title={
                        "text": "Streets with the most crashes in your filters",
                        "font": {"size": 15},
                        "x": 0.5,
                        "xanchor": "center",
                    },
                    yaxis={"title": ""},
                    coloraxis_colorbar={"title": "Serious share"},
                )
                st.plotly_chart(fig_s, width="stretch")


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
            st.plotly_chart(f1, width="stretch")
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
            st.plotly_chart(f2, width="stretch")

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


def render_model_tab(
    view: pd.DataFrame,
    data: pd.DataFrame,
    training_result: Dict[str, Any],
    pack: Dict[str, Any],
) -> None:
    if not ML_AVAILABLE:
        st.info("Install scikit-learn to use training controls, charts, and downloads on this tab.")
        return

    te = training_result.get("_train_error") if isinstance(training_result, dict) else None
    train_pack = {
        k: v
        for k, v in (training_result or {}).items()
        if k != "_train_error"
    }

    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Model</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Training always uses the full cleaned sample from the sidebar row cap. "
        "Change settings below and submit to retrain (cached per configuration). "
        "Pick which fitted estimator drives the Risk tab using the selector above the tabs."
    )

    cur = _parse_ml_config_json(st.session_state.get("ml_config_json", DEFAULT_ML_CONFIG_JSON))
    with st.expander("Training setup", expanded=True):
        with st.form("ml_train_form"):
            st.markdown("**Data & split**")
            use_coords = st.checkbox(
                "Include latitude & longitude as features",
                value=bool(cur.get("use_coords", True)),
            )
            test_size = st.slider(
                "Holdout fraction (test set)",
                min_value=0.10,
                max_value=0.40,
                value=float(cur.get("test_size", 0.2)),
                step=0.05,
                help="Stratified split, random_state=42.",
            )
            st.markdown("**Which models to fit**")
            c0, c1 = st.columns(2)
            with c0:
                tr_rf = st.checkbox("Random Forest", value=bool(cur.get("train_rf", True)))
                tr_gb = st.checkbox("Gradient Boosting", value=bool(cur.get("train_gb", True)))
            with c1:
                tr_xgb = st.checkbox(
                    "XGBoost",
                    value=bool(cur.get("train_xgb", True)),
                    disabled=not XGB_AVAILABLE,
                )
                tr_lgb = st.checkbox(
                    "LightGBM",
                    value=bool(cur.get("train_lgb", True)),
                    disabled=not LGB_AVAILABLE,
                )
            st.markdown("**Hyperparameters**")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.caption("Random Forest")
                rf_n = st.number_input("RF n_estimators", 50, 500, int(cur.get("rf_n_estimators", 150)), 25)
                rf_d = st.number_input("RF max_depth", 3, 40, int(cur.get("rf_max_depth", 15)), 1)
                rf_leaf = st.number_input("RF min_samples_leaf", 1, 50, int(cur.get("rf_min_samples_leaf", 5)), 1)
            with r2:
                st.caption("Gradient Boosting")
                gb_n = st.number_input("GB n_estimators", 20, 400, int(cur.get("gb_n_estimators", 100)), 10)
                gb_lr = st.number_input("GB learning rate", 0.01, 0.50, float(cur.get("gb_learning_rate", 0.1)), 0.01)
                gb_d = st.number_input("GB max_depth", 2, 20, int(cur.get("gb_max_depth", 8)), 1)
            with r3:
                st.caption("XGB / LGB (when enabled)")
                xgb_n = st.number_input("Boost n_estimators", 20, 400, int(cur.get("xgb_n_estimators", 100)), 10)
                xgb_d = st.number_input("Boost max_depth", 2, 20, int(cur.get("xgb_max_depth", 8)), 1)
                xgb_lr = st.number_input("Boost learning rate", 0.01, 0.50, float(cur.get("xgb_learning_rate", 0.1)), 0.01)

            submitted = st.form_submit_button("Apply settings & retrain", width="stretch")

        if submitted:
            new_cfg: Dict[str, Any] = {
                "use_coords": use_coords,
                "test_size": float(test_size),
                "train_rf": tr_rf,
                "train_gb": tr_gb,
                "train_xgb": tr_xgb and XGB_AVAILABLE,
                "train_lgb": tr_lgb and LGB_AVAILABLE,
                "rf_n_estimators": int(rf_n),
                "rf_max_depth": int(rf_d),
                "rf_min_samples_leaf": int(rf_leaf),
                "gb_n_estimators": int(gb_n),
                "gb_learning_rate": float(gb_lr),
                "gb_max_depth": int(gb_d),
                "xgb_n_estimators": int(xgb_n),
                "xgb_max_depth": int(xgb_d),
                "xgb_learning_rate": float(xgb_lr),
                "lgb_n_estimators": int(xgb_n),
                "lgb_max_depth": int(xgb_d),
                "lgb_learning_rate": float(xgb_lr),
            }
            st.session_state.ml_config_json = json.dumps(new_cfg, sort_keys=True)
            st.rerun()

    if te:
        st.error(f"Training error: {te}")

    if (
        not train_pack
        or "models" not in train_pack
        or "best_model" not in train_pack
        or not train_pack["models"]
    ):
        if not te:
            st.info(
                "No models loaded yet. You need scikit-learn, at least 1,000 rows after cleaning, "
                "and at least one model enabled above."
            )
        return

    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Holdout evaluation</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{train_pack.get('n_train', 0):,} train rows · {train_pack.get('n_test', 0):,} test rows. "
        f"Highest ROC AUC this run: {train_pack.get('best_model_name', '')}."
    )

    mt = train_pack.get("metrics_table") or {}
    if mt:
        df_m = pd.DataFrame(mt).T.round(4)
        st.dataframe(df_m, width="stretch")

    probas = train_pack.get("probas_test") or {}
    y_test = train_pack.get("y_test")
    if y_test is not None and len(y_test) and probas and roc_curve is not None:
        v1, v2 = st.columns(2)
        with v1:
            try:
                st.plotly_chart(_plot_roc_curves(y_test, probas), width="stretch")
            except ValueError:
                st.caption("ROC curves need both classes on the holdout set.")
        with v2:
            if mt:
                st.plotly_chart(_plot_metrics_comparison(mt), width="stretch")

    active = st.session_state.get("active_ml_model", train_pack.get("best_model_name"))
    if active not in train_pack["models"]:
        active = train_pack["best_model_name"]
    proba_one = probas.get(active) if probas else None
    if y_test is not None and proba_one is not None and confusion_matrix is not None:
        st.plotly_chart(
            _plot_confusion_matrix_fig(
                y_test,
                proba_one,
                f"Confusion matrix @0.5 — {active}",
            ),
            width="stretch",
        )

    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Feature importance (scoring model)</p>',
        unsafe_allow_html=True,
    )
    scoring_model = train_pack["models"][active]["model"]
    if hasattr(scoring_model, "feature_importances_"):
        names = train_pack["feature_names"]
        imp = np.asarray(scoring_model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1]
        fig_imp = go.Figure(
            go.Bar(
                x=imp[order],
                y=[names[i] for i in order],
                orientation="h",
                marker_color=CHART["accent"],
                hovertemplate="%{y}<br>Importance %{x:.4f}<extra></extra>",
            )
        )
        fig_imp.update_layout(
            **_plotly_layout_base(f"Feature importance — {active}", "", None),
        )
        fig_imp.update_xaxes(title_text="Importance (tree-based)")
        fig_imp.update_yaxes(title_text="")
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.caption("This estimator doesn’t expose feature importances here.")

    st.markdown(
        '<p class="nyc-section-kicker nyc-kicker-tight">Batch scores & report</p>',
        unsafe_allow_html=True,
    )
    cfg = get_app_config()
    max_rows = int(cfg["model_export_max_rows"])

    forward = {
        **train_pack,
        "best_model": scoring_model,
        "best_model_name": train_pack.get("best_model_name"),
        "scoring_model_label": active,
    }

    st.caption(
        f"Using {active} for P(serious) on filtered rows with complete features "
        "(same as Risk tab). Not for deployment."
    )

    scored = _score_view_with_best_model(view, forward)
    if scored is None or scored.empty:
        st.warning(
            "No rows in the current filter could be scored (missing coordinates or other inputs). "
            "Widen filters, turn on lat/lon in training, or set severity to All."
        )
        return

    n_scored = len(scored)
    st.metric("Rows scored in this filter", f"{n_scored:,}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Mean P(serious)", f"{scored['p_serious'].mean():.1%}")
    with m2:
        if "is_serious" in scored.columns:
            st.metric("Actual serious share", f"{scored['is_serious'].mean():.1%}")
    with m3:
        if "is_serious" in scored.columns and n_scored > 0 and roc_auc_score is not None:
            try:
                auc = roc_auc_score(scored["is_serious"], scored["p_serious"])
                st.metric("ROC AUC (this filter)", f"{auc:.3f}")
            except Exception:  # noqa: BLE001
                st.metric("ROC AUC (this filter)", "—")

    extra_cols = [
        c
        for c in [
            "CRASH_DATE",
            "CRASH_TIME",
            "borough",
            "LATITUDE",
            "LONGITUDE",
            "year",
            "total_injured",
            "total_killed",
        ]
        if c in view.columns
    ]
    export_df = scored.copy()
    for c in extra_cols:
        export_df[c] = view.loc[export_df.index, c].values

    out_cols = (
        extra_cols
        + [c for c in forward["feature_names"] if c not in extra_cols]
        + (["is_serious"] if "is_serious" in export_df.columns else [])
        + ["p_serious"]
    )
    out_cols = [c for c in out_cols if c in export_df.columns]
    export_trim = export_df[out_cols].copy()

    cap = st.number_input(
        "Max rows in scored CSV",
        min_value=1,
        max_value=max_rows,
        value=min(25_000, max_rows, n_scored),
        step=1_000,
        help=f"Hard ceiling {max_rows:,}. If the filter returns more, we take a random sample (seed 42).",
    )
    cap_i = int(min(cap, n_scored))
    if n_scored > cap_i:
        export_dl = export_trim.sample(n=cap_i, random_state=42)
        st.caption(f"CSV uses a random {cap_i:,} of {n_scored:,} scored rows.")
    else:
        export_dl = export_trim

    show_prev = st.toggle("Preview first 200 scored rows", value=False)
    if show_prev:
        st.dataframe(export_dl.head(200), width="stretch", height=280)

    csv_buf = io.StringIO()
    export_dl.to_csv(csv_buf, index=True, index_label="source_row_index")
    st.download_button(
        label="Download scored CSV",
        data=csv_buf.getvalue(),
        file_name="nyc_crashes_scored_by_model.csv",
        mime="text/csv",
        width="stretch",
    )

    card = _model_card_payload(forward, data, pack)
    card["scored_export"] = {
        "filter_row_count_in_view": int(len(view)),
        "rows_with_complete_features": int(n_scored),
        "csv_rows_downloaded": int(len(export_dl)),
        "columns_in_csv": list(export_dl.columns),
    }
    json_bytes = json.dumps(card, indent=2)
    st.download_button(
        label="Download model report (JSON)",
        data=json_bytes,
        file_name="nyc_crash_model_report.json",
        mime="application/json",
        width="stretch",
    )


def main() -> None:
    cfg = get_app_config()

    st.markdown('<p class="nyc-page-title">NYC motor vehicle crashes</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="nyc-lead">Browse the city’s crash export: filter by year and borough, look at timing and maps, '
        "try a simple model that guesses how often a crash counts as serious, and pull batch scores plus a JSON model "
        "report from the Model tab when training succeeds. "
        "Public data only; this isn’t official analysis from DOT or NYPD.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### How many rows to load")
        st.caption("Smaller loads are faster; raise the cap if you have RAM and want more history.")
        cap = st.slider(
            "Max rows to load",
            min_value=cfg["data_sample_min"],
            max_value=cfg["data_sample_max"],
            value=min(cfg["data_sample_default"], cfg["data_sample_max"]),
            step=5_000,
            help="With a local CSV: random sample (seed 42) when the file is larger. "
            "Without a CSV (e.g. cloud): newest records from the API, up to this many.",
        )
        if st.button("Reload file (clear cache)", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if os.path.isfile(DATA_FILE):
        mtime = os.path.getmtime(DATA_FILE)
        with st.spinner("Loading CSV and cleaning dates…"):
            pack = load_collision_data(cap, DATA_FILE, mtime)
    else:
        st.info(
            f"No `{DATA_FILE}` in the app folder (normal for Streamlit Cloud). "
            "Pulling the newest crashes from NYC Open Data up to your row cap."
        )
        api_cap = min(cap, cfg["data_sample_max"])
        tok = _socrata_app_token()
        with st.spinner("Downloading from NYC Open Data…"):
            pack = load_collision_data_from_api(api_cap, tok)

    if pack.get("error") == "missing_file":
        st.error("The file was there and then it wasn’t. Check the path and try again.")
        return
    if pack.get("error") == "api_empty":
        st.error(
            "NYC Open Data returned no rows. Try again later, lower the row cap, "
            "or add `NYC_OPEN_DATA_APP_TOKEN` in Streamlit app secrets."
        )
        return
    if pack.get("error") == "schema":
        if pack.get("source") == "api":
            st.error("The API response didn’t contain usable crash date and time fields.")
        else:
            st.error(
                "This file doesn’t look like the NYC crashes export. We need crash date and time columns "
                "(spaces or underscores both work — e.g. `CRASH DATE` or `CRASH_DATE`)."
            )
        return
    if pack.get("error") and str(pack["error"]).startswith("load_failed"):
        if pack.get("source") == "api":
            st.error(
                "Couldn’t download or parse NYC Open Data. Check connectivity, try again, "
                "or set `NYC_OPEN_DATA_APP_TOKEN` in secrets if you’re rate-limited."
            )
        else:
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

    st.session_state.setdefault("ml_config_json", DEFAULT_ML_CONFIG_JSON)

    raw_train: Dict[str, Any] = {}
    train_pack: Dict[str, Any] = {}
    if ML_AVAILABLE:
        cfg_j = st.session_state["ml_config_json"]
        with st.spinner("Training models…"):
            raw_train = train_models(data, cfg_j)
        te = raw_train.get("_train_error") if isinstance(raw_train, dict) else None
        train_pack = {
            k: v
            for k, v in (raw_train or {}).items()
            if k != "_train_error"
        }
        if te:
            st.warning(
                "Model training failed. Charts, map, and the Model tab still work; the Risk tab won’t show a score."
            )
            st.caption(str(te))
            train_pack = {}
        elif train_pack.get("models"):
            tc = train_pack.get("train_config") or {}
            ts = float(tc.get("test_size", 0.2))
            st.caption(
                f"Fit on {len(data):,} cleaned rows; {ts:.0%} held out for scoring. "
                f"Best ROC AUC: {train_pack.get('best_model_name', '')}."
            )

    if train_pack.get("models"):
        opts = list(train_pack["models"].keys())
        if st.session_state.get("active_ml_model") not in opts:
            st.session_state.active_ml_model = train_pack["best_model_name"]
        st.selectbox(
            "Model for Risk predictions & batch scores",
            options=opts,
            key="active_ml_model",
        )

    model_info: Dict[str, Any] = {}
    if train_pack.get("models"):
        act = st.session_state.get("active_ml_model", train_pack["best_model_name"])
        if act not in train_pack["models"]:
            act = train_pack["best_model_name"]
            st.session_state.active_ml_model = act
        model_info = {
            **train_pack,
            "best_model": train_pack["models"][act]["model"],
            "scoring_model_label": act,
        }

    tab_risk, tab_time, tab_map, tab_model = st.tabs(
        ["Risk", "Time", "Map", "Model"]
    )
    with tab_risk:
        render_prediction_ui(model_info, view)
    with tab_time:
        if len(view) < 10:
            st.info("Need at least 10 rows for these charts. Relax the filters.")
        else:
            render_time_charts(view)
            render_pattern_charts(view)
    with tab_map:
        render_geo(view)
    with tab_model:
        render_model_tab(view, data, raw_train if ML_AVAILABLE else {}, pack)

    st.divider()
    st.caption(
        f"Data: [NYC Open Data]({NYC_OPEN_DATA_URL}) · "
        f"[Kaggle CSV mirror]({KAGGLE_DATASET_URL}). "
        "For exploration only, not official safety planning."
    )


if __name__ == "__main__":
    main()
