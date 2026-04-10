# NYC motor vehicle collisions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Streamlit UI and a Jupyter notebook for NYC’s published [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) file. You can filter the data, plot when and where crashes happen, train a few classifiers on a “serious” label, and download a filtered CSV. It’s a local exploration tool, not a substitute for anything official.

## What’s in the repo

- `streamlit_app.py` — dashboard: filters, summary, risk tab, time charts, map, model scoring + report  
- `nyc_collisions_advanced_analysis.ipynb` — same CSV, EDA and models (optional TensorFlow section if you install it)  
- `requirements.txt` — Python dependencies  
- `.streamlit/config.toml` — light theme and accent color  

Put `Motor_Vehicle_Collisions_-_Crashes.csv` in the project root (gitignored by default).

## Setup

- Python 3.8+ and enough RAM for a multi‑million row CSV, or use the app’s row cap.  
- `pip install -r requirements.txt`  
- Download the crashes CSV from NYC Open Data and name it as above.

## Run

```bash
streamlit run streamlit_app.py
jupyter notebook nyc_collisions_advanced_analysis.ipynb   # optional
```

## Using the Streamlit app

1. **Risk** — time, place, and a probability from the best model on your loaded sample (holdout ROC AUC).  
2. **Time** — hourly volume, weekday and month charts, serious share by context.  
3. **Map** — borough bars and a Folium heat layer (sampled for speed).  
4. **Model** — tune holdout size, which algorithms run, and tree hyperparameters; compare ROC AUC / accuracy / F1 on the holdout set; pick which fitted model powers the Risk tab; batch-score the filtered slice (CSV) and download a JSON report.

Open the in-app “Notes” expander for row counts, sampling, label definitions, and limits.

### UI theme

Colors and base fonts come from [`.streamlit/config.toml`](.streamlit/config.toml). Plotly colors are set in the `CHART` dict at the top of `streamlit_app.py`. Extra layout CSS uses class names starting with `nyc-` in the first `st.markdown` block of `streamlit_app.py`.

## Models and features (notebook / app)

The app trains tree-based models (and XGBoost/LightGBM if installed) on time and location fields plus flags (weekend, rush hour, night, etc.). “Serious” is defined as at least one fatality or two or more injuries. The notebook can add a small Keras model if TensorFlow is available. Treat metrics as exploratory, not production benchmarks.

## Jupyter notebook

The notebook loads the CSV, cleans dates, plots basics, fits several classifiers, and compares ROC AUC on a test split. Install optional libraries from `requirements.txt` for full parity with the app.

## Deployment

### Local
```bash
streamlit run streamlit_app.py --server.port 8501
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Docker (optional)
```bash
docker build -t nyc-collisions .
docker run -p 8501:8501 nyc-collisions
```

## Data

Source: [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) on NYC Open Data. Fields include crash date/time, coordinates when present, injury and fatality counts, contributing factors, and vehicle types. Coverage and quality depend on how incidents are reported.

## Contributing

Issues and PRs welcome. For bugs, include OS, Python version, and how to reproduce.

## Troubleshooting

- **CSV not found** — file name and location must match above; check permissions.  
- **Models** — `pip install scikit-learn`; optional `xgboost`, `lightgbm`. On some Macs you may need OpenMP (`brew install libomp`) for XGBoost.  
- **Maps** — `pip install folium`; tiles need network access.  
- **Slow or OOM** — lower `max_rows` in the sidebar and smaller ML sample sizes.

## Performance

Streamlit caches the loaded frame and fitted model. Maps subsample points for responsiveness.

## Privacy

No accounts or telemetry in this repo. Processing is local except map tile requests.

## License

MIT — see `LICENSE`.

## Credits

NYC Open Data for the dataset; Streamlit, scikit-learn, Plotly, and Folium for the stack used here.
