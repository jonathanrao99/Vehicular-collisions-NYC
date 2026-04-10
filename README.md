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

Optional: put `Motor_Vehicle_Collisions_-_Crashes.csv` in the project root (gitignored by default). Get it from NYC Open Data or the [Kaggle mirror](https://www.kaggle.com/datasets/tush32/motor-vehicle-collisions-crashes). If the file is missing (e.g. on Streamlit Cloud), the app downloads the newest rows from NYC Open Data’s API instead, up to your row cap.

## Setup

- Python 3.8+ and enough RAM for a multi‑million row CSV, or use the app’s row cap.  
- `pip install -r requirements.txt`  
- Download the crashes CSV from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) or the [Kaggle mirror](https://www.kaggle.com/datasets/tush32/motor-vehicle-collisions-crashes) and name it as above.

### NYC Open Data API credentials (optional)

If you [sign in](https://data.cityofnewyork.us/) and open [Developer / app tokens](https://data.cityofnewyork.us/profile/edit/developer_settings), you’ll get:

| What NYC shows | Use in this project? |
|----------------|----------------------|
| **Application token** (app token) | **Yes.** The app sends it as `X-App-Token` when downloading data from the API (better rate limits on Streamlit Cloud). |
| **Secret token** | **No** for this app. Public crash data is read with GET requests; the secret is for other Socrata features. **Never commit** the secret. |

**Streamlit Cloud:** App → **Settings** → **Secrets** → add:

```toml
NYC_OPEN_DATA_APP_TOKEN = "paste-your-application-token-here"
```

**Local:** create `.streamlit/secrets.toml` with the same line (that file is gitignored). Alternatively set the environment variable `NYC_OPEN_DATA_APP_TOKEN`.

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
1. Push code to GitHub (you do **not** need to commit the CSV).
2. Connect the repository to [share.streamlit.io](https://share.streamlit.io) and deploy.
3. **Recommended:** In app **Secrets**, set `NYC_OPEN_DATA_APP_TOKEN` to your **application token** from [NYC developer settings](https://data.cityofnewyork.us/profile/edit/developer_settings) (see *NYC Open Data API credentials* above). You do **not** need to put the secret token in this app.

### Docker (optional)
```bash
docker build -t nyc-collisions .
docker run -p 8501:8501 nyc-collisions
```

## Data

Source: [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) on NYC Open Data. A downloadable CSV mirror is also on [Kaggle](https://www.kaggle.com/datasets/tush32/motor-vehicle-collisions-crashes). Fields include crash date/time, coordinates when present, injury and fatality counts, contributing factors, and vehicle types. Coverage and quality depend on how incidents are reported.

## Contributing

Issues and PRs welcome. For bugs, include OS, Python version, and how to reproduce.

## Troubleshooting

- **CSV not found (local)** — file name and location must match above; check permissions. On Streamlit Cloud, missing CSV is expected; the app uses the live API.  
- **Models** — `pip install scikit-learn`; optional `xgboost`, `lightgbm`. On some Macs you may need OpenMP (`brew install libomp`) for XGBoost.  
- **Maps** — `pip install folium`; tiles need network access.  
- **Slow or OOM** — lower `max_rows` in the sidebar and smaller ML sample sizes.  
- **API / throttling on Cloud** — add `NYC_OPEN_DATA_APP_TOKEN` (application token only) in Streamlit Secrets; keep the NYC **secret** token out of the repo.

## Performance

Streamlit caches the loaded frame and fitted model. Maps subsample points for responsiveness.

## Privacy

No accounts or telemetry in this repo. Processing is local except map tile requests.

## License

MIT — see `LICENSE`.

## Credits

NYC Open Data for the dataset; [Kaggle mirror](https://www.kaggle.com/datasets/tush32/motor-vehicle-collisions-crashes) for an easy CSV download; Streamlit, scikit-learn, Plotly, and Folium for the stack used here.
