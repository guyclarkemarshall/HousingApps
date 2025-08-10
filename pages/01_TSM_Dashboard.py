import time
import requests
import streamlit as st
import numpy as np
import pandas as pd
from utils.io_excel import load_tsm_from_excel_bytes
from utils.charts import leaderboard_bar, scatter_with_trend, distribution_box

st.set_page_config(page_title="TSM Dashboard", page_icon="🏠", layout="wide")
st.title("TSM 2024 Dashboard (Excel from URL or upload)")

DEFAULT_URL = "https://assets.publishing.service.gov.uk/media/6744943be26d6f8ca3cb35c0/2024_TSM_Full_Data_v1.1_FINAL.xlsx"

# ---- Controls (left sidebar) ----
st.sidebar.header("Data source")
use_default = st.sidebar.toggle("Use default public dataset", value=True, help="If off, upload your own Excel below.")
uploaded = st.sidebar.file_uploader("Or upload a TSM Excel", type=["xlsx","xls"])
custom_url = st.sidebar.text_input("…or fetch from a custom URL", value=DEFAULT_URL if not uploaded else "")

@st.cache_data(show_spinner=True, ttl=60*10)  # cache for 10 minutes
def fetch_excel_bytes(url: str, timeout: int = 20) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

# Resolve source of bytes
excel_bytes = None
source_label = None
try:
    if uploaded is not None and not use_default:
        excel_bytes = uploaded.getvalue()
        source_label = f"Uploaded: {uploaded.name}"
    else:
        # Use URL (default or custom)
        url = custom_url.strip() if custom_url.strip() else DEFAULT_URL
        excel_bytes = fetch_excel_bytes(url)
        source_label = f"Fetched from URL: {url}"
except Exception as e:
    st.error(f"Could not load Excel file. {e}")
    st.stop()

# Load and tidy
@st.cache_data(show_spinner=True, ttl=60*10)
def load_df(b: bytes) -> pd.DataFrame:
    return load_tsm_from_excel_bytes(b)

df = load_df(excel_bytes)
if df.empty:
    st.warning("No headline TSM proportion columns were found. Check the workbook/sheets.")
    st.stop()

st.caption(f"Data source → {source_label}")

# ---- Filters ----
st.sidebar.header("Filters")
tenures = sorted(df["tenure"].dropna().unique().tolist())
regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
metrics = sorted(df["metric_code"].dropna().unique().tolist())

tenure = st.sidebar.selectbox("Tenure", tenures, index=0)
region = st.sidebar.selectbox("Region", regions, index=0)
metric = st.sidebar.selectbox("Metric (TP01–TP12)", metrics, index=metrics.index("TP01") if "TP01" in metrics else 0)

def bounds(series):
    s = series.dropna().astype(float)
    if len(s)==0: return (0.0,1.0), True
    lo, hi = float(np.floor(s.min())), float(np.ceil(s.max()))
    if lo==hi: hi = lo+1
    return (lo,hi), False

(gb,gdis) = bounds(df["group_size"])
(sb,sdis) = bounds(df["sample_achieved"])

if not gdis:
    gmin,gmax = st.sidebar.slider("Group size (homes)", min_value=int(gb[0]), max_value=int(gb[1]), value=(int(gb[0]), int(gb[1])))
else:
    gmin,gmax = 0,0
if not sdis:
    smin,smax = st.sidebar.slider("Sample size achieved", min_value=int(sb[0]), max_value=int(sb[1]), value=(int(sb[0]), int(sb[1])))
else:
    smin,smax = 0,0

# Apply filters
d = df[(df["tenure"]==tenure) & (df["metric_code"]==metric)].copy()
if region != "All": d = d[d["region"]==region]
if not gdis: d = d[(d["group_size"].fillna(-1) >= gmin) & (d["group_size"].fillna(-1) <= gmax)]
if not sdis: d = d[(d["sample_achieved"].fillna(-1) >= smin) & (d["sample_achieved"].fillna(-1) <= smax)]

# ---- Charts ----
st.subheader("Leaderboard (Top 15)")
st.plotly_chart(leaderboard_bar(d), use_container_width=True)

st.subheader("Score vs Size")
x_choice = st.selectbox("X-axis", ["group_size","sample_achieved","relevant_population"], index=0, format_func=lambda s: s.replace("_"," ").title())
if d[x_choice].dropna().empty:
    x_choice = "sample_achieved"
st.plotly_chart(scatter_with_trend(d, x_choice), use_container_width=True)

st.subheader("Distribution (Box Plot)")
split_by = st.radio("Split by", ["region","landlord_type"], horizontal=True, index=0)
st.plotly_chart(distribution_box(d, split_by), use_container_width=True)
