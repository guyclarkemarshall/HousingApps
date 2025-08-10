import streamlit as st
import numpy as np
from utils.io_excel import load_tsm_from_excel
from utils.charts import leaderboard_bar, scatter_with_trend, distribution_box

st.set_page_config(page_title="TSM Dashboard", page_icon="🏠", layout="wide")
st.title("TSM 2024 Dashboard (Excel upload)")

uploaded = st.file_uploader("Upload the 2024 TSM Excel file", type=["xlsx","xls"])
if not uploaded:
    st.info("Upload the workbook to start (e.g., 2024_TSM_Full_Data_v1.1_FINAL.xlsx).")
    st.stop()

# Cache load
@st.cache_data(show_spinner=False)
def _load(b: bytes):
    return load_tsm_from_excel(b)

df = _load(uploaded.getvalue())
if df.empty:
    st.error("Could not find headline 'Proportion of respondents...' columns. Check the workbook.")
    st.stop()

# Sidebar filters
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

# Layout: 3 charts
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
