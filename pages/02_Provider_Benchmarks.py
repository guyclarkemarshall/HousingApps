import io, re, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from difflib import get_close_matches

# ---- CONFIG ----
st.set_page_config(page_title="Provider Benchmarks", page_icon="📈", layout="wide")
st.title("📈 Provider Benchmarks (TSM 2024)")
DEFAULT_URL = "https://assets.publishing.service.gov.uk/media/6744943be26d6f8ca3cb35c0/2024_TSM_Full_Data_v1.1_FINAL.xlsx"

# ---- HELPERS ----
HEADLINE_PATTERN = re.compile(r"Proportion of respondents who report.*\((TP\d+)\)", re.IGNORECASE)
SHEETS = [("TSM24_LCRA_Perception", "LCRA"), ("TSM24_LCHO_Perception", "LCHO")]

@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_excel_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def load_tsm_from_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    def load_sheet(sheet: str, tenure: str) -> pd.DataFrame:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=2)
        id_cols = [
            "Landlord name","Landlord code","Landlord type","Landlord predominant region",
            "Landlord group size (homes owned, LCRA and LCHO combined)",
            "Relevant\xa0tenant population size (LCRA)","Relevant\xa0tenant population size (LCHO)",
            "Required minimum sample size2,3","Total sample size achieved"
        ]
        present = [c for c in id_cols if c in df.columns]
        core = df[present].copy().rename(columns={
            "Landlord name":"landlord_name",
            "Landlord code":"landlord_code",
            "Landlord type":"landlord_type",
            "Landlord predominant region":"region",
            "Landlord group size (homes owned, LCRA and LCHO combined)":"group_size",
            "Relevant\xa0tenant population size (LCRA)":"relevant_population_lcra",
            "Relevant\xa0tenant population size (LCHO)":"relevant_population_lcho",
            "Required minimum sample size2,3":"required_min_sample",
            "Total sample size achieved":"sample_achieved",
        })
        core["relevant_population"] = np.where(
            tenure=="LCRA", core.get("relevant_population_lcra"),
            core.get("relevant_population_lcho")
        )

        metric_cols = {}
        for c in df.columns:
            if isinstance(c, str) and "for each survey method" not in c:
                m = HEADLINE_PATTERN.search(c)
                if m: metric_cols[c] = m.group(1)
        if not metric_cols:
            return pd.DataFrame()

        long = df[list(metric_cols.keys())].melt(var_name="metric_label", value_name="proportion")
        id_rep = pd.concat([core]*len(metric_cols), ignore_index=True)
        out = pd.concat([id_rep, long], axis=1)
        out["metric_code"] = out["metric_label"].map(metric_cols)
        out["tenure"] = tenure
        for c in ["group_size","relevant_population","required_min_sample","sample_achieved","proportion"]:
            if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["proportion"])
        out = out[~out["landlord_name"].isna()].reset_index(drop=True)
        return out[[
            "tenure","landlord_name","landlord_code","landlord_type","region","group_size",
            "relevant_population","required_min_sample","sample_achieved",
            "metric_code","metric_label","proportion"
        ]]

    frames = []
    for sheet, tenure in SHEETS:
        part = load_sheet(sheet, tenure)
        if not part.empty: frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def pct_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().astype(float).sort_values()
    if len(s)==0: return np.nan
    return float((s < value).sum()) / float(len(s)) * 100.0

def nearest_peers(df: pd.DataFrame, target: float, k: int = 10) -> pd.DataFrame:
    d = df.assign(abs_diff=(df["proportion"] - target).abs())
    return d.sort_values(["abs_diff","proportion"]).head(k)

# ---- DATA SOURCE (URL default, upload or custom URL override) ----
with st.sidebar:
    st.header("Data source")
    use_default = st.toggle("Use default public dataset", value=True)
    uploaded = st.file_uploader("Or upload a TSM Excel", type=["xlsx","xls"])
    custom_url = st.text_input("…or fetch from a custom URL",
                               value=DEFAULT_URL if not uploaded else "")

try:
    if uploaded is not None and not use_default:
        excel_bytes = uploaded.getvalue()
        src_label = f"Uploaded: {uploaded.name}"
    else:
        url = custom_url.strip() if custom_url.strip() else DEFAULT_URL
        excel_bytes = fetch_excel_bytes(url)
        src_label = f"Fetched from URL: {url}"
except Exception as e:
    st.error(f"Could not load Excel file. {e}")
    st.stop()

@st.cache_data(show_spinner=True, ttl=60*10)
def load_df(b: bytes) -> pd.DataFrame:
    return load_tsm_from_excel_bytes(b)

df = load_df(excel_bytes)
if df.empty:
    st.warning("No headline TSM proportion columns were found. Check the workbook/sheets.")
    st.stop()

st.caption(f"Data source → {src_label}")

# ---- TOP FILTERS ----
fcol1, fcol2, fcol3 = st.columns([1,1,2])
with fcol1:
    tenure = st.selectbox("Tenure", sorted(df["tenure"].unique().tolist()), index=0)
with fcol2:
    metrics = sorted(df["metric_code"].unique().tolist())
    metric = st.selectbox("Metric", metrics, index=(metrics.index("TP01") if "TP01" in metrics else 0))
with fcol3:
    # Fuzzy landlord search
    names = sorted(df["landlord_name"].unique().tolist())
    provider_query = st.text_input("Type provider name", value="")
    suggestions = get_close_matches(provider_query, names, n=8, cutoff=0.6) if provider_query else []
    provider = st.selectbox("Select provider", suggestions if suggestions else names, index=0)

# ---- COHORT FILTERS (build comparison set) ----
cohort = df[(df["tenure"]==tenure) & (df["metric_code"]==metric)].copy()

with st.sidebar:
    st.header("Cohort filters")
    # Region scope
    provider_region = cohort.loc[cohort["landlord_name"]==provider, "region"].dropna().unique()
    same_region_default = provider_region[0] if len(provider_region)>0 else None
    region_mode = st.radio("Region scope", ["All regions","Same as provider","Select…"], index=1 if same_region_default else 0)
    if region_mode == "Select…":
        region_sel = st.multiselect("Regions", sorted(cohort["region"].dropna().unique().tolist()))
        if region_sel:
            cohort = cohort[cohort["region"].isin(region_sel)]
    elif region_mode == "Same as provider" and same_region_default:
        cohort = cohort[cohort["region"]==same_region_default]

    # Landlord type
    type_mode = st.radio("Landlord type", ["All types","Same as provider","Select…"], index=0)
    if type_mode == "Same as provider":
        ptype = cohort.loc[cohort["landlord_name"]==provider, "landlord_type"].dropna().unique()
        if len(ptype)>0:
            cohort = cohort[cohort["landlord_type"]==ptype[0]]
    elif type_mode == "Select…":
        type_sel = st.multiselect("Types", sorted(cohort["landlord_type"].dropna().unique().tolist()))
        if type_sel:
            cohort = cohort[cohort["landlord_type"].isin(type_sel)]

    # Size bands
    def bounds(series):
        s = series.dropna().astype(float)
        if len(s)==0: return (0.0,1.0), True
        lo, hi = float(np.floor(s.min())), float(np.ceil(s.max()))
        if lo==hi: hi = lo+1
        return (lo,hi), False
    (gb, gdis) = bounds(cohort["group_size"])
    if not gdis:
        gmin,gmax = st.slider("Group size (homes)", min_value=int(gb[0]), max_value=int(gb[1]), value=(int(gb[0]), int(gb[1])))
        cohort = cohort[(cohort["group_size"].fillna(-1) >= gmin) & (cohort["group_size"].fillna(-1) <= gmax)]
    (sb, sdis) = bounds(cohort["sample_achieved"])
    if not sdis:
        smin,smax = st.slider("Sample size achieved", min_value=int(sb[0]), max_value=int(sb[1]), value=(int(sb[0]), int(sb[1])))
        cohort = cohort[(cohort["sample_achieved"].fillna(-1) >= smin) & (cohort["sample_achieved"].fillna(-1) <= smax)]

# Ensure provider row exists in the working slice
prov_row = cohort[cohort["landlord_name"]==provider]
if prov_row.empty:
    st.warning("Your current filters exclude the selected provider. Widen filters (region/type/size).")
    st.stop()

provider_score = float(prov_row["proportion"].iloc[0])

# ---- INSIGHTS STRIP ----
s = cohort["proportion"].dropna().astype(float)
mean = float(s.mean()) if len(s) else np.nan
median = float(s.median()) if len(s) else np.nan
std = float(s.std(ddof=0)) if len(s) else np.nan
p10 = float(s.quantile(0.10)) if len(s) else np.nan
p90 = float(s.quantile(0.90)) if len(s) else np.nan
z = (provider_score - mean)/std if std and std>0 else np.nan
pct = pct_rank(s, provider_score)

# Outliers (IQR)
q1, q3 = s.quantile([0.25, 0.75]) if len(s) else (np.nan, np.nan)
iqr = (q3 - q1) if (q1==q1 and q3==q3) else np.nan
lower = q1 - 1.5*iqr if iqr==iqr else np.nan
upper = q3 + 1.5*iqr if iqr==iqr else np.nan
iqr_outliers = int(((s < lower) | (s > upper)).sum()) if lower==lower and upper==upper else 0

st.markdown("### 🔎 Benchmarks (current cohort)")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Provider score", f"{provider_score:.1f}%")
k2.metric("Cohort mean", f"{mean:.1f}%")
k3.metric("Percentile", f"{pct:.0f}th")
k4.metric("Z-score", f"{z:.2f}" if z==z else "—")
k5.metric("IQR outliers", f"{iqr_outliers}")

# ---- CHARTS ----
# 1) Distribution histogram with provider line
st.subheader("Distribution (cohort)")
hist = px.histogram(cohort, x="proportion", nbins=30, labels={"proportion":"Proportion (%)"})
hist.add_vline(x=provider_score, line_color="black", line_width=2, annotation_text="Provider", annotation_position="top")
hist.add_vline(x=mean, line_dash="dash", annotation_text="Mean")
hist.add_vline(x=median, line_dash="dot", annotation_text="Median")
st.plotly_chart(hist, use_container_width=True)

# 2) Box plot by region or type (toggle)
st.subheader("Spread by grouping")
group_choice = st.radio("Split by", ["region","landlord_type"], horizontal=True)
bx = cohort.dropna(subset=["proportion"]).copy()
counts = bx[group_choice].value_counts()
keep = counts[counts >= 5].index
bx = bx[bx[group_choice].isin(keep)]
if len(bx)>0:
    med_order = bx.groupby(group_choice)["proportion"].median().sort_values(ascending=False)
    bx[group_choice] = pd.Categorical(bx[group_choice], categories=med_order.index, ordered=True)
box = px.box(bx, x=group_choice, y="proportion", points="outliers",
             labels={"proportion":"Proportion (%)", group_choice:group_choice.replace("_"," ").title()})
st.plotly_chart(box, use_container_width=True)

# 3) Nearest peers table
st.subheader("Nearest peers (by score difference)")
peers = nearest_peers(cohort[cohort["landlord_name"]!=provider], provider_score, k=10)
peers = peers[["landlord_name","region","landlord_type","group_size","sample_achieved","proportion","abs_diff"]]\
         .rename(columns={"proportion":"score","abs_diff":"|Δ|"})
st.dataframe(peers, use_container_width=True)

# ---- OPTIONAL: MULTI-METRIC PROFILE (radar) ----
with st.expander("Multi-metric profile (radar)"):
    # Pick up to 6 metrics for a quick profile
    all_metrics = sorted(df["metric_code"].unique().tolist())
    chosen = st.multiselect("Select metrics (up to 6)", all_metrics, default=["TP01","TP02"], max_selections=6)
    prof = df[(df["tenure"]==tenure) & (df["metric_code"].isin(chosen))].copy()
    # provider
    pvec = prof[prof["landlord_name"]==provider].groupby("metric_code", as_index=False)["proportion"].mean()
    # cohort (based on current filters ignoring metric)
    cohort_ids = cohort["landlord_name"].unique().tolist()
    cvec = df[(df["tenure"]==tenure) & (df["metric_code"].isin(chosen)) & (df["landlord_name"].isin(cohort_ids))]\
             .groupby("metric_code", as_index=False)["proportion"].mean()
    # align axes
    axes = chosen
    pmap = {k:v for k,v in zip(pvec["metric_code"], pvec["proportion"])}
    cmap = {k:v for k,v in zip(cvec["metric_code"], cvec["proportion"])}
    pvals = [pmap.get(a, np.nan) for a in axes]
    cvals = [cmap.get(a, np.nan) for a in axes]
    # Build radar
    rad = go.Figure()
    rad.add_trace(go.Scatterpolar(r=pvals, theta=axes, fill='toself', name=provider))
    rad.add_trace(go.Scatterpolar(r=cvals, theta=axes, fill='toself', name='Cohort mean'))
    rad.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(rad, use_container_width=True)

# ---- DOWNLOAD VIEW ----
st.download_button(
    label="📥 Download cohort (CSV)",
    data=cohort.to_csv(index=False).encode("utf-8"),
    file_name=f"tsm_benchmark_{tenure}_{metric}_{provider.replace(' ','_')}.csv",
    mime="text/csv",
)
