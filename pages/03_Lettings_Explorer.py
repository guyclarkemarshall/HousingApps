import io, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Lettings Explorer (Social Housing 2023/24)", page_icon="🏘️", layout="wide")
st.title("🏘️ Social Housing Lettings in England — Tenancies (2023/24)")

DEFAULT_ODS_URL = "https://assets.publishing.service.gov.uk/media/67505f25bcd3a46a2248c878/Social_housing_lettings_in_England_tenancies_summary_tables_April_2023_to_March_2024.ods"

st.caption("Data: DLUHC — Social housing lettings in England, tenancies summary tables (Apr 2023–Mar 2024)")

# -----------------------------------------------------------------------------
# Helpers: fetch ODS, load sheets, fuzzy column detection, tidy
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

COLUMN_SYNONYMS = {
    "region": ["region", "government office region", "gor", "itl1", "area"],
    "landlord_type": ["landlord type", "provider type", "organisation type", "landlord"],
    "need": ["need", "housing need", "needs type", "general needs", "supported"],
    "rent_type": ["rent type", "tenure type", "type of rent", "social rent", "affordable", "intermediate"],
    "tenancies": ["tenancies", "lettings", "lets", "number of tenancies", "count"],
    "homes": ["stock", "dwellings", "homes"],
}

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def find_col(df: pd.DataFrame, keys: list[str]):
    cols = { _norm(c): c for c in df.columns }
    for k in keys:
        k = k.lower()
        # exact
        if k in cols: return cols[k]
        # contains
        for norm, orig in cols.items():
            if k in norm:
                return orig
    return None

def possible_numeric(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=True, ttl=60*10)
def load_ods_to_long(ods_bytes: bytes) -> dict:
    """
    Reads all sheets via pandas.read_excel(engine='odf') and returns:
    {
      "raw": dict of sheet_name -> DataFrame,
      "candidates": long_df (best-effort tidy with columns:
           region, landlord_type, need, rent_type, tenancies, sheet)
    }
    """
    def _norm(s):  # local normaliser (guard against non-str)
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    # 1) Read all sheets (engine=odf requires odfpy)
    all_sheets = pd.read_excel(io.BytesIO(ods_bytes), sheet_name=None, engine="odf")
    raw = {}
    for name, df in all_sheets.items():
        if df is None:
            continue
        # Ensure string col names, drop all-empty rows/cols
        df.columns = [str(c) for c in df.columns]
        df = df.dropna(how="all")
        df = df.loc[:, ~df.columns.to_series().apply(lambda c: df[c].isna().all())]
        raw[name] = df

    rows = []
    for name, df in raw.items():
        if df.empty or len(df.columns) < 2:
            continue

        # --- try to detect dimension columns ---
        c_region   = find_col(df, COLUMN_SYNONYMS["region"])
        c_landlord = find_col(df, COLUMN_SYNONYMS["landlord_type"])
        c_need     = find_col(df, COLUMN_SYNONYMS["need"])
        c_rent     = find_col(df, COLUMN_SYNONYMS["rent_type"])

        # --- detect 1+ numeric "tenancies-like" columns (duplicates possible) ---
        ten_like_norms = {_norm(k) for k in COLUMN_SYNONYMS["tenancies"]}
        ten_cols = []
        for col in df.columns:
            n = _norm(col)
            if any(t in n for t in ten_like_norms) and possible_numeric(df[col]):
                ten_cols.append(col)

        dims = [c for c in [c_region, c_landlord, c_need, c_rent] if c is not None]
        if not dims or not ten_cols:
            continue

        # Keep unique column labels in order
        keep_cols = list(dict.fromkeys(dims + ten_cols))
        sub = df[keep_cols].copy()

        # Canonical rename for dims
        colmap = {}
        if c_region:   colmap[c_region] = "region"
        if c_landlord: colmap[c_landlord] = "landlord_type"
        if c_need:     colmap[c_need] = "need"
        if c_rent:     colmap[c_rent] = "rent_type"
        sub = sub.rename(columns=colmap)

        # --- coalesce multiple "tenancies-like" columns into a single 'tenancies' ---
        # Identify all columns (post-rename) that look like counts
        ten_candidates = []
        for col in sub.columns:
            n = _norm(col)
            if ("tenanc" in n) or ("lett" in n) or (n in {"count", "number of tenancies", "number"}):
                ten_candidates.append(col)

        # If nothing matched above, fall back to original ten_cols
        if not ten_candidates:
            ten_candidates = ten_cols

        # Convert candidates to numeric and row-wise sum (min_count=1 keeps NaN when all NaN)
        ten_df = sub[ten_candidates].apply(pd.to_numeric, errors="coerce")
        sub["tenancies"] = ten_df.sum(axis=1, min_count=1)

        # Drop other count columns now that we've coalesced
        sub = sub.drop(columns=[c for c in ten_candidates if c != "tenancies" and c in sub.columns], errors="ignore")

        # Clean text dims; drop obvious "total" rows
        for c in ["region","landlord_type","need","rent_type"]:
            if c in sub.columns:
                sub[c] = sub[c].astype(str).str.strip()
                sub = sub[~sub[c].str.contains(r"\btotal\b", case=False, na=False)]

        # Finalise
        sub["sheet"] = name
        rows.append(sub)

    if not rows:
        return {"raw": raw, "candidates": pd.DataFrame()}

    long_df = pd.concat(rows, ignore_index=True)

    # Normalise region names (loose)
    if "region" in long_df.columns:
        normalize = {
            "north east": "North East",
            "north west": "North West",
            "yorkshire and the humber": "Yorkshire and The Humber",
            "yorkshire & the humber": "Yorkshire and The Humber",
            "east midlands": "East Midlands",
            "west midlands": "West Midlands",
            "east of england": "East of England",
            "london": "London",
            "south east": "South East",
            "south west": "South West",
        }
        long_df["region"] = long_df["region"].str.lower().map(normalize).fillna(long_df["region"])

    # Keep only positive counts
    long_df["tenancies"] = pd.to_numeric(long_df["tenancies"], errors="coerce")
    long_df = long_df.dropna(subset=["tenancies"])
    long_df = long_df[long_df["tenancies"] > 0]

    # Ensure canonical columns present
    for c in ["region","landlord_type","need","rent_type"]:
        if c not in long_df.columns:
            long_df[c] = np.nan

    return {"raw": raw, "candidates": long_df[["region","landlord_type","need","rent_type","tenancies","sheet"]]}


# -----------------------------------------------------------------------------
# Data source controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    use_default = st.toggle("Use default GOV.UK dataset", value=True)
    upload = st.file_uploader("Or upload the ODS", type=["ods"])
    custom_url = st.text_input("…or fetch from a custom URL", value=DEFAULT_ODS_URL if not upload else "")

try:
    if upload is not None and not use_default:
        ods_bytes = upload.getvalue()
        source_label = f"Uploaded: {upload.name}"
    else:
        url = custom_url.strip() if custom_url.strip() else DEFAULT_ODS_URL
        ods_bytes = fetch_bytes(url)
        source_label = f"Fetched from URL: {url}"
except Exception as e:
    st.error(f"Could not load the ODS file. {e}")
    st.stop()

loaded = load_ods_to_long(ods_bytes)
raw_sheets = loaded["raw"]
long_df = loaded["candidates"]

if long_df.empty:
    st.error("I couldn’t automatically detect a tenancies table with region/landlord/need/rent columns. "
             "Use the expander below to inspect raw sheets and confirm column headers.")
    with st.expander("🔧 Inspect raw sheets"):
        for name, df in raw_sheets.items():
            st.write(f"**{name}**")
            st.dataframe(df.head(25), use_container_width=True)
    st.stop()

st.caption(f"Source: {source_label}")
st.success(f"Detected {len(long_df):,} tenancy rows across {long_df['sheet'].nunique()} sheet(s).")

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
left, right = st.columns([2,3])
with left:
    regions = ["All"] + sorted([r for r in long_df["region"].dropna().unique().tolist()])
    landlord_types = ["All"] + sorted([t for t in long_df["landlord_type"].dropna().unique().tolist()])
    needs = ["All"] + sorted([n for n in long_df["need"].dropna().unique().tolist()])
    rent_types = ["All"] + sorted([r for r in long_df["rent_type"].dropna().unique().tolist()])

    region = st.selectbox("Region", regions, index=0)
    landlord = st.selectbox("Landlord type", landlord_types, index=0)
    need = st.selectbox("Needs type", needs, index=0)
    rent = st.selectbox("Rent type", rent_types, index=0)

# Apply filters
d = long_df.copy()
if region != "All":   d = d[d["region"] == region]
if landlord != "All": d = d[d["landlord_type"] == landlord]
if need != "All":     d = d[d["need"] == need]
if rent != "All":     d = d[d["rent_type"] == rent]

if d.empty:
    st.warning("Your current filter combination returns no rows. Try widening filters.")
    st.stop()

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
def nsum(x): 
    return int(np.nansum(pd.to_numeric(x, errors="coerce")))

K_total = nsum(d["tenancies"])
K_by_need = d.groupby("need", dropna=False)["tenancies"].sum().sort_values(ascending=False)
K_by_rent = d.groupby("rent_type", dropna=False)["tenancies"].sum().sort_values(ascending=False)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total tenancies", f"{K_total:,}")
k2.metric("Top need", f"{K_by_need.index[0] if len(K_by_need)>0 else '—'}: {int(K_by_need.iloc[0]) if len(K_by_need)>0 else 0:,}")
k3.metric("Top rent type", f"{K_by_rent.index[0] if len(K_by_rent)>0 else '—'}: {int(K_by_rent.iloc[0]) if len(K_by_rent)>0 else 0:,}")
k4.metric("Sheets parsed", f"{d['sheet'].nunique()}")

st.divider()

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
# 1) Stacked breakdown: choose stack dimension
st.subheader("Breakdown by dimension")
stack_dim = st.selectbox("Stacked by", ["need","rent_type","landlord_type"], index=0, format_func=lambda s: s.replace("_"," ").title())

gcols = ["region", "landlord_type", "need", "rent_type"]
groupers = [c for c in gcols if c in d.columns]
agg = (d.groupby(groupers, dropna=False)["tenancies"].sum().reset_index())

# Primary category for x-axis: region if available, else landlord_type
if "region" in agg.columns and agg["region"].notna().any():
    x_axis = "region"
else:
    x_axis = "landlord_type" if "landlord_type" in agg.columns else stack_dim

fig_stack = px.bar(
    agg, x=x_axis, y="tenancies", color=stack_dim,
    labels={"tenancies":"Tenancies", x_axis:x_axis.replace("_"," ").title(), stack_dim:stack_dim.replace("_"," ").title()},
    barmode="stack"
)
fig_stack.update_xaxes(tickangle=30)
st.plotly_chart(fig_stack, use_container_width=True)

# 2) Benchmark: selected region/type vs national (or selected cohort vs total)
st.subheader("Benchmark vs overall")
bench_all = long_df.copy()
bench_overall = nsum(bench_all["tenancies"])
bench_by_region = bench_all.groupby("region", dropna=False)["tenancies"].sum().reset_index().sort_values("tenancies", ascending=False)
bench_by_type = bench_all.groupby("landlord_type", dropna=False)["tenancies"].sum().reset_index().sort_values("tenancies", ascending=False)

colA, colB = st.columns(2)
with colA:
    fig_bench_region = px.bar(
        bench_by_region, x="region", y="tenancies",
        labels={"tenancies":"Tenancies","region":"Region"},
        title="All England — Tenancies by region"
    )
    fig_bench_region.update_xaxes(tickangle=30)
    st.plotly_chart(fig_bench_region, use_container_width=True)

with colB:
    fig_bench_type = px.bar(
        bench_by_type, x="landlord_type", y="tenancies",
        labels={"tenancies":"Tenancies","landlord_type":"Landlord type"},
        title="All England — Tenancies by landlord type"
    )
    st.plotly_chart(fig_bench_type, use_container_width=True)

# 3) Geographic bubble map (region centroids)
st.subheader("Geographic distribution (bubble map)")

# England region centroids (approx)
REGION_CENTROIDS = {
    "North East": (54.8, -1.6),
    "North West": (53.8, -2.6),
    "Yorkshire and The Humber": (53.9, -1.2),
    "East Midlands": (52.9, -1.0),
    "West Midlands": (52.5, -2.0),
    "East of England": (52.2, 0.3),
    "London": (51.5, -0.1),
    "South East": (51.2, -0.9),
    "South West": (50.9, -3.7),
}

geo = (d.groupby("region", dropna=False)["tenancies"].sum()
         .reset_index().dropna(subset=["region"]))
geo["lat"] = geo["region"].map(lambda r: REGION_CENTROIDS.get(r, (np.nan, np.nan))[0])
geo["lon"] = geo["region"].map(lambda r: REGION_CENTROIDS.get(r, (np.nan, np.nan))[1])
geo = geo.dropna(subset=["lat","lon"])

if geo.empty:
    st.info("Map needs region-level rows. Select a filter set that includes regional data.")
else:
    fig_map = px.scatter_geo(
        geo, lat="lat", lon="lon", size="tenancies", hover_name="region",
        projection="natural earth",
        scope="europe",
        size_max=40,
        labels={"tenancies":"Tenancies"}
    )
    fig_map.update_geos(
        center=dict(lat=53.3, lon=-1.5),
        fitbounds="locations",
        showcountries=True,
        lataxis_range=[49.5, 56.5],
        lonaxis_range=[-6.5, 2.0]
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Tiny insights + download
# -----------------------------------------------------------------------------
st.markdown("### 🔎 Insights (filtered)")
s = pd.to_numeric(d["tenancies"], errors="coerce").dropna()
if len(s) == 0:
    st.write("No numeric tenancies in this view.")
else:
    total = int(s.sum())
    regions_in_view = d["region"].dropna().nunique()
    types_in_view = d["landlord_type"].dropna().nunique()
    needs_in_view = d["need"].dropna().nunique()
    rents_in_view = d["rent_type"].dropna().nunique()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total tenancies (filtered)", f"{total:,}")
    c2.metric("Regions in view", f"{regions_in_view}")
    c3.metric("Landlord types", f"{types_in_view}")
    c4.metric("Needs/Rents", f"{needs_in_view}/{rents_in_view}")

st.download_button(
    label="📥 Download current view (CSV)",
    data=d.to_csv(index=False).encode("utf-8"),
    file_name="lettings_view_2023_24.csv",
    mime="text/csv",
)

with st.expander("🔧 Inspect raw sheets"):
    st.caption("If something looks off, open the sheets and check headers the heuristic matched.")
    for name, df in raw_sheets.items():
        st.write(f"**{name}**")
        st.dataframe(df.head(25), use_container_width=True)
