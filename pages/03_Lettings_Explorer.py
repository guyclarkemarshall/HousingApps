
import io, re, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Lettings Explorer (Social Housing 2023/24) — Fixed", page_icon="🏘️", layout="wide")
st.title("🏘️ Social Housing Lettings in England — Tenancies (2023/24)")

DEFAULT_ODS_URL = "https://assets.publishing.service.gov.uk/media/67505f25bcd3a46a2248c878/Social_housing_lettings_in_England_tenancies_summary_tables_April_2023_to_March_2024.ods"
st.caption("Data: DLUHC — Social housing lettings in England, tenancies summary tables (Apr 2023–Mar 2024)")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm(s): return re.sub(r"\s+", " ", str(s)).strip().lower()

def possible_numeric(s):
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=True, ttl=60*10)
def fetch_bytes(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=True, ttl=60*10)
def read_all_sheets(ods_bytes: bytes):
    # engine='odf' needs odfpy
    xls = pd.read_excel(io.BytesIO(ods_bytes), sheet_name=None, engine="odf")
    out = {}
    for name, df in xls.items():
        if df is None:
            continue
        df.columns = [str(c) for c in df.columns]
        df = df.dropna(how="all")
        df = df.loc[:, ~df.columns.to_series().apply(lambda c: df[c].isna().all())]
        df = df.loc[:, ~df.columns.duplicated()]
        out[name] = df
    return out

# -----------------------------------------------------------------------------
# PRESET MANUAL MAPPINGS (best guess for this specific ODS)
# -----------------------------------------------------------------------------
PRESET_PATTERNS = [
    {
        "sheet_regex": r"(region).*|(gor)|(^table.*region)|(itl)",
        "maps": {
            "region": ["Region", "Government Office Region", "GOR", "ITL1", "Area"],
            "landlord_type": ["Landlord type", "Provider type", "Organisation type", "Landlord"],
            "need": ["Need", "Housing need", "General needs / Supported", "Needs type"],
            "rent_type": ["Rent type", "Tenure type", "Type of rent", "Social rent", "Affordable rent", "Intermediate"],
            "tenancies": ["Tenancies", "Lettings", "Number of lettings", "Count", "Number"],
        },
        "wide_mode": False
    },
    {
        "sheet_regex": r"(summary|overview|totals|all lettings)",
        "maps": {
            "region": ["Region", "Area", "GOR", "ITL1"],
            "landlord_type": ["Landlord type", "Provider type"],
            "need": ["Need", "Needs type"],
            "rent_type": ["Rent type", "Tenure type"],
            "tenancies": ["Lettings", "Tenancies", "Number of lettings", "Count", "Number"],
        },
        "wide_mode": True
    },
]

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        for pat in candidates:
            if pat.lower() in c.lower():
                return c
    return None

def apply_preset_mapping(df, preset):
    maps = preset["maps"]
    region_col   = first_existing(df, maps.get("region", []))
    landlord_col = first_existing(df, maps.get("landlord_type", []))
    need_col     = first_existing(df, maps.get("need", []))
    rent_col     = first_existing(df, maps.get("rent_type", []))

    ten_candidates = []
    for cand in maps.get("tenancies", []):
        col = first_existing(df, [cand])
        if col and col not in ten_candidates:
            ten_candidates.append(col)
    if preset.get("wide_mode", False):
        dimset = {c for c in [region_col, landlord_col, need_col, rent_col] if c}
        for col in df.columns:
            if col in dimset:
                continue
            if possible_numeric(df[col]):
                ten_candidates.append(col)
    ten_candidates = [c for c in dict.fromkeys(ten_candidates) if c in df.columns]
    if not ten_candidates:
        return None

    sub_cols = [c for c in [region_col, landlord_col, need_col, rent_col] if c] + ten_candidates
    sub = df[sub_cols].copy()
    colmap = {}
    if region_col:   colmap[region_col] = "region"
    if landlord_col: colmap[landlord_col] = "landlord_type"
    if need_col:     colmap[need_col] = "need"
    if rent_col:     colmap[rent_col] = "rent_type"
    sub = sub.rename(columns=colmap)

    ten_df = sub[ten_candidates].apply(pd.to_numeric, errors="coerce")
    sub["tenancies"] = ten_df.sum(axis=1, min_count=1)

    for c in ["region","landlord_type","need","rent_type"]:
        if c in sub.columns:
            sub[c] = sub[c].astype(str).str.strip()
            sub = sub[~sub[c].str.contains(r"\btotal\b", case=False, na=False)]
    sub = sub.dropna(subset=["tenancies"])
    sub = sub[sub["tenancies"] > 0]
    for c in ["region","landlord_type","need","rent_type"]:
        if c not in sub.columns:
            sub[c] = np.nan
    return sub[["region","landlord_type","need","rent_type","tenancies"]]

def try_presets(raw_sheets):
    for name, df in raw_sheets.items():
        for preset in PRESET_PATTERNS:
            if re.search(preset["sheet_regex"], name, flags=re.IGNORECASE):
                out = apply_preset_mapping(df, preset)
                if out is not None and not out.empty and out["tenancies"].sum() > 0:
                    out = out.copy()
                    out["sheet"] = name
                    return out
    for name, df in raw_sheets.items():
        for preset in PRESET_PATTERNS:
            out = apply_preset_mapping(df, preset)
            if out is not None and not out.empty and out["tenancies"].sum() > 0:
                out = out.copy()
                out["sheet"] = name
                return out
    return pd.DataFrame()

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

raw_sheets = read_all_sheets(ods_bytes)
preset_long = try_presets(raw_sheets)

manual_long = pd.DataFrame()
if preset_long.empty:
    with st.expander("🛠️ Manual Mapping Mode (preset failed — map the columns)", expanded=True):
        st.caption("Pick a sheet and map the columns. Only 'Tenancies' (count) is mandatory; other dimensions optional.")
        sheet_names = list(raw_sheets.keys())
        mm_sheet = st.selectbox("Sheet", sheet_names, index=0 if sheet_names else None)
        if mm_sheet:
            dfm = raw_sheets[mm_sheet].copy()
            st.write("Preview of selected sheet (first 25 rows):")
            st.dataframe(dfm.head(25), use_container_width=True)

            cols = dfm.columns.tolist()
            num_like = [c for c in cols if possible_numeric(dfm[c])]
            count_like = [c for c in cols if any(k in _norm(c) for k in ["tenanc","lett","count","number"]) and c in num_like]
            ten_col = st.selectbox("Tenancies column (required)", count_like if count_like else num_like, index=0 if (count_like or num_like) else None)

            region_col   = st.selectbox("Region column (optional)",   ["<none>"] + cols, index=0)
            landlord_col = st.selectbox("Landlord type column (optional)", ["<none>"] + cols, index=0)
            need_col     = st.selectbox("Needs type column (optional)",     ["<none>"] + cols, index=0)
            rent_col     = st.selectbox("Rent type column (optional)",      ["<none>"] + cols, index=0)

            wide_mode = st.checkbox("Sheet is wide (sum all numeric non-dimension columns)", value=False)

            if ten_col:
                sub = dfm.copy()
                colmap = {}
                if region_col and region_col != "<none>":   colmap[region_col] = "region"
                if landlord_col and landlord_col != "<none>": colmap[landlord_col] = "landlord_type"
                if need_col and need_col != "<none>":       colmap[need_col] = "need"
                if rent_col and rent_col != "<none>":       colmap[rent_col] = "rent_type"
                colmap[ten_col] = "tenancies"
                sub = sub.rename(columns=colmap)

                dims_present = [c for c in ["region","landlord_type","need","rent_type"] if c in sub.columns]

                if wide_mode:
                    numeric_cols = [c for c in sub.columns if c not in dims_present and possible_numeric(sub[c])]
                    if numeric_cols:
                        sub["tenancies"] = sub[numeric_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

                sub["tenancies"] = pd.to_numeric(sub.get("tenancies", np.nan), errors="coerce")
                for c in dims_present:
                    sub[c] = sub[c].astype(str).str.strip()
                    sub = sub[~sub[c].str.contains(r"\btotal\b", case=False, na=False)]
                sub = sub.dropna(subset=["tenancies")
                sub = sub[sub["tenancies"] > 0]
                sub["sheet"] = mm_sheet
                for c in ["region","landlord_type","need","rent_type"]:
                    if c not in sub.columns:
                        sub[c] = np.nan
                manual_long = sub[["region","landlord_type","need","rent_type","tenancies","sheet"]]
                st.success(f"Built tidy table from sheet '{mm_sheet}' with {len(manual_long):,} rows.")

if not preset_long.empty:
    long_df = preset_long
    st.caption("Mode: Preset mapping ✅")
elif not manual_long.empty:
    long_df = manual_long
    st.caption("Mode: Manual mapping ✅")
else:
    st.error("No table detected and no manual mapping provided. Please use the manual mapping expander.")
    with st.expander("🔧 Inspect raw sheets"):
        for name, df in raw_sheets.items():
            st.write(f"**{name}** — columns: {list(df.columns)}")
            st.dataframe(df.head(15), use_container_width=True)
    st.stop()

st.success(f"Using {len(long_df):,} tenancy rows from sheet: {long_df['sheet'].iloc[0]}")

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

d = long_df.copy()
if region != "All":   d = d[d["region"] == region]
if landlord != "All": d = d[d["landlord_type"] == landlord]
if need != "All":     d = d[d["need"] == need]
if rent != "All":     d = d[d["rent_type"] == rent]

if d.empty:
    st.warning("Your current filter combination returns no rows. Try widening filters or adjust manual mapping.")
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
k4.metric("Sheet", d["sheet"].iloc[0])

st.divider()

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
st.subheader("Breakdown by dimension")
stack_dim = st.selectbox("Stacked by", ["need","rent_type","landlord_type"], index=0, format_func=lambda s: s.replace("_"," ").title())

gcols = ["region", "landlord_type", "need", "rent_type"]
groupers = [c for c in gcols if c in d.columns]
agg = (d.groupby(groupers, dropna=False)["tenancies"].sum().reset_index())

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

st.subheader("Benchmark vs overall")
bench_all = long_df.copy()
bench_by_region = bench_all.groupby("region", dropna=False)["tenancies"].sum().reset_index().sort_values("tenancies", ascending=False)
bench_by_type = bench_all.groupby("landlord_type", dropna=False)["tenancies"].sum().reset_index().sort_values("tenancies", ascending=False)

colA, colB = st.columns(2)
with colA:
    if bench_by_region["region"].notna().any():
        fig_bench_region = px.bar(
            bench_by_region.dropna(subset=["region"]), x="region", y="tenancies",
            labels={"tenancies":"Tenancies","region":"Region"},
            title="All England — Tenancies by region"
        )
        fig_bench_region.update_xaxes(tickangle=30)
        st.plotly_chart(fig_bench_region, use_container_width=True)
    else:
        st.info("No region dimension available in the selected sheet.")

with colB:
    if bench_by_type["landlord_type"].notna().any():
        fig_bench_type = px.bar(
            bench_by_type.dropna(subset=["landlord_type"]), x="landlord_type", y="tenancies",
            labels={"tenancies":"Tenancies","landlord_type":"Landlord type"},
            title="All England — Tenancies by landlord type"
        )
        st.plotly_chart(fig_bench_type, use_container_width=True)
    else:
        st.info("No landlord type dimension available in the selected sheet.")

st.subheader("Geographic distribution (bubble map)")

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
    st.info("Map needs region-level rows. If your selected sheet doesn't include regions, try another sheet or use manual mapping.")
else:
    fig_map = px.scatter_geo(
        geo, lat="lat", lon="lon", size="tenancies", hover_name="region",
        projection="natural earth", scope="europe", size_max=40,
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

with st.expander("🔎 Debug: inspect raw sheets & columns"):
    for name, df in raw_sheets.items():
        st.write(f"**{name}** — columns: {list(df.columns)}")
        st.dataframe(df.head(15), use_container_width=True)
