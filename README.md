# HousingApps
Various Analytical and Data tools for Social Housing

---

# 🏠 TSM 2024 Dashboard – Streamlit

An interactive dashboard for exploring the UK Regulator of Social Housing’s **Tenant Satisfaction Measures (TSM) 2024** dataset.

Built in **Streamlit** with **Plotly** for visualisation.  
Defaults to the official TSM Excel file hosted by GOV.UK but supports uploading your own or pulling from a custom URL.

## ✨ Features
- **Default public dataset**: Auto-loads from  
  `https://assets.publishing.service.gov.uk/media/6744943be26d6f8ca3cb35c0/2024_TSM_Full_Data_v1.1_FINAL.xlsx`
- **Upload option**: Replace with your own `.xlsx` / `.xls` file.
- **Custom URL fetch**: Point to any publicly available Excel file.
- **Interactive filters**: Tenure, Region, Metric (TP01–TP12), Group size, Sample size.
- **Charts**:
  - Leaderboard (Top 15 landlords)
  - Score vs Size scatter (with trendline)
  - Distribution (Box Plot) split by Region or Landlord type
- **Insights panel**: KPIs, top/bottom performers, percentiles, outlier counts.
- **Download current view**: Export filtered data as CSV.

## 📂 Project Structure
```
app.py                   # Landing page
requirements.txt         # Python dependencies
utils/
    io_excel.py          # Excel parsing & tidying
    charts.py            # Shared Plotly chart builders
pages/
    01_TSM_Dashboard.py  # Main dashboard page
    99_About.py          # About / credits page (optional)
```

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/tsm-dashboard.git
cd tsm-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit
```bash
streamlit run app.py
```

### 4. Open in browser
Streamlit will show a local URL (e.g. `http://localhost:8501`).  
Click to open the dashboard.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Sign in with GitHub, click **New app**.
4. Choose your repo, branch, and set `app.py` as the main file.
5. Deploy. Your app will be live at  
   `https://<username>-<repo-name>.streamlit.app`
6. **Optional**: In the app settings, add a **custom domain** (e.g., `dashboards.hearthsong.ai`) and configure your DNS (CNAME record).

---

## 🧩 Adding More Dashboards
1. Create a new file in `pages/` (e.g. `02_Other_Dashboard.py`).
2. Use the same pattern: data loader → filters → charts → insights panel.
3. Push to GitHub. Community Cloud redeploys automatically.

---

## 🛠 Dependencies
See [`requirements.txt`](requirements.txt):
- streamlit
- pandas
- numpy
- plotly
- openpyxl
- requests

---

## 📄 Data Source
Official dataset: [Tenant Satisfaction Measures: landlord-level data 2024](https://www.gov.uk/government/publications/tenant-satisfaction-measures-tsm-landlord-level-data-2024)

---

## 📜 License
CC0 1.0 Universal. See `LICENSE` for details.

---

## 👤 Author
Built by Guy Marshall with GPT-5 assistance.
