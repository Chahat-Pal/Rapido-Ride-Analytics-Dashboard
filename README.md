# 🚖 Rapido Ride Analytics — Data Analytics Internship Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)
![Plotly](https://img.shields.io/badge/Plotly-5.24-brightgreen)
![Pandas](https://img.shields.io/badge/Pandas-2.2-yellow)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📌 Project Overview

End-to-end **Data Analytics** project on **50,000 Rapido ride records** covering:

- ✅ Full data cleaning & feature engineering
- ✅ 12 static charts exported to `report_images/`
- ✅ 6 data-driven business insights (auto-generated from filtered data)
- ✅ Risk · Opportunity · Action framework
- ✅ 3-page interactive Streamlit dashboard with sidebar filters
- ✅ Professional Word project report
- ✅ Ready for internship submission

---

## 📂 Project Structure

```
Data Analytics Project/
│
├── data/
│   ├── rides_data.csv              ← Raw dataset (50,000 rides)
|   ├── rides_data_cleaned.csv      
│
├── notebook/
│   └── Chahat_RideAnalysis.ipynb   ← Full EDA + KPIs + Insights
│
│── Chahat_Rapido_Ride_Analytics_Dashboard.py   ← Main Streamlit dashboard (Executive Overview, Ride & Revenue Analysis, Customer & Risk Analysis)
│
├── report_images/                  ← 12 chart PNGs (auto-generated)
│   ├── 01_ride_status.png
│   ├── 02_service_distribution.png
│   ├── 03_revenue_by_service.png
│   ├── 04_monthly_revenue_trend.png
│   ├── 05_peak_hours.png
│   ├── 06_time_of_day.png
│   ├── 07_heatmap_day_hour.png
│   ├── 08_fare_distribution.png
│   ├── 09_distance_vs_fare.png
│   ├── 10_payment_methods.png
│   ├── 11_top_source_areas.png
│   └── 12_cancellation_by_service.png
│
├── Chahat_ProjectReport.docx     ← Full internship report
├── requirements.txt                ← All Python dependencies
└── README.md
```

---

## 📊 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `services` | String | Service type: `cab economy`, `auto`, `parcel` |
| `date` | Date | Ride date (YYYY-MM-DD) |
| `time` | Time | Ride start time (HH:MM:SS) |
| `ride_status` | String | `completed` or `cancelled` |
| `source` | String | Pickup location name |
| `destination` | String | Drop location name |
| `duration` | Integer | Ride duration (minutes) |
| `ride_id` | String | Unique ride identifier |
| `distance` | Float | Distance travelled (km) |
| `ride_charge` | Float | Base fare (₹) |
| `misc_charge` | Float | Miscellaneous charges (₹) |
| `total_fare` | Float | Total amount paid (₹) |
| `payment_method` | String | Payment mode (GPay, Paytm, Amazon Pay, QR scan) |

> **Dataset:** Rapido-style ride data · 50,000 records · Bengaluru area

---

## 📌 Key KPIs (Full Dataset)

| KPI | Value |
|-----|-------|
| Total Rides | 50,000 |
| Service Types | cab economy, auto, parcel |
| Date Range | June 2024 – August 2024 |
| Ride Statuses | completed, cancelled |
| Payment Methods | GPay, Paytm, Amazon Pay, QR scan, N/A |

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Static Visualisation | Matplotlib 3.9, Seaborn 0.13 |
| Interactive Charts | Plotly 5.24 |
| Dashboard | Streamlit 1.37 |
| Trend Lines | statsmodels 0.14 |
| Reporting | python-docx 1.1 |
| Notebook | Jupyter |

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Create a virtual environment first

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option A — Run the Jupyter Notebook (EDA + Charts)

```bash
cd notebook
jupyter notebook Chahat_RideAnalysis.ipynb
```

- Select **Kernel → Restart & Run All**
- All 12 charts are saved to `report_images/`
- Cleaned CSV is saved to `data/rides_data_cleaned.csv`

### Option B — Launch the Interactive Dashboard

```bash
streamlit run Chahat_Rapido_Ride_Analytics_Dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## 📋 Dashboard Pages

| Page | What's inside |
|------|--------------|
| **🏠 Page 1 – Executive Overview** | 6 KPI cards · Service pie chart · Status bar · Monthly revenue trend with MoM growth % · Revenue by service stacked bar · Dynamic Key Insights |
| **📊 Page 2 – Ride & Revenue Analysis** | Peak hours bar · Time-of-day bar · Day×Hour heatmap · Revenue by service · Top 10 source areas · Fare histogram · Fare box plot · Distance vs Fare scatter (with OLS trendline) · Payment methods pie + bar · Key Insights |
| **⚠️ Page 3 – Customer & Risk Analysis** | 4 KPI cards · Cancellation monthly trend · Cancel rate by service · Cancellations by hour · Low-revenue zones · 6 Dynamic Insights · Risk/Opportunity/Action tab board · Raw data explorer |

**Sidebar filters** (applied globally): Date Range · Service Type · Source Area · Ride Status

All charts and insights **update dynamically** based on the active filters.

---

## 💡 Business Insights (Risk · Opportunity · Action)

| Type | Topic | Summary |
|------|-------|---------|
| 🔴 Risk | Cancellation Risk | High cancel rate reduces revenue and trust |
| 🔴 Risk | Peak-Hour Supply Gap | Driver shortage at peak hours causes lost rides |
| 🔴 Risk | Low-Revenue Zones | Several source areas under-utilised |
| 🟢 Opportunity | Scale Top Service | Leading service can grow with promotions |
| 🟢 Opportunity | Surge Pricing | Dynamic pricing at peak hours → +15–25% revenue/ride |
| 🟢 Opportunity | Parcel B2B | Parcel rides show strong revenue/km potential |
| 🔵 Action | Cancel Penalty System | Tiered driver penalties reduce cancel rate |
| 🔵 Action | Digital Payment Cashback | 5% cashback campaign to migrate cash users |
| 🔵 Action | Geo-targeted Promos | First-ride discounts in low-ride zones |

---

## 👤 Author

**[Chahat Pal]**
Data Analytics Internship
Tools: Python · Pandas · Plotly · Streamlit · Matplotlib · Seaborn

---

*Built for the Data Analytics Internship Submission · 2026*
