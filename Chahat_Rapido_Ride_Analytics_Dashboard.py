"""
Rapido Ride Analytics  ·  Chahat_Rapido_Ride_Analytics_Dashboard.py
==================================
Data Analytics project.

Sections
--------
  1. Imports
  2. Page Config & Global CSS
  3. Data Loading  (@st.cache_data)
  4. Data Cleaning
  5. Feature Engineering
  6. KPI Functions
  7. Visualization Functions
  8. UI Components  (cards, headers, insight/risk widgets)
  9. Dashboard Pages  (Page 1 · Page 2 · Page 3)
 10. Main

Run:
    streamlit run Chahat_Rapido_Ride_Analytics_Dashboard.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Rapido Ride Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* page background */
    .main, [data-testid="stAppViewContainer"] { background: #f8f9fc; }

    /* section divider */
    .section-divider {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 28px 0 20px;
    }

    /* section heading with left accent bar */
    .section-heading {
        font-size: 17px; font-weight: 700; color: #1f2328;
        margin: 0 0 14px; padding-left: 10px;
        border-left: 4px solid #3b82d4;
    }

    /* KPI card */
    .kpi-wrap  { border-radius: 12px; padding: 18px 16px; text-align: center;
                 color: #fff; margin-bottom: 4px; }
    .kpi-label { font-size: 12px; opacity: .85; margin-bottom: 6px;
                 letter-spacing: .4px; text-transform: uppercase; }
    .kpi-value { font-size: 28px; font-weight: 800; line-height: 1; }
    .kpi-delta { font-size: 12px; opacity: .8; margin-top: 6px; }

    /* insight card */
    .insight-card {
        background: #fff; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
    }
    .insight-bullet { font-size: 18px; font-weight: 700; margin-right: 8px; }
    .insight-text   { font-size: 14px; color: #1f2328; }

    /* risk / opportunity / action board cards */
    .risk-card {
        border-radius: 10px; padding: 16px 18px;
        margin-bottom: 10px; border-left: 5px solid;
    }
    .risk-card.red   { background: #fff5f5; border-color: #ef4444; }
    .risk-card.green { background: #f0fdf4; border-color: #22c55e; }
    .risk-card.blue  { background: #eff6ff; border-color: #3b82d4; }
    .risk-card h4    { margin: 0 0 6px; font-size: 14px; font-weight: 700; }
    .risk-card p     { margin: 0; font-size: 13px; color: #374151; line-height: 1.55; }

    /* footer */
    .dash-footer {
        margin-top: 40px; padding-top: 16px;
        border-top: 1px solid #e5e7eb;
        text-align: center; font-size: 12px; color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
# Dataset is always at data/rides_data.csv, relative to the project root.
# When running `streamlit run Chahat_Rapido_Ride_Analytics_Dashboard.py` from the project root this resolves correctly.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rides_data.csv")


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    """Read raw CSV and return a DataFrame with cleaned types."""
    df = pd.read_csv(path)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Coerce numeric columns to float  (handles 'nan' strings)
    • Parse date column
    • Fill missing payment_method for cancelled rides with 'N/A'
    • Remove duplicate ride IDs (keep first occurrence)
    """
    df = df.copy()

    # Numeric columns
    for col in ["ride_charge", "misc_charge", "total_fare", "distance", "duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Time → extract hour
    df["time_parsed"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce")
    df["hour"] = df["time_parsed"].dt.hour

    # Fill missing payment (cancelled rides have no payment)
    df["payment_method"] = df["payment_method"].fillna("N/A")

    # Drop duplicate rides
    df = df.drop_duplicates(subset="ride_id", keep="first")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns needed for analysis:
      • year, month, month_name, day_of_week
      • time_of_day  (Morning / Afternoon / Evening / Night)
      • revenue      (total_fare for completed rides, 0 for cancelled)
      • source_area  (first two words of source — for area-level grouping)
    """
    df = df.copy()

    # Date decomposition
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%b")
    df["day_of_week"] = df["date"].dt.day_name()

    # Time-of-day segment
    def _time_of_day(h):
        if pd.isna(h):    return "Unknown"
        if 5 <= h < 12:   return "Morning"
        if 12 <= h < 17:  return "Afternoon"
        if 17 <= h < 21:  return "Evening"
        return "Night"

    df["time_of_day"] = df["hour"].apply(_time_of_day)

    # Revenue column (zero for non-completed rides)
    df["revenue"] = np.where(df["ride_status"] == "completed", df["total_fare"], 0.0)

    # Source area: first two words of the pickup location string
    df["source_area"] = df["source"].str.split().str[:2].str.join(" ")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. KPI FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute all business KPIs from the (already filtered) DataFrame.
    Returns a dict of scalar values ready for display.
    """
    completed = df[df["ride_status"] == "completed"]
    cancelled = df[df["ride_status"] == "cancelled"]
    total = len(df)

    total_revenue   = completed["total_fare"].sum()
    avg_fare        = completed["total_fare"].mean()  if len(completed) else 0.0
    median_fare     = completed["total_fare"].median() if len(completed) else 0.0
    cancel_rate     = len(cancelled) / total * 100 if total else 0.0
    completion_rate = 100.0 - cancel_rate
    total_distance  = completed["distance"].sum()
    avg_distance    = completed["distance"].mean()    if len(completed) else 0.0
    avg_duration    = completed["duration"].mean()    if len(completed) else 0.0

    # Peak hour (hour with most rides)
    hourly = df.groupby("hour").size()
    peak_hour = int(hourly.idxmax()) if len(hourly) else 0

    # Top payment method
    top_payment = (
        completed["payment_method"].mode().iloc[0]
        if len(completed) and len(completed["payment_method"].dropna())
        else "N/A"
    )

    return {
        "total_rides":      total,
        "completed_rides":  len(completed),
        "cancelled_rides":  len(cancelled),
        "total_revenue":    total_revenue,
        "avg_fare":         avg_fare,
        "median_fare":      median_fare,
        "cancel_rate":      cancel_rate,
        "completion_rate":  completion_rate,
        "total_distance":   total_distance,
        "avg_distance":     avg_distance,
        "avg_duration":     avg_duration,
        "peak_hour":        peak_hour,
        "top_payment":      top_payment,
    }


def generate_insights(df: pd.DataFrame, completed: pd.DataFrame, cancelled: pd.DataFrame) -> list:
    """
    Produce 6 data-driven insight strings from the filtered dataset.
    All numbers are computed live — insights update when filters change.
    """
    insights = []
    total = len(df)
    if total == 0:
        return ["No data available for the current filters."]

    # 1. Peak revenue month
    if len(completed):
        monthly = (
            completed.groupby(["year", "month", "month_name"])["total_fare"]
            .sum().reset_index().sort_values("total_fare", ascending=False)
        )
        top = monthly.iloc[0]
        insights.append(
            f"📈 Revenue peaked in <b>{top['month_name']} {int(top['year'])}</b> "
            f"(₹{top['total_fare']/1e6:.2f}M), indicating the highest demand period."
        )

    # 2. Top service type
    top_svc = df["services"].value_counts()
    if len(top_svc):
        insights.append(
            f"🏆 <b>{top_svc.index[0].title()}</b> is the most popular service at "
            f"<b>{top_svc.iloc[0] / total * 100:.1f}%</b> of all rides."
        )

    # 3. Cancellation rate
    cancel_rate = len(cancelled) / total * 100 if total else 0
    level = "critically high" if cancel_rate > 15 else ("high" if cancel_rate > 8 else "moderate")
    insights.append(
        f"⚠️ Cancellation rate is <b>{cancel_rate:.1f}%</b> — "
        f"considered <b>{level}</b> and a key business concern."
    )

    # 4. Peak hour
    hourly = df.groupby("hour").size()
    if len(hourly):
        ph = int(hourly.idxmax())
        insights.append(
            f"⏰ Demand peaks at <b>{ph}:00</b> with <b>{int(hourly.max()):,}</b> rides. "
            f"Surge pricing at this hour can boost revenue by 15–25%."
        )

    # 5. Top payment method
    if len(completed):
        pay = completed["payment_method"].value_counts()
        if len(pay):
            insights.append(
                f"💳 <b>{pay.index[0]}</b> dominates payments at "
                f"<b>{pay.iloc[0] / len(completed) * 100:.1f}%</b> of completed rides."
            )

    # 6. Fare skew
    if len(completed):
        avg_f  = completed["total_fare"].mean()
        med_f  = completed["total_fare"].median()
        skew   = "right-skewed (premium rides pulling average up)" if avg_f > med_f * 1.05 else "well-distributed"
        insights.append(
            f"💰 Average fare ₹{avg_f:.0f} vs Median ₹{med_f:.0f}. "
            f"Distribution is <b>{skew}</b>."
        )

    return insights[:6]


def generate_risk_board(df: pd.DataFrame, cancelled: pd.DataFrame) -> tuple:
    """
    Return three lists of (css_class, icon, title, body) tuples
    for the Risk, Opportunity, and Action tabs.
    """
    total       = len(df)
    cancel_rate = len(cancelled) / total * 100 if total else 0
    top_svc     = df["services"].value_counts().index[0].title() if len(df) else "N/A"
    hourly      = df.groupby("hour").size()
    ph          = int(hourly.idxmax()) if len(hourly) else 0

    risks = [
        ("red", "🔴", "Cancellation Risk",
         f"Current cancellation rate of {cancel_rate:.1f}% directly reduces revenue and damages "
         "driver trust. Persistent cancellations lead to customer churn."),
        ("red", "🔴", "Peak-Hour Supply Gap",
         f"Demand spikes at {ph}:00 but driver supply does not match — causing unavailability, "
         "poor customer experience, and lost revenue."),
        ("red", "🔴", "Low-Revenue Zone Underperformance",
         "Several source areas consistently generate near-zero revenue, representing "
         "wasted driver idle time and a missed monetisation opportunity."),
    ]
    opportunities = [
        ("green", "🟢", f"Scale {top_svc} Service",
         f"{top_svc} leads ride volume. First-ride discounts and referral bonuses "
         "can grow this segment further."),
        ("green", "🟢", "Surge Pricing at Peak Hours",
         f"Dynamic pricing at {ph}:00 and adjacent hours can lift revenue by 15–25% "
         "per ride without adding fleet cost."),
        ("green", "🟢", "Parcel Service B2B Expansion",
         "Parcel rides show lower cancellation rates and strong revenue/km. "
         "Onboarding 50+ local business clients creates a stable recurring revenue stream."),
    ]
    actions = [
        ("blue", "🔵", "Introduce Driver Cancellation Penalties",
         "Tiered penalty (warning → fine → deactivation) for unjustified cancellations. "
         "Pair with a peak-hour availability bonus to incentivise supply."),
        ("blue", "🔵", "Launch Digital Payment Cashback Campaign",
         "5% cashback on GPay and Amazon Pay for 90 days migrates cash users to digital, "
         "reducing fraud risk and reconciliation overhead."),
        ("blue", "🔵", "Geo-Targeted Promotions for Low-Ride Zones",
         "First-ride discounts and in-app nudges in the bottom 10 source areas. "
         "Incentivise drivers to reposition to these zones during off-peak hours."),
    ]
    return risks, opportunities, actions


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def chart_service_pie(df):
    svc = df["services"].value_counts().reset_index()
    svc.columns = ["Service", "Rides"]
    fig = px.pie(
        svc, names="Service", values="Rides",
        color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
        hole=0.42,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      pull=[0.04] * len(svc))
    fig.update_layout(showlegend=True, height=340,
                      margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def chart_ride_status_bar(df):
    status = df["ride_status"].value_counts().reset_index()
    status.columns = ["Status", "Count"]
    fig = px.bar(
        status, x="Status", y="Count",
        color="Status",
        color_discrete_map={"completed": "#22c55e", "cancelled": "#ef4444"},
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, height=340, yaxis_title="Rides",
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=10, b=10))
    return fig


def chart_monthly_revenue(completed):
    monthly = (
        completed.groupby(["year", "month", "month_name"])["total_fare"]
        .sum().reset_index().sort_values(["year", "month"])
    )
    monthly["period"]     = monthly["month_name"] + " " + monthly["year"].astype(str)
    monthly["growth_pct"] = monthly["total_fare"].pct_change() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["total_fare"],
        mode="lines+markers+text",
        line=dict(color="#3b82d4", width=3),
        marker=dict(size=9, color="#3b82d4", line=dict(width=2, color="white")),
        text=monthly["total_fare"].apply(lambda v: f"₹{v/1e6:.2f}M"),
        textposition="top center",
        fill="tozeroy", fillcolor="rgba(59,130,212,0.10)",
        name="Revenue",
        hovertemplate="%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["growth_pct"],
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(size=6, color="#f59e0b"),
        name="MoM Growth %",
        yaxis="y2",
        hovertemplate="%{x}<br>Growth: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Month",
        yaxis=dict(title="Revenue (₹)", tickformat="₹,.0f"),
        yaxis2=dict(title="MoM Growth %", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", y=1.08),
        height=380, margin=dict(t=20, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_revenue_by_service_stacked(completed):
    rev = (
        completed.groupby(["year", "month", "month_name", "services"])["total_fare"]
        .sum().reset_index().sort_values(["year", "month"])
    )
    rev["period"] = rev["month_name"] + " " + rev["year"].astype(str)
    fig = px.bar(
        rev, x="period", y="total_fare", color="services",
        barmode="stack",
        color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
        labels={"total_fare": "Revenue (₹)", "period": "Month", "services": "Service"},
    )
    fig.update_layout(
        yaxis_tickformat="₹,.0f",
        legend=dict(orientation="h", y=1.05),
        height=360, margin=dict(t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_peak_hours(df):
    hourly = df.groupby("hour").size().reset_index(name="Rides")
    peak_h = int(hourly.loc[hourly["Rides"].idxmax(), "hour"])
    hourly["color"] = hourly["hour"].apply(
        lambda h: "#ef4444" if h == peak_h else "#3b82d4"
    )
    fig = px.bar(
        hourly, x="hour", y="Rides",
        color="color", color_discrete_map="identity",
        labels={"hour": "Hour (0–23)", "Rides": "Rides"},
        text="Rides",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1),
        annotations=[dict(
            x=peak_h, y=hourly["Rides"].max() * 1.08,
            text=f"⚡ Peak: {peak_h}:00",
            showarrow=False, font=dict(size=12, color="#ef4444"),
        )],
        height=340, margin=dict(t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_time_of_day(df):
    tod_order = ["Morning", "Afternoon", "Evening", "Night"]
    tod = df["time_of_day"].value_counts().reindex(tod_order).reset_index()
    tod.columns = ["TimeOfDay", "Rides"]
    fig = px.bar(
        tod, x="TimeOfDay", y="Rides",
        color="TimeOfDay",
        color_discrete_sequence=["#f59e0b", "#22c55e", "#f97316", "#6366f1"],
        text="Rides",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, height=320,
                      margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chart_day_hour_heatmap(df):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    hmap = (
        df.groupby(["day_of_week", "hour"]).size()
        .unstack(fill_value=0).reindex(day_order)
    )
    fig = px.imshow(
        hmap,
        labels=dict(x="Hour", y="Day", color="Rides"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=320, margin=dict(t=10, b=10))
    return fig


def chart_revenue_by_service_bar(completed):
    rev = (
        completed.groupby("services")["total_fare"]
        .sum().sort_values(ascending=False).reset_index()
    )
    rev.columns = ["Service", "Revenue"]
    fig = px.bar(
        rev, x="Service", y="Revenue",
        color="Service",
        color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
        text=rev["Revenue"].apply(lambda v: f"₹{v/1e6:.2f}M"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, yaxis_tickformat="₹,.0f",
        height=320, margin=dict(t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_top_areas(df, n=10):
    top = df["source_area"].value_counts().head(n).reset_index()
    top.columns = ["Area", "Rides"]
    fig = px.bar(
        top.sort_values("Rides"), x="Rides", y="Area",
        orientation="h",
        color="Rides", color_continuous_scale="Blues",
        text="Rides",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False, height=320,
        margin=dict(t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_fare_histogram(completed):
    fig = px.histogram(
        completed.dropna(subset=["total_fare"]),
        x="total_fare", nbins=60,
        labels={"total_fare": "Total Fare (₹)"},
        color_discrete_sequence=["#3b82d4"],
    )
    mean_val = completed["total_fare"].mean() if len(completed) else 0
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                  annotation_text=f"Mean ₹{mean_val:.0f}",
                  annotation_position="top right")
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chart_fare_boxplot(completed):
    fig = px.box(
        completed.dropna(subset=["total_fare"]),
        x="services", y="total_fare",
        color="services",
        color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
        labels={"total_fare": "Total Fare (₹)", "services": "Service"},
    )
    fig.update_layout(showlegend=False, height=320,
                      margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chart_distance_vs_fare(completed):
    sample = completed.dropna(subset=["distance", "total_fare"])
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=42)
    fig = px.scatter(
        sample, x="distance", y="total_fare",
        color="services",
        color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
        opacity=0.45,
        labels={"distance": "Distance (km)", "total_fare": "Total Fare (₹)",
                "services": "Service"},
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#ef4444",
    )
    fig.update_layout(
        legend=dict(orientation="h", y=1.05),
        height=380, margin=dict(t=10, b=30),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_payment_pie(completed):
    pay = completed["payment_method"].value_counts().reset_index()
    pay.columns = ["Method", "Count"]
    fig = px.pie(
        pay, names="Method", values="Count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.38,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(legend=dict(orientation="h", y=-0.15),
                      height=340, margin=dict(t=10, b=30))
    return fig


def chart_payment_bar(completed):
    pay = completed["payment_method"].value_counts().reset_index()
    pay.columns = ["Method", "Count"]
    fig = px.bar(
        pay.sort_values("Count"), x="Count", y="Method",
        orientation="h",
        color="Count", color_continuous_scale="Teal",
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=340,
                      margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chart_cancel_trend(df):
    ct = (
        df.groupby(["year", "month", "month_name", "ride_status"])
        .size().reset_index(name="Count").sort_values(["year", "month"])
    )
    ct["period"] = ct["month_name"] + " " + ct["year"].astype(str)
    fig = px.bar(
        ct, x="period", y="Count", color="ride_status",
        barmode="group",
        color_discrete_map={"completed": "#22c55e", "cancelled": "#ef4444"},
        labels={"Count": "Rides", "period": "Month", "ride_status": "Status"},
    )
    fig.update_layout(
        legend=dict(orientation="h", y=1.05),
        height=340, margin=dict(t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_cancel_by_service(df):
    csvc = (
        df.groupby("services")
        .apply(
            lambda x: (x["ride_status"] == "cancelled").sum() / len(x) * 100,
            include_groups=False,
        )
        .reset_index(name="Cancel Rate (%)")
    )
    fig = px.bar(
        csvc, x="services", y="Cancel Rate (%)",
        color="services",
        color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82d4"],
        text=csvc["Cancel Rate (%)"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=300,
                      margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def chart_cancel_by_hour(cancelled):
    chr_df = cancelled.groupby("hour").size().reset_index(name="Cancellations")
    fig = px.bar(
        chr_df, x="hour", y="Cancellations",
        color="Cancellations", color_continuous_scale="Reds",
        labels={"hour": "Hour (0–23)"},
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickmode="linear", dtick=1),
        height=300, margin=dict(t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def chart_low_areas(df, n=15):
    low = (
        df.groupby("source_area")["revenue"]
        .sum().sort_values().head(n).reset_index()
    )
    low.columns = ["Area", "Revenue"]
    fig = px.bar(
        low, x="Revenue", y="Area",
        orientation="h",
        color="Revenue", color_continuous_scale="Reds_r",
        text=low["Revenue"].apply(lambda v: f"₹{v/1e3:.1f}K"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=380,
                      margin=dict(t=10, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. UI COMPONENTS  (reusable widgets)
# ══════════════════════════════════════════════════════════════════════════════
def kpi_card(col, label: str, value: str, color: str = "#3b82d4", delta: str = ""):
    col.markdown(
        f"<div class='kpi-wrap' style='background:{color};'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        + (f"<div class='kpi-delta'>{delta}</div>" if delta else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str):
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)


def show_insights(df, completed, cancelled):
    """Render the dynamic Key Insights section."""
    section_header("💡 Key Insights")
    bullets  = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
    insights = generate_insights(df, completed, cancelled)
    for i, text in enumerate(insights):
        st.markdown(
            f"<div class='insight-card'>"
            f"<span class='insight-bullet'>{bullets[i]}</span>"
            f"<span class='insight-text'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def show_risk_board(df, cancelled):
    """Render the tabbed Risk · Opportunity · Action board."""
    section_header("🗂️ Risk · Opportunity · Action Board")
    risks, opps, actions = generate_risk_board(df, cancelled)

    def _card(kind, icon, title, body):
        css = {"red": "red", "green": "green", "blue": "blue"}.get(kind, "blue")
        st.markdown(
            f"<div class='risk-card {css}'>"
            f"<h4>{icon} {title}</h4>"
            f"<p>{body}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["🔴 Risks", "🟢 Opportunities", "🔵 Actions"])
    with tab1:
        for item in risks:   _card(*item)
    with tab2:
        for item in opps:    _card(*item)
    with tab3:
        for item in actions: _card(*item)


# ══════════════════════════════════════════════════════════════════════════════
# 9. DASHBOARD PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_executive_overview(df, completed, cancelled, kpis, start_dt, end_dt):
    st.title("🚖 Rapido Ride Analytics — Executive Overview")
    st.caption(
        f"Showing **{len(df):,}** rides · "
        f"{start_dt.strftime('%d %b %Y')} → {end_dt.strftime('%d %b %Y')}"
    )

    # KPI cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "🚗 Total Rides",      f"{kpis['total_rides']:,}",                  "#3b82d4")
    kpi_card(c2, "💰 Total Revenue",    f"₹{kpis['total_revenue']/1e6:.2f}M",        "#22c55e")
    kpi_card(c3, "🎯 Avg Fare",         f"₹{kpis['avg_fare']:.0f}",                  "#f59e0b")
    kpi_card(c4, "✅ Completion Rate",  f"{kpis['completion_rate']:.1f}%",           "#10b981")
    kpi_card(c5, "❌ Cancel Rate",      f"{kpis['cancel_rate']:.1f}%",               "#ef4444")
    kpi_card(c6, "📏 Total Distance",   f"{kpis['total_distance']/1e3:.1f}K km",     "#8b5cf6")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Service pie  |  Status bar
    col1, col2 = st.columns(2)
    with col1:
        section_header("🛵 Rides by Service Type")
        st.plotly_chart(chart_service_pie(df), use_container_width=True)
    with col2:
        section_header("🔖 Ride Status Distribution")
        st.plotly_chart(chart_ride_status_bar(df), use_container_width=True)

    # Monthly revenue trend
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("📈 Monthly Revenue Trend (with MoM Growth %)")
    if len(completed):
        st.plotly_chart(chart_monthly_revenue(completed), use_container_width=True)
    else:
        st.info("No completed rides in the selected filters.")

    # Revenue stacked by service
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("💰 Revenue by Service Type — Monthly Breakdown")
    if len(completed):
        st.plotly_chart(chart_revenue_by_service_stacked(completed), use_container_width=True)

    # Key insights
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    show_insights(df, completed, cancelled)


def page_ride_revenue(df, completed, cancelled):
    st.title("📊 Ride & Revenue Analysis")
    st.caption(f"Filtered dataset: **{len(df):,}** rides")

    # Peak hours
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("⏰ Peak Hours Analysis")
    st.plotly_chart(chart_peak_hours(df), use_container_width=True)

    # Time of day  |  Heatmap
    col1, col2 = st.columns(2)
    with col1:
        section_header("🌅 Rides by Time of Day")
        st.plotly_chart(chart_time_of_day(df), use_container_width=True)
    with col2:
        section_header("📅 Demand Heatmap: Day × Hour")
        st.plotly_chart(chart_day_hour_heatmap(df), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Revenue by service  |  Top source areas
    col3, col4 = st.columns(2)
    with col3:
        section_header("💰 Revenue by Service Type")
        st.plotly_chart(chart_revenue_by_service_bar(completed), use_container_width=True)
    with col4:
        section_header("📍 Top 10 Source Areas by Rides")
        st.plotly_chart(chart_top_areas(df), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Fare histogram  |  Box plot
    col5, col6 = st.columns(2)
    with col5:
        section_header("💵 Fare Distribution")
        st.plotly_chart(chart_fare_histogram(completed), use_container_width=True)
    with col6:
        section_header("📦 Fare Box Plot by Service")
        st.plotly_chart(chart_fare_boxplot(completed), use_container_width=True)

    # Distance vs Fare scatter
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("📐 Distance vs Fare Analysis")
    st.plotly_chart(chart_distance_vs_fare(completed), use_container_width=True)

    # Payment methods
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("💳 Payment Method Distribution")
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(chart_payment_pie(completed), use_container_width=True)
    with col8:
        st.plotly_chart(chart_payment_bar(completed), use_container_width=True)

    # Key insights
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    show_insights(df, completed, cancelled)


def page_customer_risk(df, completed, cancelled, kpis):
    st.title("⚠️ Customer & Risk Analysis")
    st.caption(f"Filtered dataset: **{len(df):,}** rides")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "❌ Cancelled Rides",   f"{kpis['cancelled_rides']:,}",   "#ef4444")
    kpi_card(c2, "📉 Cancellation Rate", f"{kpis['cancel_rate']:.1f}%",   "#f97316")
    kpi_card(c3, "✅ Completion Rate",   f"{kpis['completion_rate']:.1f}%", "#22c55e")
    kpi_card(c4, "📊 Total Rides",       f"{kpis['total_rides']:,}",       "#3b82d4")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Cancellation trend
    section_header("📉 Cancellation Trend — Monthly")
    st.plotly_chart(chart_cancel_trend(df), use_container_width=True)

    # Cancel by service  |  Cancel by hour
    col1, col2 = st.columns(2)
    with col1:
        section_header("🚗 Cancellation Rate by Service")
        st.plotly_chart(chart_cancel_by_service(df), use_container_width=True)
    with col2:
        section_header("⏰ Cancellations by Hour of Day")
        st.plotly_chart(chart_cancel_by_hour(cancelled), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Low-performing areas
    section_header("🔻 Low-Performing Areas (Bottom 15 by Revenue)")
    st.plotly_chart(chart_low_areas(df), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Dynamic insights
    show_insights(df, completed, cancelled)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Risk · Opportunity · Action board
    show_risk_board(df, cancelled)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Raw data explorer
    section_header("🔍 Raw Data Explorer")
    with st.expander("Show filtered data table (max 500 rows)"):
        cols_show = [
            "ride_id", "date", "services", "ride_status",
            "source", "destination", "duration", "distance",
            "total_fare", "payment_method", "time_of_day",
        ]
        st.dataframe(df[cols_show].head(500), use_container_width=True, height=380)


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load, clean & engineer ────────────────────────────────────────────────
    raw_df    = load_data(DATA_PATH)
    cleaned   = clean_data(raw_df)
    df_full   = engineer_features(cleaned)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:12px 0 6px;'>"
            "<span style='font-size:36px;'>🚖</span><br>"
            "<span style='font-size:15px;font-weight:700;color:#1f2328;'>Rapido Analytics</span><br>"
            "<span style='font-size:11px;color:#6b7280;'>Ride Insights Dashboard</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:10px 0 16px;'>", unsafe_allow_html=True)
        st.markdown("#### 🎛️ Filters")

        date_min   = df_full["date"].min().date()
        date_max   = df_full["date"].max().date()
        date_range = st.date_input("📅 Date Range",
                                   value=(date_min, date_max),
                                   min_value=date_min,
                                   max_value=date_max)

        services_all = sorted(df_full["services"].dropna().unique())
        sel_services = st.multiselect("🚗 Service Type", services_all, default=services_all)

        top20_areas = df_full["source_area"].value_counts().head(20).index.tolist()
        sel_areas   = st.multiselect("📍 Source Area (top 20)", top20_areas, default=[])

        status_all = sorted(df_full["ride_status"].dropna().unique())
        sel_status = st.multiselect("🔖 Ride Status", status_all, default=status_all)

        st.markdown("<hr style='margin:16px 0 10px;'>", unsafe_allow_html=True)

        page = st.radio(
            "📄 Navigate to Page",
            [
                "🏠  Executive Overview",
                "📊  Ride & Revenue Analysis",
                "⚠️  Customer & Risk Analysis",
            ],
        )

        st.markdown(
            "<div style='padding-top:20px;font-size:11px;color:#9ca3af;text-align:center;'>"
            "Data Analytics Internship Project<br>Built with Streamlit + Plotly"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    start_dt = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else df_full["date"].min()
    end_dt   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df_full["date"].max()

    df = df_full[(df_full["date"] >= start_dt) & (df_full["date"] <= end_dt)].copy()
    if sel_services:
        df = df[df["services"].isin(sel_services)]
    if sel_areas:
        df = df[df["source_area"].isin(sel_areas)]
    if sel_status:
        df = df[df["ride_status"].isin(sel_status)]

    completed = df[df["ride_status"] == "completed"]
    cancelled = df[df["ride_status"] == "cancelled"]

    # ── Compute KPIs ──────────────────────────────────────────────────────────
    kpis = compute_kpis(df)

    # ── Render selected page ──────────────────────────────────────────────────
    if page == "🏠  Executive Overview":
        page_executive_overview(df, completed, cancelled, kpis, start_dt, end_dt)
    elif page == "📊  Ride & Revenue Analysis":
        page_ride_revenue(df, completed, cancelled)
    elif page == "⚠️  Customer & Risk Analysis":
        page_customer_risk(df, completed, cancelled, kpis)

    # ── Footer (all pages) ────────────────────────────────────────────────────
    st.markdown(
        "<div class='dash-footer'>"
        "🚖 Rapido Ride Analytics &nbsp;·&nbsp; "
        "Data Analytics Internship Project &nbsp;·&nbsp; "
        "Built with Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
