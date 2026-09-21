"""
Rapido Ride Analytics Dashboard
================================
3-page interactive Streamlit app:
  Page 1 — Executive Overview      (KPIs, trends, service distribution)
  Page 2 — Ride & Revenue Analysis (peak hours, fare analysis, payments)
  Page 3 — Customer & Risk         (cancellations, insights, risk board)

Run:
    streamlit run dashboard/app.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rapido Ride Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── page background ── */
    .main { background: #f8f9fc; }
    [data-testid="stAppViewContainer"] { background: #f8f9fc; }

    /* ── section divider ── */
    .section-divider {
        border: none; border-top: 2px solid #e5e7eb;
        margin: 28px 0 20px;
    }

    /* ── section heading ── */
    .section-heading {
        font-size: 17px; font-weight: 700; color: #1f2328;
        margin: 0 0 14px; padding-left: 10px;
        border-left: 4px solid #3b82d4;
    }

    /* ── KPI card ── */
    .kpi-wrap { border-radius: 12px; padding: 18px 16px; text-align: center;
                color: #fff; margin-bottom: 4px; }
    .kpi-label { font-size: 12px; opacity: .85; margin-bottom: 6px; letter-spacing:.4px; text-transform:uppercase; }
    .kpi-value { font-size: 28px; font-weight: 800; line-height: 1; }
    .kpi-delta { font-size: 12px; opacity: .8; margin-top: 6px; }

    /* ── insight card ── */
    .insight-card {
        background: #fff; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
    }
    .insight-text { font-size: 14px; color: #1f2328; margin: 0; }
    .insight-bullet {
        font-size: 18px; font-weight: 700; margin-right: 8px; line-height: 1;
    }

    /* ── risk board card ── */
    .risk-card {
        border-radius: 10px; padding: 16px 18px; margin-bottom: 10px;
        border-left: 5px solid;
    }
    .risk-card.red   { background:#fff5f5; border-color:#ef4444; }
    .risk-card.green { background:#f0fdf4; border-color:#22c55e; }
    .risk-card.blue  { background:#eff6ff; border-color:#3b82d4; }
    .risk-card h4    { margin: 0 0 6px; font-size: 14px; font-weight: 700; }
    .risk-card p     { margin: 0; font-size: 13px; color: #374151; line-height: 1.55; }

    /* ── footer ── */
    .dash-footer {
        margin-top: 40px; padding-top: 16px;
        border-top: 1px solid #e5e7eb;
        text-align: center; font-size: 12px; color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rides_data.csv")


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Numeric coercion
    for col in ["ride_charge", "misc_charge", "total_fare", "distance", "duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates & times
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["time_parsed"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce")
    df["hour"] = df["time_parsed"].dt.hour

    # Payment fill
    df["payment_method"] = df["payment_method"].fillna("N/A")

    # Remove duplicate ride IDs
    df = df.drop_duplicates(subset="ride_id", keep="first")

    # Date components
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%b")
    df["day_of_week"] = df["date"].dt.day_name()

    # Time-of-day segment
    def _tod(h):
        if pd.isna(h):    return "Unknown"
        if 5 <= h < 12:   return "Morning"
        if 12 <= h < 17:  return "Afternoon"
        if 17 <= h < 21:  return "Evening"
        return "Night"

    df["time_of_day"] = df["hour"].apply(_tod)

    # Revenue (only completed rides)
    df["revenue"] = np.where(df["ride_status"] == "completed", df["total_fare"], 0.0)

    # Shortened source area (first 2 words)
    df["source_area"] = df["source"].str.split().str[:2].str.join(" ")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper components
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(col, label: str, value: str, delta: str = "", color: str = "#3b82d4"):
    delta_html = (
        f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    )
    col.markdown(
        f"<div class='kpi-wrap' style='background:{color};'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{delta_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def section_header(title: str):
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)


def insight_card(bullet: str, text: str):
    st.markdown(
        f"<div class='insight-card'>"
        f"<span class='insight-bullet'>{bullet}</span>"
        f"<span class='insight-text'>{text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def risk_board_card(kind: str, icon: str, title: str, body: str):
    css = {"red": "red", "green": "green", "blue": "blue"}.get(kind, "blue")
    st.markdown(
        f"<div class='risk-card {css}'>"
        f"<h4>{icon} {title}</h4>"
        f"<p>{body}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic insights engine  (all numbers come from the filtered df)
# ─────────────────────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, completed: pd.DataFrame, cancelled: pd.DataFrame) -> list[str]:
    insights = []
    total = len(df)
    if total == 0:
        return ["No data available for the current filters."]

    # 1. Peak revenue month
    if len(completed):
        monthly = (
            completed.groupby(["year", "month", "month_name"])["total_fare"]
            .sum()
            .reset_index()
            .sort_values("total_fare", ascending=False)
        )
        if len(monthly):
            top = monthly.iloc[0]
            insights.append(
                f"📈 Revenue peaked in <b>{top['month_name']} {int(top['year'])}</b> "
                f"(₹{top['total_fare']/1e6:.2f}M), indicating the highest demand period."
            )

    # 2. Top service type
    top_svc = df["services"].value_counts()
    if len(top_svc):
        best = top_svc.index[0]
        pct = top_svc.iloc[0] / total * 100
        insights.append(
            f"🏆 <b>{best.title()}</b> is the most popular service, contributing "
            f"<b>{pct:.1f}%</b> of all rides."
        )

    # 3. Cancellation rate
    cancel_rate = len(cancelled) / total * 100 if total else 0
    level = "critically high" if cancel_rate > 15 else ("high" if cancel_rate > 8 else "moderate")
    insights.append(
        f"⚠️ The overall cancellation rate is <b>{cancel_rate:.1f}%</b> — "
        f"considered <b>{level}</b> and a key business concern."
    )

    # 4. Peak hour
    hourly = df.groupby("hour").size()
    if len(hourly):
        ph = int(hourly.idxmax())
        insights.append(
            f"⏰ Ride demand peaks at <b>{ph}:00</b> with "
            f"<b>{int(hourly.max()):,}</b> rides — surge pricing recommended at this hour."
        )

    # 5. Top payment method
    if len(completed):
        pay = completed["payment_method"].value_counts()
        if len(pay):
            top_pay = pay.index[0]
            pay_pct = pay.iloc[0] / len(completed) * 100
            insights.append(
                f"💳 <b>{top_pay}</b> dominates as the preferred payment method "
                f"(<b>{pay_pct:.1f}%</b> of completed rides), showing strong digital adoption."
            )

    # 6. Avg fare insight
    if len(completed):
        avg_f = completed["total_fare"].mean()
        med_f = completed["total_fare"].median()
        skew = "right-skewed (premium rides pulling average up)" if avg_f > med_f * 1.05 else "well-distributed"
        insights.append(
            f"💰 Average fare is <b>₹{avg_f:.0f}</b> (median ₹{med_f:.0f}). "
            f"Fare distribution is <b>{skew}</b>."
        )

    return insights[:6]


def generate_risk_board(df: pd.DataFrame, completed: pd.DataFrame, cancelled: pd.DataFrame):
    total = len(df)
    cancel_rate = len(cancelled) / total * 100 if total else 0
    top_svc = df["services"].value_counts().index[0].title() if len(df) else "N/A"
    ph = int(df.groupby("hour").size().idxmax()) if len(df) and df["hour"].notna().any() else "N/A"

    risks = [
        ("red", "🔴", "Cancellation Risk",
         f"Current cancellation rate of {cancel_rate:.1f}% directly reduces revenue and damages "
         "driver trust. Persistent cancellations lead to customer churn and negative brand perception."),
        ("red", "🔴", "Peak-Hour Supply Gap",
         f"Demand spikes at {ph}:00 but driver supply does not scale accordingly. "
         "This creates ride unavailability, poor customer experience, and lost fare revenue."),
        ("red", "🔴", "Low-Revenue Zone Underperformance",
         "Several source areas consistently generate near-zero revenue. "
         "These zones represent wasted driver idle time and a missed monetisation opportunity."),
    ]
    opps = [
        ("green", "🟢", f"Scale {top_svc} Service",
         f"{top_svc} rides lead all service types in volume. "
         "Targeted promotions — first-ride discounts, referral bonuses — can grow this segment further."),
        ("green", "🟢", "Surge Pricing During Peak Hours",
         f"Implementing dynamic pricing between 7–9 AM and 6–9 PM can lift revenue by 15–25% "
         "per ride without adding fleet cost."),
        ("green", "🟢", "Parcel Service B2B Expansion",
         "Parcel delivery rides have competitive revenue-per-km and lower cancellation rates. "
         "Onboarding 50+ local business clients can create a reliable new revenue stream."),
    ]
    actions = [
        ("blue", "🔵", "Introduce Driver Cancellation Penalties",
         "A tiered penalty system (warning → fine → deactivation) for unjustified cancellations "
         "will reduce cancel rate within 30 days. Pair with a peak-hour availability bonus."),
        ("blue", "🔵", "Launch Digital Payment Cashback Campaign",
         "A 5% cashback on GPay and Amazon Pay rides for 90 days will shift cash users to digital, "
         "reducing reconciliation cost and fraud risk."),
        ("blue", "🔵", "Geo-Targeted Promotions for Low-Ride Zones",
         "Run in-app notifications and first-ride discounts in the bottom 10 source areas. "
         "Incentivise drivers to reposition to these zones during off-peak hours."),
    ]
    return risks, opps, actions


# ─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────
df_full = load_data(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header (no external image — avoids broken icon in offline environments)
    st.markdown(
        "<div style='text-align:center;padding:12px 0 6px;'>"
        "<span style='font-size:36px;'>🚖</span><br>"
        "<span style='font-size:15px;font-weight:700;color:#1f2328;'>Rapido Analytics</span><br>"
        "<span style='font-size:11px;color:#6b7280;'>Internship Dashboard</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:10px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("#### 🎛️ Filters")

    # Date range
    date_min = df_full["date"].min().date()
    date_max = df_full["date"].max().date()
    date_range = st.date_input(
        "📅 Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    # Service type
    services_all = sorted(df_full["services"].dropna().unique())
    sel_services = st.multiselect("🚗 Service Type", services_all, default=services_all)

    # Source area (top 20 only to avoid oversized list)
    top20_areas = df_full["source_area"].value_counts().head(20).index.tolist()
    sel_areas = st.multiselect("📍 Source Area", top20_areas, default=[])

    # Ride status
    status_all = sorted(df_full["ride_status"].dropna().unique())
    sel_status = st.multiselect("🔖 Ride Status", status_all, default=status_all)

    st.markdown("<hr style='margin:16px 0 10px;'>", unsafe_allow_html=True)

    # Page navigation
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

# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Executive Overview":
    st.title("🚖 Rapido Ride Analytics — Executive Overview")
    st.caption(
        f"Showing **{len(df):,}** rides · "
        f"{start_dt.strftime('%d %b %Y')} → {end_dt.strftime('%d %b %Y')}"
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_rides   = len(df)
    total_revenue = completed["total_fare"].sum()
    avg_fare      = completed["total_fare"].mean() if len(completed) else 0.0
    cancel_rate   = len(cancelled) / total_rides * 100 if total_rides else 0.0
    total_dist    = completed["distance"].sum()
    comp_rate     = 100 - cancel_rate

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "🚗 Total Rides",       f"{total_rides:,}",              color="#3b82d4")
    kpi_card(c2, "💰 Total Revenue",     f"₹{total_revenue/1e6:.2f}M",    color="#22c55e")
    kpi_card(c3, "🎯 Avg Fare",          f"₹{avg_fare:.0f}",              color="#f59e0b")
    kpi_card(c4, "✅ Completion Rate",   f"{comp_rate:.1f}%",             color="#10b981")
    kpi_card(c5, "❌ Cancel Rate",       f"{cancel_rate:.1f}%",           color="#ef4444")
    kpi_card(c6, "📏 Total Distance",    f"{total_dist/1e3:.1f}K km",     color="#8b5cf6")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("🛵 Rides by Service Type")
        svc_df = df["services"].value_counts().reset_index()
        svc_df.columns = ["Service", "Rides"]
        fig = px.pie(
            svc_df, names="Service", values="Rides",
            color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
            hole=0.42,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          pull=[0.04] * len(svc_df))
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
                          height=340, legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("🔖 Ride Status Distribution")
        status_df = df["ride_status"].value_counts().reset_index()
        status_df.columns = ["Status", "Count"]
        color_map = {"completed": "#22c55e", "cancelled": "#ef4444"}
        fig2 = px.bar(
            status_df, x="Status", y="Count",
            color="Status", color_discrete_map=color_map,
            text="Count",
        )
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=340,
                           yaxis_title="Number of Rides")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Monthly revenue trend ─────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("📈 Monthly Revenue Trend")

    if len(completed):
        monthly = (
            completed.groupby(["year", "month", "month_name"])["total_fare"]
            .sum().reset_index().sort_values(["year", "month"])
        )
        monthly["period"]     = monthly["month_name"] + " " + monthly["year"].astype(str)
        monthly["growth_pct"] = monthly["total_fare"].pct_change() * 100
        monthly["label"]      = monthly["total_fare"].apply(lambda v: f"₹{v/1e6:.2f}M")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=monthly["period"], y=monthly["total_fare"],
            mode="lines+markers+text",
            line=dict(color="#3b82d4", width=3),
            marker=dict(size=9, color="#3b82d4",
                        line=dict(width=2, color="white")),
            text=monthly["label"], textposition="top center",
            fill="tozeroy", fillcolor="rgba(59,130,212,0.10)",
            name="Revenue",
            hovertemplate="%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        # Growth % secondary axis
        fig3.add_trace(go.Scatter(
            x=monthly["period"], y=monthly["growth_pct"],
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=6, color="#f59e0b"),
            name="MoM Growth %",
            yaxis="y2",
            hovertemplate="%{x}<br>Growth: %{y:.1f}%<extra></extra>",
        ))
        fig3.update_layout(
            xaxis_title="Month",
            yaxis=dict(title="Revenue (₹)", tickformat="₹,.0f"),
            yaxis2=dict(title="MoM Growth %", overlaying="y", side="right",
                        ticksuffix="%", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=20, b=40),
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No completed rides in the selected filters.")

    # ── Revenue by Service (month stacked) ───────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("💰 Revenue by Service Type — Monthly Breakdown")

    if len(completed):
        rev_svc = (
            completed.groupby(["year", "month", "month_name", "services"])["total_fare"]
            .sum().reset_index().sort_values(["year", "month"])
        )
        rev_svc["period"] = rev_svc["month_name"] + " " + rev_svc["year"].astype(str)
        fig4 = px.bar(
            rev_svc, x="period", y="total_fare", color="services",
            barmode="stack",
            color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
            labels={"total_fare": "Revenue (₹)", "period": "Month", "services": "Service"},
        )
        fig4.update_layout(
            yaxis_tickformat="₹,.0f", legend=dict(orientation="h", y=1.05),
            margin=dict(t=10, b=40), height=360,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Key Insights (Page 1) ─────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("💡 Key Insights")
    insights = generate_insights(df, completed, cancelled)
    bullets = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
    for i, ins in enumerate(insights):
        insight_card(bullets[i], ins)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RIDE & REVENUE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Ride & Revenue Analysis":
    st.title("📊 Ride & Revenue Analysis")
    st.caption(f"Filtered dataset: **{len(df):,}** rides")

    # ── Peak Hours ────────────────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("⏰ Peak Hours Analysis")

    hourly = df.groupby("hour").size().reset_index(name="Rides")
    peak_h = int(hourly.loc[hourly["Rides"].idxmax(), "hour"])
    hourly["color"] = hourly["hour"].apply(
        lambda h: "#ef4444" if h == peak_h else "#3b82d4"
    )
    fig_h = px.bar(
        hourly, x="hour", y="Rides",
        color="color", color_discrete_map="identity",
        labels={"hour": "Hour of Day (0–23)", "Rides": "Number of Rides"},
        text="Rides",
    )
    fig_h.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig_h.update_layout(
        showlegend=False,
        xaxis=dict(tickmode="linear", dtick=1),
        annotations=[dict(
            x=peak_h, y=hourly["Rides"].max() * 1.08,
            text=f"⚡ Peak: {peak_h}:00",
            showarrow=False, font=dict(size=12, color="#ef4444"),
        )],
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=20, b=20), height=340,
    )
    st.plotly_chart(fig_h, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Time of Day ───────────────────────────────────────────────────────────
    with col1:
        section_header("🌅 Rides by Time of Day")
        tod_order = ["Morning", "Afternoon", "Evening", "Night"]
        tod = df["time_of_day"].value_counts().reindex(tod_order).reset_index()
        tod.columns = ["TimeOfDay", "Rides"]
        fig_tod = px.bar(
            tod, x="TimeOfDay", y="Rides",
            color="TimeOfDay",
            color_discrete_sequence=["#f59e0b", "#22c55e", "#f97316", "#6366f1"],
            text="Rides",
        )
        fig_tod.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_tod.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig_tod, use_container_width=True)

    # ── Day × Hour Heatmap ────────────────────────────────────────────────────
    with col2:
        section_header("📅 Demand Heatmap: Day × Hour")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hmap = (
            df.groupby(["day_of_week", "hour"]).size()
            .unstack(fill_value=0)
            .reindex(day_order)
        )
        fig_hmap = px.imshow(
            hmap,
            labels=dict(x="Hour", y="Day", color="Rides"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig_hmap.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig_hmap, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Revenue by Service ────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        section_header("💰 Revenue by Service Type")
        rev_svc = (
            completed.groupby("services")["total_fare"]
            .sum().sort_values(ascending=False).reset_index()
        )
        rev_svc.columns = ["Service", "Revenue"]
        fig_rsvc = px.bar(
            rev_svc, x="Service", y="Revenue",
            color="Service",
            color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
            text=rev_svc["Revenue"].apply(lambda v: f"₹{v/1e6:.2f}M"),
        )
        fig_rsvc.update_traces(textposition="outside")
        fig_rsvc.update_layout(
            showlegend=False, yaxis_tickformat="₹,.0f",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10), height=320,
        )
        st.plotly_chart(fig_rsvc, use_container_width=True)

    with col4:
        section_header("📍 Top 10 Source Areas by Rides")
        top_src = df["source_area"].value_counts().head(10).reset_index()
        top_src.columns = ["Area", "Rides"]
        fig_src = px.bar(
            top_src.sort_values("Rides"), x="Rides", y="Area",
            orientation="h",
            color="Rides", color_continuous_scale="Blues",
            text="Rides",
        )
        fig_src.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_src.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10), height=320,
        )
        st.plotly_chart(fig_src, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Fare Distribution ─────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        section_header("💵 Fare Distribution")
        fig_fd = px.histogram(
            completed.dropna(subset=["total_fare"]),
            x="total_fare", nbins=60,
            labels={"total_fare": "Total Fare (₹)"},
            color_discrete_sequence=["#3b82d4"],
        )
        mf = completed["total_fare"].mean() if len(completed) else 0
        fig_fd.add_vline(x=mf, line_dash="dash", line_color="red",
                         annotation_text=f"Mean ₹{mf:.0f}",
                         annotation_position="top right")
        fig_fd.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig_fd, use_container_width=True)

    with col6:
        section_header("📦 Fare Box Plot by Service")
        fig_box = px.box(
            completed.dropna(subset=["total_fare"]),
            x="services", y="total_fare",
            color="services",
            color_discrete_sequence=["#3b82d4", "#22c55e", "#f59e0b"],
            labels={"total_fare": "Total Fare (₹)", "services": "Service"},
        )
        fig_box.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white",
                              margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Distance vs Fare scatter ──────────────────────────────────────────────
    section_header("📐 Distance vs Fare Analysis")
    sample = completed.dropna(subset=["distance", "total_fare"])
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=42)
    fig_sc = px.scatter(
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
    fig_sc.update_layout(
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=30), height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Payment Methods ───────────────────────────────────────────────────────
    section_header("💳 Payment Method Distribution")
    pay_df = completed["payment_method"].value_counts().reset_index()
    pay_df.columns = ["Method", "Count"]
    col7, col8 = st.columns([1, 1])

    with col7:
        fig_pay = px.pie(
            pay_df, names="Method", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.38,
        )
        fig_pay.update_traces(textposition="outside", textinfo="percent+label")
        fig_pay.update_layout(legend=dict(orientation="h", y=-0.15),
                              margin=dict(t=10, b=30), height=340)
        st.plotly_chart(fig_pay, use_container_width=True)

    with col8:
        fig_pay_bar = px.bar(
            pay_df.sort_values("Count"), x="Count", y="Method",
            orientation="h",
            color="Count", color_continuous_scale="Teal",
            text="Count",
        )
        fig_pay_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_pay_bar.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                                  paper_bgcolor="white",
                                  margin=dict(t=10, b=10), height=340)
        st.plotly_chart(fig_pay_bar, use_container_width=True)

    # ── Key Insights (Page 2) ─────────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("💡 Key Insights")
    insights = generate_insights(df, completed, cancelled)
    bullets = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
    for i, ins in enumerate(insights):
        insight_card(bullets[i], ins)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CUSTOMER & RISK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚠️  Customer & Risk Analysis":
    st.title("⚠️ Customer & Risk Analysis")
    st.caption(f"Filtered dataset: **{len(df):,}** rides")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_rides  = len(df)
    cancel_total = len(cancelled)
    cancel_rate  = cancel_total / total_rides * 100 if total_rides else 0
    comp_rate    = 100 - cancel_rate

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "❌ Cancelled Rides",   f"{cancel_total:,}",  color="#ef4444")
    kpi_card(c2, "📉 Cancellation Rate", f"{cancel_rate:.1f}%", color="#f97316")
    kpi_card(c3, "✅ Completion Rate",   f"{comp_rate:.1f}%",  color="#22c55e")
    kpi_card(c4, "📊 Total Rides",       f"{total_rides:,}",   color="#3b82d4")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Cancellation trend ────────────────────────────────────────────────────
    section_header("📉 Cancellation Trend — Monthly")
    if len(df):
        ct = (
            df.groupby(["year", "month", "month_name", "ride_status"])
            .size().reset_index(name="Count")
            .sort_values(["year", "month"])
        )
        ct["period"] = ct["month_name"] + " " + ct["year"].astype(str)
        fig_ct = px.bar(
            ct, x="period", y="Count", color="ride_status",
            barmode="group",
            color_discrete_map={"completed": "#22c55e", "cancelled": "#ef4444"},
            labels={"Count": "Rides", "period": "Month", "ride_status": "Status"},
        )
        fig_ct.update_layout(
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=40), height=340,
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Cancellation by Service ───────────────────────────────────────────────
    with col1:
        section_header("🚗 Cancellation Rate by Service")
        csvc = (
            df.groupby("services")
            .apply(lambda x: (x["ride_status"] == "cancelled").sum() / len(x) * 100,
                   include_groups=False)
            .reset_index(name="Cancel Rate (%)")
        )
        fig_csvc = px.bar(
            csvc, x="services", y="Cancel Rate (%)",
            color="services",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82d4"],
            text=csvc["Cancel Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_csvc.update_traces(textposition="outside")
        fig_csvc.update_layout(showlegend=False, plot_bgcolor="white",
                               paper_bgcolor="white",
                               margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_csvc, use_container_width=True)

    # ── Cancellation by Hour ──────────────────────────────────────────────────
    with col2:
        section_header("⏰ Cancellations by Hour of Day")
        chr_df = cancelled.groupby("hour").size().reset_index(name="Cancellations")
        fig_chr = px.bar(
            chr_df, x="hour", y="Cancellations",
            color="Cancellations", color_continuous_scale="Reds",
            labels={"hour": "Hour (0–23)"},
        )
        fig_chr.update_layout(
            coloraxis_showscale=False,
            xaxis=dict(tickmode="linear", dtick=1),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10), height=300,
        )
        st.plotly_chart(fig_chr, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Low-Performing Areas ──────────────────────────────────────────────────
    section_header("🔻 Low-Performing Areas (Bottom 15 by Revenue)")
    low = (
        df.groupby("source_area")["revenue"]
        .sum().sort_values().head(15).reset_index()
    )
    low.columns = ["Area", "Revenue"]
    fig_low = px.bar(
        low, x="Revenue", y="Area",
        orientation="h",
        color="Revenue", color_continuous_scale="Reds_r",
        text=low["Revenue"].apply(lambda v: f"₹{v/1e3:.1f}K"),
    )
    fig_low.update_traces(textposition="outside")
    fig_low.update_layout(
        coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=10, b=10), height=380,
    )
    st.plotly_chart(fig_low, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Dynamic Key Insights ──────────────────────────────────────────────────
    section_header("💡 Key Insights (Data-Driven)")
    insights = generate_insights(df, completed, cancelled)
    bullets = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣"]
    for i, ins in enumerate(insights):
        insight_card(bullets[i], ins)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Risk / Opportunity / Action Board ────────────────────────────────────
    section_header("🗂️ Risk · Opportunity · Action Board")
    risks, opps, actions = generate_risk_board(df, completed, cancelled)

    tab1, tab2, tab3 = st.tabs(["🔴 Risks", "🟢 Opportunities", "🔵 Actions"])

    with tab1:
        for kind, icon, title, body in risks:
            risk_board_card(kind, icon, title, body)

    with tab2:
        for kind, icon, title, body in opps:
            risk_board_card(kind, icon, title, body)

    with tab3:
        for kind, icon, title, body in actions:
            risk_board_card(kind, icon, title, body)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Raw Data Explorer ─────────────────────────────────────────────────────
    section_header("🔍 Raw Data Explorer")
    with st.expander("Show filtered data table (max 500 rows)"):
        cols_show = [
            "ride_id", "date", "services", "ride_status",
            "source", "destination", "duration", "distance",
            "total_fare", "payment_method", "time_of_day",
        ]
        st.dataframe(
            df[cols_show].head(500),
            use_container_width=True,
            height=380,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer (all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dash-footer'>"
    "🚖 Rapido Ride Analytics Dashboard &nbsp;·&nbsp; "
    "Data Analytics Internship Project &nbsp;·&nbsp; "
    "Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
