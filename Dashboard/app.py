import os
import pandas as pd

import numpy as np
import streamlit as st
import plotly.graph_objects as go

DATA_FILE  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database", "CPC NCEP NOA ANOM.txt")
ERA5_WA_PQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database", "era5_wa_monthly.parquet")

EN_COL  = "#c0392b"
LN_COL  = "#2471a3"
NU_COL  = "#95a5a6"

SEASON_ORDER = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
SEAS_MONTH   = {s: i+1 for i, s in enumerate(SEASON_ORDER)}
MONTH_ORDER  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

st.set_page_config(page_title="Nino 3.4 — ENSO Monitor", layout="wide")
st.markdown("""
<style>
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: #ffffff }
    [data-testid="block-container"] { padding: 16px 28px 0 28px !important }
    h1 { font-size: 18px !important; font-weight: 700; color: #1a202c; margin-bottom: 0 }
    footer { display: none }
    .stTabs [data-baseweb="tab"] {
        font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase
    }
    [data-testid="stMetricValue"] { font-size: 20px !important }
    [data-testid="stMetricLabel"] {
        font-size: 10px !important; font-weight: 600 !important;
        letter-spacing: .1em !important; text-transform: uppercase; color: #718096 !important
    }
</style>
""", unsafe_allow_html=True)


# ── data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    with open(DATA_FILE) as f:
        lines = f.readlines()
    rows = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 4:
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["SEAS", "YR", "TOTAL", "ANOM"])
    df["YR"]    = df["YR"].astype(int)
    df["TOTAL"] = df["TOTAL"].astype(float)
    df["ANOM"]  = df["ANOM"].astype(float)
    df["MONTH"] = df["SEAS"].map(SEAS_MONTH)
    df["DATE"]  = pd.to_datetime({"year": df["YR"], "month": df["MONTH"], "day": 1})
    df = df.sort_values("DATE").reset_index(drop=True)
    df["LABEL"] = df["SEAS"] + " " + df["YR"].astype(str)
    return df


def phase_label(anom):
    if anom >= 1.5:  return "Strong El Nino"
    if anom >= 0.5:  return "El Nino"
    if anom <= -1.5: return "Strong La Nina"
    if anom <= -0.5: return "La Nina"
    return "Neutral"


def phase_color(anom):
    if anom >= 0.5:  return EN_COL
    if anom <= -0.5: return LN_COL
    return NU_COL


def threshold_lines(fig, levels=None):
    if levels is None:
        levels = [(0.5, EN_COL, "dash"), (-0.5, LN_COL, "dash"),
                  (1.5, EN_COL, "dot"),  (-1.5, LN_COL, "dot"),
                  (2.0, EN_COL, "dot"),  (-2.0, LN_COL, "dot")]
    for y, col, dash in levels:
        fig.add_hline(y=y, line=dict(color=col, width=0.7, dash=dash))
    fig.add_hline(y=0, line=dict(color="#718096", width=0.8))


def base_layout(fig, height=420, xmargin=20, rmargin=20, xtitle=""):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(t=10, b=50, l=55, r=rmargin),
        yaxis=dict(title="Anomaly (°C)", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title=xtitle, gridcolor="#ebebeb"),
        hovermode="x unified", showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568"),
    )


def fill_traces(fig, df_sub):
    pos = df_sub["ANOM"].clip(lower=0)
    neg = df_sub["ANOM"].clip(upper=0)
    fig.add_trace(go.Scatter(
        x=df_sub["DATE"], y=pos, fill="tozeroy",
        fillcolor="rgba(192,57,43,0.18)", line=dict(color=EN_COL, width=0.4),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=df_sub["DATE"], y=neg, fill="tozeroy",
        fillcolor="rgba(36,113,163,0.18)", line=dict(color=LN_COL, width=0.4),
        showlegend=False, hoverinfo="skip"
    ))


def classify_events(df, threshold=0.5, min_dur=5):
    phase_arr = np.where(df["ANOM"] >= threshold, "EN",
                np.where(df["ANOM"] <= -threshold, "LN", "N"))
    events = []
    i = 0
    while i < len(df):
        p = phase_arr[i]
        if p in ("EN", "LN"):
            j = i
            while j < len(df) and phase_arr[j] == p:
                j += 1
            if j - i >= min_dur:
                blk = df.iloc[i:j]
                pk  = blk["ANOM"].idxmax() if p == "EN" else blk["ANOM"].idxmin()
                events.append({
                    "Type":            "El Nino" if p == "EN" else "La Nina",
                    "Start":           df.iloc[i]["LABEL"],
                    "Peak Season":     df.loc[pk, "LABEL"],
                    "End":             df.iloc[j-1]["LABEL"],
                    "Peak Anomaly":    round(df.loc[pk, "ANOM"], 2),
                    "Duration":        j - i,
                })
            i = j
        else:
            i += 1
    return pd.DataFrame(events)


@st.cache_data
def load_era5_wa():
    wa = pd.read_parquet(ERA5_WA_PQ)
    idx = pd.to_datetime(wa.index)
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_convert(None)
    wa.index = idx
    wa = wa.sort_index()
    wa["year"]  = wa.index.year
    wa["month"] = wa.index.month
    return wa


def merge_enso_wa(enso_df, wa_df):
    """Merge ENSO + ERA5 WA on year/month, compute anomalies post-merge."""
    enso_m = enso_df[["DATE", "ANOM"]].copy()
    enso_m["year"]  = enso_m["DATE"].dt.year
    enso_m["month"] = enso_m["DATE"].dt.month

    wa_copy = wa_df[["precip_mm", "temp_c", "year", "month"]].copy()

    merged = enso_m.merge(wa_copy, on=["year", "month"], how="inner")
    merged = merged.set_index("DATE").sort_index()

    # Compute anomalies vs 1981-2010 climatology on the merged series
    base  = merged[(merged["year"] >= 1981) & (merged["year"] <= 2010)]
    clim  = base.groupby("month")[["precip_mm", "temp_c"]].mean()
    merged["precip_anom"] = merged["precip_mm"] - merged["month"].map(clim["precip_mm"])
    merged["temp_anom"]   = merged["temp_c"]    - merged["month"].map(clim["temp_c"])

    merged["month_name"] = merged["month"].apply(lambda m: MONTH_ORDER[m - 1])
    merged["phase"]      = merged["ANOM"].apply(
        lambda a: "El Nino" if a >= 0.5 else ("La Nina" if a <= -0.5 else "Neutral")
    )
    return merged


# ── load ─────────────────────────────────────────────────────────────────────
df  = load()
row = df.iloc[-1]
cur_label = row["LABEL"]
cur_anom  = row["ANOM"]
cur_phase = phase_label(cur_anom)
cur_col   = phase_color(cur_anom)
cur_seas  = row["SEAS"]
cur_yr    = int(row["YR"])

# ── header ───────────────────────────────────────────────────────────────────
st.title("Nino 3.4 — ENSO Monitor")
st.caption("NOAA CPC  |  3-Month Running Mean SST Anomaly  |  Nino 3.4 Region (5N–5S, 170W–120W)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest Reading", cur_label)
c2.metric("Anomaly", f"{cur_anom:+.2f} °C")
c3.metric("Absolute SST", f"{row['TOTAL']:.2f} °C")
c4.metric("Phase", cur_phase)
record_row = df.loc[df["ANOM"].idxmax()]
c5.metric("All-Time Peak", f"{record_row['ANOM']:+.2f} °C  ({record_row['LABEL']})")

st.markdown(
    f'<div style="height:4px;border-radius:2px;margin:12px 0 4px 0;'
    f'background:linear-gradient(90deg,{EN_COL},{NU_COL},{LN_COL})"></div>',
    unsafe_allow_html=True
)

# ── tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Full History", "Current Cycle", "Analogues",
    "Heatmap", "Event Catalogue", "Seasonal Distribution",
    "West Africa — ENSO Correlation"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1  Full History
# ════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown("**Nino 3.4 SST Anomaly — 1950 to Present**")
    st.caption("Red fill: El Nino  |  Blue fill: La Nina  |  Dashed lines: ±0.5 / ±1.5 / ±2.0 °C")

    fig1 = go.Figure()
    fill_traces(fig1, df)
    fig1.add_trace(go.Scatter(
        x=df["DATE"], y=df["ANOM"], mode="lines",
        line=dict(color="#2c3e50", width=1.1),
        customdata=df["LABEL"],
        hovertemplate="%{customdata}<br>%{y:+.2f} °C<extra></extra>"
    ))
    threshold_lines(fig1)

    for label, date_str, val, anchor in [
        ("1982/83", "1983-01-01", 2.23, "center"),
        ("1997/98", "1998-01-01", 2.40, "center"),
        ("2015/16", "2016-01-01", 2.75, "center"),
        ("2023/24", "2024-01-01", 2.06, "center"),
    ]:
        fig1.add_annotation(
            x=date_str, y=val + 0.15, text=label,
            showarrow=False, xanchor=anchor,
            font=dict(size=8, color=EN_COL)
        )

    base_layout(fig1, height=430)
    st.plotly_chart(fig1, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2  Current Cycle
# ════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown(f"**Current Cycle — Last 5 Years**  &nbsp;&nbsp; {cur_label}: "
                f"<span style='color:{cur_col};font-weight:700'>{cur_anom:+.2f} °C — {cur_phase}</span>",
                unsafe_allow_html=True)

    cutoff = df["DATE"].max() - pd.DateOffset(years=5)
    recent = df[df["DATE"] >= cutoff].copy()

    fig2 = go.Figure()
    fill_traces(fig2, recent)
    fig2.add_trace(go.Scatter(
        x=recent["DATE"], y=recent["ANOM"], mode="lines+markers",
        line=dict(color="#2c3e50", width=2),
        marker=dict(size=5, color=[phase_color(a) for a in recent["ANOM"]],
                    line=dict(color="white", width=0.8)),
        customdata=recent["LABEL"],
        hovertemplate="%{customdata}<br>%{y:+.2f} °C<extra></extra>"
    ))
    threshold_lines(fig2, levels=[
        (0.5, EN_COL, "dash"), (-0.5, LN_COL, "dash"),
        (1.5, EN_COL, "dot"),  (-1.5, LN_COL, "dot")
    ])

    fig2.add_annotation(
        x=recent["DATE"].iloc[-1], y=cur_anom,
        text=f"  {cur_label}<br>  {cur_anom:+.2f} °C",
        showarrow=False, xanchor="left",
        font=dict(size=9, color=cur_col)
    )
    base_layout(fig2, height=400, rmargin=100)
    st.plotly_chart(fig2, use_container_width=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3  Analogues
# ════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown(f"**Historical Analogues — {cur_label} ({cur_anom:+.2f} °C)**")
    st.caption(f"Historical {cur_seas} seasons with anomaly within ±0.4 °C of current. "
               f"Vertical line = now. Dashed lines = analogues, solid black = current.")

    mask = (
        (df["SEAS"] == cur_seas) &
        (df["YR"] < cur_yr) &
        (abs(df["ANOM"] - cur_anom) <= 0.4)
    )
    candidates = df[mask].copy()
    candidates["dist"] = abs(candidates["ANOM"] - cur_anom)
    top6 = candidates.nsmallest(6, "dist")

    ACOLS = ["#636e72", "#0984e3", "#6c5ce7", "#e17055", "#00b894", "#fdcb6e"]

    fig3 = go.Figure()

    for i, (idx, cand) in enumerate(top6.iterrows()):
        s = max(0, idx - 4)
        e = min(len(df) - 1, idx + 12)
        win = df.iloc[s:e+1].copy().reset_index(drop=True)
        match_pos = idx - s
        win["offset"] = range(-match_pos, len(win) - match_pos)
        fig3.add_trace(go.Scatter(
            x=win["offset"], y=win["ANOM"], mode="lines",
            line=dict(color=ACOLS[i % len(ACOLS)], width=1.5, dash="dot"),
            name=f"{cand['LABEL']}  ({cand['ANOM']:+.2f})",
            customdata=win["LABEL"],
            hovertemplate="%{customdata}<br>%{y:+.2f} °C<extra></extra>",
            opacity=0.75
        ))

    # Ensemble mean
    if len(top6) >= 2:
        all_offsets = range(-4, 13)
        mean_vals = []
        for off in all_offsets:
            vals = []
            for idx, cand in top6.iterrows():
                s = max(0, idx - 4)
                win = df.iloc[s:min(len(df)-1, idx+12)+1].copy().reset_index(drop=True)
                match_pos = idx - s
                win["offset"] = range(-match_pos, len(win) - match_pos)
                row_match = win[win["offset"] == off]
                if len(row_match):
                    vals.append(row_match["ANOM"].values[0])
            mean_vals.append(np.mean(vals) if vals else np.nan)
        fig3.add_trace(go.Scatter(
            x=list(all_offsets), y=mean_vals, mode="lines",
            line=dict(color="#2c3e50", width=2, dash="dash"),
            name="Ensemble mean",
            hovertemplate="Offset %{x}<br>Mean: %{y:+.2f} °C<extra></extra>"
        ))

    # Current trajectory
    cur_idx = df.index[-1]
    ts = max(0, cur_idx - 4)
    traj = df.iloc[ts:].copy().reset_index(drop=True)
    match_pos = cur_idx - ts
    traj["offset"] = range(-match_pos, len(traj) - match_pos)
    fig3.add_trace(go.Scatter(
        x=traj["offset"], y=traj["ANOM"], mode="lines+markers",
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=6, color="#c0392b"),
        name=f"Current — {cur_label}",
        customdata=traj["LABEL"],
        hovertemplate="%{customdata}<br>%{y:+.2f} °C<extra></extra>"
    ))

    fig3.add_vline(x=0, line=dict(color="#4a5568", width=1, dash="dash"))
    fig3.add_hline(y=0.5,  line=dict(color=EN_COL, width=0.6, dash="dash"))
    fig3.add_hline(y=-0.5, line=dict(color=LN_COL, width=0.6, dash="dash"))
    fig3.add_hline(y=0,    line=dict(color="#718096", width=0.7))

    fig3.update_layout(
        template="plotly_white", height=440,
        margin=dict(t=10, b=50, l=55, r=20),
        yaxis=dict(title="Anomaly (°C)", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title="Seasons from current (0 = now)", gridcolor="#ebebeb"),
        hovermode="closest",
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#e2e8f0", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    if len(top6) == 0:
        st.info("No analogues found within ±0.4 °C for this season.")
    else:
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4  Heatmap
# ════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("**ENSO Heatmap — Year × Season**")
    st.caption("Each cell = 3-month running mean Nino 3.4 anomaly.  Red = El Nino, Blue = La Nina.")

    pivot = df.pivot_table(index="YR", columns="SEAS", values="ANOM")
    pivot = pivot.reindex(columns=SEASON_ORDER)
    years = pivot.index.tolist()

    z    = pivot.values
    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str))

    fig4 = go.Figure(go.Heatmap(
        z=z,
        x=SEASON_ORDER,
        y=[str(y) for y in years],
        colorscale="RdBu_r",
        zmid=0, zmin=-2.5, zmax=2.5,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=7),
        colorbar=dict(
            title=dict(text="°C", font=dict(size=10)),
            thickness=14, len=0.9,
            tickfont=dict(size=9)
        ),
        hovertemplate="Season: %{x}<br>Year: %{y}<br>Anomaly: %{z:+.2f} °C<extra></extra>"
    ))

    fig4.update_layout(
        template="plotly_white", height=820,
        margin=dict(t=30, b=20, l=60, r=80),
        xaxis=dict(side="top", tickfont=dict(size=11), title=""),
        yaxis=dict(title="", tickfont=dict(size=8), autorange="reversed"),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5  Event Catalogue
# ════════════════════════════════════════════════════════════════════════════
with t5:
    events = classify_events(df, threshold=0.5, min_dur=5)

    if events.empty:
        st.info("No events classified.")
    else:
        en_evs = events[events["Type"] == "El Nino"].sort_values("Peak Anomaly", ascending=False).reset_index(drop=True)
        ln_evs = events[events["Type"] == "La Nina"].sort_values("Peak Anomaly").reset_index(drop=True)

        col1, col2 = st.columns(2)
        st.markdown("---")
        st.markdown("**Peak Anomaly by Event**")

        # Sort all events by the peak season date for a timeline bar
        events_sorted = events.copy()
        # Map peak season label to approximate date
        events_sorted["peak_seas"]  = events_sorted["Peak Season"].str[:3]
        events_sorted["peak_yr"]    = events_sorted["Peak Season"].str[-4:].astype(int)
        events_sorted["peak_month"] = events_sorted["peak_seas"].map(SEAS_MONTH)
        events_sorted["peak_date"]  = pd.to_datetime({
            "year": events_sorted["peak_yr"],
            "month": events_sorted["peak_month"],
            "day": 1
        })
        events_sorted = events_sorted.sort_values("peak_date")

        fig5 = go.Figure(go.Bar(
            x=events_sorted["Peak Season"],
            y=events_sorted["Peak Anomaly"],
            marker_color=[EN_COL if t == "El Nino" else LN_COL
                          for t in events_sorted["Type"]],
            customdata=events_sorted[["Type", "Duration", "Start", "End"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Type: %{customdata[0]}<br>"
                "Peak: %{y:+.2f} °C<br>"
                "Duration: %{customdata[1]} seasons<br>"
                "%{customdata[2]} → %{customdata[3]}<extra></extra>"
            )
        ))
        fig5.add_hline(y=0, line=dict(color="#718096", width=0.8))
        fig5.add_hline(y=0.5,  line=dict(color=EN_COL, width=0.6, dash="dash"))
        fig5.add_hline(y=-0.5, line=dict(color=LN_COL, width=0.6, dash="dash"))
        fig5.update_layout(
            template="plotly_white", height=360,
            margin=dict(t=10, b=80, l=55, r=20),
            yaxis=dict(title="Peak Anomaly (°C)", gridcolor="#ebebeb", zeroline=False),
            xaxis=dict(title="", tickangle=-45, gridcolor="#ebebeb"),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="Inter, sans-serif", color="#4a5568"),
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6  Seasonal Distribution
# ════════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown("**Seasonal Distribution of Nino 3.4 Anomaly — 1950 to Present**")
    st.caption("Box = IQR  |  Whiskers = 1.5× IQR  |  Mean = dashed line  |  Star = current reading")

    fig6 = go.Figure()

    for seas in SEASON_ORDER:
        vals = df[df["SEAS"] == seas]["ANOM"].dropna()
        fig6.add_trace(go.Box(
            y=vals, name=seas,
            marker=dict(color="#4a5568", size=3, opacity=0.5),
            line=dict(color="#4a5568", width=1.2),
            fillcolor="rgba(74,85,104,0.10)",
            boxmean=True,
            hovertemplate=f"<b>{seas}</b><br>%{{y:+.2f}} °C<extra></extra>"
        ))

    fig6.add_trace(go.Scatter(
        x=[cur_seas], y=[cur_anom],
        mode="markers",
        marker=dict(color=cur_col, size=13, symbol="star",
                    line=dict(color="white", width=1)),
        name=f"Current — {cur_label}  ({cur_anom:+.2f} °C)",
        hovertemplate=f"{cur_label}<br>{cur_anom:+.2f} °C<extra></extra>"
    ))

    fig6.add_hline(y=0.5,  line=dict(color=EN_COL, width=0.7, dash="dash"))
    fig6.add_hline(y=-0.5, line=dict(color=LN_COL, width=0.7, dash="dash"))
    fig6.add_hline(y=0,    line=dict(color="#718096", width=0.8))

    fig6.update_layout(
        template="plotly_white", height=460,
        margin=dict(t=10, b=50, l=55, r=20),
        yaxis=dict(title="Anomaly (°C)", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title="Season", gridcolor="#ebebeb"),
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568"),
        showlegend=True
    )
    st.plotly_chart(fig6, use_container_width=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 7  West Africa — ENSO Correlation
# ════════════════════════════════════════════════════════════════════════════
with t7:
    if not os.path.exists(ERA5_WA_PQ):
        st.warning("era5_wa_monthly.parquet not found. Run Database/ingest_era5_wa.py first.")
        st.stop()

    wa  = load_era5_wa()
    mrg = merge_enso_wa(df, wa)

    PHASE_COLS = {"El Nino": EN_COL, "Neutral": NU_COL, "La Nina": LN_COL}

    st.markdown("**West Africa Cocoa Belt — ENSO Impact on Rainfall & Temperature**")
    st.caption("ERA5 Reanalysis  |  Ivory Coast / Ghana region (4–10°N, 8°W–2°E)  |  1950–present  |  Climatology baseline: 1981–2010")

    # ── KPI row
    for phase_name, col in PHASE_COLS.items():
        pass
    ph_stats = mrg.groupby("phase")[["precip_mm", "temp_c"]].mean().round(1)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("El Nino avg precip",  f"{ph_stats.loc['El Nino','precip_mm']:.0f} mm/mo"  if 'El Nino'  in ph_stats.index else "—")
    k2.metric("La Nina avg precip",  f"{ph_stats.loc['La Nina','precip_mm']:.0f} mm/mo"  if 'La Nina'  in ph_stats.index else "—")
    k3.metric("El Nino avg temp",    f"{ph_stats.loc['El Nino','temp_c']:.1f} °C"         if 'El Nino'  in ph_stats.index else "—")
    k4.metric("La Nina avg temp",    f"{ph_stats.loc['La Nina','temp_c']:.1f} °C"         if 'La Nina'  in ph_stats.index else "—")

    st.markdown("---")

    # ── Section 1: Monthly precip by ENSO phase (composite bar)
    st.markdown("**Rainfall by Month — ENSO Phase Composite**")
    st.caption("Average monthly precipitation (mm) grouped by ENSO phase. Highlights which rainy seasons are most affected.")

    fig7a = go.Figure()
    for phase_name, col in PHASE_COLS.items():
        sub = mrg[mrg["phase"] == phase_name].groupby("month")["precip_mm"].mean()
        fig7a.add_trace(go.Bar(
            x=MONTH_ORDER,
            y=[sub.get(m+1, 0) for m in range(12)],
            name=phase_name,
            marker_color=col,
            hovertemplate=f"<b>{phase_name}</b><br>%{{x}}: %{{y:.1f}} mm<extra></extra>"
        ))
    fig7a.update_layout(
        template="plotly_white", height=380, barmode="group",
        margin=dict(t=10, b=50, l=55, r=20),
        yaxis=dict(title="Precip (mm/month)", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title="", gridcolor="#ebebeb"),
        legend=dict(font=dict(size=11), orientation="h", y=1.05, x=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    st.plotly_chart(fig7a, use_container_width=True)

    # ── Section 2: Lagged correlation — Nino 3.4 vs WA precip
    st.markdown("---")
    st.markdown("**Lagged Correlation — Nino 3.4 Anomaly vs West Africa Rainfall**")
    st.caption("Correlation at each lag (months). Positive lag = ENSO leads rainfall. Peak tells you how many months ahead ENSO predicts conditions.")

    max_lag = 12
    lags    = range(-max_lag, max_lag + 1)
    corr_p, corr_t = [], []
    for lag in lags:
        shifted = mrg["ANOM"].shift(lag)
        corr_p.append(shifted.corr(mrg["precip_anom"]))
        corr_t.append(shifted.corr(mrg["temp_anom"]))

    fig7b = go.Figure()
    fig7b.add_trace(go.Scatter(
        x=list(lags), y=corr_p, mode="lines+markers",
        line=dict(color=LN_COL, width=2),
        marker=dict(size=5),
        name="vs Precipitation anomaly",
        hovertemplate="Lag %{x}m<br>r = %{y:.3f}<extra></extra>"
    ))
    fig7b.add_trace(go.Scatter(
        x=list(lags), y=corr_t, mode="lines+markers",
        line=dict(color=EN_COL, width=2, dash="dash"),
        marker=dict(size=5),
        name="vs Temperature anomaly",
        hovertemplate="Lag %{x}m<br>r = %{y:.3f}<extra></extra>"
    ))
    fig7b.add_hline(y=0,     line=dict(color="#718096", width=0.8))
    fig7b.add_hline(y=0.2,   line=dict(color="#718096", width=0.6, dash="dot"))
    fig7b.add_hline(y=-0.2,  line=dict(color="#718096", width=0.6, dash="dot"))
    fig7b.add_vline(x=0,     line=dict(color="#4a5568", width=0.8, dash="dash"))
    fig7b.update_layout(
        template="plotly_white", height=360,
        margin=dict(t=10, b=50, l=55, r=20),
        yaxis=dict(title="Pearson r", gridcolor="#ebebeb", zeroline=False, range=[-0.6, 0.6]),
        xaxis=dict(title="Lag (months, positive = ENSO leads)", gridcolor="#ebebeb"),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    st.plotly_chart(fig7b, use_container_width=True)

    # ── Section 3: Scatter — Nino 3.4 vs WA precip anomaly by season
    st.markdown("---")
    st.markdown("**Scatter — Nino 3.4 vs Rainfall Anomaly by Month**")
    st.caption("Each dot = one month. Coloured by ENSO phase. Slope shows direction of relationship.")

    sel_month = st.selectbox(
        "Filter by calendar month (or All)",
        options=["All"] + MONTH_ORDER,
        index=0
    )
    scatter_df = mrg.copy()
    if sel_month != "All":
        m_num = MONTH_ORDER.index(sel_month) + 1
        scatter_df = scatter_df[scatter_df["month"] == m_num]

    fig7c = go.Figure()
    for phase_name, col in PHASE_COLS.items():
        sub = scatter_df[scatter_df["phase"] == phase_name]
        fig7c.add_trace(go.Scatter(
            x=sub["ANOM"], y=sub["precip_anom"],
            mode="markers",
            name=phase_name,
            marker=dict(color=col, size=5, opacity=0.65,
                        line=dict(color="white", width=0.4)),
            customdata=sub.index.strftime("%b %Y"),
            hovertemplate="%{customdata}<br>Nino 3.4: %{x:+.2f} °C<br>Precip anom: %{y:+.1f} mm<extra></extra>"
        ))
    # Trendline
    valid = scatter_df[["ANOM", "precip_anom"]].dropna()
    if len(valid) > 10:
        m_coef, b_coef = np.polyfit(valid["ANOM"], valid["precip_anom"], 1)
        x_range = np.linspace(valid["ANOM"].min(), valid["ANOM"].max(), 100)
        fig7c.add_trace(go.Scatter(
            x=x_range, y=m_coef * x_range + b_coef,
            mode="lines", line=dict(color="#4a5568", width=1.5, dash="dot"),
            name=f"Trend  (slope={m_coef:.1f} mm/°C)", showlegend=True
        ))
    fig7c.add_hline(y=0, line=dict(color="#718096", width=0.7))
    fig7c.add_vline(x=0, line=dict(color="#718096", width=0.7))
    fig7c.update_layout(
        template="plotly_white", height=400,
        margin=dict(t=10, b=50, l=60, r=20),
        yaxis=dict(title="Rainfall anomaly (mm vs 1981–2010 clim)", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title="Nino 3.4 anomaly (°C)", gridcolor="#ebebeb"),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    st.plotly_chart(fig7c, use_container_width=True)

    # ── Section 4: Dual time series overlay
    st.markdown("---")
    st.markdown("**Time Series Overlay — Nino 3.4 vs West Africa Rainfall Anomaly**")
    st.caption("12-month rolling mean shown for clarity. Inverted relationship visible during strong ENSO events.")

    roll = mrg[["ANOM", "precip_anom"]].rolling(12, center=True).mean().dropna()

    fig7d = go.Figure()
    fig7d.add_trace(go.Scatter(
        x=roll.index, y=roll["ANOM"],
        mode="lines", name="Nino 3.4 (12m roll)",
        line=dict(color="#2c3e50", width=1.5),
        hovertemplate="%{x|%b %Y}<br>Nino 3.4: %{y:+.2f} °C<extra></extra>"
    ))
    fig7d.add_trace(go.Scatter(
        x=roll.index, y=roll["precip_anom"] / 30,   # scale to same axis
        mode="lines", name="WA Precip anom (÷30, 12m roll)",
        line=dict(color=LN_COL, width=1.5, dash="dash"),
        hovertemplate="%{x|%b %Y}<br>Precip anom (scaled): %{y:+.2f}<extra></extra>"
    ))
    fig7d.add_hline(y=0, line=dict(color="#718096", width=0.7))
    fig7d.update_layout(
        template="plotly_white", height=360,
        margin=dict(t=10, b=50, l=55, r=20),
        yaxis=dict(title="Nino 3.4 °C  /  Precip anom ÷30", gridcolor="#ebebeb", zeroline=False),
        xaxis=dict(title="", gridcolor="#ebebeb"),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0", borderwidth=1),
        hovermode="x unified",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#4a5568")
    )
    st.plotly_chart(fig7d, use_container_width=True)

