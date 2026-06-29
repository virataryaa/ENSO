import os
import pandas as pd

import numpy as np
import streamlit as st
import plotly.graph_objects as go

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Database", "CPC NCEP NOA ANOM.txt")

EN_COL  = "#c0392b"
LN_COL  = "#2471a3"
NU_COL  = "#95a5a6"

SEASON_ORDER = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
SEAS_MONTH   = {s: i+1 for i, s in enumerate(SEASON_ORDER)}

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
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Full History", "Current Cycle", "Analogues",
    "Heatmap", "Event Catalogue", "Seasonal Distribution"
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

    # 12-month change table
    recent12 = df.tail(13).copy()
    delta = recent12["ANOM"].diff().dropna()
    tbl = recent12.tail(12).copy()
    tbl["Change"] = delta.values
    tbl = tbl[["LABEL","ANOM","Change"]].rename(columns={"LABEL":"Season","ANOM":"Anomaly (°C)","Change":"Change (°C)"})
    tbl = tbl.sort_values("Season", ascending=False).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


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
        with col1:
            st.markdown(f"**El Nino Events**  ({len(en_evs)})")
            st.dataframe(en_evs, use_container_width=True, hide_index=True)
        with col2:
            st.markdown(f"**La Nina Events**  ({len(ln_evs)})")
            st.dataframe(ln_evs, use_container_width=True, hide_index=True)

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

    # Summary stats table
    st.markdown("---")
    st.markdown("**Summary Statistics by Season**")
    stats_rows = []
    for seas in SEASON_ORDER:
        vals = df[df["SEAS"] == seas]["ANOM"].dropna()
        n_en = (vals >= 0.5).sum()
        n_ln = (vals <= -0.5).sum()
        n_ne = len(vals) - n_en - n_ln
        stats_rows.append({
            "Season": seas,
            "Mean (°C)": round(vals.mean(), 2),
            "Std (°C)":  round(vals.std(), 2),
            "Min (°C)":  round(vals.min(), 2),
            "Max (°C)":  round(vals.max(), 2),
            "El Nino (n)": int(n_en),
            "Neutral (n)": int(n_ne),
            "La Nina (n)": int(n_ln),
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
