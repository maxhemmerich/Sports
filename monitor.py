"""
monitor.py — Streamlit dashboard showing today's flagged NBA prop bets.

Run with:
    streamlit run monitor.py

Features:
  - Live "Refresh" button to re-run screener
  - Clean table of today's flagged bets
  - Bankroll input with Kelly sizing update
  - Edge % bar chart
  - Bet allocation pie chart
  - Historical results tracker
"""

import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Props Screener",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_today_results() -> pd.DataFrame:
    today = date.today().isoformat()
    path = RESULTS_DIR / f"bets_{today}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def run_screener(bankroll: float, min_edge: float, min_diff: float) -> pd.DataFrame:
    """Invoke screener pipeline and return results."""
    with st.spinner("Running screener — fetching odds and building predictions..."):
        try:
            from screener import run_screener as _run, format_output
            return _run(bankroll=bankroll, min_edge=min_edge, min_diff=min_diff)
        except Exception as e:
            st.error(f"Screener error: {e}")
            return pd.DataFrame()


def load_all_results() -> pd.DataFrame:
    """Load all historical result files for P&L tracking."""
    frames = []
    for p in sorted(RESULTS_DIR.glob("bets_*.csv")):
        df = pd.read_csv(p)
        date_str = p.stem.replace("bets_", "")
        df["date"] = date_str
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def style_side(val: str) -> str:
    if val == "OVER":
        return "color: #00c853; font-weight: bold"
    elif val == "UNDER":
        return "color: #ff5252; font-weight: bold"
    return ""


def style_edge(val):
    try:
        v = float(str(val).replace("%", "").replace("+", ""))
        if v >= 8:
            return "background-color: #1b5e20; color: white"
        elif v >= 5:
            return "background-color: #2e7d32; color: white"
        elif v >= 3:
            return "background-color: #388e3c; color: white"
    except Exception:
        pass
    return ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏀 NBA Props Model")
st.sidebar.markdown("---")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    max_value=100_000.0,
    value=float(os.getenv("BANKROLL", "100")),
    step=10.0,
    help="Your current total bankroll for Kelly sizing",
)

min_edge = st.sidebar.slider(
    "Min edge %",
    min_value=1.0,
    max_value=20.0,
    value=float(os.getenv("MIN_EDGE_PCT", "4")),
    step=0.5,
    help="Minimum model edge required to flag a bet",
)

min_diff = st.sidebar.slider(
    "Min prediction gap (pts)",
    min_value=0.5,
    max_value=5.0,
    value=float(os.getenv("MIN_LINE_DIFF", "1.5")),
    step=0.25,
    help="Minimum |prediction - line| to consider",
)

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh / Run Screener", use_container_width=True)
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏀 NBA Props Betting Dashboard")
st.caption(f"Date: {date.today().isoformat()}  |  Bankroll: **${bankroll:,.2f}**")

# Load or run
if refresh:
    bets_df = run_screener(bankroll=bankroll, min_edge=min_edge, min_diff=min_diff)
else:
    bets_df = load_today_results()
    if bets_df.empty:
        st.info("No cached results for today. Click **Refresh / Run Screener** to generate bets.")

# ── Summary metrics ───────────────────────────────────────────────────────────
if not bets_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bets Flagged", len(bets_df))
    total_allocated = bets_df["bet_dollars"].sum() if "bet_dollars" in bets_df.columns else 0
    col2.metric("Total Allocated", f"${total_allocated:.2f}")
    col3.metric("% of Bankroll", f"{total_allocated / bankroll * 100:.1f}%")
    avg_edge = bets_df["edge_pct"].mean() if "edge_pct" in bets_df.columns else 0
    col4.metric("Avg Edge", f"+{avg_edge:.1f}%")

    st.markdown("---")

    # ── Bets Table ─────────────────────────────────────────────────────────────
    st.subheader("Today's Flagged Bets")

    display = bets_df.copy()
    if "edge_pct" in display.columns:
        display["edge%"] = display["edge_pct"].apply(lambda x: f"+{x:.1f}%")
    if "kelly_pct" in display.columns:
        display["kelly%"] = display["kelly_pct"].apply(lambda x: f"{x:.2f}%")
    if "bet_dollars" in display.columns:
        display["bet_$"] = display["bet_dollars"].apply(lambda x: f"${x:.2f}")
    if "odds" in display.columns:
        display["odds"] = display["odds"].apply(
            lambda x: f"{int(float(x)):+d}" if pd.notna(x) else "N/A"
        )
    if "prediction" in display.columns:
        display["prediction"] = display["prediction"].apply(lambda x: f"{float(x):.1f}")

    show_cols = ["player", "line", "prediction", "edge%", "side", "odds", "bookmaker", "kelly%", "bet_$", "game"]
    avail_cols = [c for c in show_cols if c in display.columns]

    styled = display[avail_cols].style.applymap(style_side, subset=["side"] if "side" in avail_cols else [])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Charts ─────────────────────────────────────────────────────────────────
    if len(bets_df) > 1:
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Edge % by Player")
            if "edge_pct" in bets_df.columns and "player" in bets_df.columns:
                chart_data = bets_df.set_index("player")["edge_pct"].sort_values(ascending=True)
                st.bar_chart(chart_data)

        with chart_col2:
            st.subheader("Bet Allocation")
            if "bet_dollars" in bets_df.columns and "player" in bets_df.columns:
                alloc = bets_df.set_index("player")["bet_dollars"]
                # Show as horizontal bar chart (Streamlit doesn't have native pie)
                st.bar_chart(alloc.sort_values(ascending=True))

    # ── Raw data expander ──────────────────────────────────────────────────────
    with st.expander("Raw Data"):
        st.dataframe(bets_df, use_container_width=True)

# ── Arbitrage Opportunities ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Arbitrage Opportunities (Pinnacle vs bet365)")

try:
    from odds import get_arbitrage_opportunities
    arb_df = get_arbitrage_opportunities()
    if arb_df.empty:
        st.info("No arbitrage opportunities found. Arb requires both Pinnacle and bet365 lines for the same player/market.")
    else:
        st.success(f"{len(arb_df)} arb opportunity(ies) found!")
        st.dataframe(arb_df, use_container_width=True, hide_index=True)
        st.caption("Arb% = guaranteed profit as % of total stake. Bet both legs simultaneously.")
except Exception as e:
    st.info(f"Run screener first to check for arbitrage. ({e})")

# ── Historical P&L ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Historical Bet Log")

tracker_path = RESULTS_DIR / "bet_tracker.csv"
if tracker_path.exists():
    tracker_df = pd.read_csv(tracker_path)
    settled = tracker_df[tracker_df["result"].isin(["WIN", "LOSS", "PUSH"])]
    pending = tracker_df[~tracker_df["result"].isin(["WIN", "LOSS", "PUSH"])]

    if not settled.empty:
        won = settled[settled["result"] == "WIN"]
        lost = settled[settled["result"] == "LOSS"]
        push = settled[settled["result"] == "PUSH"]

        pnl_won = (won["entered_$"] * won["odds"].apply(
            lambda x: (abs(x) / 100) if x > 0 else (100 / abs(x))
        )).sum() if not won.empty else 0
        pnl_lost = lost["entered_$"].sum() if not lost.empty else 0
        net_pnl = pnl_won - pnl_lost

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Win Rate", f"{len(won) / len(settled) * 100:.1f}%" if settled.shape[0] else "N/A")
        h2.metric("W / L / P", f"{len(won)} / {len(lost)} / {len(push)}")
        h3.metric("Net P&L", f"${net_pnl:+.2f}")
        h4.metric("Pending Bets", len(pending))

    st.dataframe(
        tracker_df.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Update the 'result' column in results/bet_tracker.csv with WIN / LOSS / PUSH after games settle.")
else:
    hist_df = load_all_results()
    if hist_df.empty:
        st.info("No bet history yet. Run the screener and log bets to track P&L.")
    else:
        st.write(f"{len(hist_df)} screened bet records across {hist_df['date'].nunique()} day(s)")
        st.dataframe(hist_df.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
        st.caption("Run screener.py interactively to log which bets you actually placed.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ For informational purposes only. Gamble responsibly. "
    "Ontario, Canada residents: visit [iGaming Ontario](https://www.igamingontario.ca) for responsible gambling resources."
)
