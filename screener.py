"""
screener.py — Compare model predictions vs today's prop lines.
Flags +EV bets with Kelly sizing and dollar amounts.

Output format:
    player          | line  | prediction | edge%  | side | pinnacle_odds | bet365_odds | kelly% | bet_$
    Nikola Jokic    | 26.5  | 29.1       | +6.2%  | OVER | -108          | -110        | 4.2%   | $4.20

Kelly formula (half-Kelly):
    f = (bp - q) / b  * 0.5
    where b = decimal_odds - 1, p = estimated win prob from edge, q = 1-p
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data import load_gamelogs
from features import build_live_features, FEATURE_COLS
from model import load_model, predict
from odds import get_today_lines, american_to_decimal, implied_probability

# ── Config ────────────────────────────────────────────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "100"))        # starting bankroll in $
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "4"))  # minimum edge % to flag
MIN_LINE_DIFF = float(os.getenv("MIN_LINE_DIFF", "1.5"))  # minimum |pred - line| pts
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))  # cap at 5% per bet
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def no_vig_probs(over_price: float, under_price: float) -> tuple[float, float]:
    """
    Remove bookmaker vig to get fair probabilities for over and under.
    Returns (p_over, p_under) summing to 1.
    """
    p_raw_over = implied_probability(over_price)
    p_raw_under = implied_probability(under_price)
    total = p_raw_over + p_raw_under
    return p_raw_over / total, p_raw_under / total


def edge_from_prediction(
    prediction: float,
    line: float,
    p_over_fair: float,
    p_under_fair: float,
) -> tuple[str, float, float]:
    """
    Determine best bet side and edge %.

    The model predicts a point total. We translate that into a win probability
    using a simple normal distribution assumption around the line.

    sigma is estimated as ~5 pts (typical player scoring std dev at the line).

    Returns:
        (side, model_win_prob, edge_pct)
        side: 'OVER' or 'UNDER'
        model_win_prob: P(actual > line) or P(actual < line)
        edge_pct: model_prob - book_fair_prob  (as %)
    """
    from scipy import stats  # soft import for speed

    sigma = 5.0  # conservative estimate for scoring distribution

    # P(score > line) given model predicts `prediction`
    p_over_model = 1 - stats.norm.cdf(line, loc=prediction, scale=sigma)
    p_under_model = 1 - p_over_model

    edge_over = (p_over_model - p_over_fair) * 100
    edge_under = (p_under_model - p_under_fair) * 100

    if edge_over >= edge_under:
        return "OVER", p_over_model, edge_over
    else:
        return "UNDER", p_under_model, edge_under


def kelly_fraction(win_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly: f = (b*p - q) / b
    Half-Kelly: f = full_kelly * 0.5
    Capped at MAX_KELLY_FRACTION.
    """
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    if b <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    half_kelly = max(full_kelly * 0.5, 0.0)
    return min(half_kelly, MAX_KELLY_FRACTION)


def best_price_for_side(row: pd.Series, side: str) -> tuple[float | None, str | None]:
    """
    Return (best_american_price, bookmaker) for the given side.
    Checks both pinnacle and bet365 columns.
    """
    col = "over_price" if side == "OVER" else "under_price"
    candidates = {}

    # Lines df may have separate rows per bookmaker or combined columns
    if col in row.index:
        candidates[row.get("bookmaker", "unknown")] = row[col]

    if not candidates:
        return None, None

    best_bk = max(candidates, key=candidates.get)
    return candidates[best_bk], best_bk


def screen_player(
    player_name: str,
    line: float,
    over_price: float,
    under_price: float,
    bookmaker: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    df_history: pd.DataFrame,
    model,
    def_lookup=None,
) -> dict | None:
    """
    Run the full screening pipeline for one player-line.
    Returns a result dict or None if not flagged.
    """
    # Determine if the player's team is home or away
    # We need player's team; look up in history
    player_rows = df_history[df_history["player_name"] == player_name]
    if player_rows.empty:
        return None

    team_abbr = player_rows.iloc[-1].get("team_abbreviation", "")

    # Is the player's team the home team?
    is_home = home_team.upper() in (team_abbr.upper(), "")  # rough match
    # Refine using full team names from history where possible
    last_matchup = player_rows.sort_values("game_date").iloc[-1].get("matchup", "")
    if "@" in last_matchup:
        # e.g. "LAL @ GSW" — player is away
        is_home_approx = False
    else:
        is_home_approx = True

    # Opponent abbreviation
    opp_abbr = away_team[:3].upper() if is_home else home_team[:3].upper()
    game_date = commence_time[:10] if commence_time else date.today().isoformat()

    # Build features
    feats = build_live_features(
        player_name=player_name,
        opponent_team_abbr=opp_abbr,
        is_home=is_home_approx,
        game_date=game_date,
        df_history=df_history,
        def_lookup=def_lookup,
    )
    if not feats:
        return None

    # Predict
    try:
        prediction = predict(feats, model)
    except Exception:
        return None

    diff = prediction - line
    if abs(diff) < MIN_LINE_DIFF:
        return None

    # Edge calculation
    if pd.isna(over_price) or pd.isna(under_price):
        return None

    try:
        p_over_fair, p_under_fair = no_vig_probs(float(over_price), float(under_price))
    except Exception:
        return None

    try:
        side, win_prob, edge_pct = edge_from_prediction(prediction, line, p_over_fair, p_under_fair)
    except ImportError:
        # scipy not available — fallback to simple diff-based edge
        side = "OVER" if diff > 0 else "UNDER"
        win_prob = 0.55 if abs(diff) > 2 else 0.52
        edge_pct = (win_prob - 0.5) * 100

    if edge_pct < MIN_EDGE_PCT:
        return None

    # Best odds for chosen side
    price_col = float(over_price) if side == "OVER" else float(under_price)
    dec_odds = american_to_decimal(price_col)

    # Kelly sizing
    kf = kelly_fraction(win_prob, dec_odds)
    bet_dollars = round(kf * BANKROLL, 2)

    return {
        "player": player_name,
        "line": line,
        "prediction": round(prediction, 1),
        "edge_pct": round(edge_pct, 1),
        "side": side,
        "odds": price_col,
        "bookmaker": bookmaker,
        "kelly_pct": round(kf * 100, 2),
        "bet_dollars": bet_dollars,
        "game": f"{away_team} @ {home_team}",
        "commence_time": commence_time,
    }


def run_screener(
    bankroll: float = BANKROLL,
    min_edge: float = MIN_EDGE_PCT,
    min_diff: float = MIN_LINE_DIFF,
) -> pd.DataFrame:
    """
    Main screener pipeline.
    1. Load today's lines
    2. Load historical game logs
    3. Load trained model
    4. For each player-line, run screening
    5. Return flagged bets as DataFrame
    """
    global BANKROLL, MIN_EDGE_PCT, MIN_LINE_DIFF
    BANKROLL = bankroll
    MIN_EDGE_PCT = min_edge
    MIN_LINE_DIFF = min_diff

    print("=== NBA Props Screener ===")
    print(f"Bankroll: ${BANKROLL:.2f}  |  Min edge: {MIN_EDGE_PCT}%  |  Min diff: {MIN_LINE_DIFF} pts")

    # Load model
    try:
        model = load_model()
        print("[screener] Model loaded.")
    except FileNotFoundError:
        print("[screener] model.pkl not found — training now ...")
        from model import train, save_model, load_training_data
        X, y = load_training_data()
        model = train(X, y)
        save_model(model)

    # Load today's lines
    lines_df = get_today_lines()
    if lines_df.empty:
        print("[screener] No lines available for today. Exiting.")
        return pd.DataFrame()

    # Keep only player_points market
    if "market" in lines_df.columns:
        lines_df = lines_df[lines_df["market"] == "player_points"]

    print(f"[screener] {len(lines_df)} prop lines to evaluate.")

    # Load all shared data once — avoids reloading per-player
    from features import build_defense_lookup
    df_history = load_gamelogs()
    def_lookup = build_defense_lookup()
    print(f"[screener] Data loaded: {len(df_history)} game rows, {len(def_lookup)} defense entries.")

    results = []
    seen_players = set()  # screen each player once (best available line)

    for _, row in lines_df.iterrows():
        player = row.get("player_name", "")
        if not player or player in seen_players:
            continue

        line = row.get("line")
        if pd.isna(line):
            continue

        over_price = row.get("over_price")
        under_price = row.get("under_price")
        bookmaker = row.get("bookmaker", "unknown")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        commence_time = row.get("commence_time", "")

        result = screen_player(
            player_name=player,
            line=float(line),
            over_price=over_price,
            under_price=under_price,
            bookmaker=bookmaker,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            df_history=df_history,
            model=model,
            def_lookup=def_lookup,
        )

        if result:
            results.append(result)
            seen_players.add(player)

    if not results:
        print("[screener] No +EV bets found today.")
        return pd.DataFrame()

    bets_df = pd.DataFrame(results).sort_values("edge_pct", ascending=False).reset_index(drop=True)

    # Save results
    today = date.today().isoformat()
    out_path = RESULTS_DIR / f"bets_{today}.csv"
    bets_df.to_csv(out_path, index=False)

    return bets_df


def format_output(df: pd.DataFrame) -> str:
    """Format bets DataFrame as a readable table."""
    if df.empty:
        return "No +EV bets found for today."

    display = df.copy()
    display["edge%"] = display["edge_pct"].apply(lambda x: f"+{x:.1f}%")
    display["kelly%"] = display["kelly_pct"].apply(lambda x: f"{x:.2f}%")
    display["bet_$"] = display["bet_dollars"].apply(lambda x: f"${x:.2f}")
    display["odds"] = display["odds"].apply(lambda x: f"{int(x):+d}" if not pd.isna(x) else "N/A")
    display["prediction"] = display["prediction"].apply(lambda x: f"{x:.1f}")

    cols = ["player", "line", "prediction", "edge%", "side", "odds", "bookmaker", "kelly%", "bet_$", "game"]
    avail = [c for c in cols if c in display.columns]
    return display[avail].to_string(index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Props Screener")
    parser.add_argument("--bankroll", type=float, default=BANKROLL, help="Current bankroll in $")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT, help="Minimum edge %% to flag")
    parser.add_argument("--min-diff", type=float, default=MIN_LINE_DIFF, help="Min |prediction - line| pts")
    args = parser.parse_args()

    bets = run_screener(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        min_diff=args.min_diff,
    )

    print("\n" + "=" * 90)
    print("TODAY'S FLAGGED BETS")
    print("=" * 90)
    print(format_output(bets))

    if not bets.empty:
        total_bet = bets["bet_dollars"].sum()
        print(f"\nTotal allocated: ${total_bet:.2f} / ${args.bankroll:.2f} ({total_bet/args.bankroll*100:.1f}%)")
        print(f"Bets flagged: {len(bets)}")
        print(f"\nResults saved to: results/bets_{date.today().isoformat()}.csv")

    print("\n[screener.py] Done.")
