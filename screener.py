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
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data import load_gamelogs
from features import build_live_features, MARKET_CONFIG
from model import load_model, predict
from odds import get_today_lines, american_to_decimal, implied_probability

# ── Config ────────────────────────────────────────────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "100"))        # starting bankroll in $
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "4"))  # minimum edge % to flag
MIN_LINE_DIFF = float(os.getenv("MIN_LINE_DIFF", "1.5"))  # minimum |pred - line| pts
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))  # cap at 5% per bet
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "1.0"))  # max total bankroll % across all bets
MIN_GAMES = int(os.getenv("MIN_GAMES", "20"))           # min career games in DB before flagging
MIN_SEASON_GAMES = int(os.getenv("MIN_SEASON_GAMES", "15"))  # min games in current season (Oct–present)
CURRENT_SEASON_START = "2025-10-01"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Scoring distribution std dev per market — used to convert prediction gap → win probability.
SIGMA_BY_MARKET = {
    "player_points":    5.0,
    "player_rebounds":  2.5,
    "player_assists":   2.5,
    "player_threes":    1.2,
    "player_blocks":    0.8,
    "player_steals":    0.8,
    "player_turnovers": 1.2,
}


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
    sigma: float = 5.0,
) -> tuple[str, float, float]:
    """
    Determine best bet side and edge %.

    The model predicts a point total. We translate that into a win probability
    using a normal distribution around the line. Sigma varies by market:
      points: ~5.0,  rebounds: ~2.5,  assists: ~2.0

    Returns:
        (side, model_win_prob, edge_pct)
        side: 'OVER' or 'UNDER'
        model_win_prob: P(actual > line) or P(actual < line)
        edge_pct: model_prob - book_fair_prob  (as %)
    """
    from scipy import stats  # soft import for speed

    # P(score > line) given model predicts `prediction`
    p_over_model = 1 - stats.norm.cdf(line, loc=prediction, scale=sigma)
    p_under_model = 1 - p_over_model

    edge_over = (p_over_model - p_over_fair) * 100
    edge_under = (p_under_model - p_under_fair) * 100

    if edge_over >= edge_under:
        return "OVER", p_over_model, edge_over
    else:
        return "UNDER", p_under_model, edge_under


def kelly_fraction(win_prob: float, decimal_odds: float, edge_pct: float = 0.0) -> float:
    """
    Edge-proportional bet sizing: fraction = edge_pct / 200
    (4% edge → 2% of bankroll, 8% → 4%, 20% → 10%, no cap)

    Relative sizing is preserved when MAX_TOTAL_EXPOSURE scales the full slate down.
    """
    if edge_pct <= 0:
        return 0.0
    return edge_pct / 200.0


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
    target: str = "pts",
    market: str = "player_points",
) -> dict | None:
    """
    Run the full screening pipeline for one player-line.
    Returns a result dict or None if not flagged.
    """
    player_rows = df_history[df_history["player_name"] == player_name]
    if player_rows.empty:
        return None

    if len(player_rows) < MIN_GAMES:
        return None

    current_rows = player_rows[player_rows["game_date"] >= pd.Timestamp(CURRENT_SEASON_START)]
    if len(current_rows) < MIN_SEASON_GAMES:
        return None

    team_abbr = player_rows.sort_values("game_date").iloc[-1].get("team_abbreviation", "")

    home_upper = home_team.upper()
    abbr_upper = team_abbr.upper()
    is_home_approx = abbr_upper in home_upper or any(
        abbr_upper == word[:len(abbr_upper)] for word in home_upper.split()
    )

    opp_abbr = away_team[:3].upper() if is_home_approx else home_team[:3].upper()
    game_date = commence_time[:10] if commence_time else date.today().isoformat()

    feats = build_live_features(
        player_name=player_name,
        opponent_team_abbr=opp_abbr,
        is_home=is_home_approx,
        game_date=game_date,
        df_history=df_history,
        def_lookup=def_lookup,
        target=target,
    )
    if not feats:
        return None

    try:
        prediction = predict(feats, model, target=target)
    except Exception:
        return None

    diff = prediction - line
    if abs(diff) < MIN_LINE_DIFF:
        return None

    if pd.isna(over_price) or pd.isna(under_price):
        return None

    try:
        p_over_fair, p_under_fair = no_vig_probs(float(over_price), float(under_price))
    except Exception:
        return None

    sigma = SIGMA_BY_MARKET.get(market, 5.0)
    try:
        side, win_prob, edge_pct = edge_from_prediction(
            prediction, line, p_over_fair, p_under_fair, sigma=sigma
        )
    except ImportError:
        side = "OVER" if diff > 0 else "UNDER"
        win_prob = 0.55 if abs(diff) > 2 else 0.52
        edge_pct = (win_prob - 0.5) * 100

    if edge_pct < MIN_EDGE_PCT:
        return None

    price_col = float(over_price) if side == "OVER" else float(under_price)
    dec_odds = american_to_decimal(price_col)

    kf = kelly_fraction(win_prob, dec_odds, edge_pct=edge_pct)
    bet_dollars = round(kf * BANKROLL, 2)

    return {
        "player": player_name,
        "market": market,
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
    debug: bool = False,
    bookmaker_filter: str | None = None,
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
    max_exposure = MAX_TOTAL_EXPOSURE * bankroll

    print("=== NBA Props Screener ===")
    print(f"Bankroll: ${BANKROLL:.2f}  |  Min edge: {MIN_EDGE_PCT}%  |  Min diff: {MIN_LINE_DIFF} pts"
          + (f"  |  Bookmaker: {bookmaker_filter}" if bookmaker_filter else ""))

    # Load models for each market (auto-train if missing or stale)
    from model import _train_target, load_training_data as _ltd, train as _train, save_model as _save, is_model_stale
    models = {}
    for market, (feat_cols, target) in MARKET_CONFIG.items():
        if is_model_stale(target):
            print(f"[screener] model_{target} stale or feature mismatch — retraining ...")
            _train_target(target, do_eval=False, retrain=True)
        try:
            models[market] = load_model(target=target)
        except FileNotFoundError:
            print(f"[screener] model_{target}.pkl not found — training now ...")
            _train_target(target, do_eval=False, retrain=False)
            models[market] = load_model(target=target)
    print(f"[screener] Models loaded: {list(models.keys())}")

    # Load today's lines (all markets)
    lines_df = get_today_lines()
    if lines_df.empty:
        print("[screener] No lines available for today. Exiting.")
        return pd.DataFrame()

    # Filter to supported markets only
    supported_markets = set(MARKET_CONFIG.keys())
    if "market" in lines_df.columns:
        lines_df = lines_df[lines_df["market"].isin(supported_markets)]

    # Filter to specific bookmaker if requested
    if bookmaker_filter and "bookmaker" in lines_df.columns:
        mask = lines_df["bookmaker"].str.lower() == bookmaker_filter.lower()
        lines_df = lines_df[mask]
        if lines_df.empty:
            available = lines_df["bookmaker"].unique().tolist() if not lines_df.empty else []
            print(f"[screener] No lines found for bookmaker '{bookmaker_filter}'. Exiting.")
            return pd.DataFrame()

    print(f"[screener] {len(lines_df)} prop lines across {lines_df['market'].nunique() if 'market' in lines_df.columns else 1} market(s).")

    # Load all shared data once
    from features import build_defense_lookup
    df_history = load_gamelogs()
    def_lookup = build_defense_lookup()
    print(f"[screener] Data loaded: {len(df_history)} game rows, {len(def_lookup)} defense entries.")

    # Build name lookup: normalized → actual NBA name
    import re

    def _norm(name: str) -> str:
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = re.sub(r"\b([A-Z])\.([A-Z])\.", r"\1\2", name)
        name = re.sub(r"\b([A-Z])\.", r"\1", name)
        name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", lambda m: " " + m.group(1).rstrip("."), name, flags=re.IGNORECASE)
        return " ".join(name.strip().lower().split())

    nba_names = list(df_history["player_name"].unique())
    norm_to_nba = {_norm(n): n for n in nba_names}

    # Load active (unsettled) bets so we don't show bets we're already in
    active_bets: set[tuple[str, str]] = set()
    _tracker_path = RESULTS_DIR / "bet_tracker.csv"
    if _tracker_path.exists():
        try:
            _tracker = pd.read_csv(_tracker_path)
            _pending = _tracker[~_tracker["result"].isin(["WIN", "LOSS", "PUSH"])]
            for _, _row in _pending.iterrows():
                _p = str(_row.get("player", "")).strip()
                _m = str(_row.get("market", "")).strip()
                if _p and _m:
                    active_bets.add((_p, _m))
            if active_bets:
                print(f"[screener] {len(active_bets)} market(s) already in play (marked with *).")
        except Exception:
            pass

    results = []
    # seen key = (player_raw, market) so same player can have pts + reb + ast
    seen_keys = set()
    n_no_history = 0

    for _, row in lines_df.iterrows():
        player = row.get("player_name", "")
        market = row.get("market", "player_points")
        key = (player, market)
        if not player or key in seen_keys:
            continue
        seen_keys.add(key)

        line = row.get("line")
        if pd.isna(line):
            continue

        # Name resolution
        if player in norm_to_nba.values():
            nba_player = player
        else:
            nba_player = norm_to_nba.get(_norm(player))
            if nba_player is None:
                n_no_history += 1
                continue

        over_price = row.get("over_price")
        under_price = row.get("under_price")
        bookmaker = row.get("bookmaker", "unknown")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        commence_time = row.get("commence_time", "")

        _, target = MARKET_CONFIG.get(market, (None, "pts"))
        model = models.get(market)
        if model is None:
            continue

        result = screen_player(
            player_name=nba_player,
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
            target=target,
            market=market,
        )

        if result:
            nba_player_check = norm_to_nba.get(_norm(player), player)
            result["in_play"] = (nba_player_check, market) in active_bets
            results.append(result)

    unique_players = len({k[0] for k in seen_keys})
    print(f"[screener] Lines evaluated: {len(seen_keys)} | "
          f"Players matched: {unique_players} | "
          f"No history: {n_no_history} | "
          f"Flagged: {len(results)}")

    if debug:
        from model import predict as _pred
        game_date_str = date.today().isoformat()
        diag_rows = []
        for _, row in lines_df.iterrows():
            p_raw = row.get("player_name", "")
            mkt = row.get("market", "player_points")
            nba_p = norm_to_nba.get(_norm(p_raw)) if p_raw not in norm_to_nba.values() else p_raw
            if not nba_p:
                continue
            ln = row.get("line")
            if pd.isna(ln):
                continue
            _, tgt = MARKET_CONFIG.get(mkt, (None, "pts"))
            mdl = models.get(mkt)
            if mdl is None:
                continue
            feats = build_live_features(nba_p, "", False, game_date_str, df_history, def_lookup, target=tgt)
            if feats:
                pred = _pred(feats, mdl, target=tgt)
                diff = pred - float(ln)
                o_p = row.get("over_price")
                u_p = row.get("under_price")
                diag_rows.append((nba_p, mkt[-3:].upper(), float(ln), round(pred, 1), round(diff, 1),
                                  "" if pd.isna(o_p) else int(o_p),
                                  "" if pd.isna(u_p) else int(u_p)))

        if diag_rows:
            print("\n[DEBUG] Prediction vs Line (all matched players):")
            print(f"{'Player':<28} {'Mkt':>4} {'Line':>6} {'Pred':>6} {'Diff':>6} {'Over':>6} {'Under':>6}")
            print("-" * 68)
            for r in sorted(diag_rows, key=lambda x: abs(x[4]), reverse=True)[:30]:
                print(f"{r[0]:<28} {r[1]:>4} {r[2]:>6.1f} {r[3]:>6.1f} {r[4]:>+6.1f} {str(r[5]):>6} {str(r[6]):>6}")

    # Show unmatched names
    if n_no_history > 0:
        unmatched = [row.get("player_name", "") for _, row in lines_df.iterrows()
                     if row.get("player_name", "") and norm_to_nba.get(_norm(row.get("player_name", ""))) is None
                     and row.get("player_name", "") not in norm_to_nba.values()]
        unmatched = list(dict.fromkeys(unmatched))  # deduplicate
        if unmatched:
            print(f"[screener] Still unmatched (first 10): {unmatched[:10]}")

    if not results:
        print("[screener] No +EV bets found today.")
        return pd.DataFrame()

    bets_df = pd.DataFrame(results).sort_values("edge_pct", ascending=False).reset_index(drop=True)

    # Scale bets proportionally if total allocation exceeds MAX_TOTAL_EXPOSURE
    total_raw = bets_df["bet_dollars"].sum()
    if total_raw > max_exposure:
        scale = max_exposure / total_raw
        bets_df["bet_dollars"] = (bets_df["bet_dollars"] * scale).round(2)
        bets_df["kelly_pct"] = (bets_df["kelly_pct"] * scale).round(3)
        print(f"[screener] Scaled bets by {scale:.3f}x — total capped at ${max_exposure:.2f} ({MAX_TOTAL_EXPOSURE*100:.0f}% of bankroll)")

    # Save results
    today = date.today().isoformat()
    out_path = RESULTS_DIR / f"bets_{today}.csv"
    bets_df.to_csv(out_path, index=False)

    return bets_df


TRACKER_PATH = RESULTS_DIR / "bet_tracker.csv"


def prompt_update_results() -> None:
    """
    At startup: find unsettled bets in the tracker and prompt for WIN/LOSS/PUSH.
    """
    if not TRACKER_PATH.exists():
        return
    df = pd.read_csv(TRACKER_PATH)
    pending_mask = df["result"].isna() | (df["result"].astype(str).str.strip() == "")
    pending = df[pending_mask]
    if pending.empty:
        return

    print(f"\n{'=' * 60}")
    print(f"{len(pending)} unsettled bet(s) in tracker — update results?")
    pending_list = list(pending.iterrows())
    for i, (_, row) in enumerate(pending_list, 1):
        mkt = str(row.get("market", "")).replace("player_", "")
        odds_str = f"{int(row['odds']):+d}" if not pd.isna(row.get("odds")) else "?"
        book = str(row.get("bookmaker", "?"))
        print(f"  {i}. {row['date']} | {row['player']} | {mkt} | {row['side']} {row['line']} @ {odds_str} | {book} | ${float(row['entered_$']):.2f}")

    print("Enter row numbers to settle (e.g. 1,3) or Enter to skip:")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
    except ValueError:
        print("[tracker] Invalid input — skipping.")
        return

    result_map = {"W": "WIN", "L": "LOSS", "P": "PUSH", "WIN": "WIN", "LOSS": "LOSS", "PUSH": "PUSH"}
    for i in indices:
        if i < 0 or i >= len(pending_list):
            continue
        orig_idx, row = pending_list[i]
        mkt = str(row.get("market", "")).replace("player_", "")
        print(f"  {row['player']} | {mkt} | {row['side']} {row['line']}")
        try:
            raw_result = input("  Result (W/L/P): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break
        result = result_map.get(raw_result)
        if result:
            df.at[orig_idx, "result"] = result
            print(f"    → {result}")
        else:
            print(f"    Unrecognized '{raw_result}' — skipping.")

    df.to_csv(TRACKER_PATH, index=False)
    print(f"[tracker] Results saved → {TRACKER_PATH}")


def prompt_bankroll() -> float:
    """
    Ask for current cash balance, then add unsettled bet amounts to get total bankroll.
    Falls back to BANKROLL constant if skipped.
    """
    print(f"\n{'=' * 60}")
    print("Enter your current cash balance (or Enter to use default):")
    try:
        raw = input(f"  Cash balance [${BANKROLL:.2f}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return BANKROLL

    if not raw:
        cash = BANKROLL
    else:
        try:
            cash = float(raw.replace("$", "").replace(",", ""))
        except ValueError:
            print(f"  Invalid — using default ${BANKROLL:.2f}")
            cash = BANKROLL

    # Add unsettled bets (money currently at risk)
    at_risk = 0.0
    if TRACKER_PATH.exists():
        df = pd.read_csv(TRACKER_PATH)
        pending = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")]
        if not pending.empty and "entered_$" in pending.columns:
            at_risk = pending["entered_$"].fillna(0).sum()

    bankroll = cash + at_risk
    if at_risk > 0:
        print(f"  Cash: ${cash:.2f}  +  At risk (unsettled): ${at_risk:.2f}  =  Bankroll: ${bankroll:.2f}")
    else:
        print(f"  Bankroll: ${bankroll:.2f}")
    return bankroll


def prompt_bookmaker(lines_df: pd.DataFrame) -> str | None:
    """
    Show available bookmakers from today's lines and let the user pick one (or all).
    Returns the selected bookmaker string, or None for all books.
    """
    if "bookmaker" not in lines_df.columns:
        return None
    books = sorted(lines_df["bookmaker"].dropna().unique())
    if not books:
        return None

    print(f"\n{'=' * 60}")
    print("Available bookmakers:")
    for i, b in enumerate(books, 1):
        print(f"  {i}. {b}")
    print(f"  {len(books) + 1}. All bookmakers")
    print("Select (number or name, Enter for all):")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if idx == len(books):
            return None
        if 0 <= idx < len(books):
            return books[idx]
    except ValueError:
        matched = [b for b in books if raw.lower() in b.lower()]
        if len(matched) == 1:
            return matched[0]
        if matched:
            print(f"  Ambiguous: {matched} — using all.")
    return None


def prompt_and_log_bets(bets_df: pd.DataFrame) -> None:
    """
    Interactive: ask which flagged bets were placed and log them to the tracker.
    Appends to results/bet_tracker.csv with columns:
        date, player, market, line, side, odds, bookmaker, edge_pct,
        suggested_$, entered_$, result
    """
    if bets_df.empty:
        return

    print("\n" + "=" * 60)
    print("Which bets did you place?")
    for i, (_, row) in enumerate(bets_df.iterrows(), 1):
        mkt = str(row.get("market", "")).replace("player_", "")
        book = str(row.get("bookmaker", "?"))
        in_play = " *" if row.get("in_play") else ""
        print(f"  {i}. {row['player']}{in_play} | {mkt} | {row['side']} {row['line']} @ {int(row['odds']):+d} | {book} | ${row['bet_dollars']:.2f}")
    print("Enter row numbers (e.g. 1,3) or press Enter to skip:")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw:
        return

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
    except ValueError:
        print("[tracker] Invalid input — skipping.")
        return

    logged = []
    today = date.today().isoformat()
    for idx in indices:
        if idx < 0 or idx >= len(bets_df):
            print(f"  [skip] Row {idx + 1} out of range.")
            continue
        row = bets_df.iloc[idx]
        label = f"  {row['player']} | {row.get('market','').replace('player_','')} | {row['side']} {row['line']} @ {int(row['odds']):+d}"
        print(label)
        try:
            amt_raw = input(f"    Amount bet ($, or Enter to use suggested ${row['bet_dollars']:.2f}): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        entered = float(amt_raw) if amt_raw else row["bet_dollars"]
        logged.append({
            "date": today,
            "player": row["player"],
            "market": row.get("market", ""),
            "line": row["line"],
            "side": row["side"],
            "odds": row["odds"],
            "bookmaker": row.get("bookmaker", ""),
            "edge_pct": row["edge_pct"],
            "suggested_$": row["bet_dollars"],
            "entered_$": entered,
            "result": "",  # fill in later: WIN / LOSS / PUSH
        })

    if not logged:
        return

    new_df = pd.DataFrame(logged)
    if TRACKER_PATH.exists():
        existing = pd.read_csv(TRACKER_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(TRACKER_PATH, index=False)
    print(f"\n[tracker] {len(logged)} bet(s) logged → {TRACKER_PATH}")
    print("  Update the 'result' column with WIN / LOSS / PUSH after games settle.")


def format_output(df: pd.DataFrame) -> str:
    """Format bets DataFrame as a readable table."""
    if df.empty:
        return "No +EV bets found for today."

    display = df.copy()
    if "in_play" in display.columns:
        display["player"] = display.apply(
            lambda r: f"* {r['player']}" if r.get("in_play") else r["player"], axis=1
        )
    display["edge%"] = display["edge_pct"].apply(lambda x: f"+{x:.1f}%")
    display["kelly%"] = display["kelly_pct"].apply(lambda x: f"{x:.2f}%")
    display["bet_$"] = display["bet_dollars"].apply(lambda x: f"${x:.2f}")
    display["odds"] = display["odds"].apply(lambda x: f"{int(x):+d}" if not pd.isna(x) else "N/A")
    display["prediction"] = display["prediction"].apply(lambda x: f"{x:.1f}")

    cols = ["player", "market", "line", "prediction", "edge%", "side", "odds", "bookmaker", "kelly%", "bet_$", "game"]
    avail = [c for c in cols if c in display.columns]
    return display[avail].to_string(index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Props Screener")
    parser.add_argument("--bankroll", type=float, default=BANKROLL, help="Current bankroll in $")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT, help="Minimum edge %% to flag")
    parser.add_argument("--min-diff", type=float, default=MIN_LINE_DIFF, help="Min |prediction - line| pts")
    parser.add_argument("--debug", action="store_true", help="Print prediction vs line debug table")
    bk_group = parser.add_mutually_exclusive_group()
    bk_group.add_argument("--draftkings", action="store_true", help="Only show DraftKings lines (skips prompt)")
    bk_group.add_argument("--bookmaker", type=str, default=None, help="Filter to a specific bookmaker (skips prompt)")
    args = parser.parse_args()

    # Settle unsettled bets from previous sessions first
    prompt_update_results()

    # Bankroll: CLI --bankroll overrides prompt (useful for scripts/cron)
    if args.bankroll != BANKROLL:
        bankroll = args.bankroll
    else:
        bankroll = prompt_bankroll()

    # Bookmaker selection: CLI flag takes precedence, otherwise interactive prompt
    if args.draftkings:
        bookmaker = "draftkings"
    elif args.bookmaker:
        bookmaker = args.bookmaker
    else:
        # Fetch lines once just to show available books for the prompt
        from odds import get_today_lines as _get_lines
        _lines_preview = _get_lines()
        bookmaker = prompt_bookmaker(_lines_preview)

    bets = run_screener(
        bankroll=bankroll,
        min_edge=args.min_edge,
        min_diff=args.min_diff,
        debug=args.debug,
        bookmaker_filter=bookmaker,
    )

    print("\n" + "=" * 90)
    print("TODAY'S FLAGGED BETS")
    print("=" * 90)
    print(format_output(bets))

    if not bets.empty:
        total_bet = bets["bet_dollars"].sum()
        print(f"\nTotal allocated: ${total_bet:.2f} / ${bankroll:.2f} ({total_bet/bankroll*100:.1f}%)")
        print(f"Bets flagged: {len(bets)}")
        print(f"\nResults saved to: results/bets_{date.today().isoformat()}.csv")

        # Show arbitrage opportunities if any exist
        try:
            from odds import get_arbitrage_opportunities
            arb_df = get_arbitrage_opportunities()
            if not arb_df.empty:
                print("\n" + "=" * 90)
                print("ARBITRAGE OPPORTUNITIES (bet both sides across books)")
                print("=" * 90)
                print(arb_df.to_string(index=False))
        except Exception:
            pass

        prompt_and_log_bets(bets)

    print("\n[screener.py] Done.")
