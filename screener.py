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
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from data import load_gamelogs
from features import build_live_features, MARKET_CONFIG
from model import load_model, predict
from odds import get_today_lines, american_to_decimal, implied_probability


def _game_date_local(commence_time: str) -> str:
    """Convert UTC commence_time ISO string to local calendar date (YYYY-MM-DD)."""
    ct = (commence_time or "").strip()
    if not ct:
        return date.today().isoformat()
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d")
    except Exception:
        return ct[:10] if len(ct) >= 10 else date.today().isoformat()

# ── Config ────────────────────────────────────────────────────────────────────
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")   # set in .env to enable push notifications via ntfy.sh
BANKROLL = float(os.getenv("BANKROLL", "100"))        # starting bankroll in $
DEPOSIT  = float(os.getenv("DEPOSIT",  "0"))          # total cash deposited (for net-profit display)
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "4"))  # minimum edge % to flag
MIN_LINE_DIFF = float(os.getenv("MIN_LINE_DIFF", "1.5"))  # minimum |pred - line| pts
MAX_KELLY_FRACTION = float(os.getenv("MAX_KELLY_FRACTION", "0.05"))  # cap at 5% per bet
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "1.0"))  # max total bankroll % across all bets
MAX_BETS = int(os.getenv("MAX_BETS", "20"))            # max bets to show/recommend (top N by edge)
MAX_BETS_PER_PLAYER = int(os.getenv("MAX_BETS_PER_PLAYER", "2"))  # max props per player (0 = no limit)
MIN_GAMES = int(os.getenv("MIN_GAMES", "20"))           # min career games in DB before flagging
MIN_SEASON_GAMES = int(os.getenv("MIN_SEASON_GAMES", "15"))  # min games in current season (Oct–present)
CURRENT_SEASON_START = "2025-10-01"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
STATE_PATH = RESULTS_DIR / "state.json"
BALANCE_LOG_PATH = RESULTS_DIR / "balance_log.csv"
ADJUSTMENTS_PATH = RESULTS_DIR / "adjustments.csv"
DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm"]
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "60"))          # seconds between screener runs
LOOP_PRINT_EVERY = int(os.getenv("LOOP_PRINT_EVERY", "5"))     # print timestamp every N iterations
LINES_REFRESH_SECS = int(os.getenv("LINES_REFRESH_SECS", "900"))  # re-fetch odds API every 15 min

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
    Half-Kelly criterion: f* = 0.5 * (b*p - q) / b
    where b = decimal_odds - 1, p = win_prob, q = 1 - p.
    Capped at MAX_KELLY_FRACTION per bet.
    """
    if edge_pct <= 0 or win_prob <= 0:
        return 0.0
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - win_prob
    kelly = (b * win_prob - q) / b
    return max(0.0, min(kelly * 0.5, MAX_KELLY_FRACTION))


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
    game_date = _game_date_local(commence_time)

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

    # Use residual sigma from model training if available (data-driven), else fall back
    sigma = getattr(model, "residual_sigma_", None) or SIGMA_BY_MARKET.get(market, 5.0)
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
    bet_dollars = round(kf * BANKROLL)

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
    bookmaker_filter: list[str] | str | None = None,
    lines_df: "pd.DataFrame | None" = None,
    book_tradeable: "dict[str, float] | None" = None,
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

    # Normalize bookmaker_filter to a list or None
    if isinstance(bookmaker_filter, str):
        bookmaker_filter = [bookmaker_filter]

    print("=== NBA Props Screener ===")
    bk_label = ", ".join(bookmaker_filter) if bookmaker_filter else "all"
    print(f"Bankroll: ${BANKROLL:.2f}  |  Min edge: {MIN_EDGE_PCT}%  |  Min diff: {MIN_LINE_DIFF} pts  |  Books: {bk_label}")

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

    # Load today's lines (all markets) — reuse passed-in data to avoid double API fetch
    if lines_df is None:
        lines_df = get_today_lines()
    if lines_df.empty:
        print("[screener] No lines available for today. Exiting.")
        return pd.DataFrame()

    # Filter to supported markets only
    supported_markets = set(MARKET_CONFIG.keys())
    if "market" in lines_df.columns:
        lines_df = lines_df[lines_df["market"].isin(supported_markets)]

    # Filter to specific bookmaker(s) if requested (active_books toggle only — balance does not hide bets)
    if bookmaker_filter and "bookmaker" in lines_df.columns:
        lower_filter = [b.lower() for b in bookmaker_filter]
        mask = lines_df["bookmaker"].str.lower().isin(lower_filter)
        lines_df = lines_df[mask]
        if lines_df.empty:
            print(f"[screener] No lines found for bookmaker(s): {bookmaker_filter}. Exiting.")
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
    # seen key = (player_raw, market, bookmaker) — evaluate each book's line independently
    seen_keys = set()
    n_no_history = 0

    for _, row in lines_df.iterrows():
        player = row.get("player_name", "")
        market = row.get("market", "player_points")
        bookmaker_key = row.get("bookmaker", "unknown")
        key = (player, market, bookmaker_key)
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

    unique_players = len({k[0] for k in seen_keys})  # k = (player, market, bookmaker)
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

    # Per-player cap: take top N props per player by edge (prevents correlated loss)
    if MAX_BETS_PER_PLAYER > 0:
        bets_df = (
            bets_df.groupby("player", group_keys=False)
            .apply(lambda g: g.head(MAX_BETS_PER_PLAYER), include_groups=False)
            .sort_values("edge_pct", ascending=False)
            .reset_index(drop=True)
        )

    # Cap to top N bets FIRST so scaling is based only on shown bets
    if MAX_BETS > 0:
        bets_df = bets_df.head(MAX_BETS).reset_index(drop=True)

    # Scale bets proportionally if total allocation exceeds MAX_TOTAL_EXPOSURE
    total_raw = bets_df["bet_dollars"].sum()
    if total_raw > max_exposure:
        scale = max_exposure / total_raw
        bets_df["bet_dollars"] = (bets_df["bet_dollars"] * scale).round(0)
        bets_df["kelly_pct"] = (bets_df["kelly_pct"] * scale).round(3)
        print(f"[screener] Scaled bets by {scale:.3f}x — total capped at ${max_exposure:.2f} ({MAX_TOTAL_EXPOSURE*100:.0f}% of bankroll)")

    # Drop bets that round to $0 — not actionable
    bets_df = bets_df[bets_df["bet_dollars"] >= 1].reset_index(drop=True)

    # Save results
    today = date.today().isoformat()
    out_path = RESULTS_DIR / f"bets_{today}.csv"
    bets_df.to_csv(out_path, index=False)

    return bets_df


TRACKER_PATH = RESULTS_DIR / "bet_tracker.csv"

# market key → gamelog stat column
_MARKET_STAT = {
    "player_points":    "pts",
    "player_rebounds":  "reb",
    "player_assists":   "ast",
    "player_threes":    "fg3m",
    "player_blocks":    "blk",
    "player_steals":    "stl",
    "player_turnovers": "tov",
}


def auto_settle_bets(already_reported: set | None = None) -> list[str]:
    """
    Automatically settle unsettled bets whose game date is in the past.
    Updates bet_tracker.csv and book balances in state.
    Returns a list of human-readable result lines (caller decides whether to print).
    """
    if not TRACKER_PATH.exists():
        return []

    df = pd.read_csv(TRACKER_PATH)
    pending_mask = df["result"].isna() | (df["result"].astype(str).str.strip() == "")
    pending = df[pending_mask].copy()
    if pending.empty:
        return []

    today = date.today().isoformat()
    past = pending[pending["date"] <= today]
    if past.empty:
        return []

    try:
        import contextlib as _cl, io as _io
        from data import load_gamelogs, fetch_player_gamelogs, CURRENT_SEASON, DATA_DIR
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            logs = load_gamelogs()
        logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
        logs["player_name_lower"] = logs["player_name"].str.lower().str.strip()

        # If any past bets are missing from the loaded logs the cache may be stale
        # (games finished after the last refresh). Force-refresh once — only if the
        # cache itself is older than 30 min so we don't hammer the API every tick.
        past_dates = set(past["date"].astype(str))
        logged_dates = {str(d) for d in logs["game_date"]}
        if past_dates - logged_dates:
            cache_path = DATA_DIR / f"gamelogs_{CURRENT_SEASON.replace('-', '_')}.csv"
            cache_age_min = (time.time() - cache_path.stat().st_mtime) / 60 if cache_path.exists() else 999
            if cache_age_min > 30:
                if cache_path.exists():
                    cache_path.unlink()
                with _cl.redirect_stdout(_buf):
                    logs = load_gamelogs()
                logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
                logs["player_name_lower"] = logs["player_name"].str.lower().str.strip()
    except Exception as e:
        return [f"[auto-settle] Could not load game logs: {e}"]

    messages: list[str] = []
    not_found: list[str] = []

    for orig_idx, row in past.iterrows():
        market = str(row.get("market", ""))
        stat_col = _MARKET_STAT.get(market)
        if stat_col is None:
            not_found.append(f"{row['player']} ({market} unknown)")
            continue

        game_date = row["date"]
        player_lower = str(row["player"]).lower().strip()
        line = float(row["line"])
        side = str(row["side"]).upper()

        try:
            game_date_obj = date.fromisoformat(game_date)
        except ValueError:
            not_found.append(f"{row['player']} (bad date {game_date})")
            continue

        match = logs[
            (logs["player_name_lower"] == player_lower) &
            (logs["game_date"] == game_date_obj)
        ]

        if match.empty:
            key = f"{row['player']} ({game_date})"
            if already_reported is None or key not in already_reported:
                not_found.append(key)
                if already_reported is not None:
                    already_reported.add(key)
            continue

        actual = float(match.iloc[0][stat_col])
        if actual > line:
            result = "WIN" if side == "OVER" else "LOSS"
        elif actual < line:
            result = "LOSS" if side == "OVER" else "WIN"
        else:
            result = "PUSH"

        df.at[orig_idx, "result"] = result
        df.at[orig_idx, "settled_date"] = date.today().isoformat()

        book = str(row.get("bookmaker", "")).lower()
        stake = float(row.get("entered_$", 0))
        if result == "WIN":
            dec = american_to_decimal(float(row["odds"]))
            delta = stake * dec  # stake returned + profit
        elif result == "PUSH":
            delta = stake
        else:
            delta = 0.0
        new_tradeable = _update_book_balance(book, delta)

        mkt_short = market.replace("player_", "")
        messages.append(f"  [settled] {row['player']} | {mkt_short} | {side} {line} → actual {actual} → {result}  |  {book} ${new_tradeable:.0f}")

    if messages:
        df.to_csv(TRACKER_PATH, index=False)

    if not_found:
        messages.append(f"  [not found] {', '.join(not_found)}")

    return messages


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
            book = str(row.get("bookmaker", "")).lower()
            stake = float(row.get("entered_$", 0))
            if result == "WIN":
                dec = american_to_decimal(float(row["odds"]))
                delta = stake * (dec - 1)
            elif result == "PUSH":
                delta = stake
            else:
                delta = 0.0
            new_tradeable = _update_book_balance(book, delta)
            print(f"    → {result}  |  {book} tradeable ${new_tradeable:.0f}")
        else:
            print(f"    Unrecognized '{raw_result}' — skipping.")

    df.to_csv(TRACKER_PATH, index=False)
    print(f"[tracker] Results saved → {TRACKER_PATH}")


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            import json as _json
            return _json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_state(data: dict) -> None:
    import json as _json
    existing = _load_state()
    existing.update(data)
    STATE_PATH.write_text(_json.dumps(existing, indent=2))


def _get_book_balances() -> dict[str, float]:
    """Return tradeable balance per book from state (excludes at-risk pending bets)."""
    return _load_state().get("book_balances", {})


def _update_book_balance(book: str, delta: float) -> float:
    """Add delta to a book's tradeable balance. Returns the new tradeable value."""
    balances = _get_book_balances()
    balances[book] = balances.get(book, 0.0) + delta
    _save_state({"book_balances": balances})
    return balances[book]


def _at_risk_per_book() -> dict[str, float]:
    """Sum of entered_$ for unsettled bets, keyed by bookmaker."""
    if not TRACKER_PATH.exists():
        return {}
    try:
        df = pd.read_csv(TRACKER_PATH)
        pending = df[df["result"].isna() | (df["result"].astype(str).str.strip() == "")]
        if pending.empty or "entered_$" not in pending.columns:
            return {}
        return pending.groupby("bookmaker")["entered_$"].sum().to_dict()
    except Exception:
        return {}


def _total_bankroll(book_balances: dict[str, float]) -> float:
    """Total capital = tradeable across all books + at-risk pending bets."""
    at_risk = _at_risk_per_book()
    total = sum(book_balances.values())
    total += sum(at_risk.get(b, 0.0) for b in book_balances)
    return total


def _calc_pnl(since_date: str | None = None) -> float:
    """
    Calculate realised PnL from settled bets in the tracker.
    Pass since_date (ISO string, e.g. '2026-03-14') to restrict to that day.
    Returns net profit/loss in dollars.
    """
    if not TRACKER_PATH.exists():
        return 0.0
    try:
        df = pd.read_csv(TRACKER_PATH)
        settled = df[df["result"].astype(str).str.strip().isin(["WIN", "LOSS", "PUSH"])]
        if since_date:
            settled = settled[settled["date"].astype(str).str.strip() == since_date]
        if settled.empty:
            return 0.0
        pnl = 0.0
        for _, row in settled.iterrows():
            result = str(row["result"]).strip().upper()
            amt = float(row.get("entered_$", 0) or 0)
            odds = float(row.get("odds", 0) or 0)
            if result == "WIN":
                pnl += amt * (odds / 100) if odds > 0 else amt * (100 / abs(odds))
            elif result == "LOSS":
                pnl -= amt
            # PUSH: no change
        return pnl
    except Exception:
        return 0.0


def _log_balance(book_balances: dict) -> None:
    """Append today's total balance to the balance log (one row per day, last wins)."""
    try:
        total = _total_bankroll(book_balances)
        today = date.today().isoformat()
        row = f"{today},{total:.2f}\n"
        if BALANCE_LOG_PATH.exists():
            lines = BALANCE_LOG_PATH.read_text().splitlines(keepends=True)
            lines = [l for l in lines if not l.startswith(today + ",")]
        else:
            lines = ["date,balance\n"]
        lines.append(row)
        BALANCE_LOG_PATH.write_text("".join(lines))
    except Exception:
        pass


def _pnl_str() -> str:
    """Return a compact 'Today $X.XX  |  Overall $X.XX' PnL string."""
    today = date.today().isoformat()
    daily = _calc_pnl(since_date=today)
    overall = _calc_pnl()
    d_sign = "+" if daily >= 0 else ""
    o_sign = "+" if overall >= 0 else ""
    return f"Today {d_sign}${daily:.0f}  |  Overall {o_sign}${overall:.0f}"


def prompt_book_balances() -> dict[str, float]:
    """
    Ask the user for their current tradeable balance on each default book.
    Tradeable = money available to bet right now (not including pending bets).
    Saves to state and returns the dict.
    """
    saved = _get_book_balances()
    balances: dict[str, float] = {}
    print(f"\n{'=' * 60}")
    print("Enter tradeable balance for each sportsbook (money available to bet now):")
    for book in DEFAULT_BOOKS:
        default = saved.get(book, 0.0)
        try:
            raw = input(f"  {book.title()} [${default:.0f}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw:
            try:
                balances[book] = float(raw.replace("$", "").replace(",", ""))
            except ValueError:
                print(f"    Invalid — using ${default:.0f}")
                balances[book] = default
        else:
            balances[book] = default
    _save_state({"book_balances": balances})

    at_risk = _at_risk_per_book()
    print()
    for book in DEFAULT_BOOKS:
        risk = at_risk.get(book, 0.0)
        total = balances[book] + risk
        print(f"  {book.title()}: ${balances[book]:.0f} tradeable + ${risk:.0f} at risk = ${total:.0f}")
    total_br = _total_bankroll(balances)
    print(f"  Total bankroll: ${total_br:.0f}")
    return balances


def prompt_bookmaker(lines_df: pd.DataFrame) -> list[str] | None:
    """
    Show available bookmakers from today's lines and let the user pick one or more (or all).
    Returns a list of selected bookmaker strings, or None for all books.
    Supports comma-separated numbers (e.g. "1,3") or a single name.
    """
    if "bookmaker" not in lines_df.columns:
        return None
    books = sorted(lines_df["bookmaker"].dropna().unique())
    if not books:
        return None

    print(f"\n{'=' * 60}")
    print("Available bookmakers (enter number(s), e.g. 1,3):")
    for i, b in enumerate(books, 1):
        print(f"  {i}. {b}")
    print(f"  {len(books) + 1}. All bookmakers")
    print("Select (Enter for all):")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None

    # Try comma-separated numbers first
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    selected = []
    all_numeric = all(p.isdigit() for p in parts)
    if all_numeric:
        for p in parts:
            idx = int(p) - 1
            if idx == len(books):
                return None  # "All"
            if 0 <= idx < len(books):
                selected.append(books[idx])
            else:
                print(f"  [skip] {p} out of range.")
        return selected if selected else None

    # Fall back to name matching (single token)
    matched = [b for b in books if raw.lower() in b.lower()]
    if len(matched) == 1:
        return [matched[0]]
    if matched:
        print(f"  Ambiguous: {matched} — using all.")
    return None


def _print_single_bet(row, label: str = "") -> None:
    mkt = str(row.get("market", "")).replace("player_", "")
    book = str(row.get("bookmaker", "?"))
    in_play = " *IN-PLAY*" if row.get("in_play") else ""
    prefix = f"[{label}]  " if label else ""
    print(f"\n{'─'*60}", flush=True)
    print(f"{prefix}{row['player']}{in_play} | {mkt} | {row['side']} {row['line']} @ {int(row['odds']):+d} | {book}", flush=True)
    print(f"Suggested: ${int(row['bet_dollars'])}   — place it? (y / y $AMT / n):", flush=True)


def _print_bet_list(bets_df: pd.DataFrame) -> None:
    """Print numbered bet list (shared by prompt and loop input thread)."""
    print("\n" + "=" * 60)
    print("Recent bets — enter slip to log  (e.g.  1 $10 2 $5):")
    for i, (_, row) in enumerate(bets_df.iterrows(), 1):
        mkt = str(row.get("market", "")).replace("player_", "")
        book = str(row.get("bookmaker", "?"))
        in_play = " *" if row.get("in_play") else ""
        print(f"  {i}. {row['player']}{in_play} | {mkt} | {row['side']} {row['line']} @ {int(row['odds']):+d} | {book} | ${int(row['bet_dollars'])}")


def _log_slip(bets_df: pd.DataFrame, raw: str) -> None:
    """Parse slip string and log matched bets to the tracker."""
    import re as _re
    tokens = _re.sub(r"[$,]", " ", raw).split()
    slip: dict[int, float] = {}
    pending_rows: list[int] = []  # numbers seen before an amount
    for tok in tokens:
        try:
            amt = float(tok)
            # Apply this amount to all pending row numbers
            for row_num in pending_rows:
                slip[row_num] = amt
            pending_rows = []
        except ValueError:
            if tok.isdigit():
                pending_rows.append(int(tok))

    if not slip:
        print("[tracker] Could not parse slip — skipping.", flush=True)
        return

    logged = []
    for row_num, entered in sorted(slip.items()):
        idx = row_num - 1
        if idx < 0 or idx >= len(bets_df):
            print(f"  [skip] Row {row_num} out of range.", flush=True)
            continue
        row = bets_df.iloc[idx]
        _game_date = _game_date_local(str(row.get("commence_time", "") or ""))
        print(f"  {row['player']} | {row.get('market','').replace('player_','')} | {row['side']} {row['line']} @ {int(row['odds']):+d} — ${entered:.0f}", flush=True)
        logged.append({
            "date": _game_date,
            "player": row["player"],
            "market": row.get("market", ""),
            "line": row["line"],
            "side": row["side"],
            "odds": row["odds"],
            "bookmaker": row.get("bookmaker", ""),
            "edge_pct": row["edge_pct"],
            "suggested_$": row["bet_dollars"],
            "entered_$": entered,
            "result": "",
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
    for b in logged:
        _update_book_balance(b["bookmaker"].lower(), -b["entered_$"])
    balances = _get_book_balances()
    summary = "  ".join(
        f"{book.title()} ${balances.get(book, 0):.0f} avail"
        for book in DEFAULT_BOOKS
    )
    print(f"\n[tracker] {len(logged)} bet(s) logged  |  {summary}", flush=True)


def prompt_and_log_bets(bets_df: pd.DataFrame) -> None:
    """Interactive one-shot prompt (used outside of loop mode)."""
    if bets_df.empty:
        return
    _print_bet_list(bets_df)
    print("Enter slip (row# $amount ...), e.g.  1 $10 2 $5 3 $15  — or press Enter to skip:")
    try:
        raw = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return
    _log_slip(bets_df, raw)


def _ntfy(title: str, body: str, priority: str = "default", tags: str = "") -> None:
    """Send a notification via ntfy.sh. No-op if NTFY_TOPIC is not set."""
    if not NTFY_TOPIC:
        return
    try:
        import requests as _req
        headers = {"Title": title, "Priority": priority}
        if tags:
            headers["Tags"] = tags
        _req.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=body.encode("utf-8"), headers=headers, timeout=10)
    except Exception as _e:
        print(f"[notify] Failed: {_e}", flush=True)


def _notify(new_bets: pd.DataFrame) -> None:
    """Push a new-bets notification via ntfy.sh."""
    lines = []
    for _, row in new_bets.iterrows():
        mkt = str(row.get("market", "")).replace("player_", "")
        book = str(row.get("bookmaker", ""))
        lines.append(f"{row['player']} {row['side']} {row['line']} {mkt} @ {int(row['odds']):+d} ({book}) ${int(row['bet_dollars'])}")
    _ntfy(f"{len(new_bets)} new bet(s) flagged", "\n".join(lines), priority="high", tags="money_with_wings")


def format_output(df: pd.DataFrame) -> str:
    """Format bets DataFrame as a readable table."""
    if df.empty:
        return "No +EV bets found for today."

    display = df.copy()
    if "in_play" in display.columns:
        display["player"] = display.apply(
            lambda r: f"* {r['player']}" if r.get("in_play") else r["player"], axis=1
        )
    display["bet_$"] = display["bet_dollars"].apply(lambda x: f"${int(x)}")
    display["odds"] = display["odds"].apply(lambda x: f"{int(x):+d}" if not pd.isna(x) else "N/A")
    display["market"] = display["market"].apply(lambda x: str(x).replace("player_", ""))

    cols = ["bookmaker", "player", "market", "line", "side", "odds", "bet_$", "game"]
    avail = [c for c in cols if c in display.columns]
    return display[avail].to_string(index=False)


def _bet_key(r) -> tuple:
    """Normalised key so screener rows and tracker CSV rows compare equal."""
    return (
        str(r["player"]).strip(),
        str(r["market"]).strip(),
        str(r["side"]).strip().upper(),
        float(r["line"]),
        str(r["bookmaker"]).strip().lower() if "bookmaker" in r and r["bookmaker"] == r["bookmaker"] else "",
    )


def _position(r) -> tuple:
    return (str(r["player"]).strip(), str(r["market"]).strip(), str(r["side"]).strip().upper())


if __name__ == "__main__":
    import contextlib
    import time as _time

    parser = argparse.ArgumentParser(description="NBA Props Screener")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT, help="Minimum edge %% to flag")
    parser.add_argument("--min-diff", type=float, default=MIN_LINE_DIFF, help="Min |prediction - line| pts")
    parser.add_argument("--debug", action="store_true", help="Print prediction vs line debug table")
    parser.add_argument("--interval", type=int, default=LOOP_INTERVAL, help="Seconds between screener runs")
    args = parser.parse_args()

    bookmaker = DEFAULT_BOOKS  # draftkings + fanduel

    # One-time startup: auto-settle any past bets from game logs
    for _msg in auto_settle_bets():
        print(_msg)

    # Load saved balances (edit on the dashboard)
    book_balances = _get_book_balances()
    if not book_balances:
        book_balances = {book: 0.0 for book in DEFAULT_BOOKS}

    import threading as _threading

    # Mutable shared state (avoids nonlocal in module-level scope)
    _st = {
        "prev_unplaced_keys": set(),
        "notified_not_found": set(),
        "iteration": 0,
        "book_balances": book_balances,
        "latest_bets": pd.DataFrame(),
        "placed_positions": set(),   # positions already in tracker
        "skipped_keys": set(),       # keys declined this session
        "lines_cache": None,         # cached pd.DataFrame from last API fetch
        "lines_fetched_at": 0.0,     # epoch seconds of last fetch
        # dashboard-settable config (dashboard may override these at runtime)
        "min_edge": args.min_edge,
        "min_diff": args.min_diff,
        "interval": args.interval,
        "active_books": list(bookmaker),
    }
    _latest_lock = _threading.Lock()

    from dashboard import start_dashboard, broadcast_state as _broadcast
    start_dashboard(_st, _latest_lock)

    print(f"\n[screener] Running — check every {args.interval}s  |  Ctrl-C to stop  |  r + Enter to force odds refresh", flush=True)
    _ntfy("Screener started", f"Checking every {args.interval}s\nDK ${book_balances.get('draftkings', 0):.0f}  FD ${book_balances.get('fanduel', 0):.0f}  MGM ${book_balances.get('betmgm', 0):.0f}", tags="white_check_mark")

    from datetime import datetime as _dt

    _stop = _threading.Event()

    def _screener_loop():
        try:
            while not _stop.is_set():
                _st["iteration"] += 1

                try:
                    _settled_msgs = list(auto_settle_bets(already_reported=_st["notified_not_found"]))
                    for _msg in _settled_msgs:
                        print(_msg, flush=True)
                    if _settled_msgs:
                        print(f"  [pnl] {_pnl_str()}", flush=True)
                    _st["book_balances"] = _get_book_balances()

                    bankroll = _total_bankroll(_st["book_balances"]) or BANKROLL

                    import io as _io, time as _time
                    _now = _time.monotonic()
                    _cache_age = _now - _st["lines_fetched_at"]
                    if _st["lines_cache"] is None or _cache_age >= LINES_REFRESH_SECS:
                        _fetch_buf = _io.StringIO()
                        with contextlib.redirect_stdout(_fetch_buf):
                            from odds import get_today_lines as _gtl
                            _fresh_lines = _gtl()
                        _fetch_out = _fetch_buf.getvalue().strip()
                        ts = _dt.now().strftime('%H:%M:%S')
                        if _fresh_lines.empty:
                            print(f"[{ts}] lines refresh — no lines returned", flush=True)
                            for _l in _fetch_out.splitlines():
                                print(f"  {_l}", flush=True)
                        else:
                            print(f"[{ts}] lines refresh — {len(_fresh_lines)} props fetched", flush=True)
                        _st["lines_cache"] = _fresh_lines
                        _st["lines_fetched_at"] = _now

                    _buf = _io.StringIO()
                    with contextlib.redirect_stdout(_buf):
                        bets = run_screener(
                            bankroll=bankroll,
                            min_edge=_st.get("min_edge", args.min_edge),
                            min_diff=_st.get("min_diff", args.min_diff),
                            debug=args.debug,
                            bookmaker_filter=_st.get("active_books", bookmaker),
                            book_tradeable=_st["book_balances"],
                            lines_df=_st["lines_cache"].copy() if _st["lines_cache"] is not None else None,
                        )
                    _screener_out = _buf.getvalue().strip()
                    if args.debug:
                        print(_screener_out, flush=True)
                    elif bets.empty and "No lines" in _screener_out:
                        ts = _dt.now().strftime('%H:%M:%S')
                        print(f"[{ts}] No lines returned — check API", flush=True)

                    placed_positions: set = set()
                    if TRACKER_PATH.exists():
                        _tr = pd.read_csv(TRACKER_PATH)
                        _pending = _tr[_tr["result"].isna() | (_tr["result"].astype(str).str.strip() == "")]
                        for _, _r in _pending.iterrows():
                            placed_positions.add((
                                str(_r["player"]).strip(),
                                str(_r["market"]).strip(),
                                str(_r["side"]).strip().upper(),
                            ))

                    unplaced_keys = {
                        _bet_key(row) for _, row in bets.iterrows()
                        if _position(row) not in placed_positions
                    } if not bets.empty else set()

                    if not bets.empty and unplaced_keys:
                        unplaced_bets = bets[
                            [_bet_key(row) in unplaced_keys for _, row in bets.iterrows()]
                        ].reset_index(drop=True)
                    else:
                        unplaced_bets = pd.DataFrame()

                    new_keys = unplaced_keys - _st["prev_unplaced_keys"]
                    if new_keys:
                        new_bets = bets[[_bet_key(row) in new_keys for _, row in bets.iterrows()]]
                        _notify(new_bets)
                        _st["book_balances"] = _get_book_balances()

                    _st["prev_unplaced_keys"] = unplaced_keys

                    with _latest_lock:
                        _st["latest_bets"] = unplaced_bets
                        _st["placed_positions"] = placed_positions
                    _log_balance(_st["book_balances"])
                    _broadcast()

                    if _st["iteration"] % LOOP_PRINT_EVERY == 0:
                        ts = _dt.now().strftime('%H:%M:%S')
                        n = len(unplaced_bets) if not unplaced_bets.empty else 0
                        print(f"[{ts}] tick {_st['iteration']} — {n} bet{'s' if n != 1 else ''} available", flush=True)

                except Exception as _e:
                    import traceback
                    print(f"[error] {_e}", flush=True)
                    traceback.print_exc()

                _stop.wait(_st.get("interval", args.interval))

        except Exception as _fatal:
            import traceback
            msg = f"{type(_fatal).__name__}: {_fatal}"
            print(f"[fatal] {msg}", flush=True)
            traceback.print_exc()
            _ntfy("Screener crashed", msg, priority="urgent", tags="rotating_light")
            _stop.set()

    _loop_thread = _threading.Thread(target=_screener_loop, daemon=True)
    _loop_thread.start()

    # Main thread: keep alive, handle r=refresh and Ctrl-C
    try:
        while not _stop.is_set():
            try:
                raw = input().strip().lower()
            except EOFError:
                break
            if raw in ("r", "refresh"):
                with _latest_lock:
                    _st["lines_fetched_at"] = 0.0
                print("[screener] Odds refresh queued.", flush=True)
    except KeyboardInterrupt:
        pass

    _stop.set()
    print("\n[screener] Stopped.")

