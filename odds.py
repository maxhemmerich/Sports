"""
odds.py — Pull today's NBA player prop lines from Pinnacle and bet365
via The Odds API.  Stores results locally and exposes helper functions
for the screener.

Endpoints used:
  GET /v4/sports/basketball_nba/events
  GET /v4/sports/basketball_nba/events/{event_id}/odds
       ?regions=us&markets=player_points,player_rebounds,player_assists,player_threes,...
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
# Preferred books — used for display and best-price selection.
# The free API tier restricts player props to US books (DraftKings, FanDuel etc).
# We fetch all available books and prefer PREFERRED_BOOKS if present.
PREFERRED_BOOKS = {"pinnacle", "bet365"}
REGIONS = "us"          # free tier: us region has player props coverage
MARKETS = "player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_turnovers"

# ── SharpAPI config (optional) ────────────────────────────────────────────────
# SharpAPI provides Pinnacle + bet365 lines with better coverage than Odds API.
# Set SHARP_API_KEY in your .env to enable. Falls back to Odds API if not set.
# Docs: check your SharpAPI provider's documentation for endpoint format.
SHARP_API_KEY = os.getenv("SHARP_API_KEY")
SHARP_BASE_URL = os.getenv("SHARP_BASE_URL", "")  # e.g. "https://api.sharpapi.io/v1"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

REQUEST_DELAY = 0.3  # seconds between API calls


def _get(path: str, params: dict | None = None) -> dict | list:
    """Authenticated GET request to The Odds API."""
    if not API_KEY:
        raise EnvironmentError("ODDS_API_KEY not set in .env")
    url = f"{BASE_URL}{path}"
    p = {"apiKey": API_KEY}
    if params:
        p.update(params)
    resp = requests.get(url, params=p, timeout=15)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  [api] GET {path} | quota used={used} remaining={remaining}")
    resp.raise_for_status()
    return resp.json()


def fetch_today_events() -> list[dict]:
    """
    Return NBA events commencing within the next 48 hours.
    Uses a 48-hour window (not just today's date string) so UTC timestamps
    never cause games to be missed due to timezone offset.
    """
    today = date.today().isoformat()
    cache = DATA_DIR / f"events_{today}.json"
    if cache.exists():
        print(f"  [cache] Events loaded from {cache}")
        with open(cache) as f:
            return json.load(f)

    now_utc = datetime.now(timezone.utc)
    window_end = now_utc + timedelta(hours=48)

    data = _get(f"/sports/{SPORT}/events")
    print(f"  [api] Total NBA events returned: {len(data)}")

    upcoming = []
    for ev in data:
        commence_str = ev.get("commence_time", "")
        if not commence_str:
            continue
        try:
            # Format: '2026-03-10T00:30:00Z'
            commence_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if now_utc <= commence_dt <= window_end:
            upcoming.append(ev)

    with open(cache, "w") as f:
        json.dump(upcoming, f, indent=2)
    print(f"  [saved] {len(upcoming)} events in next 48h → {cache}")
    return upcoming


def fetch_event_props(event_id: str, event_date: str) -> list[dict]:
    """
    Fetch player prop lines for a single event.
    Tries SharpAPI first (better Pinnacle/bet365 coverage) then falls back to Odds API.
    Returns raw bookmaker data list.
    """
    mkt_key = MARKETS.replace(",", "_")
    cache = DATA_DIR / f"props_{event_id}_{event_date}_{mkt_key}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    # Try SharpAPI if configured
    if SHARP_API_KEY and SHARP_BASE_URL:
        sharp_books = _fetch_sharp_props_nba()
        if sharp_books:
            print(f"    [sharp] Using SharpAPI data ({len(sharp_books)} books)")
            with open(cache, "w") as f:
                json.dump(sharp_books, f, indent=2)
            return sharp_books

    time.sleep(REQUEST_DELAY)
    data = _get(
        f"/sports/{SPORT}/events/{event_id}/odds",
        params={
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": "american",
            # No bookmakers filter — take everything the API returns,
            # then prefer Pinnacle/bet365 in parse_props if available.
        },
    )
    bookmakers = data.get("bookmakers", [])
    available = [b.get("key") for b in bookmakers]
    if available:
        print(f"    [books] {available}")
    else:
        print(f"    [warn] No bookmakers returned for this event")
    with open(cache, "w") as f:
        json.dump(bookmakers, f, indent=2)
    return bookmakers


def parse_props(bookmakers: list[dict]) -> pd.DataFrame:
    """
    Parse raw bookmaker JSON into a flat DataFrame.
    Keeps ALL bookmakers — one row per player/market/line/bookmaker.
    Bookmaker filtering happens downstream in the screener.

    Returns columns:
        player_name, market, line, over_price, under_price, bookmaker
    """
    rows = []
    for bk in bookmakers:
        bk_name = bk.get("key", "")
        for market in bk.get("markets", []):
            mkt_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                name = outcome.get("description") or outcome.get("name", "")
                side = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")
                rows.append({
                    "player_name": name,
                    "market": mkt_key,
                    "side": side,
                    "line": point,
                    "price": price,
                    "bookmaker": bk_name,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    over = df[df["side"].str.lower() == "over"].rename(columns={"price": "over_price"})
    under = df[df["side"].str.lower() == "under"].rename(columns={"price": "under_price"})
    merged = pd.merge(
        over[["player_name", "market", "line", "over_price", "bookmaker"]],
        under[["player_name", "market", "line", "under_price", "bookmaker"]],
        on=["player_name", "market", "line", "bookmaker"],
        how="outer",
    )
    return merged.reset_index(drop=True)


def get_today_lines(refresh: bool = False) -> pd.DataFrame:
    """
    Master function: fetch all of today's player_points props from
    Pinnacle and bet365, return as a unified DataFrame.

    Columns:
        player_name, market, line, over_price, under_price,
        bookmaker, event_id, home_team, away_team, commence_time

    Pass refresh=True to delete today's cached files and re-fetch live odds.
    """
    today = date.today().isoformat()
    master_cache = DATA_DIR / f"lines_{today}.csv"

    if refresh:
        # Delete master cache
        if master_cache.exists():
            master_cache.unlink()
            print(f"[odds] Cache cleared: {master_cache}")
        # Delete per-event prop caches for today
        for f in DATA_DIR.glob(f"props_*_{today}_*.json"):
            f.unlink()
            print(f"[odds] Cache cleared: {f.name}")
        # Delete events cache for today
        events_cache = DATA_DIR / f"events_{today}.json"
        if events_cache.exists():
            events_cache.unlink()
            print(f"[odds] Cache cleared: {events_cache.name}")

    if master_cache.exists():
        print(f"[odds] Loading today's lines from cache: {master_cache}")
        return pd.read_csv(master_cache)

    events = fetch_today_events()
    if not events:
        print("[odds] No NBA events found for today.")
        return pd.DataFrame()

    all_frames = []
    for ev in events:
        event_id = ev["id"]
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        commence = ev.get("commence_time", "")
        ev_date = commence[:10] if commence else today
        print(f"  Fetching props: {away} @ {home}")
        try:
            bookmakers = fetch_event_props(event_id, ev_date)
            df = parse_props(bookmakers)
            if not df.empty:
                df["event_id"] = event_id
                df["home_team"] = home
                df["away_team"] = away
                df["commence_time"] = commence
                all_frames.append(df)
        except requests.HTTPError as e:
            print(f"  [warn] {event_id}: {e}")

    if not all_frames:
        print("[odds] No prop lines found for today's games.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(master_cache, index=False)
    print(f"[odds] {len(combined)} prop rows → {master_cache}")
    return combined


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal."""
    if american >= 100:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def implied_probability(american: float) -> float:
    """Implied win probability from American odds (no vig removed)."""
    dec = american_to_decimal(american)
    return 1 / dec


def _parse_props_all_books(bookmakers: list[dict]) -> pd.DataFrame:
    """
    Like parse_props but keeps every book's line — used for arbitrage detection.
    Returns columns: player_name, market, line, over_price, under_price, bookmaker
    """
    rows = []
    for bk in bookmakers:
        bk_name = bk.get("key", "")
        for market in bk.get("markets", []):
            mkt_key = market.get("key", "")
            for outcome in market.get("outcomes", []):
                name = outcome.get("description") or outcome.get("name", "")
                side = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")
                rows.append({
                    "player_name": name,
                    "market": mkt_key,
                    "side": side,
                    "line": point,
                    "price": price,
                    "bookmaker": bk_name,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    over = df[df["side"].str.lower() == "over"].rename(columns={"price": "over_price"})
    under = df[df["side"].str.lower() == "under"].rename(columns={"price": "under_price"})
    return pd.merge(
        over[["player_name", "market", "line", "over_price", "bookmaker"]],
        under[["player_name", "market", "line", "under_price", "bookmaker"]],
        on=["player_name", "market", "line", "bookmaker"],
        how="outer",
    )


def detect_arbitrage(all_books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find arbitrage opportunities across ALL available bookmakers.

    For each (player, market, line) group, finds the best over price across all
    books and the best under price across all books.  If those come from different
    books and their combined implied probability < 1.0, it's an arb.

    arb_pct = guaranteed profit as % of total stake.
    Returns rows sorted by arb_pct descending.
    """
    if all_books_df.empty:
        return pd.DataFrame()

    key_cols = ["player_name", "market", "line"]
    arb_rows = []

    for key, grp in all_books_df.groupby(key_cols):
        player, market, line = key

        over_rows = grp[grp["over_price"].notna()][["over_price", "bookmaker"]]
        under_rows = grp[grp["under_price"].notna()][["under_price", "bookmaker"]]

        if over_rows.empty or under_rows.empty:
            continue

        best_over_idx = over_rows["over_price"].idxmax()
        best_over_price = float(over_rows.loc[best_over_idx, "over_price"])
        best_over_book = over_rows.loc[best_over_idx, "bookmaker"]

        best_under_idx = under_rows["under_price"].idxmax()
        best_under_price = float(under_rows.loc[best_under_idx, "under_price"])
        best_under_book = under_rows.loc[best_under_idx, "bookmaker"]

        # Same book on both sides is not a real arb opportunity
        if best_over_book == best_under_book:
            continue

        total_implied = implied_probability(best_over_price) + implied_probability(best_under_price)
        if total_implied < 1.0:
            arb_pct = round((1 - total_implied) * 100, 2)
            arb_rows.append({
                "player": player,
                "market": market.replace("player_", ""),
                "line": line,
                "leg1": f"OVER @ {best_over_book} ({int(best_over_price):+d})",
                "leg2": f"UNDER @ {best_under_book} ({int(best_under_price):+d})",
                "arb_%": arb_pct,
            })

    if not arb_rows:
        return pd.DataFrame()
    return pd.DataFrame(arb_rows).sort_values("arb_%", ascending=False).reset_index(drop=True)


def _fetch_sharp_props_nba() -> list[dict]:
    """
    Fetch NBA player props from SharpAPI (Pinnacle + bet365 specialist).

    Set SHARP_API_KEY and SHARP_BASE_URL in your .env to enable.
    This is a stub — update the endpoint/params to match your SharpAPI provider.

    Expected return format: same bookmakers list as Odds API
        [{"key": "pinnacle", "markets": [{"key": "player_points", "outcomes": [...]}]}]
    """
    if not SHARP_API_KEY or not SHARP_BASE_URL:
        return []

    try:
        today = date.today().isoformat()
        resp = requests.get(
            f"{SHARP_BASE_URL}/nba/props",
            params={"date": today, "markets": MARKETS, "books": "pinnacle,bet365"},
            headers={"Authorization": f"Bearer {SHARP_API_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # TODO: transform data to match Odds API bookmakers format if needed
        return data.get("bookmakers", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"  [sharp] SharpAPI fetch failed: {e} — falling back to Odds API")
        return []


def get_arbitrage_opportunities() -> pd.DataFrame:
    """
    Return today's arbitrage opportunities across all available bookmakers.
    Reads from cached props files so no extra API calls are made.
    """
    today = date.today().isoformat()
    events_cache = DATA_DIR / f"events_{today}.json"
    if not events_cache.exists():
        return pd.DataFrame()

    with open(events_cache) as f:
        events = json.load(f)

    all_frames = []
    mkt_key = MARKETS.replace(",", "_")
    for ev in events:
        event_id = ev["id"]
        commence = ev.get("commence_time", today)
        ev_date = commence[:10]
        cache = DATA_DIR / f"props_{event_id}_{ev_date}_{mkt_key}.json"
        if not cache.exists():
            continue
        with open(cache) as f:
            bookmakers = json.load(f)
        all_books = _parse_props_all_books(bookmakers)
        if not all_books.empty:
            all_frames.append(all_books)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    return detect_arbitrage(combined)


def best_line(df: pd.DataFrame, player: str, side: str) -> tuple[float | None, str | None]:
    """
    Return (best_price, bookmaker) for a given player + side ('Over'/'Under').
    Best = highest payout (least negative / most positive American odds).
    """
    col = "over_price" if side.lower() == "over" else "under_price"
    subset = df[df["player_name"] == player][[col, "bookmaker"]].dropna()
    if subset.empty:
        return None, None
    idx = subset[col].idxmax()
    return subset.loc[idx, col], subset.loc[idx, "bookmaker"]


if __name__ == "__main__":
    print("=== Odds Pull ===")
    lines = get_today_lines()
    if lines.empty:
        print("No lines available — check API key or try again closer to game time.")
    else:
        print(f"\nTotal prop lines: {len(lines)}")
        print("\nBookmaker coverage:")
        print(lines["bookmaker"].value_counts())
        print("\nSample props:")
        cols = ["player_name", "line", "over_price", "under_price", "bookmaker", "home_team", "away_team"]
        avail = [c for c in cols if c in lines.columns]
        print(lines[avail].head(15).to_string(index=False))
    print("\n[odds.py] Done.")
