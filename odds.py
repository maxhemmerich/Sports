"""
odds.py — Pull today's NBA player prop lines from Pinnacle and bet365
via The Odds API.  Stores results locally and exposes helper functions
for the screener.

Endpoints used:
  GET /v4/sports/basketball_nba/events
  GET /v4/sports/basketball_nba/events/{event_id}/odds
       ?regions=us&markets=player_points&bookmakers=pinnacle,bet365
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
BOOKMAKERS = "pinnacle,bet365"
# Pinnacle = eu region, bet365 = uk region (neither is in 'us')
REGIONS = "eu,uk"
MARKETS = "player_points"

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
    Fetch player_points props for a single event from Pinnacle + bet365.
    Returns raw bookmaker data list.
    """
    cache = DATA_DIR / f"props_{event_id}_{event_date}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    time.sleep(REQUEST_DELAY)
    data = _get(
        f"/sports/{SPORT}/events/{event_id}/odds",
        params={
            "regions": REGIONS,
            "markets": MARKETS,
            "bookmakers": BOOKMAKERS,
            "oddsFormat": "american",
        },
    )
    bookmakers = data.get("bookmakers", [])
    if not bookmakers:
        # Show which bookmakers the API did return (helps diagnose plan limits)
        available = [b.get("key") for b in data.get("bookmakers", [])]
        print(f"    [warn] No Pinnacle/bet365 props. API bookmakers: {available or 'none'}")
    with open(cache, "w") as f:
        json.dump(bookmakers, f, indent=2)
    return bookmakers


def parse_props(bookmakers: list[dict]) -> pd.DataFrame:
    """
    Parse raw bookmaker JSON into a flat DataFrame.

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
                side = outcome.get("name", "")  # 'Over' or 'Under'
                point = outcome.get("point")
                price = outcome.get("price")
                rows.append(
                    {
                        "player_name": name,
                        "market": mkt_key,
                        "side": side,
                        "line": point,
                        "price": price,
                        "bookmaker": bk_name,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Pivot over/under into single row per player-line-bookmaker
    over = df[df["side"].str.lower() == "over"].rename(columns={"price": "over_price"})
    under = df[df["side"].str.lower() == "under"].rename(columns={"price": "under_price"})
    merged = pd.merge(
        over[["player_name", "market", "line", "over_price", "bookmaker"]],
        under[["player_name", "market", "line", "under_price", "bookmaker"]],
        on=["player_name", "market", "line", "bookmaker"],
        how="outer",
    )
    return merged


def get_today_lines() -> pd.DataFrame:
    """
    Master function: fetch all of today's player_points props from
    Pinnacle and bet365, return as a unified DataFrame.

    Columns:
        player_name, market, line, over_price, under_price,
        bookmaker, event_id, home_team, away_team, commence_time
    """
    today = date.today().isoformat()
    master_cache = DATA_DIR / f"lines_{today}.csv"
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
