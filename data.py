"""
data.py — Pull last 2 seasons of NBA player game logs.

Uses direct requests to stats.nba.com (bypasses nba_api's HTTP layer
which does not reliably forward custom headers). Falls back to the
balldontlie public API if the NBA site is unavailable.
"""

import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASONS = ["2022-23", "2023-24"]
SEASON_TYPE = "Regular Season"
REQUEST_DELAY = 1.2   # polite delay between requests (seconds)
NBA_TIMEOUT = 60      # per-request timeout

# Full browser headers — stats.nba.com rejects bot-like requests
_SESSION = requests.Session()
_SESSION.headers.update({
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
})

# ── Direct NBA Stats API helpers ───────────────────────────────────────────


def _nba_get(url: str, params: dict, max_retries: int = 4) -> dict:
    """
    GET request to stats.nba.com with exponential-backoff retry.
    Raises RuntimeError after max_retries failures.
    """
    delay = 2
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=NBA_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_err = e
            print(f"  [retry {attempt+1}/{max_retries}] {type(e).__name__} — waiting {delay}s ...")
            time.sleep(delay)
            delay *= 2
        except requests.HTTPError as e:
            raise RuntimeError(f"NBA API HTTP error: {e}") from e
    raise RuntimeError(f"NBA API unreachable after {max_retries} attempts: {last_err}")


def _parse_nba_response(data: dict) -> pd.DataFrame:
    """Convert stats.nba.com resultSets JSON into a DataFrame."""
    result_set = data["resultSets"][0]
    headers = result_set["headers"]
    rows = result_set["rowSet"]
    return pd.DataFrame(rows, columns=headers)


def fetch_season_gamelogs_direct(season: str) -> pd.DataFrame:
    """
    Fetch all player game logs for one season directly from stats.nba.com.
    Endpoint: /stats/playergamelogs
    """
    url = "https://stats.nba.com/stats/playergamelogs"
    params = {
        "Season": season,
        "SeasonType": SEASON_TYPE,
        "LeagueID": "00",
    }
    print(f"  [nba.com] GET playergamelogs season={season} ...")
    time.sleep(REQUEST_DELAY)
    data = _nba_get(url, params)
    df = _parse_nba_response(data)
    df.columns = [c.lower() for c in df.columns]
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ── Balldontlie fallback ───────────────────────────────────────────────────
# Free public API, no key needed, covers full game logs.

BDL_BASE = "https://api.balldontlie.io/v1"
BDL_PAGE_SIZE = 100


def _bdl_get(path: str, params: dict) -> dict:
    """GET request to balldontlie API (no auth required for v1 free tier)."""
    resp = requests.get(f"{BDL_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _season_year(season_str: str) -> int:
    """'2022-23' → 2022"""
    return int(season_str.split("-")[0])


def fetch_season_gamelogs_bdl(season: str) -> pd.DataFrame:
    """
    Fetch player game logs via balldontlie API.
    Covers regular season only. Paginates automatically.
    Returns DataFrame with nba_api-compatible column names.
    """
    year = _season_year(season)
    print(f"  [balldontlie] Fetching season {season} (year={year}) ...")

    rows = []
    cursor = None
    page = 1
    while True:
        params: dict = {
            "seasons[]": year,
            "per_page": BDL_PAGE_SIZE,
        }
        if cursor:
            params["cursor"] = cursor

        time.sleep(0.35)  # free tier rate limit
        try:
            data = _bdl_get("/stats", params)
        except requests.HTTPError as e:
            print(f"  [bdl] HTTP error on page {page}: {e}")
            break

        items = data.get("data", [])
        if not items:
            break

        for item in items:
            player = item.get("player", {})
            team = item.get("team", {})
            game = item.get("game", {})
            rows.append({
                "player_id": player.get("id"),
                "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "team_id": team.get("id"),
                "team_abbreviation": team.get("abbreviation", ""),
                "game_id": game.get("id"),
                "game_date": game.get("date", "")[:10],
                "matchup": f"{game.get('home_team_id')} vs {game.get('visitor_team_id')}",
                "wl": None,
                "pts": item.get("pts"),
                "reb": item.get("reb"),
                "ast": item.get("ast"),
                "fga": item.get("fga"),
                "fgm": item.get("fgm"),
                "fg3a": item.get("fg3a"),
                "fg3m": item.get("fg3m"),
                "fta": item.get("fta"),
                "ftm": item.get("ftm"),
                "min": item.get("min"),
                "plus_minus": item.get("plus_minus"),
                "opponent_team_id": (
                    game.get("visitor_team_id")
                    if game.get("home_team_id") == team.get("id")
                    else game.get("home_team_id")
                ),
            })

        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        print(f"    page {page}: {len(items)} rows | next_cursor={cursor}")
        page += 1
        if not cursor:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["pts"])
    return df


# ── Main fetch with automatic fallback ────────────────────────────────────


def fetch_player_gamelogs(season: str) -> pd.DataFrame:
    """
    Fetch all player game logs for a season.
    Tries stats.nba.com first; falls back to balldontlie if unreachable.
    Results are cached locally.
    """
    cache_path = DATA_DIR / f"gamelogs_{season.replace('-', '_')}.csv"
    if cache_path.exists():
        print(f"  [cache] Loading {season} from {cache_path}")
        df = pd.read_csv(cache_path, low_memory=False)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    # Try primary source
    df = pd.DataFrame()
    try:
        df = fetch_season_gamelogs_direct(season)
        print(f"  [ok] nba.com: {len(df)} rows for {season}")
    except Exception as e:
        print(f"  [warn] nba.com failed ({e}). Trying balldontlie ...")
        try:
            df = fetch_season_gamelogs_bdl(season)
            print(f"  [ok] balldontlie: {len(df)} rows for {season}")
        except Exception as e2:
            raise RuntimeError(
                f"Both data sources failed for {season}.\n"
                f"  nba.com: {e}\n  balldontlie: {e2}"
            ) from e2

    if df.empty:
        raise RuntimeError(f"No data returned for {season}")

    df.to_csv(cache_path, index=False)
    print(f"  [saved] → {cache_path}")
    return df


def fetch_all_seasons() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = fetch_player_gamelogs(season)
        df["season"] = season
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return combined


def load_gamelogs() -> pd.DataFrame:
    """
    Public entry point — returns combined, sorted game log DataFrame.
    Columns of interest:
        player_id, player_name, game_date, matchup, wl,
        pts, reb, ast, min, fga, fgm, fg3a, fg3m, fta, ftm,
        plus_minus, team_abbreviation, opponent_team_id, season
    """
    out_path = DATA_DIR / "gamelogs_combined.csv"
    if out_path.exists():
        print(f"[data] Loading combined gamelogs from cache: {out_path}")
        df = pd.read_csv(out_path, low_memory=False)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    print("[data] Building combined gamelogs ...")
    df = fetch_all_seasons()
    df.to_csv(out_path, index=False)
    print(f"[data] Combined: {len(df)} rows → {out_path}")
    return df


def fetch_team_defense_stats(season: str) -> pd.DataFrame:
    df = fetch_player_gamelogs(season)
    if "opponent_team_id" not in df.columns:
        return pd.DataFrame()
    game_team = (
        df.groupby(["game_id", "team_id", "opponent_team_id"])["pts"]
        .sum()
        .reset_index()
    )
    def_stats = (
        game_team.groupby("opponent_team_id")["pts"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"opponent_team_id": "team_id", "mean": "avg_pts_allowed", "count": "games"})
    )
    def_stats["season"] = season
    return def_stats


def build_defense_lookup() -> pd.DataFrame:
    out_path = DATA_DIR / "team_defense.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    frames = []
    for season in SEASONS:
        df = fetch_team_defense_stats(season)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"[data] Defense lookup → {out_path}")
    return combined


if __name__ == "__main__":
    print("=== NBA Data Pull ===")
    print("Fetching game logs for seasons:", SEASONS)
    df = load_gamelogs()
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    print("\nSample (first 5 rows):")
    cols = [c for c in ["player_name", "game_date", "matchup", "pts", "reb", "ast"] if c in df.columns]
    print(df[cols].head())

    print("\nBuilding team defense lookup ...")
    def_df = build_defense_lookup()
    if not def_df.empty:
        print(def_df.sort_values("avg_pts_allowed", ascending=False).head(10))
    print("\n[data.py] Done.")
