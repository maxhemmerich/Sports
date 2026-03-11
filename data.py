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

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPE = "Regular Season"
REQUEST_DELAY = 1.2   # polite delay between requests (seconds)
NBA_TIMEOUT = 60      # per-request timeout

# Past seasons never change; only the current season needs daily refresh.
CURRENT_SEASON = SEASONS[-1]
GAMELOG_CACHE_HOURS = int(__import__("os").getenv("GAMELOG_CACHE_HOURS", "12"))
COMBINED_CACHE_HOURS = int(__import__("os").getenv("COMBINED_CACHE_HOURS", "12"))

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


# ── leaguegamelog fallback ─────────────────────────────────────────────────
# Different endpoint on stats.nba.com — more reliable for recent seasons
# where playergamelogs returns 500.


def fetch_season_gamelogs_league(season: str) -> pd.DataFrame:
    """
    Fetch player game logs via leaguegamelog endpoint.
    Used as fallback when playergamelogs returns a server error.
    """
    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "Counter": "0",
        "Direction": "ASC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": SEASON_TYPE,
        "Sorter": "DATE",
    }
    print(f"  [nba.com/leaguegamelog] season={season} ...")
    time.sleep(REQUEST_DELAY)
    data = _nba_get(url, params)

    result_set = data["resultSets"][0]
    headers = result_set["headers"]
    rows = result_set["rowSet"]
    df = pd.DataFrame(rows, columns=headers)
    df.columns = [c.lower() for c in df.columns]

    # leaguegamelog uses GAME_DATE not game_date, and has slightly different cols
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    # Map leaguegamelog columns to our standard names where they differ
    rename = {
        "game_date": "game_date",
        "team_id": "team_id",
        "team_abbreviation": "team_abbreviation",
        "player_id": "player_id",
        "player_name": "player_name",
        "game_id": "game_id",
        "matchup": "matchup",
        "wl": "wl",
        "pts": "pts",
        "reb": "reb",
        "ast": "ast",
        "fga": "fga",
        "fgm": "fgm",
        "fg3a": "fg3a",
        "fg3m": "fg3m",
        "fta": "fta",
        "ftm": "ftm",
        "min": "min",
        "plus_minus": "plus_minus",
    }
    df = df[[c for c in rename if c in df.columns]]

    # Derive opponent_team_id from matchup (e.g. "LAL @ GSW" → opponent is GSW's id)
    # leaguegamelog doesn't include opponent_team_id directly
    return df


# ── Main fetch with automatic fallback ────────────────────────────────────


def fetch_player_gamelogs(season: str) -> pd.DataFrame:
    """
    Fetch all player game logs for a season.
    Tries playergamelogs first, then leaguegamelog if server errors occur.
    Results are cached locally.
    """
    cache_path = DATA_DIR / f"gamelogs_{season.replace('-', '_')}.csv"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        # Past seasons are final; only refresh the current season periodically.
        if season != CURRENT_SEASON or age_hours <= GAMELOG_CACHE_HOURS:
            print(f"  [cache] Loading {season} from {cache_path}")
            df = pd.read_csv(cache_path, low_memory=False)
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df
        print(f"  [cache] {season} cache is {age_hours:.1f}h old — refreshing ...")

    df = pd.DataFrame()
    try:
        df = fetch_season_gamelogs_direct(season)
        print(f"  [ok] playergamelogs: {len(df)} rows for {season}")
    except Exception as e:
        print(f"  [warn] playergamelogs failed ({type(e).__name__}). Trying leaguegamelog ...")
        try:
            df = fetch_season_gamelogs_league(season)
            print(f"  [ok] leaguegamelog: {len(df)} rows for {season}")
        except Exception as e2:
            raise RuntimeError(
                f"Both NBA endpoints failed for {season}.\n"
                f"  playergamelogs: {e}\n  leaguegamelog: {e2}"
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
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if age_hours <= COMBINED_CACHE_HOURS:
            print(f"[data] Loading combined gamelogs from cache: {out_path}")
            df = pd.read_csv(out_path, low_memory=False)
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df
        print(f"[data] Combined cache is {age_hours:.1f}h old — refreshing ...")
        out_path.unlink()

    print("[data] Building combined gamelogs ...")
    df = fetch_all_seasons()
    df.to_csv(out_path, index=False)
    print(f"[data] Combined: {len(df)} rows → {out_path}")
    return df


def _extract_opp_abbr(matchup: str, team_abbr: str) -> str:
    """Derive opponent team abbreviation from matchup string."""
    for sep in [" vs. ", " @ "]:
        if sep in matchup:
            left, right = matchup.split(sep, 1)
            left, right = left.strip(), right.strip()
            return right if left == team_abbr else left
    return ""


def fetch_team_defense_stats(season: str) -> pd.DataFrame:
    """
    Compute how many points each team allows per game.
    Keyed by opp_abbreviation (the defending team) so we can merge by team abbr.
    """
    df = fetch_player_gamelogs(season)
    required = {"matchup", "team_abbreviation", "game_id", "pts"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["opp_abbreviation"] = df.apply(
        lambda r: _extract_opp_abbr(str(r["matchup"]), str(r["team_abbreviation"])),
        axis=1,
    )
    df = df[df["opp_abbreviation"] != ""]

    # Total points scored against each opponent per game
    game_totals = (
        df.groupby(["game_id", "opp_abbreviation"])["pts"]
        .sum()
        .reset_index()
    )
    def_stats = (
        game_totals.groupby("opp_abbreviation")["pts"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"opp_abbreviation": "team_abbreviation", "mean": "avg_pts_allowed", "count": "games"})
    )
    def_stats["season"] = season
    return def_stats


def build_defense_lookup() -> pd.DataFrame:
    out_path = DATA_DIR / "team_defense.csv"
    if out_path.exists():
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if age_hours <= COMBINED_CACHE_HOURS:
            return pd.read_csv(out_path)
        out_path.unlink()
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
