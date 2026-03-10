"""
data.py — Pull last 2 seasons of NBA player game logs via nba_api.
Stores data locally as CSV files in ./data/ directory.
"""

import os
import time
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import playergamelogs, commonallplayers
from nba_api.stats.static import players as static_players

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASONS = ["2022-23", "2023-24"]
SEASON_TYPE = "Regular Season"
# nba_api rate limiting — be respectful
REQUEST_DELAY = 1.0  # seconds between requests

# stats.nba.com blocks requests without browser-like headers
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
NBA_TIMEOUT = 120  # seconds — bulk season endpoint is slow


def get_all_active_players() -> pd.DataFrame:
    """Return a DataFrame of all active NBA players."""
    all_players = static_players.get_active_players()
    return pd.DataFrame(all_players)


def fetch_player_gamelogs(season: str) -> pd.DataFrame:
    """
    Fetch all player game logs for a given season using the bulk endpoint.
    Returns a single DataFrame with every player-game row.
    """
    cache_path = DATA_DIR / f"gamelogs_{season.replace('-', '_')}.csv"
    if cache_path.exists():
        print(f"  [cache] Loading {season} from {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  [api]   Fetching {season} game logs ...")
    time.sleep(REQUEST_DELAY)
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=SEASON_TYPE,
        headers=NBA_HEADERS,
        timeout=NBA_TIMEOUT,
    )
    df = logs.get_data_frames()[0]

    # Normalise column names to lower-case snake_case
    df.columns = [c.lower() for c in df.columns]

    # Parse game date
    df["game_date"] = pd.to_datetime(df["game_date"])

    df.to_csv(cache_path, index=False)
    print(f"  [saved] {len(df)} rows → {cache_path}")
    return df


def fetch_all_seasons() -> pd.DataFrame:
    """Fetch and combine game logs for all configured seasons."""
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
    """
    Derive opponent defensive ratings from game logs:
    average points allowed per game by each team.
    Returns DataFrame with columns: opp_team_id, season, avg_pts_allowed
    """
    df = fetch_player_gamelogs(season)

    # Each row = player-game. We want team-level totals.
    # 'matchup' looks like 'LAL vs. GSW' or 'LAL @ GSW'
    # Use opponent_team_id if present, else derive from matchup.
    if "opponent_team_id" not in df.columns:
        return pd.DataFrame()

    # Group by game_id + opponent to get total pts allowed per game
    game_team = (
        df.groupby(["game_id", "team_id", "opponent_team_id"])["pts"]
        .sum()
        .reset_index()
    )
    # Each opponent allowed 'pts' in that game
    def_stats = (
        game_team.groupby("opponent_team_id")["pts"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"opponent_team_id": "team_id", "mean": "avg_pts_allowed", "count": "games"})
    )
    def_stats["season"] = season
    return def_stats


def build_defense_lookup() -> pd.DataFrame:
    """Build a multi-season defense lookup table and cache it."""
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
    print(df[["player_name", "game_date", "matchup", "pts", "reb", "ast"]].head())

    print("\nBuilding team defense lookup ...")
    def_df = build_defense_lookup()
    if not def_df.empty:
        print(def_df.sort_values("avg_pts_allowed", ascending=False).head(10))
    print("\n[data.py] Done.")
