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

# In-memory caches for lookup DataFrames — populated on first call, reused every subsequent call
# within the same process (survives across screener ticks without re-reading CSV)
_LOOKUP_CACHE: dict[str, "pd.DataFrame"] = {}

# Dtype map — use float32/int32 to halve memory vs pandas' default float64/int64
_GAMELOG_DTYPES: dict = {
    "player_id": "Int32",
    "team_id": "Int32",
    "pts": "float32",
    "reb": "float32",
    "ast": "float32",
    "fga": "float32",
    "fgm": "float32",
    "fg3a": "float32",
    "fg3m": "float32",
    "fta": "float32",
    "ftm": "float32",
    "stl": "float32",
    "blk": "float32",
    "tov": "float32",
    "plus_minus": "float32",
}

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
        "stl": "stl",
        "blk": "blk",
        "tov": "tov",
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
            use_dt = {k: v for k, v in _GAMELOG_DTYPES.items()}
            df = pd.read_csv(cache_path, dtype=use_dt)
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
            use_dt = {k: v for k, v in _GAMELOG_DTYPES.items()}
            df = pd.read_csv(out_path, dtype=use_dt)
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


_DEFENSE_STATS = ["pts", "reb", "ast", "fg3m", "blk", "stl", "tov"]


def fetch_team_defense_stats(season: str) -> pd.DataFrame:
    """
    Compute how many of each counting stat each team allows per game.
    Keyed by opp_abbreviation (the defending team) so we can merge by team abbr.
    Returns columns: team_abbreviation, season, games,
                     avg_pts_allowed, avg_reb_allowed, avg_ast_allowed, ...
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

    stat_cols = [c for c in _DEFENSE_STATS if c in df.columns]

    # Sum all players' stats against each opponent per game, then average across games
    game_totals = (
        df.groupby(["game_id", "opp_abbreviation"])[stat_cols]
        .sum()
        .reset_index()
    )
    def_stats = (
        game_totals.groupby("opp_abbreviation")[stat_cols]
        .mean()
        .reset_index()
        .rename(columns={"opp_abbreviation": "team_abbreviation"})
        .rename(columns={c: f"avg_{c}_allowed" for c in stat_cols})
    )
    games_count = (
        game_totals.groupby("opp_abbreviation")["pts"]
        .count()
        .reset_index()
        .rename(columns={"opp_abbreviation": "team_abbreviation", "pts": "games"})
    )
    def_stats = def_stats.merge(games_count, on="team_abbreviation")
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


def _get_opp_abbr_data(matchup: str, team_abbr: str) -> str:
    """Extract opponent abbreviation from matchup string."""
    for sep in [" vs. ", " @ "]:
        if sep in matchup:
            left, right = matchup.split(sep, 1)
            left, right = left.strip(), right.strip()
            return right if left == team_abbr else left
    return ""


_ROLLING_DEF_STATS = ["pts", "reb", "ast", "fg3m", "blk", "stl", "tov"]


def build_rolling_defense_lookup(window: int = 15) -> pd.DataFrame:
    """
    Pre-compute rolling N-game defensive averages per (game_id, defending_team).

    For each game G and team T, records the average stats T allowed per game
    in the N games BEFORE G (no lookahead). Stored as:
        game_id, defending_team, opp_pts_allowed_roll{N}, opp_reb_allowed_roll{N}, ...

    Cached to data/rolling_defense_{N}g.csv.
    """
    _key = f"rolling_def_{window}"
    if _key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[_key]
    out_path = DATA_DIR / f"rolling_defense_{window}g.csv"
    if out_path.exists():
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if age_hours <= COMBINED_CACHE_HOURS:
            df = pd.read_csv(out_path, parse_dates=["game_date"])
            _LOOKUP_CACHE[_key] = df
            return df
        out_path.unlink()

    df = load_gamelogs()
    required = {"game_id", "team_abbreviation", "game_date", "matchup"}
    stat_cols = [s for s in _ROLLING_DEF_STATS if s in df.columns]
    if not required.issubset(df.columns) or not stat_cols:
        return pd.DataFrame()

    # Sum all players' stats per (game_id, team)
    team_game = (
        df.groupby(["game_id", "team_abbreviation", "game_date"])[stat_cols]
        .sum()
        .reset_index()
    )
    team_game[stat_cols] = team_game[stat_cols].astype("float32")

    # Map (game_id, team) → opponent abbreviation
    opp_series = df[["game_id", "team_abbreviation", "matchup"]].drop_duplicates(
        subset=["game_id", "team_abbreviation"]
    )
    opp_series = opp_series.copy()
    opp_series["_opp"] = opp_series.apply(
        lambda r: _get_opp_abbr_data(str(r["matchup"]), str(r["team_abbreviation"])),
        axis=1,
    )
    team_game = team_game.merge(
        opp_series[["game_id", "team_abbreviation", "_opp"]],
        on=["game_id", "team_abbreviation"],
        how="left",
    )

    # What team X scored in game G = what X's opponent (defending_team) allowed in G
    # Rename: defending_team = _opp (the team doing the defending)
    team_game = team_game.rename(columns={"_opp": "defending_team"})
    team_game = team_game[team_game["defending_team"].notna() & (team_game["defending_team"] != "")]

    team_game = team_game.sort_values(["defending_team", "game_date"])

    roll_cols = []
    for stat in stat_cols:
        col = f"opp_{stat}_allowed_roll{window}"
        team_game[col] = (
            team_game.groupby("defending_team")[stat]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        roll_cols.append(col)

    result = team_game[["game_id", "game_date", "defending_team"] + roll_cols].copy()
    result.to_csv(out_path, index=False)
    print(f"[data] Rolling {window}-game defense lookup → {out_path} ({len(result)} rows)")
    _LOOKUP_CACHE[f"rolling_def_{window}"] = result
    return result


def build_positional_defense_lookup(window: int = 15) -> pd.DataFrame:
    """
    Pre-compute rolling N-game positional defensive averages per (game_id, defending_team, pos).

    Position group is inferred from each player's actual reb in prior games
    (guard: <4, forward: 4-6, big: >=7). Per-position avg stats allowed are
    computed with a rolling window (no lookahead).

    Columns: game_id, defending_team, pos_group,
             opp_pts_allowed_pos, opp_reb_allowed_pos, opp_ast_allowed_pos

    Cached to data/positional_defense.csv.
    """
    _key = "positional_defense"
    if _key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[_key]
    out_path = DATA_DIR / "positional_defense.csv"
    if out_path.exists():
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if age_hours <= COMBINED_CACHE_HOURS:
            df = pd.read_csv(out_path)
            _LOOKUP_CACHE[_key] = df
            return df
        out_path.unlink()

    df = load_gamelogs()
    required = {"game_id", "team_abbreviation", "game_date", "matchup", "player_id", "reb"}
    stat_cols = [s for s in ["pts", "reb", "ast"] if s in df.columns]
    if not required.issubset(df.columns) or not stat_cols:
        return pd.DataFrame()

    df = df.sort_values(["player_id", "game_date"]).copy()

    # Compute rolling reb to infer position (shifted to avoid lookahead)
    df["_reb_roll10"] = (
        df.groupby("player_id")["reb"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        .fillna(df["reb"])
    ).astype("float32")

    reb = df["_reb_roll10"]
    df["_pos"] = "guard"
    df.loc[reb >= 4, "_pos"] = "forward"
    df.loc[reb >= 7, "_pos"] = "big"

    # Opponent abbreviation for each row
    opp_series = df[["game_id", "team_abbreviation", "matchup"]].drop_duplicates(
        subset=["game_id", "team_abbreviation"]
    ).copy()
    opp_series["_opp"] = opp_series.apply(
        lambda r: _get_opp_abbr_data(str(r["matchup"]), str(r["team_abbreviation"])),
        axis=1,
    )
    df = df.merge(
        opp_series[["game_id", "team_abbreviation", "_opp"]],
        on=["game_id", "team_abbreviation"],
        how="left",
    )
    df = df[df["_opp"].notna() & (df["_opp"] != "")]

    # Per-game per-position-group totals (what each pos group scored vs defending_team)
    pos_game = (
        df.groupby(["game_id", "_opp", "_pos", "game_date"])[stat_cols]
        .mean()  # avg per player in that pos group vs that team
        .reset_index()
        .rename(columns={"_opp": "defending_team"})
    )
    pos_game = pos_game.sort_values(["defending_team", "_pos", "game_date"])

    roll_cols = []
    for stat in stat_cols:
        col = f"opp_{stat}_allowed_pos"
        pos_game[col] = (
            pos_game.groupby(["defending_team", "_pos"])[stat]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        roll_cols.append(col)

    result = pos_game[["game_id", "defending_team", "_pos"] + roll_cols].rename(
        columns={"_pos": "pos_group"}
    )
    result.to_csv(out_path, index=False)
    print(f"[data] Positional defense lookup → {out_path} ({len(result)} rows)")
    _LOOKUP_CACHE["positional_defense"] = result
    return result


def build_game_context_lookup() -> pd.DataFrame:
    """
    Pre-compute rolling 10-game team scoring and game totals per (game_id, team_abbreviation).

    Columns: game_id, team_abbreviation,
             roll_team_pts_10 (team's pts/game rolling avg before this game),
             roll_game_total_10 (both teams' pts rolling avg before this game)

    Cached to data/game_context.csv.
    """
    _key = "game_context"
    if _key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[_key]
    out_path = DATA_DIR / "game_context.csv"
    if out_path.exists():
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600
        if age_hours <= COMBINED_CACHE_HOURS:
            df = pd.read_csv(out_path, parse_dates=["game_date"])
            _LOOKUP_CACHE[_key] = df
            return df
        out_path.unlink()

    df = load_gamelogs()
    required = {"game_id", "team_abbreviation", "game_date", "pts"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Sum all players' pts per (game_id, team)
    team_game = (
        df.groupby(["game_id", "team_abbreviation", "game_date"])["pts"]
        .sum()
        .reset_index()
        .rename(columns={"pts": "_team_pts"})
    )

    # Sum both teams per game_id → game total
    game_total = (
        team_game.groupby("game_id")["_team_pts"]
        .sum()
        .reset_index()
        .rename(columns={"_team_pts": "_game_total"})
    )
    team_game = team_game.merge(game_total, on="game_id")

    team_game = team_game.sort_values(["team_abbreviation", "game_date"])
    team_game["roll_team_pts_10"] = (
        team_game.groupby("team_abbreviation")["_team_pts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    ).astype("float32")
    team_game["roll_game_total_10"] = (
        team_game.groupby("team_abbreviation")["_game_total"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    ).astype("float32")

    result = team_game[["game_id", "team_abbreviation", "game_date", "roll_team_pts_10", "roll_game_total_10"]]
    result.to_csv(out_path, index=False)
    print(f"[data] Game context lookup → {out_path} ({len(result)} rows)")
    _LOOKUP_CACHE["game_context"] = result
    return result


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
