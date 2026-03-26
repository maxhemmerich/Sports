"""
features.py — Build feature matrix for each player-game.

Features:
  - Rolling avg points/reb/ast (last 5, 10, 20 games)
  - Rolling avg minutes played (last 5, 10 games) — usage proxy
  - Opponent defensive rating (avg pts/reb/ast allowed to all players)
  - Opponent team pace (avg team FGA per game — higher = faster pace)
  - Days rest since last game
  - Back-to-back flag
  - Home / Away indicator
  - Travel distance from previous game city (approximate)
"""

import time

import numpy as np
import pandas as pd
from pathlib import Path
from data import (
    load_gamelogs,
    build_defense_lookup,
    build_rolling_defense_lookup,
    build_positional_defense_lookup,
    build_game_context_lookup,
)


def _parse_minutes(val) -> float:
    """Parse NBA minutes string ('32:45' or '32.75' or 32) to float."""
    try:
        s = str(val).strip()
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(s)
    except (ValueError, TypeError):
        return float("nan")

MAX_CACHE_AGE_DAYS = int(__import__("os").getenv("MAX_CACHE_AGE_DAYS", "7"))

DATA_DIR = Path("data")

# Approximate arena coordinates (lat, lon) for travel distance calculation
TEAM_COORDS = {
    "ATL": (33.757, -84.396),
    "BOS": (42.366, -71.062),
    "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839),
    "CHI": (41.881, -87.674),
    "CLE": (41.497, -81.688),
    "DAL": (32.790, -96.810),
    "DEN": (39.749, -105.007),
    "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388),
    "HOU": (29.751, -95.362),
    "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267),
    "LAL": (34.043, -118.267),
    "MEM": (35.138, -90.050),
    "MIA": (25.781, -80.188),
    "MIL": (43.045, -87.917),
    "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082),
    "NYK": (40.750, -73.994),
    "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384),
    "PHI": (39.901, -75.172),
    "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667),
    "SAC": (38.580, -121.499),
    "SAS": (29.427, -98.437),
    "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901),
    "WAS": (38.898, -77.021),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _team_from_matchup(matchup: str, is_home: bool) -> str:
    """
    Extract team abbreviation from matchup string.
    Format: 'LAL vs. GSW'  (home=LAL) or  'LAL @ GSW' (home=GSW).
    """
    parts = matchup.replace("@", "vs.").split("vs.")
    if len(parts) != 2:
        return ""
    left = parts[0].strip()
    right = parts[1].strip()
    # The player's team is the left side; opponent is right.
    # home/away based on '@' vs 'vs.'
    return left if is_home != ("@" in matchup) else right


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features (pts, reb, ast, min, ...) per player.
    Expects df sorted by player_id, game_date.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()
    # Parse minutes to float if present
    if "min" in df.columns:
        df["min"] = df["min"].apply(_parse_minutes)
    for stat in ["pts", "reb", "ast", "fg3m", "blk", "stl", "tov", "min"]:
        if stat not in df.columns:
            continue
        for window in [5, 10, 20]:
            df[f"roll_{stat}_{window}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        # Rolling std (consistency signal) — higher = more volatile player
        for window in [5, 10]:
            df[f"roll_{stat}_std_{window}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
            )
        # Exponentially-weighted mean (span=10 ≈ half-life of ~7 games)
        df[f"ewm_{stat}_10"] = (
            df.groupby("player_id")[stat]
            .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
        )
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_rest and back_to_back columns."""
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(7).clip(upper=14)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)
    return df


def add_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Derive home/away from matchup string (1=home, 0=away)."""
    df = df.copy()
    df["is_home"] = (~df["matchup"].str.contains("@")).astype(int)
    return df


def add_travel_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate km traveled since previous game.
    Uses team abbreviation extracted from matchup.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()

    def get_city_abbr(row):
        # The player's own team abbreviation is in team_abbreviation column
        return row.get("team_abbreviation", "")

    def get_opponent_abbr(matchup: str) -> str:
        for sep in [" vs. ", " @ "]:
            if sep in matchup:
                parts = matchup.split(sep)
                # opponent is always the right side of the matchup string
                return parts[1].strip()
        return ""

    # Venue = home team city.  If is_home=1 → player's own arena, else opponent's arena.
    def venue_coords(row) -> tuple[float, float] | None:
        team = row.get("team_abbreviation", "")
        is_home = row.get("is_home", 1)
        if is_home:
            return TEAM_COORDS.get(team)
        opp = get_opponent_abbr(row.get("matchup", ""))
        return TEAM_COORDS.get(opp)

    coords = df.apply(venue_coords, axis=1)
    prev_coords = df.groupby("player_id").apply(
        lambda g: g.apply(venue_coords, axis=1).shift(1), include_groups=False
    ).reset_index(level=0, drop=True)

    def calc_dist(row_idx):
        c1 = prev_coords.get(row_idx)
        c2 = coords.get(row_idx)
        if c1 and c2:
            return haversine_km(c1[0], c1[1], c2[0], c2[1])
        return 0.0

    df["travel_km"] = [calc_dist(i) for i in df.index]
    return df


_DEFENSE_STAT_DEFAULTS = {
    "pts": 110.0, "reb": 43.0, "ast": 24.0,
    "fg3m": 12.0, "blk": 5.0, "stl": 7.0, "tov": 13.0,
}

# Defaults for game-context features (league averages)
_GAME_TOTAL_DEFAULT = 225.0       # NBA league-avg total points per game
_TEAM_PTS_DEFAULT   = 112.5       # half of game total
_ROLL_DEF_WINDOW    = 15          # rolling opponent defense window (games)


def add_defense_features(df: pd.DataFrame, def_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-stat opponent defensive averages into game log df.
    def_lookup has columns: team_abbreviation, season,
                            avg_pts_allowed, avg_reb_allowed, avg_ast_allowed, ...
    """
    df = df.copy()

    stat_cols = [f"avg_{s}_allowed" for s in _DEFENSE_STAT_DEFAULTS]
    opp_cols  = [f"opp_{s}_allowed" for s in _DEFENSE_STAT_DEFAULTS]

    if def_lookup.empty or "matchup" not in df.columns:
        for col, (_, default) in zip(opp_cols, _DEFENSE_STAT_DEFAULTS.items()):
            df[col] = default
        # backwards-compat alias
        df["opp_avg_pts_allowed"] = _DEFENSE_STAT_DEFAULTS["pts"]
        return df

    df["opp_abbr_def"] = df.apply(
        lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))),
        axis=1,
    )

    avail_def_cols = [c for c in stat_cols if c in def_lookup.columns]
    merge_cols = ["team_abbreviation", "season"] + avail_def_cols
    merged = df.merge(
        def_lookup[merge_cols].rename(columns={"team_abbreviation": "opp_abbr_def"}),
        on=["opp_abbr_def", "season"],
        how="left",
    )

    for stat, default in _DEFENSE_STAT_DEFAULTS.items():
        src = f"avg_{stat}_allowed"
        dst = f"opp_{stat}_allowed"
        if src in merged.columns:
            league_avg = merged[src].median()
            merged[dst] = merged[src].fillna(league_avg if not pd.isna(league_avg) else default)
            merged = merged.drop(columns=[src], errors="ignore")
        else:
            merged[dst] = default

    # backwards-compat alias used by _CONTEXT_COLS
    merged["opp_avg_pts_allowed"] = merged["opp_pts_allowed"]
    merged = merged.drop(columns=["opp_abbr_def"], errors="ignore")
    return merged


def add_pace_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling FGA and usage_pct (player FGA / team FGA) per player.
    usage_pct captures the player's role share independent of team pace.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()
    if "fga" in df.columns:
        df["roll_fga_10"] = (
            df.groupby("player_id")["fga"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
    else:
        df["roll_fga_10"] = 0.0

    # usage_pct: player's share of their team's FGA
    pace_lookup = build_pace_lookup()
    if not pace_lookup.empty and "team_abbreviation" in df.columns:
        df = df.merge(
            pace_lookup.rename(columns={"team_avg_fga": "_team_fga"}),
            on="team_abbreviation", how="left"
        )
        df["_team_fga"] = df["_team_fga"].fillna(85.0)
        df["usage_pct"] = df["roll_fga_10"] / df["_team_fga"] * 100
        df = df.drop(columns=["_team_fga"], errors="ignore")
    else:
        df["usage_pct"] = 0.0
    return df


def build_pace_lookup() -> pd.DataFrame:
    """
    Compute each team's average FGA per game across all loaded seasons.
    FGA per game is a reliable pace proxy — fast teams attempt more shots.
    Returns DataFrame with columns: team_abbreviation, team_avg_fga
    """
    out_path = DATA_DIR / "team_pace.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    df = load_gamelogs()
    required = {"fga", "team_abbreviation", "game_id"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Sum all players' FGA per team per game, then average across games
    team_game = (
        df.groupby(["team_abbreviation", "game_id"])["fga"]
        .sum()
        .reset_index()
    )
    pace = (
        team_game.groupby("team_abbreviation")["fga"]
        .mean()
        .reset_index()
        .rename(columns={"fga": "team_avg_fga"})
    )
    pace.to_csv(out_path, index=False)
    print(f"[features] Team pace lookup → {out_path}")
    return pace


def _get_opp_abbr(matchup: str, team_abbr: str) -> str:
    """Extract opponent team abbreviation from matchup string."""
    for sep in [" vs. ", " @ "]:
        if sep in matchup:
            left, right = matchup.split(sep, 1)
            left, right = left.strip(), right.strip()
            return right if left == team_abbr else left
    return ""


def add_opp_pace(df: pd.DataFrame, pace_lookup: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent team's avg FGA (pace proxy) into each player-game row."""
    if pace_lookup.empty or "matchup" not in df.columns:
        df["opp_team_pace"] = 85.0
        return df

    df = df.copy()
    df["opp_abbr"] = df.apply(
        lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))),
        axis=1,
    )
    df = df.merge(
        pace_lookup.rename(columns={"team_abbreviation": "opp_abbr", "team_avg_fga": "opp_team_pace"}),
        on="opp_abbr",
        how="left",
    )
    df["opp_team_pace"] = df["opp_team_pace"].fillna(85.0)
    df = df.drop(columns=["opp_abbr"], errors="ignore")
    return df


TARGET_COL = "pts"
TARGET_COL_REB = "reb"
TARGET_COL_AST = "ast"
TARGET_COL_FG3M = "fg3m"
TARGET_COL_BLK = "blk"
TARGET_COL_STL = "stl"
TARGET_COL_TOV = "tov"

# Shared context features (non-stat-specific)
_CONTEXT_COLS = [
    "opp_avg_pts_allowed",   # general defensive strength proxy (kept for all models)
    "opp_team_pace",
    "days_rest",
    "back_to_back",
    "is_home",
    "travel_km",
    "roll_fga_10",
    "usage_pct",             # player FGA / team FGA — role signal independent of pace
    "roll_min_5",            # recent minutes — strongest usage proxy
    "roll_min_10",
    "ewm_min_10",
    "min_trend",             # roll_min_5 / roll_min_20 — detects role changes (injury/trade)
    "roll_team_pts_10",      # player's team rolling 10-game pts/game (pace/role proxy)
    "roll_game_total_10",    # rolling 10-game game total (pace context)
]

FEATURE_COLS = [
    "roll_pts_5", "roll_pts_10", "roll_pts_20", "ewm_pts_10",
    "roll_pts_std_5", "roll_pts_std_10",
    "vs_opp_pts_avg",
    "opp_pts_allowed",           # season-long opponent pts allowed
    "opp_pts_allowed_roll15",    # rolling 15-game opponent pts allowed (more current)
    "opp_pts_allowed_pos",       # position-specific opponent pts allowed
] + _CONTEXT_COLS

FEATURE_COLS_REB = [
    "roll_reb_5", "roll_reb_10", "roll_reb_20", "ewm_reb_10",
    "roll_reb_std_5", "roll_reb_std_10",
    "vs_opp_reb_avg",
    "opp_reb_allowed",
    "opp_reb_allowed_roll15",
    "opp_reb_allowed_pos",
] + _CONTEXT_COLS

FEATURE_COLS_AST = [
    "roll_ast_5", "roll_ast_10", "roll_ast_20", "ewm_ast_10",
    "roll_ast_std_5", "roll_ast_std_10",
    "vs_opp_ast_avg",
    "opp_ast_allowed",
    "opp_ast_allowed_roll15",
    "opp_ast_allowed_pos",
] + _CONTEXT_COLS

FEATURE_COLS_FG3M = ["roll_fg3m_5", "roll_fg3m_10", "roll_fg3m_20", "ewm_fg3m_10", "roll_fg3m_std_5", "roll_fg3m_std_10", "vs_opp_fg3m_avg", "opp_fg3m_allowed", "opp_fg3m_allowed_roll15"] + _CONTEXT_COLS
FEATURE_COLS_BLK  = ["roll_blk_5",  "roll_blk_10",  "roll_blk_20",  "ewm_blk_10",  "roll_blk_std_5",  "roll_blk_std_10",  "vs_opp_blk_avg",  "opp_blk_allowed",  "opp_blk_allowed_roll15"]  + _CONTEXT_COLS
FEATURE_COLS_STL  = ["roll_stl_5",  "roll_stl_10",  "roll_stl_20",  "ewm_stl_10",  "roll_stl_std_5",  "roll_stl_std_10",  "vs_opp_stl_avg",  "opp_stl_allowed",  "opp_stl_allowed_roll15"]  + _CONTEXT_COLS
FEATURE_COLS_TOV  = ["roll_tov_5",  "roll_tov_10",  "roll_tov_20",  "ewm_tov_10",  "roll_tov_std_5",  "roll_tov_std_10",  "vs_opp_tov_avg",  "opp_tov_allowed",  "opp_tov_allowed_roll15"]  + _CONTEXT_COLS

# Map market key → (feature_cols, target_col)
MARKET_CONFIG = {
    "player_points":    (FEATURE_COLS,      TARGET_COL),
    "player_rebounds":  (FEATURE_COLS_REB,  TARGET_COL_REB),
    "player_assists":   (FEATURE_COLS_AST,  TARGET_COL_AST),
    "player_threes":    (FEATURE_COLS_FG3M, TARGET_COL_FG3M),
    "player_blocks":    (FEATURE_COLS_BLK,  TARGET_COL_BLK),
    "player_steals":    (FEATURE_COLS_STL,  TARGET_COL_STL),
    "player_turnovers": (FEATURE_COLS_TOV,  TARGET_COL_TOV),
}


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling team-scoring and game-total features via pre-computed lookup.
    - roll_team_pts_10: player's team rolling 10-game pts per game
    - roll_game_total_10: rolling 10-game total (both teams' pts)
    """
    if not {"game_id", "team_abbreviation"}.issubset(df.columns):
        df["roll_team_pts_10"]   = _TEAM_PTS_DEFAULT
        df["roll_game_total_10"] = _GAME_TOTAL_DEFAULT
        return df

    lookup = build_game_context_lookup()
    if lookup.empty:
        df["roll_team_pts_10"]   = _TEAM_PTS_DEFAULT
        df["roll_game_total_10"] = _GAME_TOTAL_DEFAULT
        return df

    df = df.merge(lookup, on=["game_id", "team_abbreviation"], how="left")
    df["roll_team_pts_10"]   = df["roll_team_pts_10"].fillna(_TEAM_PTS_DEFAULT)
    df["roll_game_total_10"] = df["roll_game_total_10"].fillna(_GAME_TOTAL_DEFAULT)
    return df


def add_rolling_opp_defense(df: pd.DataFrame, window: int = _ROLL_DEF_WINDOW) -> pd.DataFrame:
    """
    Merge pre-computed rolling N-game opponent defensive averages into the DataFrame.
    Adds columns: opp_{stat}_allowed_roll{window}
    """
    roll_cols = [f"opp_{stat}_allowed_roll{window}" for stat in _DEFENSE_STAT_DEFAULTS]

    if "game_id" not in df.columns or "team_abbreviation" not in df.columns or "matchup" not in df.columns:
        for stat, default in _DEFENSE_STAT_DEFAULTS.items():
            df[f"opp_{stat}_allowed_roll{window}"] = default
        return df

    lookup = build_rolling_defense_lookup(window)
    if lookup.empty:
        for stat, default in _DEFENSE_STAT_DEFAULTS.items():
            df[f"opp_{stat}_allowed_roll{window}"] = default
        return df

    df = df.copy()
    df["_opp_abbr"] = df.apply(
        lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))),
        axis=1,
    )

    df = df.merge(
        lookup.rename(columns={"defending_team": "_opp_abbr"}),
        on=["game_id", "_opp_abbr"],
        how="left",
    )

    for stat, default in _DEFENSE_STAT_DEFAULTS.items():
        col = f"opp_{stat}_allowed_roll{window}"
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default

    df = df.drop(columns=["_opp_abbr"], errors="ignore")
    return df


def add_positional_defense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-computed positional defensive averages into the DataFrame.
    Position group inferred from roll_reb_10 (guard/forward/big).
    Adds: opp_pts_allowed_pos, opp_reb_allowed_pos, opp_ast_allowed_pos
    """
    pos_stats = ["pts", "reb", "ast"]
    pos_cols = [f"opp_{s}_allowed_pos" for s in pos_stats]

    if "game_id" not in df.columns or "matchup" not in df.columns or "roll_reb_10" not in df.columns:
        for stat in pos_stats:
            df[f"opp_{stat}_allowed_pos"] = _DEFENSE_STAT_DEFAULTS[stat]
        return df

    lookup = build_positional_defense_lookup()
    if lookup.empty:
        for stat in pos_stats:
            df[f"opp_{stat}_allowed_pos"] = _DEFENSE_STAT_DEFAULTS[stat]
        return df

    df = df.copy()

    reb = df["roll_reb_10"].fillna(0)
    df["_pos"] = "guard"
    df.loc[reb >= 4, "_pos"] = "forward"
    df.loc[reb >= 7, "_pos"] = "big"

    df["_opp_abbr"] = df.apply(
        lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))),
        axis=1,
    )

    df = df.merge(
        lookup.rename(columns={"defending_team": "_opp_abbr", "pos_group": "_pos"}),
        on=["game_id", "_opp_abbr", "_pos"],
        how="left",
    )

    for stat in pos_stats:
        col = f"opp_{stat}_allowed_pos"
        if col in df.columns:
            df[col] = df[col].fillna(_DEFENSE_STAT_DEFAULTS.get(stat, 0.0))
        else:
            df[col] = _DEFENSE_STAT_DEFAULTS.get(stat, 0.0)

    df = df.drop(columns=["_opp_abbr", "_pos"], errors="ignore")
    return df


def add_vs_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game row, add the player's EWM-weighted avg pts/reb/ast vs that
    specific opponent using only prior games (no lookahead). Using EWM (span=10)
    instead of a flat expanding mean so recent matchups have more influence.
    Falls back to ewm_stat_10 when no prior history exists against that opponent.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()
    df["_opp_abbr"] = df.apply(
        lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))),
        axis=1,
    )
    for stat in ["pts", "reb", "ast", "fg3m", "blk", "stl", "tov"]:
        if stat not in df.columns:
            continue
        col = f"vs_opp_{stat}_avg"
        df[col] = (
            df.groupby(["player_id", "_opp_abbr"])[stat]
            .transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
        )
        fallback = f"ewm_{stat}_10"
        if fallback in df.columns:
            df[col] = df[col].fillna(df[fallback])
    df = df.drop(columns=["_opp_abbr"])
    return df


def build_feature_matrix(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full pipeline: load data → add all features → return feature matrix.
    Caches result to data/feature_matrix.csv.
    """
    cache = DATA_DIR / "feature_matrix.csv"
    if cache.exists():
        age_days = (time.time() - cache.stat().st_mtime) / 86400
        cached_cols = set(pd.read_csv(cache, nrows=0).columns)
        all_required = set(FEATURE_COLS + FEATURE_COLS_REB + FEATURE_COLS_AST + FEATURE_COLS_FG3M)
        if age_days <= MAX_CACHE_AGE_DAYS and all_required.issubset(cached_cols):
            print(f"[features] Loading feature matrix from cache: {cache}")
            out = pd.read_csv(cache, low_memory=False)
            out["game_date"] = pd.to_datetime(out["game_date"])
            return out
        reason = "stale" if age_days > MAX_CACHE_AGE_DAYS else "missing new feature columns"
        print(f"[features] Cache {reason} — rebuilding ...")
        cache.unlink()

    if df is None:
        df = load_gamelogs()

    print("[features] Building feature matrix ...")
    # Pre-compute all lookup tables first (cached to CSV) before heavy in-memory ops.
    # This keeps peak memory low: lookups are small; the full df is only held once.
    def_lookup = build_defense_lookup()
    pace_lookup = build_pace_lookup()
    build_rolling_defense_lookup()     # caches rolling_defense_15g.csv
    build_positional_defense_lookup()  # caches positional_defense.csv
    build_game_context_lookup()        # caches game_context.csv

    df = add_rolling_features(df)
    df = add_rest_features(df)
    df = add_home_away(df)
    df = add_travel_distance(df)
    df = add_pace_proxy(df)
    df = add_defense_features(df, def_lookup)
    df = add_opp_pace(df, pace_lookup)
    df = add_vs_opponent_features(df)
    df = add_game_context_features(df)
    df = add_rolling_opp_defense(df)
    df = add_positional_defense(df)

    # Role-change signal: recent minutes vs longer-term average
    if "roll_min_5" in df.columns and "roll_min_20" in df.columns:
        df["min_trend"] = df["roll_min_5"] / (df["roll_min_20"] + 0.1)
    else:
        df["min_trend"] = 1.0

    # Drop rows without enough history (first few games per player)
    df = df.dropna(subset=["roll_pts_5", TARGET_COL])
    df.to_csv(cache, index=False)
    print(f"[features] Feature matrix: {len(df)} rows → {cache}")
    return df


def build_live_features(
    player_name: str,
    opponent_team_abbr: str,
    is_home: bool,
    game_date: str,
    df_history: pd.DataFrame | None = None,
    def_lookup: pd.DataFrame | None = None,
    target: str = "pts",
    game_total: float | None = None,
    team_implied_total: float | None = None,
) -> dict:
    """
    Build a single feature row for live prediction.

    Args:
        player_name: NBA player name (must match nba_api spelling)
        opponent_team_abbr: e.g. 'GSW'
        is_home: True if player's team is home
        game_date: ISO date string 'YYYY-MM-DD'
        df_history: pre-loaded game log DataFrame (avoids per-call reload)
        def_lookup: pre-loaded defense lookup DataFrame (avoids per-call reload)
        target: 'pts', 'reb', or 'ast'

    Returns:
        dict mapping feature_col → value
    """
    if df_history is None:
        df_history = load_gamelogs()

    if def_lookup is None:
        def_lookup = build_defense_lookup()

    # Filter to player's history before game_date
    player_df = df_history[
        (df_history["player_name"] == player_name)
        & (df_history["game_date"] < pd.Timestamp(game_date))
    ].sort_values("game_date")

    if player_df.empty:
        return {}

    # Rolling averages for the target stat
    stat_col = target  # pts / reb / ast
    if stat_col not in player_df.columns:
        return {}
    stat_series = player_df[stat_col]
    roll5  = float(stat_series.tail(5).mean())
    roll10 = float(stat_series.tail(10).mean())
    roll20 = float(stat_series.tail(20).mean())
    ewm10  = float(stat_series.ewm(span=10, min_periods=1).mean().iloc[-1])
    # Consistency signal: 0 when too few games to compute std
    roll_std5  = float(stat_series.tail(5).std())  if len(stat_series) >= 2 else 0.0
    roll_std10 = float(stat_series.tail(10).std()) if len(stat_series) >= 2 else 0.0
    if pd.isna(roll_std5):  roll_std5  = 0.0
    if pd.isna(roll_std10): roll_std10 = 0.0

    last_game = player_df.iloc[-1]
    last_date = pd.Timestamp(last_game["game_date"])
    days_rest = (pd.Timestamp(game_date) - last_date).days
    back_to_back = int(days_rest == 1)

    # Travel distance
    prev_is_home = "is_home" in last_game and last_game["is_home"]
    travel_km = 0.0
    if "team_abbreviation" in last_game:
        team = last_game["team_abbreviation"]
        prev_venue = TEAM_COORDS.get(team) if prev_is_home else None
        if prev_venue is None:
            opp_abbr = ""
            if "matchup" in last_game:
                matchup = last_game["matchup"]
                for sep in [" vs. ", " @ "]:
                    if sep in matchup:
                        opp_abbr = matchup.split(sep)[1].strip()
                        break
            prev_venue = TEAM_COORDS.get(opp_abbr)

        curr_venue = TEAM_COORDS.get(team) if is_home else TEAM_COORDS.get(opponent_team_abbr)
        if prev_venue and curr_venue:
            travel_km = haversine_km(prev_venue[0], prev_venue[1], curr_venue[0], curr_venue[1])

    # Opponent defense — look up per-stat averages allowed by this team
    opp_avg_pts_allowed = _DEFENSE_STAT_DEFAULTS["pts"]
    opp_stat_allowed = _DEFENSE_STAT_DEFAULTS.get(stat_col, _DEFENSE_STAT_DEFAULTS["pts"])
    if not def_lookup.empty and "team_abbreviation" in def_lookup.columns:
        opp_rows = def_lookup[def_lookup["team_abbreviation"] == opponent_team_abbr]
        if not opp_rows.empty:
            pts_col = "avg_pts_allowed" if "avg_pts_allowed" in opp_rows.columns else "opp_pts_allowed"
            if pts_col in opp_rows.columns:
                opp_avg_pts_allowed = float(opp_rows[pts_col].mean())
            stat_allowed_col = f"avg_{stat_col}_allowed"
            if stat_allowed_col in opp_rows.columns:
                opp_stat_allowed = float(opp_rows[stat_allowed_col].mean())

    roll_fga_10 = 0.0
    if "fga" in player_df.columns:
        roll_fga_10 = float(player_df["fga"].tail(10).mean())

    # Minutes played — strongest usage proxy for all counting stats
    roll_min_5 = float("nan")
    roll_min_10 = float("nan")
    roll_min_20 = float("nan")
    ewm_min_10 = float("nan")
    if "min" in player_df.columns:
        min_series = player_df["min"].apply(_parse_minutes)
        roll_min_5  = float(min_series.tail(5).mean())
        roll_min_10 = float(min_series.tail(10).mean())
        roll_min_20 = float(min_series.tail(20).mean())
        ewm_min_10  = float(min_series.ewm(span=10, min_periods=1).mean().iloc[-1])

    # Opponent team pace (avg FGA per game — higher = faster pace = more possessions)
    pace_lookup = build_pace_lookup()
    opp_team_pace = 85.0  # league avg fallback
    player_team_fga = 85.0  # fallback for usage calculation
    player_team = str(player_df.iloc[-1].get("team_abbreviation", ""))
    if not pace_lookup.empty and "team_abbreviation" in pace_lookup.columns:
        opp_row = pace_lookup[pace_lookup["team_abbreviation"] == opponent_team_abbr]
        if not opp_row.empty:
            opp_team_pace = float(opp_row["team_avg_fga"].iloc[0])
        own_row = pace_lookup[pace_lookup["team_abbreviation"] == player_team]
        if not own_row.empty:
            player_team_fga = float(own_row["team_avg_fga"].iloc[0])

    # Usage rate: player's share of team FGA (role signal independent of team pace)
    usage_pct = (roll_fga_10 / player_team_fga * 100) if player_team_fga > 0 else 0.0

    # Player's historical avg vs this specific opponent — EWM so recent games matter more
    vs_opp_avg = ewm10  # fallback: recency-weighted overall mean
    if opponent_team_abbr and "matchup" in player_df.columns and "team_abbreviation" in player_df.columns:
        opp_flags = player_df.apply(
            lambda r: _get_opp_abbr(str(r.get("matchup", "")), str(r.get("team_abbreviation", ""))) == opponent_team_abbr,
            axis=1,
        )
        vs_opp_games = player_df[opp_flags]
        if not vs_opp_games.empty:
            vs_opp_avg = float(
                vs_opp_games[stat_col].ewm(span=10, min_periods=1).mean().iloc[-1]
            )

    # ── Game context features ──────────────────────────────────────────────────
    # Rolling team scoring averages from historical logs
    roll_team_pts_10   = _TEAM_PTS_DEFAULT
    roll_game_total_10 = _GAME_TOTAL_DEFAULT
    if "pts" in player_df.columns and "game_id" in df_history.columns and player_team:
        # Compute from full game logs (not just player_df) to sum all teammates
        cutoff = pd.Timestamp(game_date)
        all_before = df_history[df_history["game_date"] < cutoff]
        team_game_pts = (
            all_before[all_before["team_abbreviation"] == player_team]
            .groupby("game_id")["pts"].sum()
        )
        # Get dates for those game IDs
        game_dates = (
            all_before[all_before["team_abbreviation"] == player_team]
            .groupby("game_id")["game_date"].first()
        )
        if not team_game_pts.empty:
            tdf = pd.DataFrame({"pts": team_game_pts, "game_date": game_dates}).sort_values("game_date")
            roll_team_pts_10 = float(tdf["pts"].iloc[:-0 if len(tdf) == 0 else None].tail(10).mean())

        # Game total: need both teams' pts per game
        game_totals_by_id = all_before.groupby("game_id")["pts"].sum()
        gt_dates = all_before.groupby("game_id")["game_date"].first()
        team_game_ids = set(team_game_pts.index)
        if team_game_ids:
            gt_df = pd.DataFrame({"total": game_totals_by_id, "game_date": gt_dates})
            gt_df = gt_df[gt_df.index.isin(team_game_ids)].sort_values("game_date")
            if not gt_df.empty:
                roll_game_total_10 = float(gt_df["total"].tail(10).mean())

    # Use live market game_total / team_implied_total if provided (more accurate than historical avg)
    if game_total is not None and not pd.isna(game_total):
        roll_game_total_10 = float(game_total)
    if team_implied_total is not None and not pd.isna(team_implied_total):
        roll_team_pts_10 = float(team_implied_total)

    # ── Rolling opponent defense (last 15 games) ───────────────────────────────
    opp_stat_allowed_roll = _DEFENSE_STAT_DEFAULTS.get(stat_col, opp_stat_allowed)
    if opponent_team_abbr and "game_id" in df_history.columns and "team_abbreviation" in df_history.columns:
        cutoff = pd.Timestamp(game_date)
        all_before = df_history[df_history["game_date"] < cutoff]
        # Find all game_ids where the opponent team played
        opp_game_ids_dated = (
            all_before[all_before["team_abbreviation"] == opponent_team_abbr]
            .groupby("game_id")["game_date"].first()
            .sort_values()
        )
        if not opp_game_ids_dated.empty:
            recent_gids = opp_game_ids_dated.index[-_ROLL_DEF_WINDOW:]
            # Stats allowed = what the non-opponent team scored in those games
            opp_allowed_rows = all_before[
                (all_before["game_id"].isin(recent_gids)) &
                (all_before["team_abbreviation"] != opponent_team_abbr)
            ]
            if not opp_allowed_rows.empty and stat_col in opp_allowed_rows.columns:
                per_game = opp_allowed_rows.groupby("game_id")[stat_col].sum()
                opp_stat_allowed_roll = float(per_game.mean())

    # ── Positional defense ────────────────────────────────────────────────────
    roll_reb_10_for_pos = float(player_df["reb"].tail(10).mean()) if "reb" in player_df.columns else 0.0
    if pd.isna(roll_reb_10_for_pos):
        roll_reb_10_for_pos = 0.0
    if roll_reb_10_for_pos >= 7:
        pos_group = "big"
    elif roll_reb_10_for_pos >= 4:
        pos_group = "forward"
    else:
        pos_group = "guard"

    # Positional defense: avg stat allowed by opponent to players in this position group
    opp_stat_allowed_pos = opp_stat_allowed  # fallback to season-long
    if opponent_team_abbr and stat_col in ["pts", "reb", "ast"] and "game_id" in df_history.columns:
        cutoff = pd.Timestamp(game_date)
        all_before = df_history[df_history["game_date"] < cutoff].copy()
        if "roll_reb_10" not in all_before.columns and "reb" in all_before.columns:
            all_before["_tmp_reb_roll"] = (
                all_before.groupby("player_id")["reb"]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
                if "player_id" in all_before.columns else all_before["reb"]
            )
            reb_col = "_tmp_reb_roll"
        else:
            reb_col = "roll_reb_10" if "roll_reb_10" in all_before.columns else None

        if reb_col:
            rr = all_before[reb_col].fillna(0)
            all_before["_pos"] = "guard"
            all_before.loc[rr >= 4, "_pos"] = "forward"
            all_before.loc[rr >= 7, "_pos"] = "big"

            opp_game_ids_dated = (
                all_before[all_before["team_abbreviation"] == opponent_team_abbr]
                .groupby("game_id")["game_date"].first()
                .sort_values()
            )
            if not opp_game_ids_dated.empty:
                recent_gids = opp_game_ids_dated.index[-15:]
                pos_rows = all_before[
                    (all_before["game_id"].isin(recent_gids)) &
                    (all_before["team_abbreviation"] != opponent_team_abbr) &
                    (all_before["_pos"] == pos_group)
                ]
                if not pos_rows.empty and stat_col in pos_rows.columns:
                    opp_stat_allowed_pos = float(pos_rows[stat_col].mean())

    prefix = stat_col  # pts / reb / ast / fg3m / blk / stl / tov
    return {
        f"roll_{prefix}_5":        roll5,
        f"roll_{prefix}_10":       roll10,
        f"roll_{prefix}_20":       roll20,
        f"ewm_{prefix}_10":        ewm10,
        f"roll_{prefix}_std_5":    roll_std5,
        f"roll_{prefix}_std_10":   roll_std10,
        f"vs_opp_{prefix}_avg":    vs_opp_avg,
        f"opp_{prefix}_allowed":          opp_stat_allowed,        # season-long
        f"opp_{prefix}_allowed_roll15":   opp_stat_allowed_roll,   # rolling 15-game
        f"opp_{prefix}_allowed_pos":      opp_stat_allowed_pos,    # positional
        "opp_avg_pts_allowed":   opp_avg_pts_allowed,
        "opp_team_pace":         opp_team_pace,
        "days_rest":             min(days_rest, 14),
        "back_to_back":          back_to_back,
        "is_home":               int(is_home),
        "travel_km":             travel_km,
        "roll_fga_10":           roll_fga_10,
        "usage_pct":             usage_pct,
        "roll_min_5":            roll_min_5,
        "roll_min_10":           roll_min_10,
        "ewm_min_10":            ewm_min_10,
        "min_trend":             (roll_min_5 / (roll_min_20 + 0.1))
                                 if not (pd.isna(roll_min_5) or pd.isna(roll_min_20))
                                 else 1.0,
        "roll_team_pts_10":      roll_team_pts_10,
        "roll_game_total_10":    roll_game_total_10,
        # Pass-through for non-pts/reb/ast markets (no positional lookup for fg3m/blk/stl/tov)
        "opp_fg3m_allowed_roll15": _DEFENSE_STAT_DEFAULTS["fg3m"] if stat_col != "fg3m" else opp_stat_allowed_roll,
        "opp_blk_allowed_roll15":  _DEFENSE_STAT_DEFAULTS["blk"]  if stat_col != "blk"  else opp_stat_allowed_roll,
        "opp_stl_allowed_roll15":  _DEFENSE_STAT_DEFAULTS["stl"]  if stat_col != "stl"  else opp_stat_allowed_roll,
        "opp_tov_allowed_roll15":  _DEFENSE_STAT_DEFAULTS["tov"]  if stat_col != "tov"  else opp_stat_allowed_roll,
        "opp_reb_allowed_pos":     opp_stat_allowed_pos if stat_col == "reb" else _DEFENSE_STAT_DEFAULTS["reb"],
        "opp_ast_allowed_pos":     opp_stat_allowed_pos if stat_col == "ast" else _DEFENSE_STAT_DEFAULTS["ast"],
    }


if __name__ == "__main__":
    print("=== Feature Matrix Build ===")
    fm = build_feature_matrix()
    print(f"\nShape: {fm.shape}")
    print(f"\nFeature columns present: {[c for c in FEATURE_COLS if c in fm.columns]}")
    print("\nSample (5 rows):")
    display_cols = ["player_name", "game_date"] + [c for c in FEATURE_COLS if c in fm.columns] + ["pts"]
    print(fm[display_cols].dropna().head(5).to_string(index=False))
    print("\n[features.py] Done.")
