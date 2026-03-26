"""
injury.py — Fetch the current NBA injury report from ESPN's public API.
Caches results for 30 minutes to avoid hammering the endpoint.

Statuses treated as "unavailable" (player excluded from screener):
  Out, Doubtful

Statuses treated as "at risk" (player flagged but not excluded):
  Questionable
"""

import time
import unicodedata
import requests

_UNAVAILABLE_STATUSES = {"out", "doubtful"}
_AT_RISK_STATUSES = {"questionable"}

_unavailable_cache: set[str] = set()
_at_risk_cache: set[str] = set()
_cache_time: float = 0.0
_TTL = 30 * 60  # 30 minutes


def _norm(name: str) -> str:
    """Normalize player name for fuzzy matching (strip accents, lowercase)."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return " ".join(name.strip().lower().split())


def get_injury_report() -> tuple[set[str], set[str]]:
    """
    Return (unavailable, at_risk) — sets of normalized player names.
    - unavailable: Out / Doubtful — skip these bets entirely
    - at_risk: Questionable — flag but don't exclude

    Cached for 30 minutes. Returns stale cache on fetch error.
    """
    global _unavailable_cache, _at_risk_cache, _cache_time

    if time.time() - _cache_time < _TTL and _cache_time > 0:
        return _unavailable_cache, _at_risk_cache

    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        unavailable: set[str] = set()
        at_risk: set[str] = set()

        for team_entry in data.get("injuries", []):
            for inj in team_entry.get("injuries", []):
                status = inj.get("status", "").lower().strip()
                name_raw = inj.get("athlete", {}).get("displayName", "").strip()
                if not name_raw:
                    continue
                name = _norm(name_raw)
                if status in _UNAVAILABLE_STATUSES:
                    unavailable.add(name)
                elif status in _AT_RISK_STATUSES:
                    at_risk.add(name)

        _unavailable_cache = unavailable
        _at_risk_cache = at_risk
        _cache_time = time.time()

        n_out = len(unavailable)
        n_q = len(at_risk)
        if n_out or n_q:
            print(f"[injury] {n_out} out/doubtful, {n_q} questionable")
        return unavailable, at_risk

    except Exception as e:
        print(f"[injury] fetch failed: {e} — using {'stale cache' if _cache_time else 'empty set'}")
        return _unavailable_cache, _at_risk_cache
