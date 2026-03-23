"""
NBA Props — Web Dashboard
Flask server that shares in-process state with the screener loop.

Usage (automatic):  started by screener.py --loop
External access:    ngrok http 5050
                    cloudflared tunnel --url http://localhost:5050
"""
from __future__ import annotations

import json
import queue
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from flask import Flask, Response, jsonify, request, stream_with_context
except ImportError:
    raise SystemExit("pip install flask  (or: pip install -r requirements.txt)")

# ── Shared state (set by start_dashboard) ─────────────────────────────────────
_st: dict = {}
_lock: threading.Lock = threading.Lock()
_clients: list[queue.Queue] = []

app = Flask(__name__)
app.config["SECRET_KEY"] = "nba-props-dash"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def _build_state() -> dict:
    from screener import (  # lazy — avoids circular import at module load
        TRACKER_PATH, DEFAULT_BOOKS, DEPOSIT, ADJUSTMENTS_PATH,
        _at_risk_per_book,
        _bet_key, _position,
    )

    with _lock:
        bets = _st.get("latest_bets", pd.DataFrame()).copy()
        balances = dict(_st.get("book_balances", {}))
        skipped = set(_st.get("skipped_keys", set()))
        placed = set(_st.get("placed_positions", set()))
        active_books = list(_st.get("active_books", DEFAULT_BOOKS))
        cfg = {
            "min_edge": _st.get("min_edge", 4.0),
            "min_diff": _st.get("min_diff", 1.5),
            "interval": _st.get("interval", 60),
        }

    # ── Book totals (avail + at-risk) ─────────────────────────────────────────
    at_risk = _at_risk_per_book()
    book_info: dict[str, dict] = {}
    all_books = sorted(set(list(balances.keys()) + list(at_risk.keys()) + active_books))
    for book in all_books:
        avail = balances.get(book, 0.0)
        risk = at_risk.get(book, 0.0)
        book_info[book] = {
            "avail": round(avail, 2),
            "at_risk": round(risk, 2),
            "total": round(avail + risk, 2),
            "active": book in active_books,
        }
    total_balance = sum(b["avail"] for b in book_info.values())

    # ── Open bets ─────────────────────────────────────────────────────────────
    open_bets: list[dict] = []
    wagered = to_gain = to_lose = 0.0
    if TRACKER_PATH.exists():
        try:
            tr = pd.read_csv(TRACKER_PATH)
            pending = tr[tr["result"].isna() | (tr["result"].astype(str).str.strip() == "")]
            for orig_idx, row in pending.iterrows():
                stake = float(row.get("entered_$", 0) or 0)
                odds = float(row.get("odds", -110) or -110)
                dec = _american_to_decimal(odds)
                profit = round(stake * (dec - 1), 2)
                wagered += stake
                to_gain += profit
                to_lose += stake
                mkt = str(row.get("market", "")).replace("player_", "")
                open_bets.append({
                    "tracker_idx": int(orig_idx),
                    "date": str(row.get("date", "")),
                    "player": str(row.get("player", "")),
                    "market": mkt,
                    "line": float(row.get("line", 0)),
                    "side": str(row.get("side", "")),
                    "odds": int(odds),
                    "bookmaker": str(row.get("bookmaker", "")),
                    "entered": round(stake, 2),
                    "to_win": profit,
                })
        except Exception:
            pass

    # ── Potential bets ────────────────────────────────────────────────────────
    def _sf(v, d=0.0):
        """Safe float: return d if v is NaN/None/non-numeric."""
        try:
            f = float(v)
            return f if f == f else d  # NaN check
        except (TypeError, ValueError):
            return d

    def _si(v, d=0):
        """Safe int."""
        try:
            f = float(v)
            return int(f) if f == f else d
        except (TypeError, ValueError):
            return d

    potential: list[dict] = []
    if not bets.empty:
        for _, row in bets.iterrows():
            key = _bet_key(row)
            mkt = str(row.get("market", "")).replace("player_", "")
            potential.append({
                "key": list(key),
                "player": str(row["player"]),
                "market": mkt,
                "line": _sf(row.get("line")),
                "side": str(row["side"]),
                "odds": _si(row.get("odds"), -110),
                "bookmaker": str(row.get("bookmaker", "")),
                "suggested": _si(row.get("bet_dollars")),
                "edge_pct": round(_sf(row.get("edge_pct")), 1),
                "prediction": round(_sf(row.get("prediction")), 1),
                "game": str(row.get("game", "")),
                "skipped": key in skipped,
                "placed": key in placed,
            })

    # ── PnL + chart history (single tracker read so chart == overall_pnl) ─────
    today_pnl = 0.0
    overall_pnl = 0.0
    chart_dates: list[str] = []
    chart_values: list[float] = []
    trade_labels: list[str] = []
    trade_pnl: list[float] = []
    try:
        # collect per-trade events
        _events: list[tuple[str, float, str]] = []  # (date, pnl, label)
        if TRACKER_PATH.exists():
            _tr = pd.read_csv(TRACKER_PATH)
            _settled = _tr[_tr["result"].astype(str).str.strip().isin(["WIN", "LOSS", "PUSH"])].copy()
            _settled = _settled.sort_values("date", kind="stable")
            _today = date.today().isoformat()
            for _, _r in _settled.iterrows():
                _res = str(_r.get("result", "")).strip().upper()
                _amt = _sf(_r.get("entered_$", 0))
                _odds = _sf(_r.get("odds", 0))
                _d = str(_r.get("date", "")).strip()
                if _res == "WIN":
                    _profit = _amt * (_odds / 100) if _odds > 0 else (_amt * (100 / abs(_odds)) if _odds != 0 else 0.0)
                elif _res == "LOSS":
                    _profit = -_amt
                else:
                    _profit = 0.0
                overall_pnl += _profit
                if _d == _today:
                    today_pnl += _profit
                _mkt = str(_r.get("market", "")).replace("player_", "")
                _lbl = f"{_r.get('player', '')} {_mkt} {_r.get('side', '')} {_r.get('line', '')}".strip()
                _events.append((_d, round(_profit, 2), _lbl))
        # merge manual adjustments
        if ADJUSTMENTS_PATH.exists():
            _adj = pd.read_csv(ADJUSTMENTS_PATH)
            for _, _r in _adj.iterrows():
                _d = str(_r.get("date", "")).strip()
                _amt = float(_r.get("amount", 0) or 0)
                _note = str(_r.get("note", "Adjustment")).strip() or "Adjustment"
                _events.append((_d, round(_amt, 2), _note))
        # sort by date and build cumulative series
        _events.sort(key=lambda e: e[0])
        _running = 0.0
        for _d, _v, _lbl in _events:
            _running += _v
            chart_dates.append(_d)
            chart_values.append(round(_running, 2))
            trade_labels.append(_lbl)
            trade_pnl.append(_v)
    except Exception:
        pass

    return {
        "book_info": book_info,
        "total_balance": round(total_balance, 2),
        "deposit": round(DEPOSIT, 2),
        "net_profit": round(total_balance - DEPOSIT, 2) if DEPOSIT > 0 else None,
        "wagered": round(wagered, 2),
        "to_gain": round(to_gain, 2),
        "to_lose": round(to_lose, 2),
        "today_pnl": round(today_pnl, 2),
        "overall_pnl": round(overall_pnl, 2),
        "potential_bets": potential,
        "open_bets": open_bets,
        "active_books": active_books,
        "config": cfg,
        "chart_dates": chart_dates,
        "chart_values": chart_values,
        "trade_labels": trade_labels,
        "trade_pnl": trade_pnl,
    }


# ── SSE broadcast ─────────────────────────────────────────────────────────────

def broadcast_state() -> None:
    """Push current state to all connected SSE clients."""
    try:
        data = _build_state()
    except Exception:
        return
    dead = []
    for q in _clients:
        try:
            q.put_nowait(data)
        except queue.Full:
            dead.append(q)
    for q in dead:
        try:
            _clients.remove(q)
        except ValueError:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return _HTML


@app.route("/api/state")
def api_state():
    return jsonify(_build_state())


@app.route("/events")
def sse():
    q: queue.Queue = queue.Queue(maxsize=5)
    _clients.append(q)

    def generate():
        # initial snapshot
        try:
            yield f"data: {json.dumps(_build_state())}\n\n"
        except Exception as _e:
            import traceback
            print(f"[dashboard] SSE init error: {_e}", flush=True)
            traceback.print_exc()
            yield 'data: {"error":"state build failed"}\n\n'
        while True:
            try:
                data = q.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                yield 'data: {"ping":1}\n\n'
            except GeneratorExit:
                break

    resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/bet_stats")
def api_bet_stats():
    from screener import TRACKER_PATH
    if not TRACKER_PATH.exists():
        return jsonify({"rows": []})
    try:
        df = pd.read_csv(TRACKER_PATH)
        settled = df[df["result"].astype(str).str.strip().isin(["WIN", "LOSS", "PUSH"])].copy()
        if settled.empty:
            return jsonify({"rows": []})
        def pnl(row):
            result = str(row["result"]).strip().upper()
            amt = float(row.get("entered_$", 0) or 0)
            odds = float(row.get("odds", 0) or 0)
            if result == "WIN":
                return amt * (odds / 100) if odds > 0 else amt * (100 / abs(odds))
            elif result == "LOSS":
                return -amt
            return 0.0
        settled = settled.copy()
        settled["_pnl"] = settled.apply(pnl, axis=1)
        settled["_market"] = settled["market"].astype(str).str.replace(r"^player_", "", regex=True).str.strip().str.lower()
        settled["_book"] = settled["bookmaker"].astype(str).str.strip().str.lower()
        out = {}
        for group_col in ("_book", "_market"):
            rows = []
            for key, grp in settled.groupby(group_col):
                wins   = (grp["result"].str.upper() == "WIN").sum()
                losses = (grp["result"].str.upper() == "LOSS").sum()
                pushes = (grp["result"].str.upper() == "PUSH").sum()
                total  = wins + losses + pushes
                net    = round(grp["_pnl"].sum(), 2)
                roi    = round(net / grp["entered_$"].sum() * 100, 1) if grp["entered_$"].sum() else 0
                rows.append({
                    "name":    key,
                    "bets":    int(total),
                    "wins":    int(wins),
                    "losses":  int(losses),
                    "pushes":  int(pushes),
                    "win_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0,
                    "net":     net,
                    "roi":     roi,
                    "avg_edge": round(grp["edge_pct"].mean(), 1) if "edge_pct" in grp.columns else None,
                    "staked":  round(grp["entered_$"].sum(), 2),
                })
            rows.sort(key=lambda r: r["net"], reverse=True)
            out[group_col.lstrip("_")] = rows
        return jsonify(out)
    except Exception:
        import traceback; traceback.print_exc()
        return jsonify({"rows": []})


@app.route("/api/pnl_debug")
def api_pnl_debug():
    import traceback
    from screener import TRACKER_PATH, BALANCE_LOG_PATH, DEPOSIT
    out = {"tracker_exists": TRACKER_PATH.exists(), "balance_exists": BALANCE_LOG_PATH.exists(),
           "tracker_path": str(TRACKER_PATH), "error": None, "columns": [], "rows": 0,
           "settled_rows": 0, "sample_result_vals": []}
    if TRACKER_PATH.exists():
        try:
            import pandas as pd
            df = pd.read_csv(TRACKER_PATH)
            out["columns"] = df.columns.tolist()
            out["rows"] = len(df)
            out["sample_result_vals"] = df["result"].astype(str).str.strip().unique().tolist()[:10] if "result" in df.columns else []
            settled = df[df["result"].astype(str).str.strip().isin(["WIN", "LOSS", "PUSH"])]
            out["settled_rows"] = len(settled)
        except Exception:
            out["error"] = traceback.format_exc()
    return jsonify(out)


@app.route("/api/pnl_history")
def api_pnl_history():
    from screener import TRACKER_PATH, BALANCE_LOG_PATH, DEPOSIT

    # ── Try settled-bet cumulative PnL first (one point per trade) ───────────
    def _row_pnl(row):
        result = str(row["result"]).strip().upper()
        amt = float(row.get("entered_$", 0) or 0)
        odds = float(row.get("odds", 0) or 0)
        if result == "WIN":
            return amt * (odds / 100) if odds > 0 else amt * (100 / abs(odds))
        elif result == "LOSS":
            return -amt
        return 0.0

    group_labels: dict[str, list[str]] = {"book": [], "market": []}
    # per-trade data: list of (date_label, pnl, book_key, market_key)
    trades: list[tuple[str, float, str, str]] = []

    if TRACKER_PATH.exists():
        try:
            df = pd.read_csv(TRACKER_PATH)
            # Labels from ALL rows for the dropdown
            for col, gk in [("bookmaker", "book"), ("market", "market")]:
                if col in df.columns:
                    vals = df[col].dropna().astype(str).str.strip().str.lower()
                    if gk == "market":
                        vals = vals.str.replace(r"^player_", "", regex=True)
                    vals = vals[vals.str.len() > 0]
                    vals = vals[vals != "nan"]
                    group_labels[gk] = sorted(vals.unique().tolist())
            # One data point per settled trade, in CSV order (chronological)
            settled = df[df["result"].astype(str).str.strip().isin(["WIN", "LOSS", "PUSH"])].copy()
            settled["date"] = settled["date"].astype(str).str.strip()
            day_count: dict[str, int] = {}
            for _, row in settled.iterrows():
                d = str(row["date"])
                day_count[d] = day_count.get(d, 0) + 1
                label = d if day_count[d] == 1 else f"{d} #{day_count[d]}"
                pnl = _row_pnl(row)
                bk = str(row.get("bookmaker", "unknown") or "unknown").strip().lower()
                mk = str(row.get("market", "unknown") or "unknown").strip().lower()
                mk = mk.replace("player_", "", 1) if mk.startswith("player_") else mk
                if not bk or bk == "nan": bk = "unknown"
                if not mk or mk == "nan": mk = "unknown"
                trades.append((label, pnl, bk, mk))
        except Exception:
            import traceback; traceback.print_exc()

    if trades:
        labels = [t[0] for t in trades]
        # cumulative all-trades
        running = 0.0
        values = []
        for _, pnl, _, _ in trades:
            running += pnl
            values.append(round(running, 2))
        # per-group cumulative series (same length, 0 for trades not in group)
        def _group_series(gk, name):
            r = 0.0
            out = []
            for _, pnl, bk, mk in trades:
                key = bk if gk == "book" else mk
                if key == name:
                    r += pnl
                out.append(round(r, 2))
            return out

        groups: dict = {}
        for gk in ("book", "market"):
            groups[gk] = {name: _group_series(gk, name) for name in group_labels[gk]}

        return jsonify({"dates": labels, "values": values, "mode": "pnl", "groups": groups})

    # ── Fall back to balance log (shows total balance over time) ──────────────
    if BALANCE_LOG_PATH.exists():
        try:
            bl = pd.read_csv(BALANCE_LOG_PATH)
            bl = bl.dropna().sort_values("date")
            if not bl.empty:
                dates = bl["date"].astype(str).tolist()
                deposit = DEPOSIT or float(bl["balance"].iloc[0])
                values = [round(float(b) - deposit, 2) for b in bl["balance"]]
                return jsonify({"dates": dates, "values": values, "mode": "balance", "groups": {}})
        except Exception:
            pass

    return jsonify({"dates": [], "values": [], "mode": "empty", "groups": {}})


@app.route("/api/place", methods=["POST"])
def api_place():
    from screener import _bet_key, _get_book_balances, _update_book_balance, TRACKER_PATH, DEFAULT_BOOKS

    body = request.json or {}
    key_raw = body.get("key")
    amount = float(body.get("amount", 0))
    if not key_raw or not amount:
        return jsonify({"error": "key and amount required"}), 400

    key = tuple(key_raw[:4]) + (str(key_raw[4]).lower(),) if len(key_raw) >= 5 else tuple(key_raw)

    with _lock:
        bets = _st.get("latest_bets", pd.DataFrame()).copy()

    mask = [_bet_key(r) == key for _, r in bets.iterrows()]
    row_df = bets[mask].reset_index(drop=True)
    if row_df.empty:
        return jsonify({"error": "bet not found in current list"}), 404

    row = row_df.iloc[0]
    entry = {
        "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "player": row["player"],
        "market": row.get("market", ""),
        "line": row["line"],
        "side": row["side"],
        "odds": row["odds"],
        "bookmaker": row.get("bookmaker", ""),
        "edge_pct": row["edge_pct"],
        "suggested_$": row["bet_dollars"],
        "entered_$": amount,
        "result": "",
    }
    new_df = pd.DataFrame([entry])
    if TRACKER_PATH.exists():
        combined = pd.concat([pd.read_csv(TRACKER_PATH), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(TRACKER_PATH, index=False)
    _update_book_balance(entry["bookmaker"].lower(), -amount)

    with _lock:
        _st.setdefault("placed_positions", set()).add(key)

    balances = _get_book_balances()
    summary = "  ".join(
        f"{book.title()} ${balances.get(book, 0):.0f} avail" for book in DEFAULT_BOOKS
    )
    print(f"\n[tracker] Logged {row['player']} {row['side']} {row['line']} ${amount:.0f}  |  {summary}", flush=True)

    with _lock:
        _st["book_balances"] = balances

    broadcast_state()
    return jsonify({"ok": True})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    body = request.json or {}
    key_raw = body.get("key")
    if not key_raw:
        return jsonify({"error": "key required"}), 400
    key = tuple(key_raw[:4]) + (str(key_raw[4]).lower(),) if len(key_raw) >= 5 else tuple(key_raw)
    with _lock:
        _st.setdefault("skipped_keys", set()).add(key)
    broadcast_state()
    return jsonify({"ok": True})


@app.route("/api/settle", methods=["POST"])
def api_settle():
    from screener import TRACKER_PATH, _update_book_balance, _get_book_balances

    body = request.json or {}
    tracker_idx = body.get("tracker_idx")
    result = str(body.get("result", "")).strip().upper()
    if result not in ("WIN", "LOSS", "PUSH"):
        return jsonify({"error": "result must be WIN, LOSS, or PUSH"}), 400
    if tracker_idx is None:
        return jsonify({"error": "tracker_idx required"}), 400

    if not TRACKER_PATH.exists():
        return jsonify({"error": "tracker not found"}), 404

    df = pd.read_csv(TRACKER_PATH)
    idx = int(tracker_idx)
    if idx not in df.index:
        return jsonify({"error": f"row {idx} not found"}), 404

    row = df.loc[idx]
    stake = float(row.get("entered_$", 0) or 0)
    odds = float(row.get("odds", -110) or -110)
    book = str(row.get("bookmaker", "")).lower()

    df.at[idx, "result"] = result
    df.to_csv(TRACKER_PATH, index=False)

    dec = _american_to_decimal(odds)
    if result == "WIN":
        delta = stake * dec  # stake returned + profit
    elif result == "PUSH":
        delta = stake  # return stake
    else:
        delta = 0.0

    _update_book_balance(book, delta)

    with _lock:
        _st["book_balances"] = _get_book_balances()

    broadcast_state()
    return jsonify({"ok": True})


@app.route("/api/balances", methods=["POST"])
def api_balances():
    from screener import _save_state, _get_book_balances
    body = request.json or {}
    balances = _get_book_balances()
    for book, amount in body.items():
        try:
            balances[str(book).lower()] = float(amount)
        except (TypeError, ValueError):
            pass
    _save_state({"book_balances": balances})
    with _lock:
        _st["book_balances"] = balances
    broadcast_state()
    return jsonify({"ok": True})


@app.route("/api/adjustment", methods=["POST"])
def api_adjustment():
    from screener import ADJUSTMENTS_PATH
    body = request.json or {}
    try:
        amount = float(body.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid amount"}), 400
    note = str(body.get("note", "")).strip() or "Adjustment"
    d = date.today().isoformat()
    row = pd.DataFrame([{"date": d, "amount": round(amount, 2), "note": note}])
    if ADJUSTMENTS_PATH.exists():
        row = pd.concat([pd.read_csv(ADJUSTMENTS_PATH), row], ignore_index=True)
    row.to_csv(ADJUSTMENTS_PATH, index=False)
    broadcast_state()
    return jsonify({"ok": True})


@app.route("/api/config", methods=["POST"])
def api_config():
    body = request.json or {}
    with _lock:
        if "min_edge" in body:
            _st["min_edge"] = float(body["min_edge"])
        if "min_diff" in body:
            _st["min_diff"] = float(body["min_diff"])
        if "interval" in body:
            _st["interval"] = int(body["interval"])
        if "books" in body:
            _st["active_books"] = [str(b).lower() for b in body["books"]]
        if body.get("refresh_lines"):
            _st["lines_fetched_at"] = 0.0
    broadcast_state()
    return jsonify({"ok": True})


# ── Start ─────────────────────────────────────────────────────────────────────

def start_dashboard(shared_st: dict, shared_lock: threading.Lock, port: int = 5050) -> None:
    """Start Flask dashboard in a daemon thread, sharing screener _st."""
    global _st, _lock
    _st = shared_st
    _lock = shared_lock

    import logging
    import socket
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"

    print(f"[dashboard] http://localhost:{port}  |  http://{local_ip}:{port}", flush=True)
    print(f"[dashboard] Remote: ngrok http {port}", flush=True)

    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()


# ── Embedded HTML ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NBA Props</title>
<style>
:root {
  --bg:#0d1117; --card:#161b22; --border:#30363d;
  --green:#3fb950; --red:#f85149; --yellow:#d29922; --blue:#58a6ff;
  --text:#e6edf3; --muted:#8b949e; --r:8px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:12px;max-width:900px;margin:0 auto}
a{color:inherit;text-decoration:none}
/* header */
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:8px}
h1{font-size:1.15rem;color:var(--green)}
#pnl{font-size:.8rem;color:var(--muted)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
/* cards */
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
@media(min-width:520px){.cards{grid-template-columns:repeat(3,1fr)}}
@media(min-width:700px){.cards{grid-template-columns:repeat(5,1fr)}}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:14px 12px}
.card-label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
.card-value{font-size:1.35rem;font-weight:700}
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)} .blue{color:var(--blue)}
/* book pills */
.books-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;align-items:center}
.book-pill{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:5px 13px;font-size:.78rem;cursor:pointer;user-select:none;transition:border-color .15s,color .15s}
.book-pill.on{border-color:var(--green);color:var(--green)}
.book-pill.off{border-color:var(--border);color:var(--muted)}
/* sections */
section{background:var(--card);border:1px solid var(--border);border-radius:var(--r);margin-bottom:14px;overflow:hidden}
.sec-hdr{padding:10px 14px;border-bottom:1px solid var(--border);font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);display:flex;align-items:center;justify-content:space-between}
.pot-tile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;padding:10px 14px}
.pot-tile{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;gap:5px}
.player{font-weight:600;font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.edge{font-size:.68rem;background:rgba(63,185,80,.15);color:var(--green);border-radius:4px;padding:2px 6px}
.game-label{font-size:.72rem;color:var(--muted)}
.bet-meta{font-size:.78rem;color:var(--muted);margin-bottom:6px}
.over{color:var(--green)} .under{color:var(--blue)}
.actions{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
input[type=number]{background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:4px 7px;width:65px;font-size:.82rem}
button{border:none;border-radius:4px;padding:6px 13px;font-size:.78rem;font-weight:700;cursor:pointer;transition:opacity .15s}
button:active{opacity:.7}
.btn-place{background:var(--green);color:#000}
.btn-skip{background:var(--border);color:var(--muted)}
.btn-win{background:var(--green);color:#000}
.btn-loss{background:var(--red);color:#fff}
.btn-push{background:var(--yellow);color:#000}
.btn-blue{background:var(--blue);color:#000}
.empty{padding:20px 14px;color:var(--muted);font-size:.82rem;text-align:center}
/* stats table */
.stats-tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.stats-tbl th{text-align:left;padding:6px 14px;color:var(--muted);font-weight:600;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border)}
.stats-tbl td{padding:7px 14px;border-bottom:1px solid var(--border)}
.stats-tbl tr:last-child td{border-bottom:none}
.stats-tbl tr:hover td{background:rgba(255,255,255,.03)}
.stats-win{color:var(--green)}.stats-loss{color:var(--red)}.stats-neu{color:var(--muted)}
/* open bets tiles */
.open-books{display:flex;gap:10px;padding:10px 14px;flex-wrap:wrap;align-items:flex-start}
.open-book-col{flex:1;min-width:300px}
.open-book-hdr{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:5px;margin-bottom:6px}
.open-tile-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.open-tile{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px}
.tile-player{font-weight:600;font-size:.85rem;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile-meta{font-size:.72rem;color:var(--muted);margin-bottom:5px;line-height:1.4}
.tile-actions{display:flex;gap:4px}
.tile-actions button{flex:1;padding:4px 0;font-size:.7rem}
/* config */
.cfg-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;padding:14px}
@media(min-width:520px){.cfg-grid{grid-template-columns:repeat(3,1fr)}}
.cfg-item label{display:block;font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.cfg-item input{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:6px 8px;font-size:.88rem}
.cfg-foot{padding:0 14px 14px;display:flex;gap:8px;flex-wrap:wrap}
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>NBA Props</h1>
  <div id="pnl">connecting...</div>
</header>

<div class="cards">
  <div class="card"><div class="card-label">Tradeable Balance</div><div class="card-value" id="c-bal">—</div></div>
  <div class="card"><div class="card-label">Net Profit</div><div class="card-value" id="c-net">—</div></div>
  <div class="card"><div class="card-label">Wagered</div><div class="card-value yellow" id="c-wag">—</div></div>
  <div class="card"><div class="card-label">To Gain</div><div class="card-value green" id="c-gain">—</div></div>
  <div class="card"><div class="card-label">At Risk</div><div class="card-value red" id="c-risk">—</div></div>
</div>

<div class="books-row" id="books-row"></div>

<section id="chart-section">
  <div class="sec-hdr">
    <span id="chart-title">Cumulative P&amp;L</span>
    <span id="chart-range" style="font-size:.72rem;color:var(--muted)"></span>
    <div style="display:flex;gap:6px;margin-left:auto;align-items:center">
      <select id="chart-filter" onchange="setChartGroup(this.value)" style="font-size:.7rem;padding:2px 6px;border-radius:4px;border:1px solid var(--border);background:var(--card);color:var(--fg);cursor:pointer">
        <option value="all">All trades</option>
      </select>
      <button onclick="showAdjModal()" style="font-size:.7rem;padding:2px 8px;border-radius:4px;border:1px solid var(--border);background:var(--card);color:var(--muted);cursor:pointer">+ Adjustment</button>
    </div>
  </div>
  <div style="padding:14px 14px 6px"><canvas id="pnl-chart" style="width:100%;height:260px"></canvas></div>
  <div id="chart-legend" style="display:flex;flex-wrap:wrap;gap:8px;padding:0 14px 10px;font-size:.7rem"></div>
</section>

<div id="adj-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center">
  <div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:20px;width:280px;max-width:90vw">
    <div style="font-weight:600;margin-bottom:12px">Log Adjustment</div>
    <label style="font-size:.75rem;color:var(--muted)">Amount ($, negative for withdrawal)</label>
    <input id="adj-amount" type="number" step="0.01" placeholder="e.g. 25.00 or -10.00" style="width:100%;margin:4px 0 10px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:6px 8px;font-size:.88rem">
    <label style="font-size:.75rem;color:var(--muted)">Note (optional)</label>
    <input id="adj-note" type="text" placeholder="e.g. Promo bonus" style="width:100%;margin:4px 0 14px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);padding:6px 8px;font-size:.88rem">
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button onclick="hideAdjModal()" style="padding:6px 14px;border-radius:4px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer">Cancel</button>
      <button onclick="submitAdj()" style="padding:6px 14px;border-radius:4px;border:none;background:var(--green);color:#000;font-weight:600;cursor:pointer">Save</button>
    </div>
  </div>
</div>

<section>
  <div class="sec-hdr"><span>Potential Bets</span><span id="pot-count"></span></div>
  <div id="pot-list"><div class="empty">No bets yet — screener starting...</div></div>
</section>

<section>
  <div class="sec-hdr"><span>Open Bets</span><span id="open-count"></span></div>
  <div id="open-list"><div class="empty">No open bets</div></div>
</section>

<section id="stats-section">
  <div class="sec-hdr"><span>Bet Stats</span>
    <div style="display:flex;gap:4px">
      <button class="chart-tog active" onclick="setStatsGroup('book')">By Book</button>
      <button class="chart-tog" onclick="setStatsGroup('market')">By Market</button>
    </div>
  </div>
  <div id="stats-body"><div class="empty">No settled bets yet</div></div>
</section>

<section>
  <div class="sec-hdr"><span>Balances</span>
    <span style="font-size:.72rem;color:var(--muted)">tradeable balance per book</span>
  </div>
  <div class="cfg-grid" id="bal-inputs"></div>
  <div class="cfg-foot">
    <button class="btn-blue" onclick="saveBalances()">Save Balances</button>
  </div>
</section>

<section>
  <div class="sec-hdr"><span>Config</span>
    <span style="font-size:.72rem;color:var(--muted)">changes apply on next screener tick</span>
  </div>
  <div class="cfg-grid">
    <div class="cfg-item"><label>Min Edge %</label><input type="number" id="cfg-edge" step="0.5" min="0"></div>
    <div class="cfg-item"><label>Min Line Diff</label><input type="number" id="cfg-diff" step="0.5" min="0"></div>
    <div class="cfg-item"><label>Interval (s)</label><input type="number" id="cfg-int" step="30" min="30"></div>
  </div>
  <div class="cfg-foot">
    <button class="btn-blue" id="cfg-save" onclick="saveConfig()">Save Config</button>
    <button class="btn-skip" onclick="refreshLines()">Force Lines Refresh</button>
  </div>
</section>

<script>
let _state = {};

// ── Initial load ──────────────────────────────────────────────────────────────
fetch('/api/state').then(r => r.json()).then(d => {
  if (!d.error) {
    _state = d;
    try { render(d); } catch(err) { console.error('render error:', err); }
    if (d.chart_dates && d.chart_dates.length) {
      _chartDates = d.chart_dates; _chartValues = d.chart_values; _chartMode = 'pnl';
      _tradeLabels = d.trade_labels || []; _tradePnl = d.trade_pnl || [];
      const title = $('chart-title'); if (title) title.textContent = 'Cumulative P&L';
    }
  }
}).catch(() => {}).finally(() => loadChart());

// ── SSE ───────────────────────────────────────────────────────────────────────
const es = new EventSource('/events');
es.onmessage = e => {
  let d;
  try { d = JSON.parse(e.data); } catch(_) { return; }
  if (d.ping || d.error) return;
  _state = d;
  try { render(d); } catch(err) { console.error('render error:', err, d); }
  _populateChartDropdown();
  // Update chart from SSE state (guaranteed same data as overall_pnl)
  if (d.chart_dates && d.chart_dates.length) {
    _chartDates = d.chart_dates;
    _chartValues = d.chart_values;
    _tradeLabels = d.trade_labels || [];
    _tradePnl = d.trade_pnl || [];
    _chartMode = 'pnl';
    const title = $('chart-title');
    if (title) title.textContent = 'Cumulative P&L';
    drawChart();
  }
};

// ── Utils ─────────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const sgn = n => (n >= 0 ? '+' : '') + '$' + Math.abs(n).toFixed(2);
const fmtOdds = n => (n > 0 ? '+' : '') + n;
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
function safeId(key) { return JSON.stringify(key).replace(/[^a-z0-9]/gi,'_'); }

// ── Render ────────────────────────────────────────────────────────────────────
function render(d) {
  // Cards
  const balClass = d.total_balance >= 0 ? 'green' : 'red';
  $('c-bal').className = 'card-value ' + balClass;
  $('c-bal').textContent = '$' + d.total_balance.toFixed(2);
  if (d.net_profit !== null && d.net_profit !== undefined) {
    const np = d.net_profit;
    $('c-net').className = 'card-value ' + (np >= 0 ? 'green' : 'red');
    $('c-net').textContent = (np >= 0 ? '+' : '') + '$' + np.toFixed(2);
  } else {
    $('c-net').className = 'card-value muted';
    $('c-net').textContent = 'set DEPOSIT';
  }
  $('c-wag').textContent = '$' + d.wagered.toFixed(2);
  $('c-gain').textContent = '+$' + d.to_gain.toFixed(2);
  $('c-risk').textContent = '-$' + d.to_lose.toFixed(2);
  $('pnl').textContent = `Today ${sgn(d.today_pnl)}  |  Overall ${sgn(d.overall_pnl)}`;

  // Books row
  const bi = d.book_info || {};
  const row = $('books-row');
  row.innerHTML = '';
  Object.entries(bi).forEach(([book, info]) => {
    const pill = document.createElement('div');
    pill.className = 'book-pill ' + (info.active ? 'on' : 'off');
    pill.innerHTML = `<strong>${cap(book)}</strong>  $${info.avail.toFixed(2)} avail / $${info.total.toFixed(2)} total`;
    pill.onclick = () => toggleBook(book, info.active);
    row.appendChild(pill);
  });

  // Potential bets
  const pot = (d.potential_bets || []).filter(b => !b.skipped && !b.placed);
  $('pot-count').textContent = `${pot.length} available`;
  const pl = $('pot-list');
  if (!pot.length) {
    pl.innerHTML = '<div class="empty">No bets available right now</div>';
  } else {
    pl.innerHTML = '<div class="pot-tile-grid">' + pot.map(b => {
      const sid = safeId(b.key);
      const keyAttr = JSON.stringify(b.key).replace(/"/g, '&quot;');
      const sideClass = b.side === 'OVER' ? 'over' : 'under';
      return `<div class="pot-tile">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
          <span class="player">${b.player}</span>
          <span class="edge">${b.edge_pct}% edge</span>
        </div>
        <div class="tile-meta">
          ${b.market} · <span class="${sideClass}">${b.side} ${b.line}</span> · ${fmtOdds(b.odds)}<br>
          ${cap(b.bookmaker)} · pred: ${b.prediction}<br>
          <span style="color:var(--muted);font-size:.67rem">${b.game}</span>
        </div>
        <div class="actions">
          $<input type="number" id="amt-${sid}" value="${b.suggested.toFixed(2)}" min="1" step="1">
          <button class="btn-place" onclick="placeBet(${keyAttr})">Place</button>
          <button class="btn-skip" onclick="skipBet(${keyAttr})">Skip</button>
        </div>
      </div>`;
    }).join('') + '</div>';
  }

  // Open bets — tiles grouped by book
  const open = d.open_bets || [];
  $('open-count').textContent = `${open.length} pending`;
  const ol = $('open-list');
  if (!open.length) {
    ol.innerHTML = '<div class="empty">No open bets</div>';
  } else {
    const byBook = {};
    open.forEach(b => { (byBook[b.bookmaker] = byBook[b.bookmaker] || []).push(b); });
    ol.innerHTML = '<div class="open-books">' +
      Object.entries(byBook).map(([book, bets]) =>
        `<div class="open-book-col">
          <div class="open-book-hdr">${cap(book)} (${bets.length})</div>
          <div class="open-tile-grid">` +
          bets.map(b => {
            const sideClass = b.side === 'OVER' ? 'over' : 'under';
            return `<div class="open-tile">
              <div class="tile-player">${b.player}</div>
              <div class="tile-meta">
                ${b.market} · <span class="${sideClass}">${b.side} ${b.line}</span> · ${fmtOdds(b.odds)}<br>
                <strong>$${b.entered.toFixed(2)}</strong> → <span class="green">+$${b.to_win.toFixed(2)}</span> · ${b.date}
              </div>
              <div class="tile-actions">
                <button class="btn-win"  onclick="settle(${b.tracker_idx},'WIN')">W</button>
                <button class="btn-loss" onclick="settle(${b.tracker_idx},'LOSS')">L</button>
                <button class="btn-push" onclick="settle(${b.tracker_idx},'PUSH')">P</button>
              </div>
            </div>`;
          }).join('') +
          '</div>' +
        '</div>'
      ).join('') +
    '</div>';
  }

  // Config
  if (d.config) {
    $('cfg-edge').value = d.config.min_edge;
    $('cfg-diff').value = d.config.min_diff;
    $('cfg-int').value  = d.config.interval;
  }

  // Balance inputs — only repopulate if user isn't focused on one
  const focused = document.activeElement;
  const balGrid = $('bal-inputs');
  if (!balGrid.contains(focused)) {
    balGrid.innerHTML = Object.entries(d.book_info || {}).map(([book, info]) =>
      `<div class="cfg-item"><label>${cap(book)} ($)</label>` +
      `<input type="number" id="bal-${book}" value="${info.avail.toFixed(2)}" min="0" step="0.01"></div>`
    ).join('');
  }
}

// ── Actions ───────────────────────────────────────────────────────────────────
async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  if (!r.ok) alert(d.error || 'Error');
  return r.ok;
}

async function refreshState() {
  try {
    const r = await fetch('/api/state');
    if (r.ok) { const d = await r.json(); _state = d; render(d); }
  } catch(_) {}
}

async function placeBet(key) {
  const sid = safeId(key);
  const amt = parseFloat($('amt-' + sid)?.value || 0);
  if (!amt || amt <= 0) return alert('Enter an amount > 0');
  if (await post('/api/place', {key, amount: amt})) await refreshState();
}

async function skipBet(key) {
  if (await post('/api/skip', {key})) await refreshState();
}

async function settle(idx, result) {
  if (await post('/api/settle', {tracker_idx: idx, result})) { await refreshState(); await loadChart(); await loadStats(); }
}

async function toggleBook(book, currently_on) {
  const books = [...(_state.active_books || [])];
  const i = books.indexOf(book);
  if (currently_on && i >= 0) books.splice(i, 1);
  else if (!currently_on && i < 0) books.push(book);
  await post('/api/config', {books});
}

async function saveBalances() {
  const body = {};
  Object.keys(_state.book_info || {}).forEach(book => {
    const el = $('bal-' + book);
    if (el) body[book] = parseFloat(el.value) || 0;
  });
  await post('/api/balances', body);
}

async function saveConfig() {
  const ok = await post('/api/config', {
    min_edge: parseFloat($('cfg-edge').value),
    min_diff: parseFloat($('cfg-diff').value),
    interval: parseInt($('cfg-int').value),
  });
  if (ok) {
    const btn = $('cfg-save');
    btn.textContent = 'Saved \u2713';
    setTimeout(() => btn.textContent = 'Save Config', 2000);
  }
}

async function refreshLines() {
  await post('/api/config', {refresh_lines: true});
}

// ── P&L Chart ─────────────────────────────────────────────────────────────────
let _chartDates = [], _chartValues = [];
let _tradeLabels = [], _tradePnl = [];
let _chartMode = 'empty';
let _chartGroups = {};    // { book: {name: [values]}, market: {name: [values]} }
let _chartGroupDates = []; // date axis for group series (may differ from main dates)
let _chartFilter = 'all'; // 'all' | 'book:fanduel' | 'market:points' etc.

function setChartGroup(val) {
  _chartFilter = val;
  drawChart();
}

function _populateChartDropdown() {
  const sel = $('chart-filter');
  if (!sel) return;
  const prev = sel.value;

  // Collect labels from settled groups AND from all bets in _state
  const allBets = [...(_state.open_bets || []), ...(_state.settled_bets || [])];
  const fromState = {
    book:   [...new Set(allBets.map(b => (b.bookmaker||'').trim().toLowerCase()).filter(Boolean))],
    market: [...new Set(allBets.map(b => (b.market||'').trim().toLowerCase()).filter(Boolean))]
  };
  const fromGroups = {
    book:   Object.keys(_chartGroups.book   || {}),
    market: Object.keys(_chartGroups.market || {})
  };
  const merged = {
    book:   [...new Set([...fromState.book,   ...fromGroups.book  ])].sort(),
    market: [...new Set([...fromState.market, ...fromGroups.market])].sort()
  };

  sel.innerHTML = '<option value="all">All trades</option>';
  const labels = {book: 'Sportsbook', market: 'Market'};
  for (const [gk, gname] of Object.entries(labels)) {
    if (!merged[gk].length) continue;
    const og = document.createElement('optgroup');
    og.label = gname;
    merged[gk].forEach(name => {
      const opt = document.createElement('option');
      opt.value = gk + ':' + name;
      opt.textContent = name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g,' ');
      og.appendChild(opt);
    });
    sel.appendChild(og);
  }
  const validVals = [...sel.options].map(o => o.value);
  sel.value = validVals.includes(prev) ? prev : 'all';
  _chartFilter = sel.value;
}

function showAdjModal() {
  const m = $('adj-modal'); if (m) { m.style.display = 'flex'; $('adj-amount').focus(); }
}
function hideAdjModal() {
  const m = $('adj-modal'); if (m) m.style.display = 'none';
  $('adj-amount').value = ''; $('adj-note').value = '';
}
async function submitAdj() {
  const amount = parseFloat($('adj-amount').value);
  if (isNaN(amount)) { alert('Enter a valid amount'); return; }
  const note = $('adj-note').value.trim();
  await apiFetch('/api/adjustment', {amount, note});
  hideAdjModal();
}
async function loadChart() {
  try {
    const r = await fetch('/api/pnl_history');
    const d = await r.json();
    _chartDates = d.dates || [];
    _chartValues = d.values || [];
    _chartMode = d.mode || 'empty';
    _chartGroups = d.groups || {};
    _chartGroupDates = d.group_dates || d.dates || [];
    _populateChartDropdown();
    const title = $('chart-title');
    if (title) title.textContent = _chartMode === 'balance' ? 'Balance vs Deposit' : 'Cumulative P&L';
    drawChart();
  } catch(e) {}
}

function drawChart() {
  const sec = $('chart-section');
  sec.style.display = '';
  const canvas = $('pnl-chart');
  if (canvas.offsetWidth === 0) { requestAnimationFrame(drawChart); return; }

  if (!_chartDates.length) {
    const ctx2 = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth || 800; canvas.height = 120;
    ctx2.clearRect(0,0,canvas.width,canvas.height);
    ctx2.fillStyle='#8b949e'; ctx2.font='13px sans-serif'; ctx2.textAlign='center';
    ctx2.fillText('No data yet — chart appears after first screener tick', canvas.width/2, 44);
    $('chart-range').textContent = '';
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth || 800, H = 260;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const toGain = (_state.to_gain || 0);
  const toLose = (_state.to_lose || 0);
  const deposit = _state.deposit || 0;
  const n = _chartValues.length; // always use main series length for layout
  // offset all historical values by deposit so chart starts at deposit level
  const chartVals = _chartValues.map(v => v + deposit);
  const lastVal = chartVals[n - 1];
  // fork starts from actual tradeable balance, not the settled-close value
  const forkBase = (_state.total_balance !== undefined && _state.total_balance !== null)
    ? _state.total_balance : lastVal;
  const gainVal = lastVal + toGain;
  const lossVal = forkBase;

  // filtered single series (book:fanduel or market:points)
  let filteredVals = null;
  let filteredDates = null;
  if (_chartFilter !== 'all') {
    const [gk, ...rest] = _chartFilter.split(':');
    const name = rest.join(':');
    const raw = (_chartGroups[gk] || {})[name];
    if (raw && raw.length) {
      filteredVals = raw.map(v => v + deposit);
      filteredDates = _chartGroupDates;
    } else {
      const label = name.replace(/_/g, ' ');
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#8b949e'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('No settled bets for "' + label + '" yet', W / 2, H / 2 - 8);
      ctx.font = '11px sans-serif';
      ctx.fillText('P&L will appear here once bets settle', W / 2, H / 2 + 12);
      $('chart-range').textContent = '';
      return;
    }
  }

  const activeDates = filteredDates || _chartDates;
  const hasFork = !filteredVals && (toGain > 0 || toLose > 0);
  const pad = {top:12, right: hasFork ? 58 : 12, bottom:28, left:52};
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top  - pad.bottom;

  // history fills 82% of width; fork gets 18% — fixed regardless of trade count
  const histW = hasFork ? cw * 0.82 : cw;
  const toX = i => { const len = displayVals ? displayVals.length : n; return pad.left + (len <= 1 ? 0 : (i / (len - 1)) * histW); };
  const forkTipX = pad.left + cw;

  const displayVals = filteredVals || chartVals;
  const allVals = hasFork ? [...displayVals, forkBase, gainVal, lossVal] : displayVals;
  const min = Math.min(deposit, ...allVals), max = Math.max(deposit, ...allVals);
  const range = max - min || 1;
  const toY = v => pad.top  + ch - ((v - min) / range) * ch;
  const zero = toY(deposit);

  // compute ~5 nicely-rounded grid ticks spanning [min, max]
  const tickCount = 5;
  const rawStep = (max - min) / tickCount;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep || 1)));
  const step = Math.ceil(rawStep / mag) * mag || 1;
  const tickStart = Math.floor(min / step) * step;
  const ticks = [];
  for (let t = tickStart; t <= max + step * 0.01; t += step) ticks.push(parseFloat(t.toFixed(10)));

  // grid lines + y-axis labels
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
  ticks.forEach(t => {
    const y = toY(t);
    if (y < pad.top - 2 || y > pad.top + ch + 2) return;
    ctx.beginPath(); ctx.moveTo(pad.left - 4, y); ctx.lineTo(pad.left + cw, y); ctx.stroke();
    ctx.fillText('$' + t.toFixed(0), pad.left - 6, y + 3);
  });

  {
    const vals = displayVals;
    const tip = vals[n - 1];
    const lineColor = tip >= deposit ? '#3fb950' : '#f85149';
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, tip >= deposit ? 'rgba(63,185,80,.25)' : 'rgba(248,81,73,.25)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.moveTo(toX(0), zero);
    vals.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
    ctx.lineTo(toX(n - 1), zero);
    ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = lineColor; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.setLineDash([]);
    vals.forEach((v, i) => i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v)));
    ctx.stroke();
  }

  if (hasFork) {
    const forkX   = toX(n - 1);
    const tipX    = forkTipX;
    const originY = toY(displayVals[n - 1]);    // fork vertex at history line end
    const gainY   = toY(gainVal);
    const lossY   = toY(lossVal);

    // cone fill between branches
    ctx.beginPath();
    ctx.moveTo(forkX, originY);
    ctx.lineTo(tipX, gainY);
    ctx.lineTo(tipX, lossY);
    ctx.closePath();
    ctx.fillStyle = 'rgba(139,148,158,0.1)';
    ctx.fill();

    // vertical separator at fork origin
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(forkX, pad.top); ctx.lineTo(forkX, pad.top + ch); ctx.stroke();

    // gain branch (green dashed)
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(forkX, originY); ctx.lineTo(tipX, gainY); ctx.stroke();

    // loss branch (red dashed)
    ctx.strokeStyle = '#f85149';
    ctx.beginPath(); ctx.moveTo(forkX, originY); ctx.lineTo(tipX, lossY); ctx.stroke();
    ctx.restore();

    // tip labels — show absolute balance at each outcome
    ctx.font = '10px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#3fb950';
    ctx.fillText('$' + gainVal.toFixed(0), tipX + 4, gainY + 4);
    ctx.fillStyle = '#f85149';
    ctx.fillText('$' + lossVal.toFixed(0),  tipX + 4, lossY + 4);

    // "Today" x-axis label at fork tip
    ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Today', tipX, H - 6);
  }

  // x-axis labels (first, mid, last historical — skip last when fork replaces it)
  const dn = displayVals.length;
  ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
  const showIdx = [0, Math.floor((dn - 1) / 2), dn - 1].filter((v, i, a) => a.indexOf(v) === i && activeDates[v]);
  showIdx.filter(i => !(hasFork && i === dn - 1)).forEach(i => ctx.fillText(activeDates[i].slice(5), toX(i), H - 6));

  $('chart-range').textContent = (activeDates[0] || '') + ' → ' + (hasFork ? 'Today' : (activeDates[dn - 1] || '')) + '  (' + dn + ' event' + (dn === 1 ? '' : 's') + ')';
}

// Chart updates via SSE state; redraw on resize only
window.addEventListener('resize', drawChart);

// ── Bet Stats ─────────────────────────────────────────────────────────────────
let _statsData = {};
let _statsGroup = 'book';

function setStatsGroup(g) {
  _statsGroup = g;
  document.querySelectorAll('#stats-section .chart-tog').forEach(b =>
    b.classList.toggle('active', b.getAttribute('onclick').includes("'" + g + "'")));
  renderStats();
}

function renderStats() {
  const rows = (_statsData[_statsGroup] || []);
  const el = $('stats-body');
  if (!el) return;
  if (!rows.length) { el.innerHTML = '<div class="empty">No settled bets yet</div>'; return; }
  const fmtNet = n => `<span class="${n > 0 ? 'stats-win' : n < 0 ? 'stats-loss' : 'stats-neu'}">${n >= 0 ? '+' : ''}$${n.toFixed(2)}</span>`;
  el.innerHTML = `<table class="stats-tbl"><thead><tr>
    <th>${_statsGroup === 'book' ? 'Book' : 'Market'}</th>
    <th>Bets</th><th>W / L / P</th><th>Win %</th>
    <th>Net P&amp;L</th><th>ROI</th><th>Staked</th>${rows[0].avg_edge != null ? '<th>Avg Edge</th>' : ''}
  </tr></thead><tbody>` +
  rows.map(r => `<tr>
    <td style="font-weight:600;text-transform:capitalize">${r.name.replace(/_/g,' ')}</td>
    <td>${r.bets}</td>
    <td><span class="stats-win">${r.wins}W</span> / <span class="stats-loss">${r.losses}L</span>${r.pushes ? ` / <span class="stats-neu">${r.pushes}P</span>` : ''}</td>
    <td>${r.win_pct}%</td>
    <td>${fmtNet(r.net)}</td>
    <td class="${r.roi > 0 ? 'stats-win' : r.roi < 0 ? 'stats-loss' : 'stats-neu'}">${r.roi > 0 ? '+' : ''}${r.roi}%</td>
    <td>$${r.staked.toFixed(2)}</td>
    ${r.avg_edge != null ? `<td>${r.avg_edge}%</td>` : ''}
  </tr>`).join('') + '</tbody></table>';
}

async function loadStats() {
  try {
    const r = await fetch('/api/bet_stats');
    _statsData = await r.json();
    renderStats();
  } catch(e) {}
}

loadStats();
</script>
</body>
</html>"""
