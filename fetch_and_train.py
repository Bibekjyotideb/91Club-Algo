"""
fetch_and_train.py
==================
1. Paginate the WinGo oracle API to pull maximum available history
   for all three timers (30sec / 1min / 3min).
2. Upsert new rows into both databases
      • data/predictions.db   — live server DB
      • vps_predictions.db    — backtester DB (project root)
   using INSERT OR IGNORE so no duplicate inflation.
3. Clear stale model checkpoints to prevent weight leakage.
4. Run a strict WALK-FORWARD backtest on fresh data:
      • Train on first TRAIN_SPLIT% of rows
      • Predict remaining rows ONE-AT-A-TIME (never looks ahead)
      • Online updates use only chronologically past data
      • Fresh EnsemblePredictor instance — no checkpoint loaded
5. Retrain the PRODUCTION model on ALL available data and save
   a clean checkpoint (server loads it on next restart or /api/reset).

Usage:
    python fetch_and_train.py [--timers 1min 3min] [--pages 20] [--no-backtest]
"""
import asyncio
import sqlite3
import os
import sys
import time
import datetime
import argparse

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Constants ────────────────────────────────────────────────────────────────
API_ENDPOINTS = {
    "30sec": "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
    "1min":  "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "3min":  "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json",
}

# Timer intervals in seconds (used to compute round timestamps)
TIMER_INTERVALS = {"30sec": 30, "1min": 60, "3min": 180}

# Max pages to fetch per timer (each page = up to PAGE_SIZE results)
DEFAULT_MAX_PAGES = 20
PAGE_SIZE         = 100   # records per API page

ROOT = os.path.dirname(os.path.abspath(__file__))
VPS_DB   = os.path.join(ROOT, "vps_predictions.db")
LIVE_DB  = os.path.join(ROOT, "data", "predictions.db")
CKPT_DIR = os.path.join(ROOT, "model", "checkpoints")

TRAIN_SPLIT    = 0.75   # 75 % train, 25 % walk-forward test
BACKTEST_EPOCHS = 10
PROD_EPOCHS     = 30    # epochs for the production re-train saved to checkpoint


# ── Helpers ───────────────────────────────────────────────────────────────────
def digit_to_label(d: int) -> str:
    return "big" if d >= 5 else "small"


def digit_to_color(d: int) -> str:
    if d == 0: return "red,violet"
    if d == 5: return "green,violet"
    return "red" if d % 2 == 0 else "green"


IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def parse_issue_ts(issue: str, timer: str) -> float:
    """
    Derive a Unix timestamp from a WinGo issue number.

    Standard format: YYYYMMDD + NNNN (round number within that day).
    Examples:
      20260509001440  →  date=2026-05-09, round=1440
      202605090001    →  date=2026-05-09, round=1
    Round timestamps are computed as:
      start_of_day_IST + (round_num - 1) * timer_interval_seconds
    """
    s = str(issue).strip()
    if len(s) >= 12 and s[:4].isdigit() and s[:4] >= "2020":
        try:
            date_part  = s[:8]
            round_part = s[8:]
            # Round numbers like '100051022' have a game-ID prefix ('10005') before
            # the actual intra-day round number ('1022').  Using the full value gives
            # timestamps in year 2100+.  Cap to the last 4 digits (≤ 9999 rounds/day).
            round_num  = int(round_part[-4:]) if len(round_part) > 4 else int(round_part)
            day_start  = datetime.datetime.strptime(date_part, "%Y%m%d").replace(tzinfo=IST)
            interval   = TIMER_INTERVALS.get(timer, 60)
            return day_start.timestamp() + (round_num - 1) * interval
        except Exception:
            pass
    return time.time()


# ── Database ──────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id  TEXT UNIQUE,
    digit     INTEGER NOT NULL,
    label     TEXT NOT NULL,
    color     TEXT DEFAULT '',
    timestamp REAL NOT NULL,
    source    TEXT DEFAULT 'fetch_and_train',
    timer     TEXT DEFAULT '1min'
);
CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp);
CREATE INDEX IF NOT EXISTS idx_results_timer ON results(timer);
"""


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(_SCHEMA)
    # Migration: add any missing columns
    existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    for col, defn in [("timer", "TEXT DEFAULT '1min'"), ("color", "TEXT DEFAULT ''")]:
        if col not in existing:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} {defn}")
    conn.commit()


def count_rows(conn: sqlite3.Connection, timer: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM results WHERE timer=?", (timer,)).fetchone()[0]


def upsert_rows(conn: sqlite3.Connection, rows: list) -> int:
    """INSERT OR IGNORE — returns number of rows actually inserted (not duplicates)."""
    n = 0
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO results"
            " (round_id, digit, label, color, timestamp, source, timer)"
            " VALUES (?, ?, ?, ?, ?, 'fetch_and_train', ?)",
            (r["round_id"], r["digit"], r["label"], r["color"], r["timestamp"], r["timer"]),
        )
        if conn.execute("SELECT changes()").fetchone()[0]:
            n += 1
    conn.commit()
    return n


# ── API Fetch ─────────────────────────────────────────────────────────────────
async def fetch_page(client: httpx.AsyncClient, url: str, page_no: int) -> list:
    try:
        resp = await client.get(url, params={
            "pageNo": page_no, "pageSize": PAGE_SIZE, "ts": int(time.time() * 1000),
        }, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            return []
        return data.get("data", {}).get("list", [])
    except Exception as e:
        print(f"      Page {page_no} error: {str(e)[:80]}")
        return []


async def download_timer(timer: str, max_pages: int) -> list:
    """Paginate API and return all records chronologically (oldest first)."""
    url = API_ENDPOINTS[timer]
    records = []
    seen_issues = set()

    async with httpx.AsyncClient() as client:
        for page in range(1, max_pages + 1):
            items = await fetch_page(client, url, page)
            if not items:
                print(f"    [{timer}] Page {page}: empty — stopping.")
                break

            new_this_page = 0
            for item in items:
                issue = str(item.get("issueNumber", "")).strip()
                if not issue or issue in seen_issues:
                    continue
                seen_issues.add(issue)
                digit = int(item.get("number", item.get("digit", 0)))
                records.append({
                    "round_id":  issue,
                    "digit":     digit,
                    "label":     digit_to_label(digit),
                    "color":     item.get("color", digit_to_color(digit)),
                    "timestamp": parse_issue_ts(issue, timer),
                    "timer":     timer,
                })
                new_this_page += 1

            print(f"    [{timer}] Page {page}: +{new_this_page} (cumulative: {len(records)})")
            # API returns only the latest page; stop early if nothing new on page 2+
            if page >= 2 and new_this_page == 0:
                print(f"    [{timer}] API has no deeper history — stopping pagination.")
                break
            await asyncio.sleep(0.3)  # gentle on the oracle server

    records.sort(key=lambda x: x["timestamp"])
    return records


# ── Checkpoint management ─────────────────────────────────────────────────────
def clear_checkpoints():
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True)
        print("  Checkpoint directory created (was missing).")
        return
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt") or f.endswith(".pkl")]
    for f in files:
        os.remove(os.path.join(CKPT_DIR, f))
    if files:
        print(f"  Cleared {len(files)} stale checkpoint(s).")
    else:
        print("  No stale checkpoints found (already clean).")


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def run_clean_backtest(db_path: str, timer: str) -> dict | None:
    """
    Strict walk-forward backtest — anti-inflation guarantees:
      • Train on first TRAIN_SPLIT fraction only
      • Predict rest ONE-AT-A-TIME; outcome revealed before next prediction
      • No checkpoint loading (CKPT_DIR is cleared before this call)
      • online_update uses only indices < current position
      • Never modifies or saves the predictor
    """
    from model.predictor import EnsemblePredictor

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT digit, timestamp, label FROM results WHERE timer=? ORDER BY timestamp ASC",
        (timer,),
    ).fetchall()
    conn.close()

    n = len(rows)
    if n < 200:
        print(f"  [{timer}] Only {n} rows — need ≥ 200 to backtest. Skipping.")
        return None

    digits     = [int(r[0]) for r in rows]
    timestamps = [float(r[1]) for r in rows]
    labels     = [r[2] for r in rows]

    train_size = max(200, int(n * TRAIN_SPLIT))
    test_size  = n - train_size - 1
    print(f"\n  [{timer}] Rows: {n}  |  Train: {train_size}  |  Test: {test_size}")
    print(f"  [{timer}] Training fresh predictor (epochs={BACKTEST_EPOCHS})...")

    # ── Fresh predictor — checkpoints cleared so initialize_lstm won't load any file
    predictor = EnsemblePredictor(timer)
    predictor.train_bulk(
        digits[:train_size], epochs=BACKTEST_EPOCHS, timestamps=timestamps[:train_size]
    )

    correct = total = 0
    max_wins = max_losses = cur_wins = cur_losses = 0
    loss_streaks: list[int] = []

    # Skip-mode stats
    skip_correct = skip_total = skip_max_losses = skip_cur_losses = 0
    skip_loss_streaks: list[int] = []
    skipped_count = 0

    # Per-model accuracy
    per_model = {"lstm": [0, 0], "markov": [0, 0], "pattern": [0, 0], "frequency": [0, 0]}

    # Safest-hour tracking (IST)
    h_stats = {h: {"c": 0, "t": 0, "cur_loss": 0, "max_loss": 0} for h in range(24)}

    for i in range(train_size, n - 1):
        pred       = predictor.predict(digits[:i], timestamps[:i])
        pred_lbl   = pred["prediction"]
        actual_lbl = labels[i]
        is_correct = (pred_lbl == actual_lbl)
        skip_flag  = pred.get("should_skip", False)

        # Reveal outcome — online update uses only past data
        predictor.record_outcome(actual_lbl)
        if i % 10 == 0:
            predictor.online_update(digits[max(0, i - 50):i],
                                    timestamps=timestamps[max(0, i - 50):i])

        total += 1
        hr = datetime.datetime.fromtimestamp(timestamps[i], tz=IST).hour
        h_stats[hr]["t"] += 1

        # Per-model raw accuracy
        actual_big = (actual_lbl == "big")
        for mname, mdata in pred.get("models", {}).items():
            if mname not in per_model:
                continue
            model_big = mdata.get("prob_big", 0.5) >= 0.5
            per_model[mname][1] += 1
            if model_big == actual_big:
                per_model[mname][0] += 1

        # Overall streaks
        if is_correct:
            correct += 1; cur_wins += 1
            if cur_losses:
                loss_streaks.append(cur_losses)
            cur_losses = 0
            if cur_wins > max_wins:
                max_wins = cur_wins
            h_stats[hr]["c"] += 1
            h_stats[hr]["cur_loss"] = 0
        else:
            cur_losses += 1; cur_wins = 0
            if cur_losses > max_losses:
                max_losses = cur_losses
            h_stats[hr]["cur_loss"] += 1
            if h_stats[hr]["cur_loss"] > h_stats[hr]["max_loss"]:
                h_stats[hr]["max_loss"] = h_stats[hr]["cur_loss"]

        # Skip-mode streaks
        if skip_flag:
            skipped_count += 1
        else:
            skip_total += 1
            if is_correct:
                skip_correct += 1
                if skip_cur_losses:
                    skip_loss_streaks.append(skip_cur_losses)
                skip_cur_losses = 0
            else:
                skip_cur_losses += 1
                if skip_cur_losses > skip_max_losses:
                    skip_max_losses = skip_cur_losses

        if total % 500 == 0:
            print(f"    {total}/{test_size} | Acc {correct/total*100:.1f}% | MaxLoss {max_losses}")

    if cur_losses:
        loss_streaks.append(cur_losses)

    avg_loss = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0

    # Safest hours
    valid_hours = []
    for h, s in h_stats.items():
        if s["t"] >= 20:
            valid_hours.append({
                "hour": h, "max_loss": s["max_loss"],
                "acc": s["c"] / s["t"] * 100, "total": s["t"],
            })
    valid_hours.sort(key=lambda x: (x["max_loss"], -x["acc"]))

    # ── Print results ────────────────────────────────────────────────────────
    W = 62
    print("\n" + "=" * W)
    print(f"  WALK-FORWARD BACKTEST — {timer.upper()}")
    print("=" * W)
    print(f"  Training rows       : {train_size}  (no future data used)")
    print(f"  Test rows           : {total}")
    print(f"  Overall Accuracy    : {correct/total*100:.1f}%  ({correct}/{total})")
    print(f"  Max Win Streak      : {max_wins}")
    print(f"  Max Loss Streak     : {max_losses}")
    print(f"  Avg Loss Streak     : {avg_loss:.2f}")
    print(f"  Loss Streaks ≥ 3    : {sum(1 for s in loss_streaks if s >= 3)}")
    print(f"  Loss Streaks ≥ 5    : {sum(1 for s in loss_streaks if s >= 5)}")
    print(f"  Loss Streaks ≥ 7    : {sum(1 for s in loss_streaks if s >= 7)}")

    if skip_total:
        skip_avg = sum(skip_loss_streaks) / len(skip_loss_streaks) if skip_loss_streaks else 0.0
        print(f"\n  --- SKIP-MODE (abstain when AI flags low confidence) ---")
        print(f"  Played / Skipped    : {skip_total} / {skipped_count}")
        print(f"  Skip Accuracy       : {skip_correct/skip_total*100:.1f}%")
        print(f"  Skip Max Loss       : {skip_max_losses}")
        print(f"  Skip Avg Loss       : {skip_avg:.2f}")
        print(f"  Skip Loss Streaks≥3 : {sum(1 for s in skip_loss_streaks if s >= 3)}")
        print(f"  Skip Loss Streaks≥5 : {sum(1 for s in skip_loss_streaks if s >= 5)}")

    print(f"\n  --- PER-MODEL RAW ACCURACY (> 50 % = real signal) ---")
    for mname, (c, t) in per_model.items():
        if t:
            acc = c / t * 100
            print(f"  {mname:<10}: {acc:.2f}%  (edge {acc-50:+.2f} pp)")

    if valid_hours:
        print(f"\n  --- SAFEST TRADING HOURS (IST, lowest max-loss first) ---")
        print(f"  {'Hour':<10}{'Max Loss':<12}{'Accuracy':<12}{'Rounds'}")
        for item in valid_hours[:8]:
            h = item["hour"]; h12 = h % 12 or 12; ampm = "PM" if h >= 12 else "AM"
            print(f"  {h12}:00 {ampm:<5}  {item['max_loss']:<12}{item['acc']:.1f}%{'':<8}{item['total']}")

    print("=" * W)

    return {
        "timer": timer, "total": total,
        "accuracy": round(correct / total * 100, 1),
        "max_loss_streak": max_losses, "avg_loss_streak": round(avg_loss, 2),
        "safest_hours": valid_hours[:8],
    }


# ── Production retrain ────────────────────────────────────────────────────────
def retrain_production(db_path: str, timer: str):
    """
    Train on ALL available data and save a clean checkpoint.
    The live server loads this checkpoint on next restart or /api/reset.
    """
    from model.predictor import EnsemblePredictor

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT digit, timestamp FROM results WHERE timer=? ORDER BY timestamp ASC",
        (timer,),
    ).fetchall()
    conn.close()

    if len(rows) < 50:
        print(f"  [{timer}] Too few rows ({len(rows)}) — skipping production retrain.")
        return

    digits     = [int(r[0]) for r in rows]
    timestamps = [float(r[1]) for r in rows]

    print(f"  [{timer}] Production retrain on {len(digits)} rows (epochs={PROD_EPOCHS})...")
    predictor = EnsemblePredictor(timer)
    predictor.train_bulk(digits, epochs=PROD_EPOCHS, timestamps=timestamps)
    predictor.save()
    print(f"  [{timer}] Checkpoint saved to {CKPT_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Fetch WinGo oracle data, upsert into DBs, clean backtest + retrain."
    )
    parser.add_argument("--timers",        nargs="+", default=["30sec", "1min", "3min"])
    parser.add_argument("--pages",         type=int,  default=DEFAULT_MAX_PAGES,
                        help=f"Max API pages per timer (default {DEFAULT_MAX_PAGES})")
    parser.add_argument("--no-backtest",   action="store_true", help="Skip walk-forward backtest")
    parser.add_argument("--no-prod-train", action="store_true", help="Skip production checkpoint save")
    args = parser.parse_args()

    print()
    print("=" * 62)
    print("  WinGo — Fetch + Backtest + Retrain")
    print("  Anti-inflation guarantees:")
    print("    • INSERT OR IGNORE  — no duplicate rows")
    print("    • Walk-forward split — train never sees test data")
    print("    • Checkpoints cleared before backtest")
    print("    • Fresh EnsemblePredictor — no prior weight leakage")
    print("    • Online updates use only chronologically past data")
    print("=" * 62)

    # ── Open DBs ─────────────────────────────────────────────────────────────
    vps_conn  = sqlite3.connect(VPS_DB)
    live_conn = sqlite3.connect(LIVE_DB) if os.path.exists(LIVE_DB) else None
    ensure_schema(vps_conn)
    if live_conn:
        ensure_schema(live_conn)

    # ── Step 1: Fetch & upsert ────────────────────────────────────────────────
    print("\n[Step 1] Fetching from oracle API...")
    for timer in args.timers:
        before_vps  = count_rows(vps_conn,  timer)
        before_live = count_rows(live_conn, timer) if live_conn else 0
        print(f"\n  Timer={timer}  (DB has {before_vps} rows before fetch)")

        records = await download_timer(timer, args.pages)
        if not records:
            print(f"  [{timer}] No records returned by API.")
            continue

        n_vps  = upsert_rows(vps_conn,  records)
        n_live = upsert_rows(live_conn, records) if live_conn else 0
        after_vps = count_rows(vps_conn, timer)

        print(f"  [{timer}] +{n_vps} new rows → vps_predictions.db now {after_vps} rows")
        if live_conn:
            print(f"  [{timer}] +{n_live} new rows → data/predictions.db")

    vps_conn.close()
    if live_conn:
        live_conn.close()

    # ── Step 2: Clear stale checkpoints ──────────────────────────────────────
    print("\n[Step 2] Clearing stale model checkpoints...")
    clear_checkpoints()

    # ── Step 3: Walk-forward backtest — prefer live DB (more rows, correct timestamps)
    results = {}
    backtest_db = LIVE_DB if os.path.exists(LIVE_DB) else VPS_DB
    if not args.no_backtest:
        print(f"\n[Step 3] Walk-forward backtest (using {os.path.basename(backtest_db)})...")
        for timer in args.timers:
            res = run_clean_backtest(backtest_db, timer)
            if res:
                results[timer] = res

    # ── Step 4: Production retrain + save checkpoint ─────────────────────────
    if not args.no_prod_train:
        print("\n[Step 4] Clearing backtest checkpoints before production retrain...")
        clear_checkpoints()   # backtest saved a 75%-data checkpoint — wipe it first
        print("\n[Step 4] Production retrain (ALL data → save checkpoint for live server)...")
        # Retrain on the larger DB (live if exists, else vps)
        src_db = LIVE_DB if os.path.exists(LIVE_DB) else VPS_DB
        for timer in args.timers:
            retrain_production(src_db, timer)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print("\n" + "=" * 62)
        print("  SUMMARY")
        print(f"  {'Timer':<8} {'Acc':>7}  {'MaxLoss':>9}  {'AvgLoss':>9}  {'Rows':>6}")
        print("  " + "-" * 54)
        for timer, r in results.items():
            print(f"  {timer:<8} {r['accuracy']:>6.1f}%  {r['max_loss_streak']:>9}  "
                  f"{r['avg_loss_streak']:>9.2f}  {r['total']:>6}")
        print("=" * 62)

    print("\n  NOTE: Dashboard stats auto-update when the live server")
    print("  receives the next result (poller is always running).")
    print("  To force an immediate refresh, open the dashboard and")
    print("  click 'Retrain Model' — or restart the server.\n")


if __name__ == "__main__":
    asyncio.run(main())
