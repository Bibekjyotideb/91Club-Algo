"""
SQLite database layer for storing results and predictions.
Supports per-timer separation (30sec, 1min, 3min).
Uses a persistent connection with WAL mode for fast concurrent reads.
"""
import aiosqlite
import os
import time
import datetime
from typing import Optional

from config import DB_PATH

# Persistent connection — avoids opening/closing on every query
_db_connection = None


async def get_db():
    """Get or create persistent database connection."""
    global _db_connection
    if _db_connection is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _db_connection = await aiosqlite.connect(DB_PATH)
        _db_connection.row_factory = aiosqlite.Row
        await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _db_connection.execute("PRAGMA synchronous=NORMAL")
    return _db_connection


async def close_db():
    """Close the persistent connection (call on shutdown)."""
    global _db_connection
    if _db_connection:
        await _db_connection.close()
        _db_connection = None


async def init_db():
    """Initialize database tables."""
    db = await get_db()
    # Create tables without the timer column first (handles existing DBs)
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id TEXT UNIQUE,
            digit INTEGER NOT NULL CHECK(digit >= 0 AND digit <= 9),
            label TEXT NOT NULL CHECK(label IN ('small', 'big')),
            color TEXT DEFAULT '',
            timestamp REAL NOT NULL,
            source TEXT DEFAULT 'manual'
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_id TEXT,
            predicted_label TEXT NOT NULL CHECK(predicted_label IN ('small', 'big')),
            confidence REAL NOT NULL,
            actual_label TEXT CHECK(actual_label IN ('small', 'big', NULL)),
            correct INTEGER DEFAULT NULL,
            model_used TEXT DEFAULT 'ensemble',
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_size INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            total_predictions INTEGER NOT NULL,
            correct_predictions INTEGER NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp);
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
    """)

    # Migration: add timer column to existing tables if missing
    for table in ["results", "predictions", "model_metrics"]:
        try:
            await db.execute(f"SELECT timer FROM {table} LIMIT 1")
        except Exception:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN timer TEXT DEFAULT '3min'")

    # Migration: add number/color prediction columns
    for col, default in [("predicted_number", "'-1'"), ("predicted_color", "''"),
                         ("actual_digit", "'-1'"), ("actual_color", "''")]:
        try:
            await db.execute(f"SELECT {col} FROM predictions LIMIT 1")
        except Exception:
            await db.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT DEFAULT {default}")

    # Create timer indexes (safe now that column exists)
    await db.executescript("""
        CREATE INDEX IF NOT EXISTS idx_results_timer ON results(timer);
        CREATE INDEX IF NOT EXISTS idx_predictions_timer ON predictions(timer);
    """)

    await db.commit()


def extract_timer(source: str) -> str:
    """Extract timer name from source string like 'scraper_30sec' -> '30sec'."""
    if not source:
        return "3min"
    source = source.lower()
    if "30sec" in source:
        return "30sec"
    elif "1min" in source:
        return "1min"
    elif "3min" in source:
        return "3min"
    elif "5min" in source:
        return "5min"
    return "3min"  # default


async def add_result(digit: int, round_id: str = None, color: str = "",
                     source: str = "manual", timer: str = None) -> dict:
    """Add a new result to the database."""
    label = "small" if digit <= 4 else "big"
    ts = time.time()
    if not round_id:
        round_id = f"R{int(ts * 1000)}"

    # Auto-detect timer from source if not explicitly set
    if not timer:
        timer = extract_timer(source)

    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO results (round_id, digit, label, color, timestamp, source, timer) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (round_id, digit, label, color, ts, source, timer)
    )
    await db.commit()
    return {"round_id": round_id, "digit": digit, "label": label, "color": color,
            "timestamp": ts, "source": source, "timer": timer}


async def add_prediction(predicted_label: str, confidence: float, round_id: str = None,
                         model_used: str = "ensemble", timer: str = "3min",
                         predicted_number: int = -1, predicted_color: str = "") -> int:
    """Record a prediction with number and color."""
    ts = time.time()
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO predictions
           (round_id, predicted_label, confidence, model_used, timestamp, timer, predicted_number, predicted_color)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (round_id, predicted_label, confidence, model_used, ts, timer, str(predicted_number), predicted_color)
    )
    await db.commit()
    return cursor.lastrowid


async def update_prediction_result(prediction_id: int, actual_label: str,
                                    actual_digit: int = -1, actual_color: str = ""):
    """Update a prediction with the actual result including digit and color."""
    correct = None
    db = await get_db()
    row = await db.execute_fetchall(
        "SELECT predicted_label FROM predictions WHERE id = ?", (prediction_id,)
    )
    if row:
        correct = 1 if row[0][0] == actual_label else 0
    await db.execute(
        "UPDATE predictions SET actual_label = ?, correct = ?, actual_digit = ?, actual_color = ? WHERE id = ?",
        (actual_label, correct, str(actual_digit), actual_color, prediction_id)
    )
    await db.commit()
    return correct


async def get_recent_results(limit: int = 100, timer: str = None) -> list:
    """Get the most recent results, optionally filtered by timer."""
    db = await get_db()
    if timer:
        rows = await db.execute_fetchall(
            "SELECT * FROM results WHERE timer = ? ORDER BY timestamp DESC LIMIT ?", (timer, limit)
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT * FROM results ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
    results = []
    for row in rows:
        r = {"id": row[0], "round_id": row[1], "digit": row[2],
             "label": row[3], "color": row[4], "timestamp": row[5], "source": row[6]}
        try:
            r["timer"] = row[7]
        except (IndexError, KeyError):
            r["timer"] = "3min"
        results.append(r)
    return results


async def get_all_results_ordered(timer: str = None) -> list:
    """Get all results ordered by timestamp (oldest first) for training."""
    db = await get_db()
    if timer:
        rows = await db.execute_fetchall(
            "SELECT digit, label, timestamp FROM results WHERE timer = ? ORDER BY timestamp ASC", (timer,)
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT digit, label, timestamp FROM results ORDER BY timestamp ASC"
        )
    return [{"digit": r[0], "label": r[1], "timestamp": r[2]} for r in rows]


async def get_recent_predictions(limit: int = 50, timer: str = None) -> list:
    """Get recent predictions with their outcomes, including number and color."""
    db = await get_db()
    if timer:
        rows = await db.execute_fetchall(
            "SELECT * FROM predictions WHERE timer = ? ORDER BY timestamp DESC LIMIT ?", (timer, limit)
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
    results = []
    for row in rows:
        r = {"id": row[0], "round_id": row[1], "predicted_label": row[2],
             "confidence": row[3], "actual_label": row[4], "correct": row[5],
             "model_used": row[6], "timestamp": row[7]}
        try:
            r["timer"] = row[8]
        except (IndexError, KeyError):
            r["timer"] = "3min"
        try:
            r["predicted_number"] = row[9]
            r["predicted_color"] = row[10]
        except (IndexError, KeyError):
            r["predicted_number"] = "-1"
            r["predicted_color"] = ""
        try:
            r["actual_digit"] = row[11]
            r["actual_color"] = row[12]
        except (IndexError, KeyError):
            r["actual_digit"] = "-1"
            r["actual_color"] = ""
        results.append(r)
    return results


async def get_accuracy_stats(timer: str = None) -> dict:
    """Calculate accuracy statistics, optionally per timer."""
    db = await get_db()
    timer_filter = " WHERE timer = ?" if timer else ""
    timer_pred_filter = " WHERE timer = ? AND correct IS NOT NULL" if timer else " WHERE correct IS NOT NULL"
    params = (timer,) if timer else ()

    total_row = await db.execute_fetchall(
        f"SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) FROM predictions{timer_pred_filter}",
        params
    )
    total = total_row[0][0] if total_row[0][0] else 0
    correct = total_row[0][1] if total_row[0][1] else 0

    if timer:
        recent_row = await db.execute_fetchall(
            """SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)
               FROM (SELECT correct FROM predictions WHERE timer = ? AND correct IS NOT NULL ORDER BY timestamp DESC LIMIT 50)""",
            (timer,)
        )
    else:
        recent_row = await db.execute_fetchall(
            """SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)
               FROM (SELECT correct FROM predictions WHERE correct IS NOT NULL ORDER BY timestamp DESC LIMIT 50)"""
        )
    recent_total = recent_row[0][0] if recent_row[0][0] else 0
    recent_correct = recent_row[0][1] if recent_row[0][1] else 0

    if timer:
        last20_row = await db.execute_fetchall(
            """SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)
               FROM (SELECT correct FROM predictions WHERE timer = ? AND correct IS NOT NULL ORDER BY timestamp DESC LIMIT 20)""",
            (timer,)
        )
    else:
        last20_row = await db.execute_fetchall(
            """SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)
               FROM (SELECT correct FROM predictions WHERE correct IS NOT NULL ORDER BY timestamp DESC LIMIT 20)"""
        )
    last20_total = last20_row[0][0] if last20_row[0][0] else 0
    last20_correct = last20_row[0][1] if last20_row[0][1] else 0

    if timer:
        result_count = await db.execute_fetchall("SELECT COUNT(*) FROM results WHERE timer = ?", (timer,))
    else:
        result_count = await db.execute_fetchall("SELECT COUNT(*) FROM results")

    return {
        "total_predictions": total,
        "total_correct": correct,
        "overall_accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "recent_50_accuracy": round(recent_correct / recent_total * 100, 1) if recent_total > 0 else 0,
        "last_20_accuracy": round(last20_correct / last20_total * 100, 1) if last20_total > 0 else 0,
        "total_results": result_count[0][0] if result_count else 0,
        "timer": timer or "all"
    }


async def get_result_count(timer: str = None) -> int:
    """Get total number of results stored."""
    db = await get_db()
    if timer:
        row = await db.execute_fetchall("SELECT COUNT(*) FROM results WHERE timer = ?", (timer,))
    else:
        row = await db.execute_fetchall("SELECT COUNT(*) FROM results")
    return row[0][0] if row else 0


async def get_timer_counts() -> dict:
    """Get result counts per timer."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT timer, COUNT(*) FROM results GROUP BY timer ORDER BY COUNT(*) DESC"
    )
    return {row[0]: row[1] for row in rows}


async def get_advanced_stats(timer: str = None, tz_offset_minutes: int = -330) -> dict:
    """Calculate advanced stats: streaks, trend, best times based on local timezone."""
    db = await get_db()
    # Get predictions for the last 30 days
    now_ts = time.time()
    thirty_days_ago = now_ts - (30 * 24 * 3600)
    
    timer_filter = "timer = ? AND correct IS NOT NULL AND timestamp >= ?" if timer else "correct IS NOT NULL AND timestamp >= ?"
    params = (timer, thirty_days_ago) if timer else (thirty_days_ago,)
    
    rows = await db.execute_fetchall(
        f"SELECT correct, timestamp FROM predictions WHERE {timer_filter} ORDER BY timestamp ASC",
        params
    )
    
    def to_local_date(ts):
        local_ts = ts - (tz_offset_minutes * 60)
        return datetime.datetime.utcfromtimestamp(local_ts).date()
        
    def to_local_hour(ts):
        local_ts = ts - (tz_offset_minutes * 60)
        return datetime.datetime.utcfromtimestamp(local_ts).hour
    
    today_date = to_local_date(now_ts)
    start_of_week = today_date - datetime.timedelta(days=today_date.weekday())
    start_of_month = today_date.replace(day=1)
    
    buckets = {
        "today": {"rows": []},
        "week": {"rows": []},
        "month": {"rows": []}
    }
    
    daily_accuracy = {}
    
    for row in rows:
        correct = row[0]
        ts = row[1]
        row_date = to_local_date(ts)
        
        date_str = row_date.strftime('%m/%d')
        if date_str not in daily_accuracy:
            daily_accuracy[date_str] = {"correct": 0, "total": 0}
        daily_accuracy[date_str]["total"] += 1
        if correct == 1:
            daily_accuracy[date_str]["correct"] += 1
            
        if row_date == today_date:
            buckets["today"]["rows"].append((correct, ts))
        if row_date >= start_of_week:
            buckets["week"]["rows"].append((correct, ts))
        if row_date >= start_of_month:
            buckets["month"]["rows"].append((correct, ts))
            
    def calc_streaks(bucket_rows):
        max_w, max_l = 0, 0
        cur_w, cur_l = 0, 0
        for r in bucket_rows:
            c = r[0]
            if c == 1:
                cur_w += 1
                cur_l = 0
                if cur_w > max_w: max_w = cur_w
            else:
                cur_l += 1
                cur_w = 0
                if cur_l > max_l: max_l = cur_l
        return {"max_wins": max_w, "max_losses": max_l}
        
    def calc_best_times(bucket_rows):
        window = []
        hour_stats = {h: {"above_55_count": 0, "total": 0} for h in range(24)}
        for r in bucket_rows:
            c, ts = r
            window.append(c)
            if len(window) > 20:
                window.pop(0)
            if len(window) == 20:
                acc = sum(window) / 20.0
                hr = to_local_hour(ts)
                hour_stats[hr]["total"] += 1
                if acc > 0.55:
                    hour_stats[hr]["above_55_count"] += 1
        
        valid_hours = []
        for h, stats in hour_stats.items():
            if stats["total"] >= 5:
                pct = stats["above_55_count"] / stats["total"]
                if pct > 0:
                    valid_hours.append({"hour": h, "pct": pct, "count": stats["above_55_count"]})
        
        valid_hours.sort(key=lambda x: (x["pct"], x["count"]), reverse=True)
        
        if not valid_hours:
            return "Not enough data"
            
        top_hours = [x["hour"] for x in valid_hours[:3]]
        top_hours.sort()
        
        def format_hr(h):
            h = h % 24
            ampm = "PM" if h >= 12 else "AM"
            h12 = h % 12
            if h12 == 0: h12 = 12
            return f"{h12}{ampm}"
            
        formatted = []
        for h in top_hours:
            formatted.append(f"{format_hr(h)}-{format_hr(h+1)}")
        
        return ", ".join(formatted)

    def calc_safest_times(bucket_rows):
        hour_stats = {h: {"max_loss": 0, "current_loss": 0, "total": 0} for h in range(24)}
        for r in bucket_rows:
            c, ts = r
            hr = to_local_hour(ts)
            hour_stats[hr]["total"] += 1
            if c == 0: # Loss
                hour_stats[hr]["current_loss"] += 1
                if hour_stats[hr]["current_loss"] > hour_stats[hr]["max_loss"]:
                    hour_stats[hr]["max_loss"] = hour_stats[hr]["current_loss"]
            else: # Win
                hour_stats[hr]["current_loss"] = 0
                
        valid_hours = []
        for h, stats in hour_stats.items():
            if stats["total"] >= 20: 
                valid_hours.append({"hour": h, "max_loss": stats["max_loss"]})
                
        if not valid_hours:
            return "Not enough data"
            
        valid_hours.sort(key=lambda x: x["max_loss"])
        best_3 = valid_hours[:3]
        best_3.sort(key=lambda x: x["hour"])
        
        def format_hr(h):
            h = h % 24
            ampm = "PM" if h >= 12 else "AM"
            h12 = h % 12
            if h12 == 0: h12 = 12
            return f"{h12}{ampm}"
            
        formatted = []
        for item in best_3:
            h = item["hour"]
            ml = item["max_loss"]
            formatted.append(f"{format_hr(h)}-{format_hr(h+1)} (L:{ml})")
            
        return ", ".join(formatted)

    def calc_optimal_times(bucket_rows):
        hour_stats = {h: {"max_loss": 0, "current_loss": 0, "wins": 0, "total": 0} for h in range(24)}
        for r in bucket_rows:
            c, ts = r
            if c is None: continue
            hr = to_local_hour(ts)
            hour_stats[hr]["total"] += 1
            if c == 1: # Win
                hour_stats[hr]["wins"] += 1
                hour_stats[hr]["current_loss"] = 0
            elif c == 0: # Loss
                hour_stats[hr]["current_loss"] += 1
                if hour_stats[hr]["current_loss"] > hour_stats[hr]["max_loss"]:
                    hour_stats[hr]["max_loss"] = hour_stats[hr]["current_loss"]
                    
        valid_hours = []
        for h, stats in hour_stats.items():
            if stats["total"] >= 20:
                acc = round((stats["wins"] / stats["total"]) * 100, 1)
                valid_hours.append({"hour": h, "max_loss": stats["max_loss"], "acc": acc})
                
        if not valid_hours:
            return "Not enough data"
            
        # Always show top 3 sorted by lowest max_loss then highest accuracy.
        # Hours with acc >= 55% get a gold border ("true golden"),
        # others get a muted border ("best available").
        valid_hours.sort(key=lambda x: (x["max_loss"], -x["acc"]))
        best_3 = valid_hours[:3]

        # Sort chronologically for display
        best_3.sort(key=lambda x: x["hour"])

        def format_hr(h):
            h = h % 24
            ampm = "PM" if h >= 12 else "AM"
            h12 = h % 12
            if h12 == 0: h12 = 12
            return f"{h12}{ampm}"

        formatted = []
        for item in best_3:
            h = item["hour"]
            ml = item["max_loss"]
            acc = item["acc"]
            is_golden = acc >= 55.0
            border = "rgba(245,158,11,0.6)" if is_golden else "rgba(255,255,255,0.1)"
            label = "<span style='font-size:0.65rem;color:#f59e0b;'>&#9733; GOLDEN</span><br>" if is_golden else ""
            formatted.append(
                f"{label}{format_hr(h)}-{format_hr(h+1)} "
                f"<br><span style='color:#10b981;font-size:0.8rem;'>WR: {acc}%</span> "
                f"| <span style='color:#ef4444;font-size:0.8rem;'>L: {ml}</span>"
            )

        return ("<div style='display:flex; gap:10px; margin-top:5px; flex-wrap:wrap;'>" +
                "".join([
                    f"<div style='background:rgba(255,255,255,0.05); padding:8px; border-radius:6px; "
                    f"border:1px solid {('rgba(245,158,11,0.6)' if item['acc']>=55 else 'rgba(255,255,255,0.1)')}; "
                    f"font-size:0.85rem; text-align:center;'>{x}</div>"
                    for x, item in zip(formatted, best_3)
                ]) +
                "</div>")

    def calc_golden_hours_raw(bucket_rows, min_sample=15):
        """Return every hour (with enough data) sorted by lowest max_loss streak,
        then highest accuracy. The JS client uses this list to gate paper trading."""
        hour_stats = {h: {"max_loss": 0, "current_loss": 0, "wins": 0, "total": 0}
                      for h in range(24)}
        for r in bucket_rows:
            c, ts = r
            if c is None:
                continue
            hr = to_local_hour(ts)
            hour_stats[hr]["total"] += 1
            if c == 1:
                hour_stats[hr]["wins"] += 1
                hour_stats[hr]["current_loss"] = 0
            elif c == 0:
                hour_stats[hr]["current_loss"] += 1
                if hour_stats[hr]["current_loss"] > hour_stats[hr]["max_loss"]:
                    hour_stats[hr]["max_loss"] = hour_stats[hr]["current_loss"]

        result = []
        for h, s in hour_stats.items():
            if s["total"] >= min_sample:
                acc = round((s["wins"] / s["total"]) * 100, 1)
                result.append({
                    "hour": h,
                    "max_loss": s["max_loss"],
                    "acc": acc,
                    "total": s["total"]
                })

        # Sort: lowest max_loss first, then highest accuracy
        result.sort(key=lambda x: (x["max_loss"], -x["acc"]))
        return result

    def calc_today_optimal(bucket_rows, min_sample=5):
        """Today's best hours with a lower sample threshold — for live tracking."""
        hour_stats = {h: {"max_loss": 0, "current_loss": 0, "wins": 0, "total": 0} for h in range(24)}
        for r in bucket_rows:
            c, ts = r
            if c is None: continue
            hr = to_local_hour(ts)
            hour_stats[hr]["total"] += 1
            if c == 1:
                hour_stats[hr]["wins"] += 1
                hour_stats[hr]["current_loss"] = 0
            elif c == 0:
                hour_stats[hr]["current_loss"] += 1
                if hour_stats[hr]["current_loss"] > hour_stats[hr]["max_loss"]:
                    hour_stats[hr]["max_loss"] = hour_stats[hr]["current_loss"]

        valid_hours = []
        for h, s in hour_stats.items():
            if s["total"] >= min_sample:
                acc = round((s["wins"] / s["total"]) * 100, 1)
                valid_hours.append({"hour": h, "max_loss": s["max_loss"], "acc": acc, "total": s["total"]})

        if not valid_hours:
            return "<span style='color:rgba(255,255,255,0.35);font-size:0.8rem;'>Not enough rounds today yet</span>"

        valid_hours.sort(key=lambda x: (x["max_loss"], -x["acc"]))
        best_3 = valid_hours[:3]
        best_3.sort(key=lambda x: x["hour"])

        def format_hr(h):
            h = h % 24
            ampm = "PM" if h >= 12 else "AM"
            h12 = h % 12
            if h12 == 0: h12 = 12
            return f"{h12}{ampm}"

        cards = []
        for item in best_3:
            h, ml, acc, n = item["hour"], item["max_loss"], item["acc"], item["total"]
            glow = "0.55" if acc >= 52.0 else "0.2"
            cards.append(
                f"<div style='background:rgba(6,182,212,0.07);padding:7px 10px;border-radius:6px;"
                f"border:1px solid rgba(6,182,212,{glow});font-size:0.82rem;text-align:center;'>"
                f"{format_hr(h)}-{format_hr(h+1)}"
                f"<br><span style='color:#10b981;font-size:0.78rem;'>WR: {acc}%</span>"
                f" | <span style='color:#ef4444;font-size:0.78rem;'>L:{ml}</span>"
                f"<br><span style='color:rgba(255,255,255,0.35);font-size:0.7rem;'>{n} rounds</span></div>"
            )
        return "<div style='display:flex;gap:8px;margin-top:5px;flex-wrap:wrap;'>" + "".join(cards) + "</div>"

    trend_data = []
    # Sort properly by taking the latest 10 days
    sorted_dates = sorted(daily_accuracy.keys())[-10:]
    for d in sorted_dates:
        stats = daily_accuracy[d]
        acc = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        trend_data.append({"date": d, "accuracy": acc})

    return {
        "streaks": {
            "today": calc_streaks(buckets["today"]["rows"]),
            "week": calc_streaks(buckets["week"]["rows"]),
            "month": calc_streaks(buckets["month"]["rows"]),
        },
        "best_times": {
            "week": calc_best_times(buckets["week"]["rows"]),
            "month": calc_best_times(buckets["month"]["rows"])
        },
        "safest_times": {
            "week": calc_safest_times(buckets["week"]["rows"]),
            "month": calc_safest_times(buckets["month"]["rows"])
        },
        "optimal_times": {
            "today": calc_today_optimal(buckets["today"]["rows"]),
            "week": calc_optimal_times(buckets["week"]["rows"]),
            "month": calc_optimal_times(buckets["month"]["rows"])
        },
        "safest_hours_raw": {
            "week":  calc_golden_hours_raw(buckets["week"]["rows"]),
            "month": calc_golden_hours_raw(buckets["month"]["rows"])
        },
        "trend": trend_data
    }
