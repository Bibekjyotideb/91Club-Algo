"""
SQLite database layer for storing results and predictions.
Supports per-timer separation (30sec, 1min, 3min).
"""
import aiosqlite
import os
import time
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "predictions.db")


async def get_db():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    return db


async def init_db():
    """Initialize database tables."""
    db = await get_db()
    try:
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
    finally:
        await db.close()


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
    try:
        await db.execute(
            "INSERT OR IGNORE INTO results (round_id, digit, label, color, timestamp, source, timer) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (round_id, digit, label, color, ts, source, timer)
        )
        await db.commit()
        return {"round_id": round_id, "digit": digit, "label": label, "color": color,
                "timestamp": ts, "source": source, "timer": timer}
    finally:
        await db.close()


async def add_prediction(predicted_label: str, confidence: float, round_id: str = None,
                         model_used: str = "ensemble", timer: str = "3min",
                         predicted_number: int = -1, predicted_color: str = "") -> int:
    """Record a prediction with number and color."""
    ts = time.time()
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO predictions
               (round_id, predicted_label, confidence, model_used, timestamp, timer, predicted_number, predicted_color)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (round_id, predicted_label, confidence, model_used, ts, timer, str(predicted_number), predicted_color)
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def update_prediction_result(prediction_id: int, actual_label: str,
                                    actual_digit: int = -1, actual_color: str = ""):
    """Update a prediction with the actual result including digit and color."""
    correct = None
    db = await get_db()
    try:
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
    finally:
        await db.close()


async def get_recent_results(limit: int = 100, timer: str = None) -> list:
    """Get the most recent results, optionally filtered by timer."""
    db = await get_db()
    try:
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
            # Timer column may or may not exist in older rows
            try:
                r["timer"] = row[7]
            except (IndexError, KeyError):
                r["timer"] = "3min"
            results.append(r)
        return results
    finally:
        await db.close()


async def get_all_results_ordered(timer: str = None) -> list:
    """Get all results ordered by timestamp (oldest first) for training."""
    db = await get_db()
    try:
        if timer:
            rows = await db.execute_fetchall(
                "SELECT digit, label, timestamp FROM results WHERE timer = ? ORDER BY timestamp ASC", (timer,)
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT digit, label, timestamp FROM results ORDER BY timestamp ASC"
            )
        return [{"digit": r[0], "label": r[1], "timestamp": r[2]} for r in rows]
    finally:
        await db.close()


async def get_recent_predictions(limit: int = 50, timer: str = None) -> list:
    """Get recent predictions with their outcomes, including number and color."""
    db = await get_db()
    try:
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
    finally:
        await db.close()


async def get_accuracy_stats(timer: str = None) -> dict:
    """Calculate accuracy statistics, optionally per timer."""
    db = await get_db()
    try:
        timer_filter = " WHERE timer = ?" if timer else ""
        timer_pred_filter = " WHERE timer = ? AND correct IS NOT NULL" if timer else " WHERE correct IS NOT NULL"
        params = (timer,) if timer else ()

        # Overall stats
        total_row = await db.execute_fetchall(
            f"SELECT COUNT(*), SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) FROM predictions{timer_pred_filter}",
            params
        )
        total = total_row[0][0] if total_row[0][0] else 0
        correct = total_row[0][1] if total_row[0][1] else 0

        # Last 50 predictions
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

        # Last 20 predictions
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

        # Total results count
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
    finally:
        await db.close()


async def get_result_count(timer: str = None) -> int:
    """Get total number of results stored."""
    db = await get_db()
    try:
        if timer:
            row = await db.execute_fetchall("SELECT COUNT(*) FROM results WHERE timer = ?", (timer,))
        else:
            row = await db.execute_fetchall("SELECT COUNT(*) FROM results")
        return row[0][0] if row else 0
    finally:
        await db.close()


async def get_timer_counts() -> dict:
    """Get result counts per timer."""
    db = await get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT timer, COUNT(*) FROM results GROUP BY timer ORDER BY COUNT(*) DESC"
        )
        return {row[0]: row[1] for row in rows}
    finally:
        await db.close()
