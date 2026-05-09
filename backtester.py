"""
Walk-forward backtester for the WinGo prediction system.

Strict separation: trains on initial chunk, then predicts one-at-a-time
moving forward. Online updates use only past data. No future leakage.
"""
import sqlite3
import sys
import os
import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.predictor import EnsemblePredictor


def run_backtest(timer_name='1min', train_size=500, epochs=10):
    db_path = 'vps_predictions.db'
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Please download it from the VPS first.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT digit, timestamp, label FROM results WHERE timer = ? ORDER BY timestamp ASC", (timer_name,))
    rows = c.fetchall()
    conn.close()

    if len(rows) < train_size + 100:
        print(f"Error: Not enough data. Found {len(rows)} rows, need at least {train_size + 100}.")
        return

    print(f"Loaded {len(rows)} rounds for {timer_name} timer.")

    digits = [int(r[0]) for r in rows]
    timestamps = [float(r[1]) for r in rows]
    labels = [r[2] for r in rows]

    predictor = EnsemblePredictor(timer_name)
    print(f"Initial Bulk Training on first {train_size} records... (This takes a moment)")
    predictor.train_bulk(digits[:train_size], epochs=epochs, timestamps=timestamps[:train_size])

    correct = 0
    total = 0
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0

    # Skip-mode parallel stats (abstain when should_skip=True)
    skip_correct = 0
    skip_total = 0
    skip_max_losses = 0
    skip_cur_losses = 0
    skip_cur_wins = 0
    skip_max_wins = 0
    skip_loss_streaks = []
    skipped_count = 0
    flips_taken = 0
    flips_correct = 0

    # Per-model accuracy
    per_model = {"lstm": [0, 0], "markov": [0, 0], "pattern": [0, 0], "frequency": [0, 0]}

    # Detailed tracking
    loss_streaks = []  # list of all loss streak lengths
    win_streaks = []
    high_conf_correct = 0
    high_conf_total = 0
    low_conf_correct = 0
    low_conf_total = 0

    hour_stats = {h: {"correct": 0, "total": 0, "cur_loss": 0, "max_loss": 0} for h in range(24)}

    test_size = len(digits) - train_size - 1
    print(f"\nStarting Walk-Forward Backtest on {test_size} unseen rounds...\n")

    for i in range(train_size, len(digits) - 1):
        history_digits = digits[:i]
        history_ts = timestamps[:i]

        pred = predictor.predict(history_digits, history_ts)
        pred_size = pred["prediction"]
        actual_size = labels[i]
        actual_ts = timestamps[i]

        # Record outcome so streak tracking and weight adjustment work
        predictor.record_outcome(actual_size)

        dt = datetime.datetime.fromtimestamp(actual_ts, tz=datetime.timezone.utc)
        dt += datetime.timedelta(hours=5, minutes=30)
        hr = dt.hour

        if i % 10 == 0:
            predictor.online_update(history_digits[-50:], timestamps=history_ts[-50:])

        total += 1
        hour_stats[hr]["total"] += 1

        # Track confidence levels
        conf_level = pred.get("confidence_level", "HIGH")
        should_skip = pred.get("should_skip", False)
        was_flipped = pred.get("flipped", False)
        is_correct = (pred_size == actual_size)

        if was_flipped:
            flips_taken += 1
            if is_correct:
                flips_correct += 1

        # Per-model raw accuracy (each model's standalone direction call)
        actual_is_big = (actual_size == "big")
        for mname, mdata in pred.get("models", {}).items():
            if mname not in per_model:
                continue
            model_big = mdata.get("prob_big", 0.5) >= 0.5
            per_model[mname][1] += 1
            if model_big == actual_is_big:
                per_model[mname][0] += 1

        if conf_level == "HIGH":
            high_conf_total += 1
            if is_correct:
                high_conf_correct += 1
        else:
            low_conf_total += 1
            if is_correct:
                low_conf_correct += 1

        # Skip-mode bookkeeping: only count rounds where we DIDN'T abstain
        if should_skip:
            skipped_count += 1
        else:
            skip_total += 1
            if is_correct:
                skip_correct += 1
                skip_cur_wins += 1
                if skip_cur_losses > 0:
                    skip_loss_streaks.append(skip_cur_losses)
                skip_cur_losses = 0
                if skip_cur_wins > skip_max_wins:
                    skip_max_wins = skip_cur_wins
            else:
                skip_cur_losses += 1
                skip_cur_wins = 0
                if skip_cur_losses > skip_max_losses:
                    skip_max_losses = skip_cur_losses

        if is_correct:
            correct += 1
            cur_wins += 1
            if cur_losses > 0:
                loss_streaks.append(cur_losses)
            cur_losses = 0
            if cur_wins > max_wins:
                max_wins = cur_wins

            hour_stats[hr]["correct"] += 1
            hour_stats[hr]["cur_loss"] = 0
        else:
            cur_losses += 1
            if cur_wins > 0:
                win_streaks.append(cur_wins)
            cur_wins = 0
            if cur_losses > max_losses:
                max_losses = cur_losses

            hour_stats[hr]["cur_loss"] += 1
            if hour_stats[hr]["cur_loss"] > hour_stats[hr]["max_loss"]:
                hour_stats[hr]["max_loss"] = hour_stats[hr]["cur_loss"]

        if total % 500 == 0:
            print(f"Progress: {total}/{test_size} | Acc: {(correct/total)*100:.1f}% | Max Loss Streak: {max_losses}")

    # Final loss/win streak
    if cur_losses > 0:
        loss_streaks.append(cur_losses)
    if cur_wins > 0:
        win_streaks.append(cur_wins)

    # Calculate stats
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    streaks_above_3 = sum(1 for s in loss_streaks if s >= 3)
    streaks_above_5 = sum(1 for s in loss_streaks if s >= 5)

    print("\n" + "=" * 60)
    print("    WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Timer:                  {timer_name}")
    print(f"  Training Set:           {train_size} rounds")
    print(f"  Test Set:               {total} rounds")
    print(f"  Overall Accuracy:       {(correct/total)*100:.1f}%")
    print(f"  Max Win Streak:         {max_wins}")
    print(f"  Max Loss Streak:        {max_losses}")
    print(f"  Avg Loss Streak:        {avg_loss_streak:.1f}")
    print(f"  Loss Streaks >= 3:      {streaks_above_3}")
    print(f"  Loss Streaks >= 5:      {streaks_above_5}")

    if high_conf_total > 0:
        print(f"\n  HIGH Confidence Preds:  {high_conf_total} ({(high_conf_correct/high_conf_total)*100:.1f}% acc)")
    if low_conf_total > 0:
        print(f"  LOW Confidence Preds:   {low_conf_total} ({(low_conf_correct/low_conf_total)*100:.1f}% acc)")

    # --- Skip-mode summary (paper-trader behavior: abstain on should_skip) ---
    if skip_total > 0:
        skip_acc = (skip_correct / skip_total) * 100
        skip_avg_loss_streak = sum(skip_loss_streaks) / len(skip_loss_streaks) if skip_loss_streaks else 0
        skip_streaks_3 = sum(1 for s in skip_loss_streaks if s >= 3)
        skip_streaks_5 = sum(1 for s in skip_loss_streaks if s >= 5)
        print("\n--- SKIP-MODE (paper trader abstains when should_skip=True) ---")
        print(f"  Rounds Played:          {skip_total} (skipped {skipped_count})")
        print(f"  Accuracy (played only): {skip_acc:.1f}%")
        print(f"  Max Loss Streak:        {skip_max_losses}")
        print(f"  Avg Loss Streak:        {skip_avg_loss_streak:.1f}")
        print(f"  Loss Streaks >= 3:      {skip_streaks_3}")
        print(f"  Loss Streaks >= 5:      {skip_streaks_5}")

    if flips_taken > 0:
        flip_acc = (flips_correct / flips_taken) * 100
        print(f"\n  Streak Flips Triggered: {flips_taken} ({flip_acc:.1f}% correct)")

    print("\n--- PER-MODEL RAW ACCURACY (above 50% = real signal) ---")
    for mname, (c, t) in per_model.items():
        if t > 0:
            acc = (c / t) * 100
            edge = acc - 50
            mark = " *" if abs(edge) > 1.5 else ""
            print(f"  {mname:<10}: {acc:.2f}% (edge {edge:+.2f}pp){mark}")

    print(f"\n  Final Ensemble Weights: {predictor.weights}")

    print("\n--- SAFEST TRADING HOURS ---")
    print("-" * 60)

    valid_hours = []
    for h, stats in hour_stats.items():
        if stats["total"] >= 20:
            acc = (stats["correct"] / stats["total"]) * 100
            valid_hours.append({
                "hour": h,
                "acc": acc,
                "max_loss": stats["max_loss"],
                "total": stats["total"]
            })

    valid_hours.sort(key=lambda x: (x["max_loss"], -x["acc"]))

    for item in valid_hours[:8]:
        h = item["hour"]
        ampm = "PM" if h >= 12 else "AM"
        h12 = h % 12
        if h12 == 0:
            h12 = 12
        time_str = f"{h12}:00 {ampm}"
        print(f"  {time_str:<10} | Max Loss: {item['max_loss']:<2} | Acc: {item['acc']:.1f}% | (n={item['total']})")

    print("=" * 60)

    return {
        "accuracy": round(correct / total * 100, 1),
        "max_loss_streak": max_losses,
        "max_win_streak": max_wins,
        "avg_loss_streak": round(avg_loss_streak, 1),
        "streaks_gte_3": streaks_above_3,
        "streaks_gte_5": streaks_above_5,
        "total_tested": total,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer", default="1min")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_backtest(args.timer, args.train, args.epochs)
