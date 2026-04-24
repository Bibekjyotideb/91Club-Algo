import sqlite3
db = sqlite3.connect('data/predictions.db')

print("=== Timer breakdown ===")
for r in db.execute("SELECT timer, source, COUNT(*) FROM results GROUP BY timer, source"):
    print(f"  timer={r[0]}, source={r[1]}, count={r[2]}")

print("\n=== Last 15 results ===")
for r in db.execute("SELECT id, round_id, digit, source, timer FROM results ORDER BY id DESC LIMIT 15"):
    print(f"  id={r[0]} round={r[1][:30]} digit={r[2]} src={r[3]} timer={r[4]}")

print("\n=== Duplicate round_ids ===")
for r in db.execute("SELECT round_id, COUNT(*) as c FROM results GROUP BY round_id HAVING c > 1 LIMIT 5"):
    print(f"  {r}")

print("\n=== Total results ===")
for r in db.execute("SELECT COUNT(*) FROM results"):
    print(f"  {r[0]}")

db.close()
