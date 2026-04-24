"""Fix existing database records: set timer based on source field."""
import sqlite3

db = sqlite3.connect('data/predictions.db')

# Fix results table
print("Fixing results table...")
db.execute("UPDATE results SET timer = '30sec' WHERE source LIKE '%30sec%' AND timer = '3min'")
db.execute("UPDATE results SET timer = '1min' WHERE source LIKE '%1min%' AND timer = '3min'")
# Leave scraper_3min and bulk/manual as 3min (correct default)

print("After fix:")
for r in db.execute("SELECT timer, source, COUNT(*) FROM results GROUP BY timer, source ORDER BY timer, source"):
    print(f"  timer={r[0]}, source={r[1]}, count={r[2]}")

# Fix predictions table too
db.execute("UPDATE predictions SET timer = '30sec' WHERE round_id IN (SELECT round_id FROM results WHERE timer = '30sec')")
db.execute("UPDATE predictions SET timer = '1min' WHERE round_id IN (SELECT round_id FROM results WHERE timer = '1min')")

print("\nPredictions by timer:")
for r in db.execute("SELECT timer, COUNT(*) FROM predictions GROUP BY timer"):
    print(f"  timer={r[0]}, count={r[1]}")

db.commit()
db.close()
print("\nDone!")
