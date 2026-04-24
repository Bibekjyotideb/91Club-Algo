"""Find exact selectors for game result elements."""
import re

with open("data/page_dump.html", "r", encoding="utf-8") as f:
    content = f.read()

idx = content.find("IFRAME 0")
iframe = content[idx:] if idx >= 0 else content

# We know the page has Vue data-v attributes
# Find all unique class names that contain meaningful words
all_classes = re.findall(r'class="([^"]*)"', iframe)
unique = set()
for cls in all_classes:
    for c in cls.split():
        # Remove data-v- attributes
        if not c.startswith('data-v-') and len(c) > 3:
            unique.add(c)

print("=== ALL UNIQUE CSS CLASSES (non data-v) ===")
for c in sorted(unique):
    safe = c.encode('ascii', errors='replace').decode('ascii')
    print(f"  {safe}")

# Find the result row structure
# We know format: "20260423100020371 6 Big"
# Look for the class around period numbers
print("\n\n=== HTML STRUCTURE AROUND PERIOD NUMBERS ===")
for m in re.finditer(r'2026042310002\d+', iframe):
    pos = m.start()
    snippet = iframe[max(0, pos-500):pos+300]
    safe = snippet.encode('ascii', errors='replace').decode('ascii')
    print(f"\nPeriod: {m.group()}")
    print(f"Context:")
    print(safe)
    print("---")
    break  # Just first one

# Find all class names that appear near "Big" or "Small" text
print("\n\n=== CLASSES NEAR Big/Small TEXT ===")
for m in re.finditer(r'class="([^"]*)"[^>]*>[^<]*(Big|Small)[^<]*<', iframe):
    safe = m.group(0).encode('ascii', errors='replace').decode('ascii')
    print(f"  {safe[:150]}")

# Find the timer/tab selectors
print("\n\n=== TIMER TABS ===")
for m in re.finditer(r'(?:30sec|1\s*Min|3\s*Min|5\s*Min)', iframe):
    pos = m.start()
    snippet = iframe[max(0,pos-200):pos+50]
    safe = snippet.encode('ascii', errors='replace').decode('ascii')
    print(f"\n  Timer '{m.group()}' context:")
    print(f"    ...{safe[-150:]}")
