"""
Win Go API Poller — Replaces Selenium scraper entirely.

Polls the public draw API for results on all timers (30sec, 1min, 3min).
No Chrome, no login, no captcha — just HTTP requests.

Can run:
  1. As a background task inside FastAPI (preferred)
  2. Standalone: python scraper/api_poller.py
"""
import os
import sys
import time
import asyncio
import httpx
from datetime import datetime
from typing import Optional, Callable, Awaitable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API endpoints for each timer
API_ENDPOINTS = {
    "30sec": "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
    "1min":  "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "3min":  "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json",
}

# How often to poll each timer (seconds)
POLL_INTERVALS = {
    "30sec": 4,    # poll every 4s for 30sec timer
    "1min":  5,    # poll every 5s for 1min timer
    "3min":  10,   # poll every 10s for 3min timer
}

# Color mapping from API
def parse_color(color_str: str) -> str:
    """Parse API color string to our format."""
    if not color_str:
        return ""
    # API returns "green", "red", "green,violet", "red,violet"
    return color_str.strip().lower()


def digit_to_color(digit: int) -> str:
    """Derive color from digit (WinGo rules)."""
    if digit == 0:
        return "red,violet"
    elif digit == 5:
        return "green,violet"
    elif digit in (2, 4, 6, 8):
        return "red"
    else:  # 1, 3, 7, 9
        return "green"


class WinGoPoller:
    """
    Polls the WinGo draw API and detects new results.
    
    Usage:
        poller = WinGoPoller(on_new_result=my_callback)
        await poller.run()  # runs forever
    """

    def __init__(
        self,
        timers: list[str] = None,
        on_new_result: Optional[Callable] = None,
    ):
        self.timers = timers or list(API_ENDPOINTS.keys())
        self.on_new_result = on_new_result  # async callback(digit, round_id, color, timer)
        
        # Track last seen issue per timer to detect new results
        self._last_issues: dict[str, str] = {}
        self._started = False
        self._total_captured: dict[str, int] = {t: 0 for t in self.timers}

    async def fetch_results(self, timer: str) -> list[dict]:
        """Fetch latest results for a timer from the API."""
        url = API_ENDPOINTS.get(timer)
        if not url:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={"ts": int(time.time() * 1000)})
                resp.raise_for_status()
                data = resp.json()
            
            if data.get("code") != 0:
                print(f"  [API] [{timer}] API error: {data.get('msg', 'unknown')}")
                return []
            
            items = data.get("data", {}).get("list", [])
            results = []
            for item in items:
                digit = int(item.get("number", "0"))
                results.append({
                    "issueNumber": item.get("issueNumber", ""),
                    "digit": digit,
                    "color": item.get("color", digit_to_color(digit)),
                    "premium": item.get("premium", ""),
                })
            return results
            
        except httpx.TimeoutException:
            print(f"  [API] [{timer}] Request timeout")
            return []
        except Exception as e:
            print(f"  [API] [{timer}] Error: {str(e)[:80]}")
            return []

    async def poll_timer(self, timer: str):
        """Poll a single timer and detect new results."""
        results = await self.fetch_results(timer)
        if not results:
            return
        
        # Results come newest first — the first item is the latest
        latest = results[0]
        latest_issue = latest["issueNumber"]
        
        # First poll — backfill all available history from the API page
        if timer not in self._last_issues:
            self._last_issues[timer] = latest_issue
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] [{timer:>5s}] Initial: #{latest_issue[-6:]} = {latest['digit']} ({latest['color']})")
            
            # Ingest all results from this page as backfill (oldest first)
            backfill = list(reversed(results))
            backfill_count = 0
            for r in backfill:
                if self.on_new_result:
                    try:
                        await self.on_new_result(
                            digit=r["digit"],
                            round_id=r["issueNumber"],
                            color=r["color"],
                            timer=timer,
                        )
                        backfill_count += 1
                    except Exception as e:
                        pass  # duplicate round_ids will be rejected by DB
            if backfill_count > 0:
                print(f"  [{ts}] [{timer:>5s}] Backfilled {backfill_count} historical results")
            return
        
        # No new result
        if latest_issue == self._last_issues[timer]:
            return
        
        # New result detected! Find all new results since last seen
        new_results = []
        for r in results:
            if r["issueNumber"] == self._last_issues[timer]:
                break
            new_results.append(r)
        
        # Process in chronological order (oldest first)
        new_results.reverse()
        
        for r in new_results:
            digit = r["digit"]
            color = r["color"]
            issue = r["issueNumber"]
            label = "SMALL" if digit <= 4 else "BIG"
            ts = datetime.now().strftime("%H:%M:%S")
            
            self._total_captured[timer] += 1
            print(f"  [{ts}] [{timer:>5s}] >>> {digit} ({label}) [{color}]  issue: #{issue[-6:]}")
            
            # Call the callback
            if self.on_new_result:
                try:
                    await self.on_new_result(
                        digit=digit,
                        round_id=issue,
                        color=color,
                        timer=timer,
                    )
                except Exception as e:
                    print(f"  [API] Callback error: {str(e)[:80]}")
        
        self._last_issues[timer] = latest_issue

    async def _poll_loop(self, timer: str):
        """Continuously poll a single timer."""
        interval = POLL_INTERVALS.get(timer, 15)
        
        while self._started:
            try:
                await self.poll_timer(timer)
            except Exception as e:
                print(f"  [API] [{timer}] Poll error: {str(e)[:80]}")
            
            await asyncio.sleep(interval)

    async def run(self):
        """Start polling all timers concurrently."""
        self._started = True
        
        print()
        print("  +-----------------------------------------------+")
        print("  |     Win Go API Poller v1.0                    |")
        print("  |     No Chrome - No Login - Pure API           |")
        print("  +-----------------------------------------------+")
        print()
        print(f"  Timers: {', '.join(self.timers)}")
        print(f"  Poll intervals: {POLL_INTERVALS}")
        print("=" * 55)
        
        # Do an initial fetch for all timers
        for timer in self.timers:
            await self.poll_timer(timer)
        
        print()
        print("  Watching for new results... (Ctrl+C to stop)")
        print("=" * 55)
        
        # Run poll loops concurrently
        tasks = [self._poll_loop(timer) for timer in self.timers]
        await asyncio.gather(*tasks)

    def stop(self):
        """Stop the poller."""
        self._started = False
        total = sum(self._total_captured.values())
        stats = " | ".join(f"{t}: {c}" for t, c in self._total_captured.items())
        print(f"\n  Stopped. Captured: {stats} | total: {total}")

    async def backfill(self, timer: str) -> list[dict]:
        """
        Fetch the latest 10 results for a timer (useful for initial data load).
        Returns results in chronological order (oldest first).
        """
        results = await self.fetch_results(timer)
        results.reverse()  # oldest first
        return results


# ═══════════════════════════════════════════
#  STANDALONE MODE
# ═══════════════════════════════════════════

async def _standalone_callback(digit, round_id, color, timer):
    """Post result to local API server when running standalone."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post("http://127.0.0.1:8000/api/result", json={
                "digit": digit,
                "round_id": round_id,
                "color": color,
                "source": f"api_poller_{timer}",
                "timer": timer,
            })
            data = resp.json()
            if data.get("status") == "ok":
                c = data.get("result", {}).get("prediction_correct")
                if c is not None:
                    sym = "HIT!" if c else "MISS"
                    print(f"           Prev prediction: {sym}")
    except Exception as e:
        print(f"  [POST] Error posting to server: {str(e)[:60]}")


async def main():
    """Run the poller standalone (posts results to local FastAPI server)."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Win Go API Poller")
    parser.add_argument("--timer", default="all",
                        help="30sec, 1min, 3min, or 'all' (default: all)")
    args = parser.parse_args()
    
    if args.timer == "all":
        timers = ["30sec", "1min", "3min"]
    else:
        timers = [args.timer]
    
    poller = WinGoPoller(
        timers=timers,
        on_new_result=_standalone_callback,
    )
    
    try:
        await poller.run()
    except KeyboardInterrupt:
        poller.stop()


if __name__ == "__main__":
    asyncio.run(main())
